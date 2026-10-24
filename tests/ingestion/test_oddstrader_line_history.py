"""OddsTrader raw archive -> SBR-shaped ``lh_*`` tables.

Pins what would go wrong silently: which side is left and right per market,
replaying sub-minute updates into one tick per minute, the book slugs shared
with SBR, and the insert-only, resumable raw archive.
"""

from __future__ import annotations

import gzip
import io
import json
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from nba_ou.fetch_data.odds_oddstrader import backfill as bf
from nba_ou.fetch_data.odds_oddstrader import build as build_mod
from nba_ou.fetch_data.odds_oddstrader import store as store_mod
from nba_ou.fetch_data.odds_oddstrader.books import PAID_BOOKS, book_for_paid
from nba_ou.fetch_data.odds_oddstrader.client import OddsTraderError
from nba_ou.fetch_data.odds_oddstrader.parse import (
    BUNDLE_FORMAT,
    american_price,
    parse_bundle,
    ticks_from_history,
)
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    slugify_bookmaker,
)

# 2024-01-16 00:00 UTC is 19:00 Eastern on the 15th: the game belongs to the 15th.
TIPOFF = datetime(2024, 1, 16, 0, 0, tzinfo=UTC)
TIPOFF_MS = int(TIPOFF.timestamp() * 1000)
HOME, AWAY = 1166, 1167  # CLE, CHI


def _ms(minutes_before_tip: float) -> int:
    return TIPOFF_MS - int(minutes_before_tip * 60_000)


def _line(partid, minutes_before_tip, adj, ap, sequence=0, **extra):
    return {
        "partid": partid,
        "tim": _ms(minutes_before_tip),
        "adj": adj,
        "ap": ap,
        "sequence": sequence,
        **extra,
    }


def _event(eid=4710773, dt=TIPOFF_MS) -> dict:
    return {
        "eid": eid,
        "dt": dt,
        "des": "Chicago Bulls@Cleveland Cavaliers",
        "es": "complete",
        "scores": [
            {"partid": HOME, "val": "60", "pn": 1},
            {"partid": HOME, "val": "49", "pn": 2},
            {"partid": AWAY, "val": "50", "pn": 1},
            {"partid": AWAY, "val": "45", "pn": 2},
        ],
        "participants": [
            {"partid": HOME, "ih": True, "source": {"nam": "Cleveland", "abbr": "CLE"}},
            {"partid": AWAY, "ih": False, "source": {"nam": "Chicago", "abbr": "CHI"}},
        ],
    }


def _history(mtid, paid, lines, partid):
    return {
        "mtid": mtid,
        "paid": paid,
        "partid": partid,
        # One bucket is enough: the parser ignores the bucketing.
        "result": {"data": {"lineHistory": [{"tim": 0, "lines": lines}]}},
    }


def _bundle(event=None, history=None) -> dict:
    return {
        "format": BUNDLE_FORMAT,
        "fetched_at": "2026-10-22T00:00:00+00:00",
        "event": event or _event(),
        "current_lines": [],
        "history": (
            history
            if history is not None
            else [
                _history(
                    402,
                    106,
                    [_line(15143, 120, 222.5, -110), _line(15144, 120, 222.5, -110)],
                    [15143, 15144],
                ),
                _history(
                    401,
                    106,
                    [_line(AWAY, 120, 4, -105), _line(HOME, 120, -4, -115)],
                    [AWAY, HOME],
                ),
                _history(
                    83,
                    106,
                    [_line(AWAY, 120, 0, 150), _line(HOME, 120, 0, -180)],
                    [AWAY, HOME],
                ),
            ]
        ),
    }


class TestPrices:
    def test_american_price_is_used_as_is(self):
        assert american_price({"ap": -115, "pri": 1.87}) == -115

    def test_missing_american_price_is_derived_from_decimal(self):
        assert american_price({"ap": None, "pri": 2.5}) == 150
        assert american_price({"ap": None, "pri": 1.5}) == -200

    def test_values_inside_minus_100_to_100_are_not_prices(self):
        assert american_price({"ap": 0, "pri": None}) is None
        assert american_price({"ap": None, "pri": None}) is None


class TestTicks:
    def _ticks(self, lines, market="totals", left=15143, right=15144):
        return ticks_from_history(
            [{"tim": 0, "lines": lines}],
            market=market,
            left_partid=left,
            right_partid=right,
            book_slug="bet365",
            book_name="bet365",
            tipoff_utc=TIPOFF,
        )

    def test_one_tick_per_minute_and_the_latest_quote_wins(self):
        base = TIPOFF - timedelta(minutes=30)
        start = int(base.timestamp() * 1000)
        ticks = self._ticks(
            [
                _line(15143, 0, 222.0, -110, sequence=1) | {"tim": start + 5_000},
                _line(15144, 0, 222.0, -110, sequence=2) | {"tim": start + 5_000},
                _line(15143, 0, 223.0, -110, sequence=3) | {"tim": start + 40_000},
                _line(15144, 0, 223.0, -110, sequence=4) | {"tim": start + 40_000},
            ]
        )
        assert len(ticks) == 1
        assert ticks[0].line_ts == base
        assert ticks[0].left_line == ticks[0].right_line == 223.0
        assert ticks[0].minutes_to_tip == -30

    def test_a_one_sided_update_keeps_the_other_side(self):
        ticks = self._ticks(
            [
                _line(15143, 60, 222.0, -110),
                _line(15144, 60, 222.0, -110),
                _line(15143, 30, 222.0, -120),
            ]
        )
        assert [(t.left_price, t.right_price) for t in ticks] == [
            (-110, -110),
            (-120, -110),
        ]

    def test_order_follows_tim_not_payload_order(self):
        ticks = self._ticks(
            [_line(15143, 30, 224.0, -110), _line(15143, 60, 222.0, -110)]
        )
        assert [t.left_line for t in ticks] == [222.0, 224.0]
        assert [t.is_opener for t in ticks] == [True, False]

    def test_moneyline_carries_no_line(self):
        ticks = self._ticks(
            [_line(AWAY, 60, 0, 150), _line(HOME, 60, 0, -180)],
            market="money_line",
            left=AWAY,
            right=HOME,
        )
        assert (ticks[0].left_line, ticks[0].right_line) == (None, None)
        assert (ticks[0].left_price, ticks[0].right_price) == (150, -180)

    def test_other_participants_are_ignored(self):
        assert self._ticks([_line(999, 60, 1.0, -110)]) == []


class TestParseBundle:
    def test_left_and_right_follow_the_store_convention(self):
        game = parse_bundle(_bundle())
        by_market = {t.market: t for t in game.ticks}
        totals = by_market["totals"]
        assert (totals.left_line, totals.right_line) == (222.5, 222.5)
        spread = by_market["point_spread"]
        # left = AWAY, right = HOME, as a handicap: the home favourite is negative.
        assert (spread.left_line, spread.right_line) == (4.0, -4.0)
        assert (spread.left_price, spread.right_price) == (-105, -115)
        moneyline = by_market["money_line"]
        assert (moneyline.left_price, moneyline.right_price) == (150, -180)

    def test_game_fields(self):
        game = parse_bundle(_bundle())
        assert game.event_id == 4710773
        assert game.game_date == date(2024, 1, 15)
        assert game.season_year == 2023
        assert (game.team_away, game.team_home) == ("Chicago", "Cleveland")
        assert (game.away_score, game.home_score) == (95, 109)
        assert {t.book_slug for t in game.ticks} == {"bet365"}

    def test_a_stored_error_is_not_parsed_as_an_empty_history(self):
        bundle = _bundle()
        bundle["history"][0]["result"] = {"errors": [{"message": "boom"}]}
        with pytest.raises(OddsTraderError):
            parse_bundle(bundle)

    def test_an_event_without_one_home_and_one_away_team_raises(self):
        event = _event()
        event["participants"][1]["ih"] = True
        with pytest.raises(OddsTraderError):
            parse_bundle(_bundle(event=event))


class TestBooks:
    @pytest.mark.parametrize(
        ("paid", "sbr_name"),
        [
            (106, "bet365"),
            (99, "DraftKings"),
            (98, "FanDuel"),
            (109, "BetMGM"),
            (113, "BetRivers"),
            (101, "Caesars"),
        ],
    )
    def test_books_sbr_carries_use_sbr_slugs(self, paid, sbr_name):
        assert book_for_paid(paid)[0] == slugify_bookmaker(sbr_name)

    def test_unknown_books_keep_their_paid(self):
        assert book_for_paid(127)[0] == "oddstrader_127"

    def test_slugs_are_unique(self):
        slugs = [slug for slug, _ in PAID_BOOKS.values()]
        assert len(slugs) == len(set(slugs))


class FakeS3:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def put_object(self, *, Bucket, Key, Body, **_):
        self.objects[Key] = Body

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise _missing()
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise _missing()
        return {}

    def get_paginator(self, _name):
        objects = self.objects

        class Paginator:
            def paginate(self, *, Bucket, Prefix):
                yield {
                    "Contents": [{"Key": k} for k in objects if k.startswith(Prefix)]
                }

        return Paginator()


def _missing():
    exc = Exception("missing")
    exc.response = {"Error": {"Code": "NoSuchKey"}}
    return exc


class FakeClient:
    """Answers like the service for one day holding ``_event()``."""

    def __init__(self, *, failing_batches=0):
        self.requests_made = 0
        self.failing_batches = failing_batches
        self.batches: list[list[str]] = []

    def query(self, query):
        self.requests_made += 1
        if "eventsByDateNew" in query:
            return {"eventsByDateNew": {"events": [_event()]}}
        return {
            "currentLines": [
                {"eid": 4710773, "mtid": 402, "paid": 106},
                {"eid": 4710773, "mtid": 401, "paid": 106},
                {"eid": 4710773, "mtid": 401, "paid": 106},
            ]
        }

    def batch(self, queries):
        self.requests_made += 1
        self.batches.append(list(queries))
        if self.failing_batches:
            self.failing_batches -= 1
            return [{"errors": [{"message": "proxy"}]} for _ in queries]
        return [{"data": {"lineHistory": []}} for _ in queries]


DAY = date(2024, 1, 15)


class TestBackfill:
    def test_history_is_asked_once_per_market_and_book_with_the_right_sides(self):
        store = store_mod.OddsTraderStore(FakeS3(), "bucket")
        client = FakeClient()
        result = bf.backfill_days([DAY], store=store, client=client, existing=set())

        assert result.games_written == 1
        (queries,) = client.batches
        assert len(queries) == 2
        assert "mtid: 401, paid: 106, partid: [1167, 1166]" in queries[0]
        assert "mtid: 402, paid: 106, partid: [15143, 15144]" in queries[1]
        bundle = store.get_raw(store_mod.game_key(DAY, 4710773))
        assert [(h["mtid"], h["paid"]) for h in bundle["history"]] == [
            (401, 106),
            (402, 106),
        ]

    def test_a_rerun_makes_no_requests(self):
        store = store_mod.OddsTraderStore(FakeS3(), "bucket")
        bf.backfill_days([DAY], store=store, client=FakeClient(), existing=set())

        client = FakeClient()
        result = bf.backfill_days([DAY], store=store, client=client)
        assert client.requests_made == 0
        assert result.games_present == 1

    def test_errored_queries_are_asked_once_more(self, monkeypatch):
        monkeypatch.setattr(bf, "RETRY_PAUSE_S", 0)
        store = store_mod.OddsTraderStore(FakeS3(), "bucket")
        client = FakeClient(failing_batches=1)
        result = bf.backfill_days([DAY], store=store, client=client, existing=set())

        assert result.games_written == 1
        assert client.batches[1] == client.batches[0]

    def test_a_failed_game_is_not_stored_and_the_run_continues(self, monkeypatch):
        monkeypatch.setattr(bf, "RETRY_PAUSE_S", 0)
        store = store_mod.OddsTraderStore(FakeS3(), "bucket")
        result = bf.backfill_days(
            [DAY, DAY + timedelta(days=1)],
            store=store,
            client=FakeClient(failing_batches=4),
            existing=set(),
        )
        assert result.games_written == 0
        assert len(result.failed_events) == 2
        assert not store.exists(store_mod.game_key(DAY, 4710773))
        assert store.exists(store_mod.events_key(DAY))

    def test_a_game_that_has_not_settled_is_left_for_later(self, monkeypatch):
        monkeypatch.setattr(bf, "now_utc", lambda: TIPOFF + timedelta(hours=1))
        store = store_mod.OddsTraderStore(FakeS3(), "bucket")
        result = bf.backfill_days(
            [DAY], store=store, client=FakeClient(), existing=set()
        )
        assert result.games_unsettled == 1
        assert result.games_written == 0

    def test_raw_objects_are_never_overwritten(self):
        store = store_mod.OddsTraderStore(FakeS3(), "bucket")
        assert store.put_raw("k", {"a": 1}) is True
        assert store.put_raw("k", {"a": 2}) is False
        assert store.get_raw("k") == {"a": 1}

    def test_season_dates_cover_october_to_september(self):
        days = bf.season_dates(2019, last_day=date(2030, 1, 1))
        assert days[0] == date(2019, 10, 1)
        assert days[-1] == date(2020, 9, 30)


def _games_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["0022300555", "0022300555"],
            "game_date": ["2024-01-15", "2024-01-15"],
            "season_year": [2023, 2023],
            "team_name": ["Cleveland Cavaliers", "Chicago Bulls"],
            "home": [True, False],
        }
    )


def _store_with(bundles: dict[int, dict]) -> store_mod.OddsTraderStore:
    s3 = FakeS3()
    for eid, bundle in bundles.items():
        key = store_mod.game_key(DAY, eid)
        s3.objects[key] = gzip.compress(json.dumps(bundle).encode())
    s3.objects[store_mod.events_key(DAY)] = gzip.compress(b"{}")
    return store_mod.OddsTraderStore(s3, "bucket")


class TestBuild:
    def test_tables_have_the_aiven_shape(self):
        store = _store_with({4710773: _bundle()})
        manifest = build_mod.build_tables(store, games_df=_games_df(), progress=False)

        assert manifest["seasons"] == [2023]
        lines = store.get_table("lh_line_2023")
        assert list(lines.columns) == [
            "game_id", "season_year", "market_id", "book_id", "line_ts",
            "mins_to_tip", "is_pregame", "is_opener",
            "left_line", "left_price", "right_line", "right_price",
        ]  # fmt: skip
        assert len(lines) == 3
        assert set(lines["book_id"]) == {106}
        totals = lines[lines["market_id"] == 1].iloc[0]
        assert totals["left_line"] == 445  # 222.5 doubled
        assert totals["mins_to_tip"] == -120
        assert bool(totals["is_pregame"]) and bool(totals["is_opener"])

        game = store.get_table("lh_game").iloc[0]
        assert (game["game_id"], game["event_id"]) == ("0022300555", 4710773)
        assert (game["team_home"], game["team_away"]) == ("CLE", "CHI")
        book = store.get_table("lh_book").iloc[0]
        assert (book["book_id"], book["slug"], book["source_book_id"]) == (
            106,
            "bet365",
            106,
        )
        meta = store.get_table("lh_load_meta").iloc[0]
        assert (meta["timezone"], meta["confidence"]) == ("UTC", "high")

    def test_a_relisted_game_keeps_the_event_with_more_ticks(self):
        thin = _bundle(event=_event(eid=1), history=[_bundle()["history"][0]])
        store = _store_with({1: thin, 4710773: _bundle()})
        manifest = build_mod.build_tables(store, games_df=_games_df(), progress=False)

        assert store.get_table("lh_game")["event_id"].tolist() == [4710773]
        assert manifest["stats_by_date_season"]["2023"]["dropped"] == {
            "duplicate_event_for_game": 1
        }

    def test_games_missing_from_nba_games_are_counted_not_stored(self):
        store = _store_with({4710773: _bundle()})
        manifest = build_mod.build_tables(
            store, games_df=_games_df().assign(game_date="2024-01-14"), progress=False
        )
        assert manifest["seasons"] == []
        assert manifest["stats_by_date_season"]["2023"]["unmatched_games"] == 1

    def test_load_ticks_decodes_like_the_aiven_reader(self):
        store = _store_with({4710773: _bundle()})
        build_mod.build_tables(store, games_df=_games_df(), progress=False)
        ticks = store_mod.load_ticks([2023], store=store)

        assert set(ticks["market"]) == {"totals", "point_spread", "money_line"}
        spread = ticks[ticks["market"] == "point_spread"].iloc[0]
        assert (spread["left_line"], spread["right_line"]) == (4.0, -4.0)
        assert spread["minutes_before_tip"] == 120.0
        assert spread["book"] == "bet365"
