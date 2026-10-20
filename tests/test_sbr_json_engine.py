"""The SBR JSON engine must hand downstream the same raw frames the DOM did.

Verified against the browser engine on 33 dates (2019-2026, incl. scheduled
games) with zero differences; these tests pin the rules that made that true.
"""

import json
from datetime import date

import pandas as pd
import pytest
from nba_ou.fetch_data.odds_sportsbook import scrape_sportsbook as s
from nba_ou.postgre_db.odds_sportsbook.update_sportsbook import (
    update_sportsbook_database as updater,
)

DAY = date(2022, 1, 12)
BOOKS = [
    "BetMGM",
    "bet365",
    "DraftKings",
    "FanDuel",
    "Fanatics Sportsbook",
    "BetRivers",
    "Caesars",
    "Hard Rock Bet",
]


def _line(**kw):
    base = dict.fromkeys(
        [
            "odds",
            "homeOdds",
            "awayOdds",
            "overOdds",
            "underOdds",
            "drawOdds",
            "homeSpread",
            "awaySpread",
            "total",
        ]
    )
    base.update(kw)
    return base


def _view(book, current, opening=None):
    return {
        "sportsbook": book,
        "currentLine": _line(**current),
        "openingLine": _line(**(opening or current)),
    }


def _game(game_id, away, home, views, opening=None, consensus=None, score=(0, 0)):
    return {
        "gameView": {
            "gameId": game_id,
            "startDate": "2022-01-13T00:00:00+00:00",
            "awayTeam": {"displayName": away},
            "homeTeam": {"displayName": home},
            "awayTeamScore": score[0],
            "homeTeamScore": score[1],
            "consensus": consensus,
        },
        "oddsViews": views,
        "openingLineViews": [opening] if opening else [],
    }


def _html(games, league="NBA"):
    payload = {
        "props": {
            "pageProps": {
                "oddsTables": [
                    {
                        "league": league,
                        "oddsTableModel": {
                            "sportsbooks": [{"name": b} for b in BOOKS],
                            "gameRows": games,
                        },
                    }
                ]
            }
        }
    }
    return (
        '<html><script id="__NEXT_DATA__" type="application/json">'
        f"{json.dumps(payload)}</script></html>"
    )


def _model(games):
    return s.parse_sbr_odds_table(_html(games))


class TestPayload:
    def test_missing_payload_is_loud(self):
        with pytest.raises(ValueError):
            s.parse_sbr_odds_table("<html>no payload</html>")

    def test_no_games_is_none(self):
        assert s.parse_sbr_odds_table(_html([])) is None

    def test_other_league_is_none(self):
        game = _game(1, "A", "B", [])
        assert s.parse_sbr_odds_table(_html([game], league="WNBA")) is None


class TestTotals:
    def _frame(self):
        views = [None] * 8
        views[0] = _view(
            "betmgm", {"total": 210.5, "overOdds": -110, "underOdds": -105}
        )
        views[5] = _view(
            "bet_rivers_co", {"total": 210, "overOdds": -112, "underOdds": -109}
        )
        views[7] = _view(
            "hardrock", {"total": 211, "overOdds": -110, "underOdds": -110}
        )
        opening = _view("betmgm", {"total": 211.5, "overOdds": -115, "underOdds": -105})
        consensus = {"overPickPercent": 53.32, "underPickPercent": 46.68}
        game = _game(251170, "Boston", "Indiana", views, opening, consensus, (119, 100))
        return s.parse_sbr_totals_rows(_model([game]), DAY)

    def test_columns_match_the_dom_extractor(self):
        df = self._frame()
        book_cols = [
            f"{b}_{k}"
            for b in [
                "betmgm",
                "bet365",
                "draftkings",
                "fanduel",
                "fanatics_sportsbook",
                "betrivers",
                "caesars",
            ]
            for k in ("line", "price")
        ]
        assert list(df.columns) == [
            "date",
            "event_id",
            "start_time",
            "matchup_url",
            "row_index",
            "team_name",
            "score",
            "consensus_pct",
            "consensus_opener_side",
            "consensus_opener_line",
            "consensus_opener_price",
            *book_cols,
        ]

    def test_values(self):
        df = self._frame()
        away, home = df.iloc[0], df.iloc[1]
        assert (away["team_name"], away["score"]) == ("Boston", "119")
        assert (away["consensus_pct"], home["consensus_pct"]) == (53, 47)
        assert (away["consensus_opener_side"], home["consensus_opener_side"]) == (
            "O",
            "U",
        )
        assert away["consensus_opener_line"] == 211.5
        assert (away["consensus_opener_price"], home["consensus_opener_price"]) == (
            -115,
            -105,
        )
        assert (away["betrivers_line"], away["betrivers_price"]) == (210, -112)
        assert home["betrivers_price"] == -109
        assert pd.isna(away["bet365_line"])

    def test_the_undrawn_eighth_book_is_not_a_column(self):
        assert not any("hard_rock" in c for c in self._frame().columns)


class TestSpread:
    def test_signs_and_price_rules(self):
        views = [None] * 8
        views[0] = _view(
            "betmgm",
            {"awaySpread": 1.5, "homeSpread": -1.5, "awayOdds": -110, "homeOdds": -110},
        )
        # A price under 90 is not a price to the DOM regex; a 0 spread renders "PK".
        views[1] = _view(
            "bet365",
            {"awaySpread": 0, "homeSpread": 0, "awayOdds": 85, "homeOdds": -105},
        )
        game = _game(1, "A", "B", views)
        df = s.parse_sbr_spread_rows(_model([game]), DAY)
        away, home = df.iloc[0], df.iloc[1]
        assert list(df.columns[4:6]) == ["row_index", "team_row"]
        assert (away["team_row"], home["team_row"]) == ("away", "home")
        assert (away["betmgm_spread_line"], home["betmgm_spread_line"]) == (1.5, -1.5)
        assert pd.isna(away["bet365_spread_price"])
        assert pd.isna(away["bet365_spread_line"])
        assert home["bet365_spread_price"] == -105


class TestMoneyline:
    def test_sentinel_is_truncated_like_the_dom(self):
        views = [None] * 8
        views[0] = _view("betmgm", {"awayOdds": 105, "homeOdds": -125})
        opening = _view("betmgm", {"awayOdds": 2147483647, "homeOdds": -10000})
        consensus = {"awayMoneyLinePickPercent": 0, "homeMoneyLinePickPercent": 0}
        game = _game(1, "A", "B", views, opening, consensus)
        df = s.parse_sbr_moneyline_rows(_model([game]), DAY)
        away, home = df.iloc[0], df.iloc[1]
        assert away["consensus_opener_price"] == 21474
        assert home["consensus_opener_price"] == -10000
        assert pd.isna(away["consensus_pct"])
        assert (away["betmgm_price"], home["betmgm_price"]) == (105, -125)


class TestNoOddsGame:
    def test_keeps_its_own_teams(self):
        views = [None] * 8
        views[0] = _view("betmgm", {"total": 220, "overOdds": -110, "underOdds": -110})
        priced = _game(1, "Atlanta", "Orlando", views)
        empty = _game(2, "Chicago", "Toronto", [None] * 8)
        df = s.parse_sbr_totals_rows(_model([priced, empty]), DAY)
        ghost = df[df["event_id"] == 2]
        assert list(ghost["team_name"]) == ["Chicago", "Toronto"]
        assert ghost["betmgm_line"].isna().all()


class TestBatchedWrites:
    def test_writes_after_each_chunk(self, monkeypatch):
        games = pd.DataFrame(
            {
                "game_id": ["1", "2", "3"],
                "game_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            }
        )
        monkeypatch.setattr(
            updater, "create_odds_sportsbook_table", lambda drop_existing: True
        )
        monkeypatch.setattr(
            updater, "load_games_for_sportsbook_update", lambda season_year: games
        )
        monkeypatch.setattr(
            updater, "get_missing_game_ids_to_scrape", lambda ids: ["1", "2", "3"]
        )

        scraped_chunks = []

        async def fake_scrape(days, *, headless, engine):
            scraped_chunks.append(list(days))
            return pd.DataFrame({"game_id": [d.isoformat() for d in days]})

        def fake_merge(scraped, games_df):
            ids = {"2024-01-01": "1", "2024-01-02": "2", "2024-01-03": "3"}
            return scraped.assign(game_id=scraped["game_id"].map(ids))

        writes = []
        monkeypatch.setattr(updater, "scrape_sportsbook_days", fake_scrape)
        monkeypatch.setattr(updater, "merge_sportsbook_with_games", fake_merge)
        monkeypatch.setattr(
            updater,
            "upsert_odds_sportsbook_df",
            lambda df, fill_columns=None: writes.append(list(df["game_id"])) or len(df),
        )

        summary = updater.update_odds_sportsbook_database(write_every_dates=2)

        assert [len(c) for c in scraped_chunks] == [2, 1]
        assert writes == [["1", "2"], ["3"]]
        assert summary["inserted_rows"] == 3


class TestAllSeasons:
    GAMES = pd.DataFrame(
        {
            "game_id": ["old", "mid", "new"],
            "game_date": pd.to_datetime(["2018-01-01", "2020-01-01", "2023-01-01"]),
        }
    )

    def _run(self, monkeypatch, **kwargs):
        monkeypatch.setattr(
            updater, "create_odds_sportsbook_table", lambda drop_existing: True
        )
        monkeypatch.setattr(
            updater, "load_games_for_sportsbook_update", lambda season_year: self.GAMES
        )
        monkeypatch.setattr(
            updater, "get_missing_game_ids_to_scrape", lambda ids: ["old", "new"]
        )
        monkeypatch.setattr(
            updater, "get_game_ids_missing_columns", lambda ids, cols: ["mid", "new"]
        )
        starts = {False: date(2019, 5, 3), True: date(2021, 10, 19)}
        monkeypatch.setattr(
            updater,
            "get_first_stored_game_date",
            lambda columns=None: starts[bool(columns)],
        )
        scraped = []

        async def fake_scrape(days, *, headless, engine):
            scraped.append(list(days))
            return pd.DataFrame()

        monkeypatch.setattr(updater, "scrape_sportsbook_days", fake_scrape)
        summary = updater.update_odds_sportsbook_database(
            backfill_books=["betrivers"], **kwargs
        )
        return summary, scraped

    def test_skips_dates_before_coverage(self, monkeypatch):
        summary, scraped = self._run(monkeypatch, skip_before_coverage=True)
        # "old" predates any stored odds; "mid" predates BetRivers.
        assert summary["missing_game_ids"] == 1
        assert summary["backfill_game_ids"] == 1
        assert scraped == [[date(2023, 1, 1)]]

    def test_single_season_mode_checks_everything(self, monkeypatch):
        summary, _ = self._run(monkeypatch)
        assert (summary["missing_game_ids"], summary["backfill_game_ids"]) == (2, 2)

    def test_dry_run_never_scrapes(self, monkeypatch):
        summary, scraped = self._run(
            monkeypatch, skip_before_coverage=True, dry_run=True
        )
        assert scraped == []
        assert summary["backfill_game_ids"] == 1
