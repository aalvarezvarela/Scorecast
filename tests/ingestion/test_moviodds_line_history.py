"""MoviOdds timelines, clocks, complete downloads and independent persistence."""

from contextlib import contextmanager
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pandas as pd
import pytest
import requests
from nba_ou.fetch_data.odds_moviodds import client as cm
from nba_ou.fetch_data.odds_moviodds.archive import bundle_key
from nba_ou.fetch_data.odds_moviodds.parse import american_odds, history_ticks
from nba_ou.postgre_db.line_history_aiven import schema as shared_schema
from nba_ou.postgre_db.line_history_aiven.fetch import GAME_COLUMNS, TICK_COLUMNS
from nba_ou.postgre_db.line_history_aiven.transform import OUTPUT_COLUMNS
from nba_ou.postgre_db.line_history_moviodds import compare, schema, store, update

from tests.support.paths import REPO_ROOT

DAY = date(2026, 10, 20)
TIP = pd.Timestamp("2026-10-21T01:30:00Z")
NOW = datetime(2026, 10, 24, 12, tzinfo=UTC)


def point(selection="Over", ts="2026-10-21T00:00:00Z", **kw):
    fields = dict(
        ts=ts,
        group_id="1453",
        market_id="1454",
        sel_id=selection,
        side=selection,
        line="Total",
        handicap="230.5",
        odds_dec=1.9,
        suspended=False,
    )
    fields.update(kw)
    return fields


def pair(ts="2026-10-21T00:00:00Z", **kw):
    return [point(s, ts, **kw) for s in ("Over", "Under")]


def parse(points, tip=TIP):
    return history_ticks(points, home="SAS", away="OKC", tipoff_utc=tip)


def target(**kw):
    fields = dict(
        game_id="0022600003",
        season_year=2026,
        game_date=DAY,
        team_home="SAS",
        team_away="OKC",
        tipoff_utc=TIP,
    )
    fields.update(kw)
    return SimpleNamespace(**fields)


def test_storage_contract_and_encoding():
    assert schema.DDL_STATEMENTS is shared_schema.DDL_STATEMENTS
    assert schema.MARKETS is shared_schema.MARKETS
    assert schema.BOOKS == {1: ("bet365", "bet365")}
    ticks = parse(pair()).ticks
    rows, games = update.build_frames(target(), "199419311", ticks)
    assert list(rows) == OUTPUT_COLUMNS
    assert rows.iloc[0].left_line == rows.iloc[0].right_line == 461
    assert rows.iloc[0].left_price == -111
    assert rows.iloc[0].mins_to_tip == -90
    assert games.iloc[0].tipoff_utc == TIP
    assert rows.iloc[0].is_opener


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.5, -200),
        (2, 100),
        (2.7, 170),
        (1.909090909, -110),
        (None, None),
        (1, None),
        (float("nan"), None),
        (500, None),
    ],
)
def test_decimal_prices(value, expected):
    assert american_odds(value) == expected


def test_orientation_and_atomic_simultaneous_changes():
    points = []
    for mid, home_line, away_line in (("1453", "-4.5", "+4.5"), ("960", None, None)):
        for side, line in (("SA Spurs", home_line), ("OKC Thunder", away_line)):
            points.append(
                point(
                    side,
                    market_id=mid,
                    sel_id=mid + side,
                    line="Line" if mid == "1453" else "To Win",
                    handicap=line,
                )
            )
    ticks = parse(list(reversed(points))).ticks
    spread = next(t for t in ticks if t.market == "point_spread")
    assert (spread.left_line, spread.right_line) == (4.5, -4.5)
    assert len(ticks) == 2
    assert all(t.is_opener for t in ticks)
    assert next(t for t in ticks if t.market == "money_line").left_line is None


def test_latest_full_timestamp_in_minute_not_input_order():
    earlier = pair("2026-10-21T00:00:01.900Z", odds_dec=1.95)
    later = pair("2026-10-21T00:00:59.100Z", odds_dec=1.8)
    ticks = parse(later + earlier).ticks
    assert len(ticks) == 1 and ticks[0].left_price == -125
    assert ticks[0].line_ts.second == 0


def test_suspension_reopening_and_no_stale_snapshot(tmp_path):
    points = pair("2026-10-21T00:00:00Z")
    points += [point("Over", "2026-10-21T00:10:00Z", suspended=True)]
    points += [point("Over", "2026-10-21T00:20:00Z")]
    parsed = parse(points)
    assert len(parsed.ticks) == 3
    assert parsed.ticks[1].left_price is parsed.ticks[1].right_price is None
    assert parsed.ticks[1].left_line is parsed.ticks[1].right_line is None
    assert sum(t.is_opener for t in parsed.ticks) == 1
    rows, _ = update.build_frames(target(), "1", parsed.ticks)
    rows.to_parquet(tmp_path / "lh_line.parquet")
    decoded = compare.ticks_from_dir(tmp_path)
    assert list(decoded) == TICK_COLUMNS
    # 75 minutes before 01:30 is 00:15, in the suspension.
    seen = compare.asof(decoded, 75)
    assert pd.isna(seen.iloc[0].left_price)
    comparison = compare.snapshot_comparison(decoded, decoded, horizons=(75, 65))
    assert comparison.moviodds_available.tolist() == [False, True]
    assert pd.isna(comparison.iloc[0].same_quote)
    # Verify the real feature snapshot builder also preserves this boundary.
    from nba_ou.data_processing.line_history.snapshots import build_snapshot_panel

    panel = build_snapshot_panel(decoded, grid=(75, 65))
    assert panel.loc[panel.snapshot_minutes == 75, "has_quote"].tolist() == [0]
    assert panel.loc[panel.snapshot_minutes == 65, "has_quote"].tolist() == [1]


def test_incoherent_handicap_emits_null_until_both_sides_change():
    points = pair()
    points += [point("Over", "2026-10-21T00:10:00Z", handicap="231.5")]
    points += [point("Under", "2026-10-21T00:20:00Z", handicap="231.5")]
    ticks = parse(points).ticks
    assert [t.left_line for t in ticks] == [230.5, None, 231.5]


def test_identity_can_be_resolved_but_future_handicap_cannot():
    first = [
        point(s, market_id=None, side="未知", line="未知", handicap=None)
        for s in ("Over", "Under")
    ]
    later = pair("2026-10-21T00:20:00Z")
    ticks = parse(first + later).ticks
    assert ticks[0].left_line is None and not ticks[0].is_opener
    assert ticks[1].left_line == 230.5 and ticks[1].is_opener


def test_swapped_historical_fields_and_numeric_line():
    old = [
        point(s, market_id=None, side="Total", line=s, handicap="229.5")
        for s in ("Over", "Under")
    ]
    ticks = parse(old).ticks
    assert len(ticks) == 1 and ticks[0].left_line == 229.5
    new = [point(s, line="231.5", handicap=None) for s in ("Over", "Under")]
    assert parse(new).ticks[0].left_line == 231.5


def test_ambiguous_selection_invalidates_only_affected_market():
    spread = [
        point("SA Spurs", market_id="1453", sel_id="bad", handicap="-2.5"),
        point(
            "OKC Thunder",
            "2026-10-21T00:01:00Z",
            market_id=None,
            line="Line",
            sel_id="bad",
            handicap="2.5",
        ),
    ]
    parsed = parse(pair() + spread)
    assert {t.market for t in parsed.ticks} == {"totals"}
    assert parsed.dropped["ambiguous_market:point_spread"] == 1


def team_points(ts, *, labels=("OKC Thunder", "SA Spurs"), away=None, home=None):
    """One spread and one moneyline snapshot; OKC (away) is the favourite."""
    away = away or {"handicap": "-2.5", "odds_dec": 1.5}
    home = home or {"handicap": "2.5", "odds_dec": 2.6}
    points = []
    for sel, label, numbers in (("a", labels[0], away), ("h", labels[1], home)):
        points.append(
            point(
                f"{sel}_spread",
                ts,
                market_id="1453",
                side=label,
                line="Line",
                handicap=numbers["handicap"],
            )
        )
        points.append(
            point(
                f"{sel}_ml",
                ts,
                market_id="960",
                side=label,
                line="To Win",
                handicap=None,
                odds_dec=numbers["odds_dec"],
            )
        )
    return points


def relabelled_history(swapped_away=None, swapped_home=None, *, usual=19):
    points = []
    for minute in range(usual):
        points += team_points(f"2026-10-21T00:{minute:02d}:00Z")
    points += team_points(
        f"2026-10-21T00:{usual:02d}:00Z",
        labels=("SA Spurs", "OKC Thunder"),
        away=swapped_away,
        home=swapped_home,
    )
    return points


def test_relabelled_selection_keeps_its_own_side():
    parsed = parse(
        relabelled_history(
            {"handicap": "-3.5", "odds_dec": 1.45},
            {"handicap": "3.5", "odds_dec": 2.75},
        )
    )
    spread = [t for t in parsed.ticks if t.market == "point_spread"]
    money = [t for t in parsed.ticks if t.market == "money_line"]
    assert (spread[-1].left_line, spread[-1].right_line) == (-3.5, 3.5)
    assert (money[-1].left_price, money[-1].right_price) == (-222, 175)
    assert parsed.repaired == {"relabelled_selection": 4}
    assert not any(k.startswith("ambiguous") for k in parsed.dropped)


@pytest.mark.parametrize(
    "swapped_away",
    [
        {"handicap": "2.5", "odds_dec": 1.5},  # spread numbers from the home side
        {"handicap": "-2.5", "odds_dec": 2.6},  # moneyline price from the home side
    ],
)
def test_relabelled_selection_with_other_side_numbers_is_ambiguous(swapped_away):
    parsed = parse(relabelled_history(swapped_away))
    market = "point_spread" if swapped_away["odds_dec"] == 1.5 else "money_line"
    assert market not in {t.market for t in parsed.ticks}
    assert parsed.dropped[f"ambiguous_market:{market}"] == 1


@pytest.mark.parametrize("numbers_swapped", [False, True])
def test_near_even_relabelling_stays_ambiguous(numbers_swapped):
    # Both sides' numbers fit either selection: a swap cannot be ruled out.
    away = {"handicap": "-0.5", "odds_dec": 1.9}
    home = {"handicap": "0.5", "odds_dec": 2.0}
    points = []
    for minute in range(19):
        points += team_points(f"2026-10-21T00:{minute:02d}:00Z", away=away, home=home)
    swapped = (home, away) if numbers_swapped else (away, home)
    points += team_points(
        "2026-10-21T00:19:00Z",
        labels=("SA Spurs", "OKC Thunder"),
        away=swapped[0],
        home=swapped[1],
    )
    parsed = parse(points)
    assert {t.market for t in parsed.ticks} == set()
    assert parsed.repaired == {}


@pytest.mark.parametrize("numbers_swapped", [False, True])
def test_relabelling_checked_against_last_complete_reference(numbers_swapped):
    away = {"handicap": "-10.5", "odds_dec": 1.1}
    home = {"handicap": "10.5", "odds_dec": 8.0}
    empty = {"handicap": "", "odds_dec": None}
    points = []
    for minute in range(18):
        points += team_points(f"2026-10-21T00:{minute:02d}:00Z", away=away, home=home)
    points += team_points("2026-10-21T00:18:00Z", away=empty, home=empty)
    swapped = (home, away) if numbers_swapped else (away, home)
    points += team_points(
        "2026-10-21T00:19:00Z",
        labels=("SA Spurs", "OKC Thunder"),
        away=swapped[0],
        home=swapped[1],
    )
    parsed = parse(points)
    spread = [t for t in parsed.ticks if t.market == "point_spread"]
    if numbers_swapped:
        assert not spread and parsed.repaired == {}
    else:
        assert (spread[-1].left_line, spread[-1].right_line) == (-10.5, 10.5)
        assert parsed.repaired == {"relabelled_selection": 4}


def test_frequent_relabelling_is_ambiguous():
    parsed = parse(relabelled_history(usual=5))
    assert {t.market for t in parsed.ticks} == set()
    assert parsed.repaired == {}


def test_exclude_live_props_and_conflicting_same_timestamp():
    points = pair() + pair(group_id="112") + pair(market_id="99999")
    points += [point("Over", odds_dec=1.8)]
    parsed = parse(points)
    assert parsed.dropped == {
        "non_main_group": 2,
        "non_main_market": 2,
        "conflicting_same_timestamp": 1,
    }
    assert parsed.ticks[0].left_price is None


@pytest.mark.parametrize(
    "tip",
    [
        "2026-11-01T05:30:00Z",
        "2026-11-01T06:30:00Z",
        "2027-03-14T06:30:00Z",
        "2027-03-14T07:30:00Z",
    ],
)
def test_dst_uses_utc_not_eastern_wall_clock(tip):
    tip = pd.Timestamp(tip)
    ticks = parse(pair((tip - pd.Timedelta(minutes=60)).isoformat()), tip).ticks
    assert ticks[0].minutes_to_tip == -60
    assert ticks[0].is_pregame


def test_naive_timestamp_refused_and_post_tip_tick_retained():
    with pytest.raises(ValueError):
        parse(pair("2026-10-21T00:00:00"))
    ticks = parse(pair() + pair("2026-10-21T01:35:00Z", odds_dec=1.8)).ticks
    assert ticks[-1].minutes_to_tip == 5 and not ticks[-1].is_pregame


def test_subminute_nba_tip_cannot_turn_post_tip_update_into_pregame():
    tip = pd.Timestamp("2026-10-21T01:30:59Z")
    ticks = parse(pair("2026-10-21T01:30:59.5Z"), tip=tip).ticks
    assert ticks[0].line_ts == pd.Timestamp("2026-10-21T01:30:00Z")
    assert not ticks[0].is_pregame and ticks[0].minutes_to_tip == 0


class FakeClient:
    def __init__(self, points=None, fail=False):
        self.records, self.days, self.fixtures = [], [], []
        self.points = pair() if points is None else points
        self.fail = fail

    def results(self, day):
        self.days.append(day)
        return {
            "results": (
                [
                    dict(
                        fixture_id="199419311",
                        league="NBA",
                        status="finished",
                        home="SA Spurs",
                        away="OKC Thunder",
                        finished_at="2026-10-21T04:00:00Z",
                        kickoff_utc="2026-10-21T01:40:00Z",
                    )
                ]
                if day == DAY.replace(day=21)
                else []
            )
        }

    def history(self, fixture):
        self.fixtures.append(fixture)
        if self.fail:
            raise cm.MoviOddsError("truncated, no progress")
        return cm.History(self.points, (fixture,), [])


def run(client=None, stored=None, **kw):
    return update.update_moviodds_line_history(
        pd.DataFrame([vars(target(**kw))]),
        stored=stored or {},
        client=client or FakeClient(),
        dry_run=True,
        now=NOW,
        progress=lambda _: None,
    )


def test_midnight_results_and_nba_clock_over_provider_clock():
    client = FakeClient()
    result = run(client)
    assert client.days == [DAY, date(2026, 10, 21)]
    assert not result.failures
    rows, games = result.frames[0]
    assert games.iloc[0].tipoff_utc == TIP
    assert rows.iloc[0].mins_to_tip == -90  # not -100 from MoviOdds


def test_missing_and_changed_nba_clock_postpone_without_fetch():
    client = FakeClient()
    assert not run(client, tipoff_utc=pd.NaT).frames
    assert not run(
        client, stored={"0022600003": (TIP + pd.Timedelta(minutes=10), True)}
    ).frames
    assert client.fixtures == []


def test_empty_and_failed_downloads_do_not_prepare_or_write():
    assert run(FakeClient(points=[])).failures
    assert run(FakeClient(fail=True)).failures
    assert not run(FakeClient(fail=True)).frames


def test_request_plan_absent_old_incomplete_recent_and_explicit():
    targets = pd.DataFrame(
        [
            vars(target(game_id=str(i), game_date=date(2026, 10, d)))
            for i, d in enumerate((20, 20, 20, 23))
        ]
    )
    stored = {"1": (TIP, False), "2": (TIP, True), "3": (TIP, False)}
    planned = update.plan_requests(targets, stored, today=date(2026, 10, 24))
    assert planned.game_id.tolist() == ["0", "3"]
    explicit = update.plan_requests(
        targets, stored, today=date(2026, 10, 24), dates=[DAY]
    )
    assert explicit.game_id.tolist() == ["0", "1", "2"]


def test_history_pagination_overlap_and_aliases(monkeypatch):
    pages = [
        dict(
            points=pair("2026-10-21T00:10:00Z"), truncated=True, fixture_ids=["1", "2"]
        ),
        dict(points=pair() + pair("2026-10-21T00:10:00Z"), truncated=False),
    ]
    calls = []
    client = cm.MoviOddsClient()

    def get(path, params):
        calls.append((path, params.copy()))
        return pages.pop(0)

    monkeypatch.setattr(client, "get_json", get)
    history = client.history("1")
    assert len(history.points) == 4 and history.fixture_ids == ("1", "2")
    assert calls[1][1]["to"] == "2026-10-21T00:10:00+00:00"
    assert calls[0][1]["limit"] == 50_000


@pytest.mark.parametrize(
    "payload",
    [
        dict(points=[], truncated=True),
        dict(points=pair()),
        dict(points=pair(), truncated=True),
    ],
)
def test_incomplete_history_refused(monkeypatch, payload):
    client = cm.MoviOddsClient()
    monkeypatch.setattr(client, "get_json", lambda *a: payload)
    with pytest.raises(cm.MoviOddsError):
        client.history("1")


def test_rate_limit_transient_retries_retry_after_and_permanent_error(monkeypatch):
    elapsed, starts = [0.0], []
    monkeypatch.setattr(cm.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(
        cm.time, "sleep", lambda n: elapsed.__setitem__(0, elapsed[0] + n)
    )
    responses = [429, 503, 200, 200, 404]

    def request(*a, **kw):
        starts.append(elapsed[0])
        response = requests.Response()
        response.status_code = responses.pop(0)
        response._content = b'{"results": []}'
        response.headers["Retry-After"] = "12"
        return response

    client = cm.MoviOddsClient()
    monkeypatch.setattr(client.session, "get", request)
    client.results(DAY)
    client.results(DAY)
    with pytest.raises(cm.MoviOddsError, match="404"):
        client.results(DAY)
    assert client.requests_made == 5
    assert starts[1] - starts[0] >= 12 and starts[2] - starts[1] >= 12
    assert all(b - a >= 1 for a, b in zip(starts, starts[1:], strict=False))


class MemoryConnection:
    """Exercise actual SQL builders and COPY merge without remote PostgreSQL."""

    def __init__(self):
        self.games, self.lines, self.staging, self.queries = {}, {}, [], []
        self.commits = 0

    def commit(self):
        self.commits += 1

    def cursor(self):
        return MemoryCursor(self)


class MemoryCursor:
    rowcount = 0

    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def execute(self, query, params=None):
        text = query.as_string() if hasattr(query, "as_string") else query
        self.conn.queries.append((text, params))
        self.rowcount = 0
        if "CREATE UNLOGGED" in text:
            self.conn.staging = []
        if 'INSERT INTO "line_history_moviodds".lh_game' in text:
            gid = params[0]
            self.rowcount = int(gid not in self.conn.games)
            self.conn.games.setdefault(gid, params)
        if 'INSERT INTO "line_history_moviodds".lh_line' in text:
            for row in self.conn.staging:
                d = dict(zip(OUTPUT_COLUMNS, row, strict=True))
                key = tuple(
                    d[c]
                    for c in (
                        "game_id",
                        "market_id",
                        "book_id",
                        "line_ts",
                        "season_year",
                    )
                )
                self.rowcount += int(key not in self.conn.lines)
                self.conn.lines.setdefault(key, row)

    @contextmanager
    def copy(self, query):
        assert '"line_history_moviodds"' in query.as_string()
        yield SimpleNamespace(write_row=self.conn.staging.append)

    def fetchall(self):
        return []


def test_insert_only_rerun_no_overwrite_shared_keys():
    rows, games = update.build_frames(target(), "1", parse(pair()).ticks)
    conn = MemoryConnection()
    assert store.insert_games(conn, games) == 1
    assert store.insert_lines(conn, rows, 2026) == 1
    first = list(conn.lines.values())[0]
    rows["left_price"] = -200
    games["event_id"] = 2
    assert store.insert_games(conn, games) == store.insert_lines(conn, rows, 2026) == 0
    assert list(conn.lines.values())[0] == first
    assert all("UPDATE" not in sql for sql, _ in conn.queries)
    assert all("line_history_oddslogic" not in sql for sql, _ in conn.queries)


def test_schema_sql_readers_and_unique_archive_keys():
    conn = MemoryConnection()
    schema.create_schema(conn)
    rendered = [q for q, _ in conn.queries]
    assert sum("CREATE TABLE IF NOT EXISTS" in q for q in rendered) == len(
        schema.DDL_STATEMENTS
    )
    assert store.fetch_games([2026], conn).columns.tolist() == GAME_COLUMNS
    assert store.fetch_pregame_ticks([2026], conn).columns.tolist() == TICK_COLUMNS
    assert bundle_key(DAY, "fixture", NOW) != bundle_key(DAY, "fixture", NOW)
    assert bundle_key(DAY, "fixture", NOW).startswith(
        "train_data/odds_sources/moviodds/raw/v1/"
    )


def nba_rows():
    return pd.DataFrame(
        [
            dict(
                game_id="0022600003",
                game_date=DAY,
                season_year=2026,
                home=True,
                team_name="San Antonio",
            ),
            dict(
                game_id="0022600003",
                game_date=DAY,
                season_year=2026,
                home=False,
                team_name="Oklahoma City",
            ),
        ]
    )


def test_authoritative_clock_join_and_ambiguous_schedule():
    games = nba_rows()
    schedule = pd.DataFrame([vars(target())])
    assert update.game_targets(games, schedule).iloc[0].tipoff_utc == TIP
    assert pd.isna(update.game_targets(games, None).iloc[0].tipoff_utc)
    conflict = pd.DataFrame([vars(target(tipoff_utc=TIP + pd.Timedelta(minutes=10)))])
    assert pd.isna(
        update.game_targets(games, pd.concat([schedule, conflict])).iloc[0].tipoff_utc
    )
    bad_team = pd.DataFrame([vars(target(team_home="DET"))])
    assert pd.isna(update.game_targets(games, bad_team).iloc[0].tipoff_utc)
    duplicate = games.copy()
    duplicate["game_id"] = "different-id"
    assert update.game_targets(pd.concat([games, duplicate]), schedule).empty


def test_partial_game_failure_does_not_stop_next_game_and_archives_failures():
    class TwoGames(FakeClient):
        def results(self, day):
            payload = super().results(day)
            if payload["results"]:
                payload["results"].append(
                    dict(
                        fixture_id="2",
                        home="NY Knicks",
                        away="PHI 76ers",
                        league="NBA",
                        status="finished",
                        finished_at="2026-10-21T02:00:00Z",
                    )
                )
            return payload

        def history(self, fixture):
            if fixture == "199419311":
                raise cm.MoviOddsError("unavailable")
            return super().history(fixture)

    class BrokenArchive:
        def put(self, *a):
            raise OSError("test archive unavailable")

    plan = pd.DataFrame(
        [
            vars(target()),
            vars(
                target(
                    game_id="second",
                    team_home="NYK",
                    team_away="PHI",
                    tipoff_utc=pd.Timestamp("2026-10-20T23:00:00Z"),
                )
            ),
        ]
    )
    result = update.update_moviodds_line_history(
        plan,
        stored={},
        client=TwoGames(),
        archive=BrokenArchive(),
        dry_run=False,
        now=NOW,
        progress=lambda _: None,
    )
    assert len(result.failures) == 1 and len(result.frames) == 1
    assert result.frames[0][1].iloc[0].game_id == "second"
    assert result.archive_failures == 4  # two discovery days and both games


def test_cli_dry_run_never_writes_even_with_create_schema(monkeypatch, tmp_path):
    import importlib.util

    path = (
        REPO_ROOT
        / "scripts/update_databases/update_moviodds_line_history.py"
    )
    spec = importlib.util.spec_from_file_location("moviodds_cli", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    conn = MemoryConnection()
    conn.read_only = False

    @contextmanager
    def connection():
        yield conn

    class ContextClient(FakeClient):
        requests_made = 3

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def forbidden(*a, **kw):
        raise AssertionError("write attempted in dry-run")

    monkeypatch.setattr(cli, "connect_line_history_db", connection)
    monkeypatch.setattr(cli, "load_games_for_line_history_creation", nba_rows)
    monkeypatch.setattr(
        cli, "fetch_schedules", lambda _: pd.DataFrame([vars(target())])
    )
    monkeypatch.setattr(cli, "MoviOddsClient", ContextClient)
    monkeypatch.setattr(schema, "schema_exists", lambda _: False)
    monkeypatch.setattr(schema, "create_schema", forbidden)
    monkeypatch.setattr(store, "insert_games", forbidden)
    monkeypatch.setattr(store, "insert_lines", forbidden)
    monkeypatch.setattr(store, "record_load", forbidden)
    monkeypatch.setattr(cli, "make_archive", forbidden)
    monkeypatch.setattr(update, "today_eastern", lambda: date(2026, 10, 24))
    assert (
        cli.main(
            [
                "--dry-run",
                "--create-schema",
                "--dates",
                "2026-10-20",
                "--save-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert conn.read_only and conn.commits == 0
    assert (tmp_path / "lh_line.parquet").is_file()
