"""Incremental ingestion must equal a full timeline, including late reports."""

from contextlib import contextmanager
from copy import deepcopy

import pandas as pd
import pytest
from nba_ou.postgre_db.injury_report_aiven import live_ingest as live
from nba_ou.postgre_db.injury_report_aiven.ingest import _Read
from nba_ou.postgre_db.injury_report_aiven.parse import ParsedReport
from nba_ou.postgre_db.injury_report_aiven.spans import FILING_COLUMNS, SPAN_COLUMNS

DAY = pd.Timestamp("2026-10-16").date()
TIP = pd.Timestamp("2026-10-16T23:00:00Z")
GAME = "0022600001"
TEAMS = pd.DataFrame(
    {
        "report_name": ["Boston Celtics", "New York Knicks"],
        "nba_team_id": [1610612738, 1610612752],
        "tricode": ["BOS", "NYK"],
    }
)
GAMES = pd.DataFrame(
    {
        "game_id": [GAME],
        "game_date": [DAY],
        "season_year": [2026],
        "tipoff_utc": [TIP],
        "team_home": ["BOS"],
        "team_away": ["NYK"],
    }
)
ROSTER = pd.DataFrame(
    {
        "game_id": ["past"],
        "nba_team_id": [1610612738],
        "player_id": [2544],
        "season_year": [2026],
        "norm_name": ["james|lebron"],
        "norm_initial": ["james|l"],
    }
)


def report(clock, status=3, *, nys=False, vanished=False, name="James, LeBron"):
    at = pd.Timestamp(f"2026-10-16T{clock}:00Z")
    keys = {"game_date": DAY, "team_home": "BOS", "team_away": "NYK"}
    players = (
        pd.DataFrame(
            [
                dict(
                    **keys,
                    raw_team="Boston Celtics",
                    raw_name=name,
                    status_id=status,
                    reason_category="Injury/Illness",
                    reason_detail="Knee",
                )
            ]
        )
        if status is not None and not nys
        else pd.DataFrame()
    )
    not_submitted = (
        pd.DataFrame([dict(**keys, raw_team="Boston Celtics")])
        if nys
        else pd.DataFrame()
    )
    games = (
        pd.DataFrame([dict(**keys, game_time_raw="07:00 (ET)")])
        if not vanished
        else pd.DataFrame()
    )
    return _Read(
        ParsedReport(at.to_pydatetime(), players, not_submitted, games), 4, 2026
    )


class MemoryConnection:
    """Transaction-aware stand-in; the real span builders are never mocked."""

    def __init__(self):
        self.state = {
            "reports": {},
            "done": set(),
            "reasons": {},
            "labels": {},
            "spans": pd.DataFrame(columns=SPAN_COLUMNS),
            "filings": pd.DataFrame(columns=FILING_COLUMNS),
            "unresolved": [],
        }
        self.fail_insert = False
        self.calls = []

    @contextmanager
    def transaction(self):
        saved = deepcopy(self.state)
        try:
            yield
        except Exception:
            self.state = saved
            raise

    def cursor(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        self.rows = []
        if "SELECT source_url" in sql:
            self.rows = [(url,) for url in self.state["done"] if url in params[0]]
        elif "SELECT DISTINCT game_id" in sql:
            f = self.state["filings"]
            self.rows = [
                (g,)
                for g in f.loc[
                    (f.valid_from < params[0]) & (f.valid_to > params[1]), "game_id"
                ].unique()
            ]
        elif "SELECT max(observed_at)" in sql:
            self.rows = [(max(self.state["reports"], default=None),)]
        elif "SELECT observed_at FROM" in sql:
            self.rows = [
                (at,) for at in self.state["reports"] if params[0] <= at <= params[1]
            ]
        elif "DELETE FROM" in sql:
            kind = "filings" if "ir_filing_span" in sql else "spans"
            self.state[kind] = (
                self.state[kind].loc[~self.state[kind].game_id.isin(params[0])].copy()
            )

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0]


def project_history(conn, ids, end):
    """Same event-time expansion as the SQL reader, evaluated in memory."""
    observations, listed, nys = [], [], []
    common = dict(
        game_id=GAME,
        game_date=DAY,
        season_year=2026,
        tipoff_utc=TIP,
        team_home="BOS",
        team_away="NYK",
    )
    for at in sorted(conn.state["reports"]):
        if at > end:
            continue
        f = conn.state["filings"]
        active = f.loc[f.game_id.isin(ids) & (f.valid_from <= at) & (f.valid_to > at)]
        if not active.empty:
            listed.append(dict(**common, observed_at=at))
        for filing in active.itertuples():
            team = TEAMS.iloc[filing.team_id - 1]
            if not filing.submitted:
                nys.append(
                    dict(
                        game_id=GAME,
                        team_id=filing.team_id,
                        raw_team=team.report_name,
                        observed_at=at,
                    )
                )
                continue
            s = conn.state["spans"]
            current = s.loc[
                (s.team_id == filing.team_id)
                & (s.valid_from <= at)
                & (s.valid_to > at)
                & s.status_id.notna()
            ]
            for row in current.itertuples():
                label, detail = conn.state["labels"][row.reason_id]
                observations.append(
                    dict(
                        **common,
                        observed_at=at,
                        team_id=row.team_id,
                        nba_team_id=str(team.nba_team_id),
                        raw_team=team.report_name,
                        raw_name="",
                        player_id=row.player_id,
                        status_id=row.status_id,
                        reason_category=label,
                        reason_detail=detail,
                    )
                )
    return (
        pd.DataFrame(
            observations,
            columns=[
                *common,
                "observed_at",
                "team_id",
                "nba_team_id",
                "raw_team",
                "raw_name",
                "player_id",
                "status_id",
                "reason_category",
                "reason_detail",
            ],
        ),
        pd.DataFrame(listed, columns=[*common, "observed_at"]),
        pd.DataFrame(nys, columns=["game_id", "team_id", "raw_team", "observed_at"]),
    )


@pytest.fixture
def conn(monkeypatch):
    value = MemoryConnection()
    monkeypatch.setattr(live, "read_existing_observations", project_history)
    monkeypatch.setattr(
        live.load,
        "upsert_teams",
        lambda conn, teams, *, commit: {"Boston Celtics": 1, "New York Knicks": 2},
    )
    monkeypatch.setattr(live.load, "upsert_games", lambda *a, **kw: None)
    monkeypatch.setattr(live, "create_season_partition", lambda *a, **kw: None)

    def reports(conn, rows, *, commit):
        assert not commit
        for at in rows.observed_at:
            conn.state["reports"].setdefault(at, len(conn.state["reports"]) + 1)
        return conn.state["reports"].copy()

    def reasons(conn, pairs):
        for code, label, detail in pairs:
            key = (code, detail)
            rid = conn.state["reasons"].setdefault(key, len(conn.state["reasons"]) + 1)
            conn.state["labels"][rid] = (label, detail)
        return conn.state["reasons"].copy()

    def spans(conn, rows, *, commit):
        assert not commit
        if conn.fail_insert:
            raise RuntimeError("database write failed")
        conn.state["spans"] = rows.copy()

    def filings(conn, rows, *, commit):
        assert not commit
        conn.state["filings"] = rows.copy()

    monkeypatch.setattr(live.load, "upsert_reports", reports)
    monkeypatch.setattr(live.load, "upsert_reasons", reasons)
    monkeypatch.setattr(live.load, "insert_spans", spans)
    monkeypatch.setattr(live.load, "insert_filing_spans", filings)

    def unresolved(conn, rows, *, commit):
        assert not commit
        conn.state["unresolved"].extend(rows)

    monkeypatch.setattr(live.load, "record_unresolved", unresolved)
    monkeypatch.setattr(
        live.documents, "mark_ingested", lambda c, urls: c.state["done"].update(urls)
    )
    return value


def ingest(conn, *reports):
    urls = [str(r.report.observed_at) for r in reports]
    live.ingest_live_reports(
        conn, list(reports), urls, teams=TEAMS, games=GAMES, rosters=ROSTER
    )


def test_unchanged_reports_are_registered_without_duplicate_spans(conn):
    ingest(conn, report("14:00"))
    ingest(conn, report("14:15"))
    assert len(conn.state["reports"]) == len(conn.state["done"]) == 2
    assert len(conn.state["spans"]) == 1
    assert conn.state["spans"].iloc[0].valid_from == pd.Timestamp("2026-10-16T14:00Z")


def test_late_report_rebuilds_only_the_interval_between_known_reports(conn):
    ingest(conn, report("14:00", 3), report("15:00", 5))
    ingest(conn, report("14:30", 4))
    spans = conn.state["spans"]
    assert spans.status_id.tolist() == [3, 4, 5]
    assert spans.valid_to.tolist() == [
        pd.Timestamp("2026-10-16T14:30Z"),
        pd.Timestamp("2026-10-16T15:00Z"),
        TIP,
    ]


@pytest.mark.parametrize("vanished", [False, True])
def test_removal_is_preserved_when_new_reports_are_appended(conn, vanished):
    ingest(conn, report("14:00", 5))
    ingest(conn, report("14:15", None, vanished=vanished))
    ingest(conn, report("14:30", 3))
    spans = conn.state["spans"]
    assert spans.iloc[0].valid_to == pd.Timestamp("2026-10-16T14:15Z")
    assert pd.isna(spans.iloc[1].status_id)
    assert spans.iloc[2].status_id == 3


def test_not_submitted_is_unknown_and_not_a_player_removal(conn):
    ingest(conn, report("14:00", 5))
    ingest(conn, report("14:15", None, nys=True))
    assert conn.state["spans"].status_id.tolist() == [5]
    filings = conn.state["filings"]
    assert filings.loc[filings.team_id.eq(1), "submitted"].tolist() == [True, False]


def test_failed_span_write_rolls_back_report_record_and_old_timeline(conn):
    ingest(conn, report("14:00"))
    before = deepcopy(conn.state)
    conn.fail_insert = True
    with pytest.raises(RuntimeError, match="database write failed"):
        ingest(conn, report("14:15", 5))
    assert conn.state["reports"] == before["reports"]
    assert conn.state["done"] == before["done"]
    pd.testing.assert_frame_equal(conn.state["spans"], before["spans"])


def test_already_completed_report_is_a_no_op(conn):
    sample = report("14:00")
    ingest(conn, sample)
    before = deepcopy(conn.state)
    ingest(conn, sample)
    assert conn.state["reports"] == before["reports"]
    pd.testing.assert_frame_equal(conn.state["spans"], before["spans"])


def test_exact_nba_name_resolves_before_first_box_score(conn):
    empty = ROSTER.iloc[:0]
    sample = report("14:00")
    live.ingest_live_reports(
        conn, [sample], ["new"], teams=TEAMS, games=GAMES, rosters=empty
    )
    assert conn.state["spans"].player_id.tolist() == [2544]


def test_unknown_player_is_recorded_and_does_not_block_the_report(conn):
    # As in the historical ingest: the name is dropped and recorded, and the
    # report (filings included) still completes.
    ingest(conn, report("14:00", name="Unknown, Nobody"))
    assert len(conn.state["reports"]) == len(conn.state["done"]) == 1
    assert conn.state["spans"].empty
    assert not conn.state["filings"].empty
    assert [r["raw_name"] for r in conn.state["unresolved"]] == ["Unknown, Nobody"]


def test_unknown_player_does_not_drop_resolved_players():
    rows = pd.DataFrame(
        {
            "game_id": [GAME, GAME],
            "raw_team": ["Boston Celtics"] * 2,
            "raw_name": ["James, LeBron", "Unknown, Nobody"],
            "observed_at": [pd.Timestamp("2026-10-16T14:00Z")] * 2,
            "nba_team_id": [1610612738] * 2,
            "season_year": [2026] * 2,
        }
    )
    report = live.R.ResolutionReport()
    resolved = live.resolve_live_players(rows, ROSTER.iloc[:0], report)
    assert resolved.player_id.tolist() == [2544]
    assert [r["raw_name"] for r in report.unresolved] == ["Unknown, Nobody"]


def test_live_roster_resolves_a_rookie_missing_from_static_catalog():
    rows = pd.DataFrame(
        {
            "game_id": [GAME],
            "raw_team": ["Boston Celtics"],
            "raw_name": ["Rookie, New"],
            "observed_at": [pd.Timestamp("2026-10-16T14:00Z")],
            "nba_team_id": [1610612738],
            "season_year": [2026],
        }
    )
    report = live.R.ResolutionReport()
    resolved = live.resolve_live_players(
        rows, ROSTER.iloc[:0], report, {(GAME, "1610612738"): {"9999999": "New Rookie"}}
    )
    assert resolved.player_id.tolist() == [9999999]
    assert resolved.method.tolist() == ["live_team_roster"]
    assert not report.unresolved


@pytest.mark.parametrize("vanished,nys", [(True, False), (False, True)])
def test_first_report_without_players_is_still_recorded(conn, vanished, nys):
    ingest(conn, report("14:00", None, vanished=vanished, nys=nys))
    assert len(conn.state["reports"]) == len(conn.state["done"]) == 1
    assert conn.state["spans"].empty


def test_empty_game_between_listings_is_a_removal_even_in_one_batch(conn):
    ingest(
        conn,
        report("14:00", 5),
        report("14:15", None, vanished=True),
        report("14:30", 3),
    )
    spans = conn.state["spans"]
    assert spans.iloc[0].valid_to == pd.Timestamp("2026-10-16T14:15Z")
    assert pd.isna(spans.iloc[1].status_id)
    assert spans.iloc[2].status_id == 3
