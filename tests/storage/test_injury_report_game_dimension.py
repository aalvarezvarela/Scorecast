"""Pending NBA Cup fixtures must not break the daily injury-report load."""

from datetime import date

import pandas as pd
import pytest
from nba_ou.fetch_data.nba_schedule import fetch_nba_schedule as feed
from nba_ou.postgre_db.injury_report_aiven import ingest, resolve
from nba_ou.postgre_db.injury_report_aiven.parse import ParsedReport


def _game(game_id, day, home="NYK", away="BOS"):
    return {
        "game_id": game_id,
        "game_date": date.fromisoformat(day),
        "season_year": 2026,
        # A late Eastern game tips on the following UTC day.
        "tipoff_utc": pd.Timestamp(day, tz="America/New_York") + pd.Timedelta(hours=22),
        "team_home": home,
        "team_away": away,
    }


@pytest.mark.parametrize("side", ["team_home", "team_away"])
@pytest.mark.parametrize("missing", [None, pd.NA, "", "   "])
def test_pending_fixture_is_excluded_without_dropping_known_future_games(
    monkeypatch, side, missing
):
    pending = _game("0022601201", "2026-12-04")
    pending[side] = missing
    schedule = pd.DataFrame(
        [_game("played", "2026-10-22"), pending, _game("future", "2026-12-05")]
    )
    monkeypatch.setattr(feed, "fetch_schedules", lambda seasons: schedule)

    games = resolve.load_game_dimension([2026])

    assert games["game_id"].tolist() == ["played", "future"]


def test_index_can_resolve_a_schedule_fixture_with_pending_teams(monkeypatch):
    schedule = pd.DataFrame([_game("0022601201", "2026-12-04", None, None)])
    indexed = pd.DataFrame([_game("0022601201", "2026-12-04")])
    monkeypatch.setattr(feed, "fetch_schedules", lambda seasons: schedule)
    monkeypatch.setattr(resolve, "_load_game_time_index", lambda *args: indexed)
    monkeypatch.setattr(
        resolve, "_drop_stale_schedule_rows", lambda frame, *args: frame
    )

    games = resolve.load_game_dimension([2026], object())

    assert games["game_id"].tolist() == ["0022601201"]
    assert games.iloc[0].team_home == "NYK"
    assert games.iloc[0].tipoff_source == resolve.TIPOFF_GAME_TIME_INDEX


@pytest.mark.parametrize(
    "window, expected",
    [
        ((date(2026, 10, 22), date(2026, 10, 23)), ["first", "last"]),
        ((date(2026, 10, 25), date(2026, 10, 26)), []),
        (None, ["older", "first", "last", "tonight", "future"]),
    ],
)
def test_ingest_scopes_game_dimension_writes_to_eastern_game_dates(
    monkeypatch, window, expected
):
    games = pd.DataFrame(
        [
            _game("older", "2026-10-21"),
            _game("first", "2026-10-22"),
            _game("last", "2026-10-23"),
            _game("tonight", "2026-10-24"),
            _game("future", "2026-12-04"),
        ]
    )
    teams = pd.DataFrame(
        {
            "report_name": ["New York Knicks"],
            "nba_team_id": [1610612752],
            "tricode": ["NYK"],
        }
    )
    report = ParsedReport(
        pd.Timestamp("2026-10-21T22:00:00Z").to_pydatetime(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(
            [{"game_date": date(2026, 10, 22), "team_home": "NYK", "team_away": "BOS"}]
        ),
    )
    monkeypatch.setattr(
        ingest,
        "read_reports",
        lambda *args, **kwargs: ([ingest._Read(report, 4, 2026)], 0, 0),
    )
    monkeypatch.setattr(resolve, "load_team_dimension", lambda conn: teams)
    monkeypatch.setattr(resolve, "load_game_dimension", lambda *args: games.copy())
    monkeypatch.setattr(resolve, "load_unscheduled_games", lambda *args: games.iloc[:0])
    monkeypatch.setattr(resolve, "tipoffs_from_reports", lambda *args: games.iloc[:0])
    monkeypatch.setattr(resolve, "load_rosters", lambda *args: pd.DataFrame())
    monkeypatch.setattr(
        ingest.loader, "upsert_teams", lambda *args: {"New York Knicks": 1}
    )
    monkeypatch.setattr(ingest, "create_season_partition", lambda *args: None)
    writes = []

    class StopAtWriteBoundary(Exception):
        pass

    def capture_games(conn, rows):
        writes.extend(rows["game_id"].tolist())
        raise StopAtWriteBoundary

    # Exercise the real ingest up to its DB write boundary, with no DB or PDF IO.
    monkeypatch.setattr(ingest.loader, "upsert_games", capture_games)
    with pytest.raises(StopAtWriteBoundary):
        ingest.ingest_season(
            "2026-27",
            manifest=pd.DataFrame(),
            storage=None,
            aiven=object(),
            source=object(),
            write_game_dates=window,
        )

    assert writes == expected
