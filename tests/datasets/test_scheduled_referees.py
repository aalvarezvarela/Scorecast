"""Unpublished assignments must leave tonight's referee features missing."""

import pandas as pd
import pytest
import requests
from nba_ou.config.constants import TEAM_ID_MAP
from nba_ou.create_training_data import intermediate_referees
from nba_ou.data_processing.referees import add_referee_features as legacy
from nba_ou.data_processing.referees import process_refs_scheduled_game as scheduled
from nba_ou.data_processing.referees.referee_tendencies import (
    add_referee_tendency_features,
)
from nba_ou.fetch_data.referees import fetch_refs_data as fetch
from nba_ou.fetch_data.referees import load_assignments
from nba_ou.fetch_data.referees.assignment_store import AssignmentStore

from tests.support.fake_s3 import FakeS3Client


@pytest.fixture(autouse=True)
def isolated_s3(monkeypatch):
    store = AssignmentStore(FakeS3Client(), "test")
    monkeypatch.setattr(load_assignments, "_store", lambda: store)
    return store


def _page(monkeypatch, html, error=None):
    class Response:
        text = html.replace(
            "<h1>NBA Referee Assignments</h1>",
            "<h1>NBA Referee Assignments</h1><p>November 4, 2025</p>",
        )

        def raise_for_status(self):
            if error:
                raise error

    monkeypatch.setattr(fetch.requests, "get", lambda *args, **kwargs: Response())


def _table(rows=""):
    return (
        "<table><thead><tr><th>Game</th><th>Crew Chief</th>"
        "<th>Referee</th><th>Umpire</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


@pytest.mark.parametrize("table", ["", _table()])
def test_unpublished_nba_assignments_return_an_empty_frame(monkeypatch, table):
    # WNBA can have assignments even when NBA assignments are unavailable.
    _page(
        monkeypatch,
        f"<h1>NBA Referee Assignments</h1>{table}"
        "<h1>WNBA Referee Assignments</h1>"
        + _table("<tr><td>A @ B</td><td>Other Crew</td><td>B</td><td>C</td></tr>"),
    )
    result = fetch.fetch_nba_referee_assignments_today("https://example.test")
    assert result.empty
    assert list(result.columns) == fetch.ASSIGNMENT_COLUMNS


def test_published_assignments_still_return_the_nba_crew(monkeypatch):
    _page(
        monkeypatch,
        "<h1>NBA Referee Assignments</h1>"
        + _table(
            "<tr><td>Boston @ New York</td><td>Ann One</td><td>Bob Two</td><td>Cy Three</td></tr>"
        ),
    )
    result = fetch.fetch_nba_referee_assignments_today("https://example.test")
    assert result["Crew Chief"].tolist() == ["Ann One"]
    assert result["Alternate"].isna().all()


@pytest.mark.parametrize(
    "html",
    [
        "<html>Unexpected response</html>",
        "<h1>NBA Referee Assignments</h1><table><tr><th>Changed</th></tr><tr><td>Layout</td></tr></table>",
    ],
)
def test_unrecognized_pages_still_raise(monkeypatch, html):
    _page(monkeypatch, html)
    with pytest.raises(RuntimeError, match="Could not locate"):
        fetch.fetch_nba_referee_assignments_today("https://example.test")


def test_http_errors_are_not_treated_as_unpublished_assignments(monkeypatch):
    _page(monkeypatch, "", requests.HTTPError("503"))
    with pytest.raises(requests.HTTPError):
        fetch.fetch_nba_referee_assignments_today("https://example.test")


@pytest.fixture
def schedule():
    return pd.DataFrame(
        {
            "GAME_ID": ["tonight1", "tonight2"],
            "GAME_DATE_EST": ["2025-11-04"] * 2,
            "HOME_TEAM_ID": [
                TEAM_ID_MAP["New York Knicks"],
                TEAM_ID_MAP["Boston Celtics"],
            ],
            "VISITOR_TEAM_ID": [
                TEAM_ID_MAP["Boston Celtics"],
                TEAM_ID_MAP["New York Knicks"],
            ],
        }
    )


@pytest.mark.parametrize("has_assignment", [False, True])
def test_unassigned_games_are_retained(monkeypatch, schedule, has_assignment):
    _page(
        monkeypatch,
        "<h1>NBA Referee Assignments</h1>"
        + _table(
            "<tr><td>Boston @ New York</td><td>Ann One (#1)</td><td>Bob Two (#2)</td><td>Cy Three (#3)</td></tr>"
            if has_assignment
            else ""
        ),
    )
    result = scheduled.process_scheduled_referee_assignments(schedule)
    assert result.GAME_ID.tolist() == schedule.GAME_ID.tolist()
    slots = ["REF_1", "REF_2", "REF_3"]
    assert result.loc[1, slots].isna().all()
    if has_assignment:
        assert result.loc[0, slots].tolist() == ["Ann One", "Bob Two", "Cy Three"]
    else:
        assert result[slots].isna().all().all()


@pytest.mark.parametrize("dataset", ["closing", "intermediate"])
def test_unpublished_crews_leave_all_referee_features_nan(
    monkeypatch, schedule, dataset
):
    _page(monkeypatch, "<h1>NBA Referee Assignments</h1>" + _table())
    crews = scheduled.process_scheduled_referee_assignments(schedule)
    games = pd.DataFrame(
        {
            "GAME_ID": ["past", "past"],
            "GAME_DATE": pd.to_datetime(["2025-11-01"] * 2),
            "SEASON_YEAR": [2025] * 2,
            "TEAM_ID": [1, 2],
            "HOME": [True, False],
            "PTS": [100, 110],
            "FTA": [20, 22],
            "PF": [15, 17],
            "POSS": [100, 100],
            "MIN": [240, 240],
        }
    )
    refs = pd.DataFrame(
        {
            "GAME_ID": ["past"] * 3,
            "OFFICIAL_ID": ["1", "2", "3"],
            "FIRST_NAME": ["Ann", "Bob", "Cy"],
            "LAST_NAME": ["One", "Two", "Three"],
            "GAME_DATE": pd.to_datetime(["2025-11-01"] * 3),
        }
    )
    odds = pd.DataFrame({"game_id": ["past"], "total_bet365_line_over": [205.0]})
    base = pd.DataFrame(
        {
            "GAME_ID": schedule.GAME_ID,
            "GAME_DATE": pd.to_datetime(schedule.GAME_DATE_EST),
            "SEASON_YEAR": 2025,
            "TOTAL_POINTS": float("nan"),
            "TOTAL_PF": float("nan"),
            "ODDS_TOTAL_LINE_bet365": 215.0,
        }
    )
    if dataset == "closing":
        monkeypatch.setattr(legacy, "get_refs_data_from_db", lambda *a, **kw: refs)
        result = legacy.add_referee_features_to_training_data(
            ["2025-26"], base, df_referees_scheduled=crews
        )
        with pytest.warns(UserWarning, match="without a referee assignment"):
            result = add_referee_tendency_features(
                result,
                df_team_games=games,
                df_refs=refs,
                df_odds=odds,
                total_line_book="bet365",
                spread_book="bet365",
                df_referees_scheduled=crews,
            )
    else:
        monkeypatch.setattr(
            intermediate_referees, "load_games_from_db", lambda **kw: games.copy()
        )
        monkeypatch.setattr(
            intermediate_referees, "get_refs_data_from_db", lambda *a: refs
        )
        monkeypatch.setattr(
            intermediate_referees,
            "load_and_merge_odds_yahoo_sportsbookreview",
            lambda **kw: odds.copy(),
        )
        monkeypatch.setattr(intermediate_referees, "get_main_book", lambda: "bet365")
        with pytest.warns(UserWarning, match="without a referee assignment"):
            result = intermediate_referees.add_intermediate_referee_features(
                base, df_referees_scheduled=crews
            )
    assert result.GAME_ID.tolist() == schedule.GAME_ID.tolist()
    features = [c for c in result if c.startswith("REF_")]
    assert features
    assert result[features].isna().all().all()


@pytest.mark.parametrize("date", ["2025-11-03", "2025-11-05"])
def test_stale_or_future_nba_page_is_rejected(monkeypatch, date):
    _page(
        monkeypatch,
        "<h1>NBA Referee Assignments</h1>"
        + _table(
            "<tr><td>A @ B</td><td>Ann One</td><td>Bob Two</td><td>Cy Three</td></tr>"
        ),
    )
    with pytest.raises(ValueError, match="expected"):
        fetch.fetch_nba_referee_assignments_today(
            "https://example.test", expected_assignment_date=date
        )


def test_undated_nonempty_page_is_rejected(monkeypatch):
    _page(
        monkeypatch,
        _table(
            "<tr><td>A @ B</td><td>Ann One</td><td>Bob Two</td><td>Cy Three</td></tr>"
        ),
    )
    with pytest.raises(ValueError, match="Could not verify"):
        fetch.fetch_nba_referee_assignments_today(
            "https://example.test", expected_assignment_date="2025-11-04"
        )


def test_s3_fallback_reaches_scheduled_games_without_postgres(
    monkeypatch, schedule, isolated_s3
):
    from nba_ou.postgre_db.config import db_config

    def no_postgres(*args, **kwargs):
        raise AssertionError("Live referee loading must not connect to PostgreSQL")

    monkeypatch.setattr(db_config, "connect_nba_db", no_postgres)
    frame = pd.DataFrame(
        {
            "Game": ["Boston @ New York"],
            "Crew Chief": ["Ann One (#1)"],
            "Referee": ["Bob Two (#2)"],
            "Umpire": ["Cy Three (#3)"],
        }
    )
    isolated_s3.save("2025-11-04", frame, pd.Timestamp("2025-11-04T15:00:00Z"))
    _page(monkeypatch, "", requests.Timeout("NBA unavailable"))
    result = scheduled.process_scheduled_referee_assignments(schedule)
    slots = ["REF_1", "REF_2", "REF_3"]
    assert result.loc[0, slots].tolist() == ["Ann One", "Bob Two", "Cy Three"]
    assert result.loc[1, slots].isna().all()
