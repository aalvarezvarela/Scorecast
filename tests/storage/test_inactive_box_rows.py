"""Inactive players stored in nba_injuries, never in nba_players; a failed
inactive upload leaves its games to be fetched again; and which games the
refs update counts as done."""

import pandas as pd
import pytest
from nba_ou.postgre_db.games.update_games import update_database
from nba_ou.postgre_db.injuries_refs.inactive_backfill import stored_rows_to_injuries
from nba_ou.postgre_db.injuries_refs.mapping_v3_v2_injuries_refs import (
    V2_REQUIRED_INJURIES_COLUMNS,
)
from nba_ou.postgre_db.injuries_refs.update_ref_injuries_database import (
    update_refs_injuries_database as refs_update,
)
from nba_ou.postgre_db.injuries_refs.update_ref_injuries_database import (
    update_refs_injuries_database_utils as refs_utils,
)
from nba_ou.postgre_db.players import upload_data_players


def _connect_must_not_run(*args, **kwargs):
    raise AssertionError("connected to the database")


def test_player_upload_refuses_inactive_rows(monkeypatch):
    monkeypatch.setattr(upload_data_players, "connect_nba_db", _connect_must_not_run)
    df = pd.DataFrame(
        {
            "GAME_ID": ["0022600001"] * 2,
            "TEAM_ID": ["1"] * 2,
            "PLAYER_ID": ["7", "8"],
            "COMMENT": ["", "Inactive - Injury/Illness"],
            "MIN": ["30:00", ""],
        }
    )

    with pytest.raises(ValueError, match="inactive-player rows"):
        upload_data_players.upload_players_data_to_db(df)


def _team_rows(game_id: str, team_id: str, minutes: str) -> list[dict]:
    return [
        {
            "GAME_ID": game_id,
            "TEAM_ID": team_id,
            "PLAYER_ID": f"{team_id}{i}",
            "MIN": minutes,
        }
        for i in range(5)
    ]


def test_inactives_uploaded_only_for_uploaded_games(monkeypatch):
    # 0001 uploads; 0002 is excluded (live); 0003 fails the team-minutes check.
    players = pd.DataFrame(
        _team_rows("0001", "A", "48:00")
        + _team_rows("0001", "B", "48:00")
        + _team_rows("0002", "A", "48:00")
        + _team_rows("0002", "B", "48:00")
        + _team_rows("0003", "A", "48:00")
        + _team_rows("0003", "B", "10:00")
    )
    inactives = pd.DataFrame(
        {
            "GAME_ID": ["0001", "0002", "0003"],
            "PLAYER_ID": ["90", "91", "92"],
        }
    )
    uploaded = []
    monkeypatch.setattr(
        update_database,
        "upload_injuries_data_to_db",
        lambda df, if_exists: uploaded.append(df.copy()) or True,
    )

    assert update_database.upload_inactives_to_postgresql(
        inactives, players, players_game_ids_to_exclude=["0002"]
    )

    assert uploaded[0]["GAME_ID"].tolist() == ["0001"]
    assert "MIN_NUMERIC" not in players.columns


def _stub_season_update(monkeypatch, inactives_ok: bool) -> list[str]:
    """Stub everything around the uploads; returns the upload call order."""
    players = pd.DataFrame(
        _team_rows("0001", "A", "48:00") + _team_rows("0001", "B", "48:00")
    )
    inactives = pd.DataFrame({"GAME_ID": ["0001"], "PLAYER_ID": ["90"]})
    calls = []
    monkeypatch.setattr(
        update_database, "get_existing_game_ids_from_db", lambda y: set()
    )
    monkeypatch.setattr(
        update_database, "get_existing_player_game_ids_from_db", lambda y: set()
    )
    monkeypatch.setattr(
        update_database,
        "fetch_nba_data",
        lambda *args, **kwargs: (None, players, inactives, False),
    )
    monkeypatch.setattr(
        update_database,
        "upload_injuries_data_to_db",
        lambda df, if_exists: calls.append("inactives") or inactives_ok,
    )
    monkeypatch.setattr(
        update_database,
        "upload_players_data_to_db",
        lambda df, exclude_game_ids: calls.append("players") or True,
    )
    return calls


def test_inactives_stored_before_players(monkeypatch):
    calls = _stub_season_update(monkeypatch, inactives_ok=True)

    update_database.update_team_players_database(season_year="2026")

    assert calls == ["inactives", "players"]


def test_failed_inactive_upload_leaves_game_to_refetch(monkeypatch):
    # Player rows mark a game as stored; without them the next run fetches it
    # again, inactives included.
    calls = _stub_season_update(monkeypatch, inactives_ok=False)

    with pytest.raises(RuntimeError, match="fetches these games again"):
        update_database.update_team_players_database(season_year="2026")

    assert calls == ["inactives"]


def test_stored_rows_to_injuries():
    rows = pd.DataFrame(
        {
            "player_id": ["7", "8"],
            "firstname": ["AJ", "Ann"],
            "familyname": ["Dybantsa", "Other"],
            "jerseynum": ["1", "2"],
            "team_id": ["1610612764"] * 2,
            "team_city": ["Washington"] * 2,
            "team_name": ["Wizards"] * 2,
            "team_abbreviation": ["WAS"] * 2,
            "game_id": ["0022600097", "0022600097"],
            "comment": ["Inactive - Injury/Illness", "DNP - Coach's Decision"],
        }
    )
    games = pd.DataFrame(
        {"game_id": ["0022600097"], "season_id": ["22026"], "game_date": ["2026-10-23"]}
    )

    player_rows, injuries = stored_rows_to_injuries(rows, games)

    assert player_rows["PLAYER_ID"].tolist() == ["7"]
    assert injuries.columns.tolist() == V2_REQUIRED_INJURIES_COLUMNS
    assert injuries.iloc[0][
        ["PLAYER_ID", "FIRST_NAME", "LAST_NAME", "GAME_DATE"]
    ].tolist() == [
        "7",
        "AJ",
        "Dybantsa",
        "2026-10-23",
    ]


@pytest.mark.parametrize(
    "season_year, done",
    [
        # From 2026-27 the crew alone marks a game done.
        (2026, {"0001", "0002"}),
        # Before, the summary is the only inactive-list source: a game whose
        # injuries are missing is fetched again.
        (2025, {"0001"}),
    ],
)
def test_refs_update_games_done(monkeypatch, season_year, done):
    seen = {}

    def fake_fetch(season_nullable, existing_game_ids, n_tries):
        seen["existing"] = existing_game_ids
        return pd.DataFrame(), pd.DataFrame(), False

    monkeypatch.setattr(
        refs_update, "get_existing_ref_game_ids_from_db", lambda year: {"0001", "0002"}
    )
    monkeypatch.setattr(
        refs_update, "get_existing_injury_game_ids_from_db", lambda year: {"0001"}
    )
    monkeypatch.setattr(refs_update, "fetch_refs_injuries_data", fake_fetch)

    assert refs_update.update_refs_injuries_database(season_year) is False
    assert seen["existing"] == done


def test_refs_fetch_in_game_id_order(monkeypatch):
    games = pd.DataFrame(
        {
            "GAME_ID": ["0022600009", "0022600001", "0022600005", "0012600001"],
            "SEASON_ID": ["22026"] * 4,
            "GAME_DATE": ["2026-10-25"] * 4,
        }
    )

    class FakeFinder:
        def __init__(self, **kwargs):
            pass

        def get_data_frames(self):
            return [games]

    fetched = []

    def fake_summary(game_id, season_id, game_date, n_tries):
        fetched.append(game_id)
        return None, None, False

    monkeypatch.setattr(refs_utils, "LeagueGameFinder", FakeFinder)
    monkeypatch.setattr(refs_utils, "fetch_box_score_summary", fake_summary)
    monkeypatch.setattr(refs_utils.time, "sleep", lambda seconds: None)

    refs_utils.fetch_refs_injuries_data("2026-27", existing_game_ids={"0022600005"})

    assert fetched == ["0022600001", "0022600009"]
