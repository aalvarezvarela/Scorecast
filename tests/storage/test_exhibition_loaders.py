"""The feature builders' loaders never return a preseason or All-Star game."""

import pandas as pd
import pytest
from nba_ou.postgre_db.config import db_loader
from nba_ou.postgre_db.injuries_refs.fetch_injury_db import get_injury_data_from_db
from nba_ou.postgre_db.injury_report_aiven import fetch

#: Preseason, regular season, All-Star, playoffs, play-in, Cup final.
GAME_IDS = [
    "0012400001",
    "0022400001",
    "0032400001",
    "0042400101",
    "0052400101",
    "0062400001",
]
KEPT = ["0022400001", "0042400101", "0052400101", "0062400001"]


def test_games_and_players_without_exhibitions(monkeypatch):
    games = pd.DataFrame({"game_id": GAME_IDS, "team_id": ["1"] * 6})
    players = pd.DataFrame({"game_id": GAME_IDS, "player_id": ["7"] * 6})
    monkeypatch.setattr(db_loader, "load_games_from_db", lambda **kw: games.copy())
    monkeypatch.setattr(db_loader, "load_players_from_db", lambda **kw: players.copy())

    df_games, df_players = db_loader.load_all_nba_data_from_db(seasons=["2024-25"])

    assert df_games["GAME_ID"].tolist() == KEPT
    assert df_players["GAME_ID"].tolist() == KEPT


def test_injuries_without_exhibitions(monkeypatch):
    rows = pd.DataFrame(
        {"game_id": GAME_IDS, "team_id": [1] * 6, "player_id": range(6)}
    )
    module = get_injury_data_from_db
    monkeypatch.setattr(module, "connect_nba_db", lambda: None)
    monkeypatch.setattr(module, "read_through", lambda *args: rows.copy())

    injuries = module.get_injury_data_from_db(seasons=["2024-25"])

    assert injuries["GAME_ID"].tolist() == KEPT


@pytest.mark.parametrize(
    "reader",
    [
        fetch.last_status_before_tip,
        fetch.filing_at_tip,
        fetch.report_age_at_tip,
        fetch.listed_status_events,
        fetch.listed_pairs,
    ],
)
@pytest.mark.parametrize("from_s3", [True, False])
def test_report_history_without_exhibitions(monkeypatch, reader, from_s3):
    rows = pd.DataFrame({"game_id": GAME_IDS, "team_id": ["1"] * 6})
    monkeypatch.setattr(
        fetch, "_through_s3", lambda *args, **kw: rows.copy() if from_s3 else None
    )
    monkeypatch.setattr(fetch, "read_sql_canonical", lambda *args, **kw: rows.copy())

    assert reader(conn=None)["game_id"].tolist() == KEPT


@pytest.mark.parametrize("reader", [fetch.status_spans, fetch.filing_spans])
def test_report_spans_without_exhibitions(monkeypatch, reader):
    rows = pd.DataFrame({"game_id": GAME_IDS, "team_id": ["1"] * 6})
    monkeypatch.setattr(fetch, "read_sql_canonical", lambda *args, **kw: rows.copy())

    assert reader(conn=None)["game_id"].tolist() == KEPT
