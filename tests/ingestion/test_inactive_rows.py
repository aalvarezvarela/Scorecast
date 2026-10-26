"""From 2026-27 the box score lists inactive players; ingestion splits them out."""

import numpy as np
import pandas as pd
from nba_ou.fetch_data.fetch_nba_data.inactive_rows import (
    is_inactive_comment,
    split_inactive_rows,
)
from nba_ou.postgre_db.injuries_refs.mapping_v3_v2_injuries_refs import (
    V2_REQUIRED_INJURIES_COLUMNS,
)

COMMENTS = [
    "",
    "DNP - Coach's Decision",
    "DND - Injury/Illness",
    "Inactive - Injury/Illness",
    "Inactive - G League - Two-Way",
    "Inactive - Coach's Decision",
    np.nan,
]
INACTIVE = [False, False, False, True, True, True, False]


def _players() -> pd.DataFrame:
    n = len(COMMENTS)
    return pd.DataFrame(
        {
            "GAME_ID": ["0022600001"] * 4 + ["0022600002"] * 3,
            "TEAM_ID": [1610612764] * n,
            "TEAM_CITY": ["Washington"] * n,
            "TEAM_NAME": ["Wizards"] * n,
            "TEAM_ABBREVIATION": ["WAS"] * n,
            "PLAYER_ID": list(range(100, 100 + n)),
            "firstName": [f"First{i}" for i in range(n)],
            "familyName": [f"Last{i}" for i in range(n)],
            "jerseyNum": [str(i) for i in range(n)],
            "PLAYER_NAME": [f"F. Last{i}" for i in range(n)],
            "COMMENT": COMMENTS,
            "MIN": ["30:00", "", "", "", "", "", "12:00"],
        }
    )


def _games() -> pd.DataFrame:
    # LeagueGameFinder: one row per team.
    return pd.DataFrame(
        {
            "GAME_ID": ["0022600001", "0022600001", "0022600002", "0022600002"],
            "SEASON_ID": ["22026"] * 4,
            "GAME_DATE": ["2026-10-20", "2026-10-20", "2026-10-21", "2026-10-21"],
        }
    )


def test_is_inactive_comment():
    assert is_inactive_comment(pd.Series(COMMENTS)).tolist() == INACTIVE


def test_split_keeps_dressed_players_untouched():
    players = _players()

    kept, _ = split_inactive_rows(players, _games())

    expected = players.loc[[not flag for flag in INACTIVE]].reset_index(drop=True)
    pd.testing.assert_frame_equal(kept, expected)


def test_split_builds_injuries_layout():
    _, injuries = split_inactive_rows(_players(), _games())

    assert injuries.columns.tolist() == V2_REQUIRED_INJURIES_COLUMNS
    assert injuries["PLAYER_ID"].tolist() == ["103", "104", "105"]
    assert injuries["FIRST_NAME"].tolist() == ["First3", "First4", "First5"]
    assert injuries["LAST_NAME"].tolist() == ["Last3", "Last4", "Last5"]
    assert injuries["JERSEY_NUM"].tolist() == ["3", "4", "5"]
    assert injuries["TEAM_ID"].tolist() == ["1610612764"] * 3
    assert injuries["GAME_ID"].tolist() == ["0022600001", "0022600002", "0022600002"]
    assert injuries["GAME_DATE"].tolist() == ["2026-10-20", "2026-10-21", "2026-10-21"]
    assert injuries["SEASON_ID"].tolist() == ["22026"] * 3


def test_old_format_game_has_no_inactives():
    players = _players().loc[
        lambda df: ~df["COMMENT"].fillna("").str.startswith("Inactive")
    ]

    kept, injuries = split_inactive_rows(players, _games())

    assert len(kept) == len(players)
    assert injuries.empty
    assert injuries.columns.tolist() == V2_REQUIRED_INJURIES_COLUMNS


def test_game_missing_from_games_frame_is_dropped():
    games = _games().loc[lambda df: df["GAME_ID"] == "0022600001"]

    _, injuries = split_inactive_rows(_players(), games)

    assert injuries["GAME_ID"].tolist() == ["0022600001"]
