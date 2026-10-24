"""Starter history must be available before, never from, the target game."""

import pandas as pd
import pytest
from nba_ou.data_processing.players.starter_history import add_starter_history_features


def _players(game_id, date, starters, *, team="A", reserve="bench"):
    rows = [
        {
            "GAME_ID": game_id,
            "TEAM_ID": team,
            "GAME_DATE": date,
            "PLAYER_ID": player,
            "START_POSITION": "G",
            "MIN": 40,
        }
        for player in starters
    ]
    rows.append(
        {
            "GAME_ID": game_id,
            "TEAM_ID": team,
            "GAME_DATE": date,
            "PLAYER_ID": reserve,
            "START_POSITION": None,
            "MIN": 40,
        }
    )
    return rows


def test_prior_starters_and_minutes_exclude_target_game_and_same_date():
    games = pd.DataFrame(
        [
            ("g1", "A", "2025-01-01"),
            ("g2", "A", "2025-01-02"),
            ("g3", "A", "2025-01-03"),
            ("g4", "A", "2025-01-03"),
        ],
        columns=["GAME_ID", "TEAM_ID", "GAME_DATE"],
    )
    players = pd.DataFrame(
        _players("g1", "2025-01-01", "abcde")
        + _players("g2", "2025-01-02", "abcdf")
        + _players("g3", "2025-01-03", "ghijk")
        + _players("g4", "2025-01-03", "lmnop")
    )
    got = add_starter_history_features(games, players).set_index("GAME_ID")

    assert pd.isna(got.loc["g1", "STARTER_UNIQUE_LAST_5_GAMES_BEFORE"])
    assert got.loc["g2", "STARTER_UNIQUE_LAST_5_GAMES_BEFORE"] == 5
    for game_id in ("g3", "g4"):
        assert got.loc[game_id, "STARTER_OVERLAP_LAST_TWO_GAMES_BEFORE"] == 4
        assert got.loc[game_id, "STARTER_UNIQUE_LAST_5_GAMES_BEFORE"] == 6
        assert got.loc[game_id, "STARTER_REPEAT_RATE_LAST_5_GAMES_BEFORE"] == 0
        assert got.loc[
            game_id, "STARTER_LATEST_FIVE_MINUTES_SHARE_LAST_5_GAMES_BEFORE"
        ] == pytest.approx(360 / 480)

    # A changed final box score on the target date cannot change its features.
    changed = players.copy()
    changed.loc[changed.GAME_ID.eq("g3"), "START_POSITION"] = None
    changed.loc[changed.GAME_ID.eq("g3"), "MIN"] = 0
    after = add_starter_history_features(games, changed).set_index("GAME_ID")
    pd.testing.assert_series_equal(got.loc["g3"], after.loc["g3"])


def test_incomplete_lineups_and_scheduled_placeholders_do_not_enter_history():
    games = pd.DataFrame(
        [("g1", "A", "2025-01-01"), ("g2", "A", "2025-01-02"),
         ("g3", "A", "2025-01-03")],
        columns=["GAME_ID", "TEAM_ID", "GAME_DATE"],
    )
    players = pd.DataFrame(
        _players("g1", "2025-01-01", "abcde")
        + _players("g2", "2025-01-02", "abcdf")[:4]
        + [dict(row, GAME_ID="g3", GAME_DATE="2025-01-03", MIN=0)
           for row in _players("g1", "2025-01-01", "abcde")]
    )
    got = add_starter_history_features(games, players).set_index("GAME_ID")
    assert got.loc["g3", "STARTER_UNIQUE_LAST_5_GAMES_BEFORE"] == 5
    assert pd.isna(got.loc["g3", "STARTER_OVERLAP_LAST_TWO_GAMES_BEFORE"])
