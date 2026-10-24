"""Causal feature for the game model: labels enter only after tip-off."""

from __future__ import annotations

import pandas as pd
import pytest
from nba_ou.create_training_data.historical_ridge_movement import (
    EXPECTED_TOTAL_MOVE_COLUMN,
    add_historical_ridge_movement,
)
from nba_ou.create_training_data.select_intermediate_columns import is_kept_column


def _game(game: str, day: str, season: int, current: float) -> tuple[list[dict], dict]:
    tip = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=20)
    rows = [
        {
            "GAME_ID": game,
            "SEASON_YEAR": season,
            "TIME_TO_MATCH_MIN": horizon,
            "TIPOFF_UTC": tip,
            "SNAPSHOT_TS_UTC": tip - pd.Timedelta(minutes=horizon),
            "ODDS_TOTAL_LINE_bet365": current if horizon else current + 2.0,
            "ODDS_SNAP_TOT_BET365_MOVE_FROM_OPEN": 1.0,
            "ODDS_SNAP_TOT_BET365_LINE_AGE_MINUTES": 10.0,
        }
        for horizon in (360, 0)
    ]
    return rows, {"GAME_ID": game, "CLOSING_LINE": current + 2.0}


def test_only_completed_prior_games_affect_a_prediction(monkeypatch):
    import nba_ou.create_training_data.historical_ridge_movement as ridge

    monkeypatch.setattr(ridge, "MIN_TRAIN_GAMES", 1)
    games = [
        _game("a", "2024-01-01", 2023, 220.0),
        _game("b", "2024-01-02", 2023, 220.0),
        _game("c", "2024-01-03", 2023, 220.0),
    ]
    frame = pd.DataFrame([row for rows, _ in games for row in rows])
    closes = pd.DataFrame([close for _, close in games])
    actual = add_historical_ridge_movement(frame, closes, anchor="bet365")

    # The first game has no history; closing snapshots have no remaining move.
    assert actual.loc[0, EXPECTED_TOTAL_MOVE_COLUMN] == 0.0
    assert actual.loc[1::2, EXPECTED_TOTAL_MOVE_COLUMN].eq(0.0).all()
    assert actual.loc[2, EXPECTED_TOTAL_MOVE_COLUMN] > 0.0

    changed_closes = closes.copy()
    changed_closes.loc[changed_closes.GAME_ID.eq("b"), "CLOSING_LINE"] = 228.0
    changed = add_historical_ridge_movement(frame, changed_closes, anchor="bet365")
    # A game's own eventual closing line cannot change either of its features.
    assert changed.loc[2, EXPECTED_TOTAL_MOVE_COLUMN] == pytest.approx(
        actual.loc[2, EXPECTED_TOTAL_MOVE_COLUMN]
    )
    assert (
        changed.loc[4, EXPECTED_TOTAL_MOVE_COLUMN]
        > actual.loc[4, EXPECTED_TOTAL_MOVE_COLUMN]
    )


def test_snapshot_cannot_use_a_game_that_tips_off_later(monkeypatch):
    import nba_ou.create_training_data.historical_ridge_movement as ridge

    monkeypatch.setattr(ridge, "MIN_TRAIN_GAMES", 1)
    # At 14:00 UTC, game b's 20:00 close is unknown; a finished at 12:00.
    a_rows, a_close = _game("a", "2024-01-01", 2023, 220.0)
    a_rows = [a_rows[0]]
    a_rows[0]["TIPOFF_UTC"] = pd.Timestamp("2024-01-01 12:00", tz="UTC")
    a_rows[0]["SNAPSHOT_TS_UTC"] = pd.Timestamp("2024-01-01 06:00", tz="UTC")
    b_rows, b_close = _game("b", "2024-01-01", 2023, 220.0)
    c_rows, c_close = _game("c", "2024-01-01", 2023, 220.0)
    c_rows = [c_rows[0]]
    c_rows[0]["TIPOFF_UTC"] = pd.Timestamp("2024-01-01 20:00", tz="UTC")
    c_rows[0]["SNAPSHOT_TS_UTC"] = pd.Timestamp("2024-01-01 14:00", tz="UTC")
    frame = pd.DataFrame(a_rows + b_rows + c_rows)
    closes = pd.DataFrame([a_close, b_close, c_close])
    before = add_historical_ridge_movement(frame, closes, anchor="bet365")
    closes.loc[closes.GAME_ID.eq("b"), "CLOSING_LINE"] = 228.0
    after = add_historical_ridge_movement(frame, closes, anchor="bet365")
    assert before.loc[3, EXPECTED_TOTAL_MOVE_COLUMN] == pytest.approx(
        after.loc[3, EXPECTED_TOTAL_MOVE_COLUMN]
    )


def test_uses_previous_and_current_season_only(monkeypatch):
    import nba_ou.create_training_data.historical_ridge_movement as ridge

    monkeypatch.setattr(ridge, "MIN_TRAIN_GAMES", 1)
    old_rows, old_close = _game("old", "2022-01-01", 2021, 220.0)
    previous_rows, previous_close = _game("previous", "2023-01-01", 2022, 220.0)
    current_rows, current_close = _game("current", "2024-01-01", 2023, 220.0)
    frame = pd.DataFrame(old_rows + previous_rows + current_rows)
    closes = pd.DataFrame([old_close, previous_close, current_close])
    first = add_historical_ridge_movement(frame, closes, anchor="bet365")
    closes.loc[closes.GAME_ID.eq("old"), "CLOSING_LINE"] = 228.0
    changed = add_historical_ridge_movement(frame, closes, anchor="bet365")
    assert first.loc[4, EXPECTED_TOTAL_MOVE_COLUMN] == pytest.approx(
        changed.loc[4, EXPECTED_TOTAL_MOVE_COLUMN]
    )


def test_rejects_duplicate_closing_lines():
    rows, close = _game("a", "2024-01-01", 2023, 220.0)
    with pytest.raises(ValueError, match="one row per game"):
        add_historical_ridge_movement(
            pd.DataFrame(rows), pd.DataFrame([close, close]), anchor="bet365"
        )


def test_feature_passes_the_intermediate_dataset_gate():
    assert is_kept_column(EXPECTED_TOTAL_MOVE_COLUMN)
