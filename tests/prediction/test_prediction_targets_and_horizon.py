"""Target resolution, the spread branch, and the horizon guard.

The first test here is the regression that motivated the whole change: the old
``_infer_prediction_target_from_metadata`` searched a model's name for
substrings and returned line-error for anything it did not recognise, so a
spread model would have been served -- silently -- as a totals model.
"""

import pandas as pd
import pytest
from nba_ou.prediction.prediction import (
    PREDICTION_TARGET_LINE_ERROR,
    PREDICTION_TARGET_SPREAD_ERROR,
    PREDICTION_TARGET_TOTAL_POINTS,
    resolve_prediction_target,
    select_rows_for_horizon,
)

from scripts.retire_legacy_models import is_legacy_key

# --- target resolution -----------------------------------------------------


@pytest.mark.parametrize(
    ("registry_target", "expected"),
    [
        ("line_error", PREDICTION_TARGET_LINE_ERROR),
        ("total_points", PREDICTION_TARGET_TOTAL_POINTS),
        ("spread_error", PREDICTION_TARGET_SPREAD_ERROR),
    ],
)
def test_each_declared_target_maps_to_its_own_prediction_target(
    registry_target, expected
):
    assert resolve_prediction_target(registry_target) == expected


def test_an_unknown_target_raises_instead_of_defaulting_to_line_error():
    """The old path returned PRED_LINE_ERROR for anything it did not match, so
    a spread model was served as a totals model with no warning."""
    with pytest.raises(ValueError, match="Unknown model target"):
        resolve_prediction_target("margin")
    with pytest.raises(ValueError, match="Unknown model target"):
        resolve_prediction_target("")


# --- horizon guard ---------------------------------------------------------


def _frame(minutes):
    return pd.DataFrame({"TIME_TO_MATCH_MINUTES": minutes})


def test_a_horizon_model_only_takes_rows_near_its_own_horizon():
    df = _frame([30, 60, 90, 360, 720])
    eligible = select_rows_for_horizon(df, horizon_minutes=60, tolerance_minutes=45)
    assert list(df.loc[eligible, "TIME_TO_MATCH_MINUTES"]) == [30, 60, 90]


def test_a_closing_model_has_no_upper_bound_on_lateness():
    """T-0 means 'at or after the last snapshot', not 'exactly at tip'."""
    df = _frame([0, 10, 44, 200])
    eligible = select_rows_for_horizon(df, horizon_minutes=0, tolerance_minutes=45)
    assert list(df.loc[eligible, "TIME_TO_MATCH_MINUTES"]) == [0, 10, 44]


def test_a_pooled_model_is_exempt():
    """It conditions on time-to-tip itself, so every row is fair game."""
    df = _frame([0, 60, 720, 1440])
    assert select_rows_for_horizon(df, horizon_minutes=None).all()


def test_rows_with_an_unknown_time_to_tip_are_not_silently_dropped():
    df = _frame([60, None])
    eligible = select_rows_for_horizon(df, horizon_minutes=60, tolerance_minutes=45)
    assert eligible.tolist() == [True, True]


def test_a_frame_without_the_column_is_left_alone():
    assert select_rows_for_horizon(
        pd.DataFrame({"GAME_ID": ["1", "2"]}), horizon_minutes=360
    ).all()


# --- retirement sweep ------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "models/line_error_full_dataset/production/all_seasons_xgb_line_error.json",
        "models/xgboost_model_recent_games/production/old.joblib",
        "models/total_points_last_3_seasons/archive/20260407T155218Z/m.meta.json",
    ],
)
def test_the_sweep_selects_the_old_flat_families(key):
    assert is_legacy_key(key, root="models/")


@pytest.mark.parametrize(
    "key",
    [
        "models/2_5/line_error/t0060/main/channels/production.json",
        "models/2_6/spread_error/t0000/main/specs/a3f9c21d4b0e.json",
        "models/retired/20260921/line_error_full_dataset/production/m.json",
    ],
)
def test_the_sweep_leaves_live_trees_and_previous_sweeps_alone(key):
    """Running it twice must not nest retired/ inside retired/."""
    assert not is_legacy_key(key, root="models/")
