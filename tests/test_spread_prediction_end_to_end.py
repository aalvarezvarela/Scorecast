"""The spread target through the real prediction function.

Written after the target-aware branches were added, because four separate
places in this function assumed the totals market: the mandatory-line check,
the summary column list, the SHAP confidence margin and the final dropna. Each
would have raised on the first spread model to reach production, and none was
visible from the target-resolution unit tests.

The Postgres upload is monkeypatched out -- what is under test is the frame
handed to it.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from nba_ou.prediction import prediction as prediction_module
from nba_ou.prediction.prediction import (
    PREDICTION_TARGET_LINE_ERROR,
    PREDICTION_TARGET_SPREAD_ERROR,
    _prepare_required_features_for_prediction,
    load_and_predict_model_for_nba_games,
)
from xgboost import XGBRegressor

FEATURES = ["FEAT_A", "FEAT_B"]
SPREAD_COL = "ODDS_SPREAD_LINE_HOME_bet365"
# The canonical totals line. clean_dataframe_for_training's basic cleaning
# needs it whatever market the model is in, so a real frame always carries one.
ODDS_TOTAL_COL = "ODDS_TOTAL_LINE_bet365"
# Timezone-aware, as every real caller passes: the time-to-tip subtraction is
# against a tz-aware game time.
PREDICTION_AT = datetime(2026, 9, 21, 18, 0, tzinfo=ZoneInfo("Europe/Madrid"))


@pytest.fixture
def captured_uploads(monkeypatch):
    uploads = []
    monkeypatch.setattr(
        prediction_module, "upload_predictions_to_postgre", uploads.append
    )
    return uploads


def _model(seed: int = 0) -> XGBRegressor:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(64, len(FEATURES))), columns=FEATURES)
    y = X["FEAT_A"] * 2.0 - X["FEAT_B"]
    model = XGBRegressor(n_estimators=6, max_depth=2, verbosity=0)
    model.fit(X, y)
    return model


def _frame(n: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "GAME_ID": [f"002260000{i}" for i in range(n)],
            "SEASON_TYPE": ["Regular Season"] * n,
            "GAME_DATE": ["2026-09-21"] * n,
            "GAME_TIME": ["7:00 pm ET"] * n,
            "TEAM_NAME_TEAM_HOME": [f"Home {i}" for i in range(n)],
            "TEAM_NAME_TEAM_AWAY": [f"Away {i}" for i in range(n)],
            "MATCHUP_TEAM_HOME": [f"HOM{i} vs. AWY{i}" for i in range(n)],
            SPREAD_COL: rng.normal(scale=6.0, size=n).round(1),
            ODDS_TOTAL_COL: rng.normal(loc=225, scale=8.0, size=n).round(1),
            "FEAT_A": rng.normal(size=n),
            "FEAT_B": rng.normal(size=n),
        }
    )


def _predict(target, **kwargs):
    return load_and_predict_model_for_nba_games(
        df=_frame(),
        regressor=_model(),
        model_name="spread_error_t0000_main_2_5_21_09_26",
        model_type="spread_error_t0000",
        model_version="21_09_26",
        required_features=FEATURES,
        prediction_target=target,
        prediction_datetime=PREDICTION_AT,
        shap_top_n=2,
        **kwargs,
    )


def test_a_spread_model_produces_a_spread_residual_and_a_side(captured_uploads):
    result = _predict(PREDICTION_TARGET_SPREAD_ERROR)

    assert not result.empty
    assert "PRED_SPREAD_ERROR" in result.columns
    assert "PRED_HOME_MARGIN" in result.columns
    assert "SPREAD_LINE_AT_PREDICTION" in result.columns
    assert set(result["PRED_PICK"]).issubset({"HOME", "AWAY", "PUSH"})
    assert (result["PREDICTION_VALUE_TYPE"] == "SPREAD_ERROR").all()


def test_the_predicted_margin_is_the_line_plus_the_residual(captured_uploads):
    result = _predict(PREDICTION_TARGET_SPREAD_ERROR)
    expected = result["SPREAD_LINE_AT_PREDICTION"] + result["PRED_SPREAD_ERROR"]
    pd.testing.assert_series_equal(
        result["PRED_HOME_MARGIN"], expected, check_names=False
    )


def test_the_mandatory_line_check_is_skipped_when_no_line_is_named():
    """A spread model passes None here, because demanding the totals line
    would reject it for lacking a column its market does not have.

    Tested at this layer rather than end to end: basic cleaning needs a
    canonical totals column whatever the market, so a frame without one never
    reaches this check in practice.
    """
    frame = _frame().drop(columns=[ODDS_TOTAL_COL])
    required, X, na_count, _ = _prepare_required_features_for_prediction(
        frame, FEATURES, mandatory_main_book_col=None
    )
    assert required == FEATURES
    assert list(X.columns) == FEATURES
    assert len(X) == len(frame)


def test_the_mandatory_line_check_still_fires_when_one_is_named():
    frame = _frame().drop(columns=[ODDS_TOTAL_COL])
    with pytest.raises(ValueError, match="mandatory"):
        _prepare_required_features_for_prediction(
            frame, FEATURES, mandatory_main_book_col=ODDS_TOTAL_COL
        )


def test_slot_identity_reaches_the_database_row(captured_uploads):
    """The upload happens inside this function, so columns attached to the
    RETURNED frame afterwards would never be stored."""
    identity = {
        "MODEL_TARGET": "spread_error",
        "MODEL_HORIZON_MINUTES": 0,
        "MODEL_SCHEMA_VERSION": "2_5",
        "MODEL_VARIANT": "main",
        "SPEC_ID": "a3f9c21d4b0e",
        "FIT_ID": "20260921T154203Z-a3f9c21d",
    }
    _predict(PREDICTION_TARGET_SPREAD_ERROR, extra_summary_columns=identity)

    assert len(captured_uploads) == 1
    uploaded = captured_uploads[0]
    for column, value in identity.items():
        assert column in uploaded.columns, column
        assert (uploaded[column] == value).all(), column


def test_a_line_error_model_is_unaffected_by_the_spread_branches(captured_uploads):
    result = _predict(PREDICTION_TARGET_LINE_ERROR)
    assert set(result["PRED_PICK"]).issubset({"OVER", "UNDER", "PUSH"})
    assert (result["PREDICTION_VALUE_TYPE"] == "DIFF_FROM_LINE").all()
    assert "PRED_SPREAD_ERROR" not in result.columns
