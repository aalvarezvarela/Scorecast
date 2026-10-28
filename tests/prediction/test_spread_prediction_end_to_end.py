"""The spread target through the real prediction function.

Written after the target-aware branches were added, because four separate
places in this function assumed the totals market: the mandatory-line check,
the summary column list, the SHAP confidence margin and the final dropna. Each
would have raised on the first spread model to reach production, and none was
visible from the target-resolution unit tests.
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
    PREDICTION_TARGET_TOTAL_POINTS,
    _prepare_required_features_for_prediction,
    clean_prediction_frame,
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
        **kwargs,
    )


def test_a_spread_model_produces_a_spread_residual_and_a_side():
    result = _predict(PREDICTION_TARGET_SPREAD_ERROR)

    assert not result.empty
    assert "PRED_SPREAD_ERROR" in result.columns
    assert "PRED_HOME_MARGIN" in result.columns
    assert "SPREAD_LINE_AT_PREDICTION" in result.columns
    assert set(result["PRED_PICK"]).issubset({"HOME", "AWAY", "PUSH"})
    assert (result["PREDICTION_VALUE_TYPE"] == "SPREAD_ERROR").all()


def test_the_predicted_margin_is_the_line_plus_the_residual():
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


def test_slot_identity_reaches_the_output_rows():
    """Which slot, spec and build produced each row: what the prediction
    store's recorder reads."""
    identity = {
        "MODEL_TARGET": "spread_error",
        "MODEL_HORIZON_MINUTES": 0,
        "MODEL_SCHEMA_VERSION": "2_5",
        "MODEL_VARIANT": "main",
        "SPEC_ID": "a3f9c21d4b0e",
        "FIT_ID": "20260921T154203Z-a3f9c21d",
    }
    result = _predict(PREDICTION_TARGET_SPREAD_ERROR, extra_summary_columns=identity)

    for column, value in identity.items():
        assert column in result.columns, column
        assert (result[column] == value).all(), column


def test_a_line_error_model_is_unaffected_by_the_spread_branches():
    result = _predict(PREDICTION_TARGET_LINE_ERROR)
    assert set(result["PRED_PICK"]).issubset({"OVER", "UNDER", "PUSH"})
    assert (result["PREDICTION_VALUE_TYPE"] == "DIFF_FROM_LINE").all()
    assert "PRED_SPREAD_ERROR" not in result.columns


@pytest.mark.parametrize(
    "target",
    [
        PREDICTION_TARGET_LINE_ERROR,
        PREDICTION_TARGET_TOTAL_POINTS,
        PREDICTION_TARGET_SPREAD_ERROR,
    ],
)
@pytest.mark.parametrize(
    "missing_feature",
    ["REF_CREW_FTA_TENDENCY_BEFORE", "N_REPORT_OUT_PLAYERS_BEFORE_TEAM_HOME"],
)
def test_predictions_continue_with_nan_live_features(target, missing_feature):
    features = [missing_feature, "FEAT_B"]
    training = _frame(32).rename(columns={"FEAT_A": missing_feature})
    model = XGBRegressor(n_estimators=6, max_depth=2, verbosity=0)
    model.fit(training[features], training[missing_feature] * 2 - training.FEAT_B)
    frame = _frame().rename(columns={"FEAT_A": missing_feature})
    frame[missing_feature] = np.nan

    result = load_and_predict_model_for_nba_games(
        df=frame,
        regressor=model,
        model_name="missing_live_features",
        model_type="test",
        model_version="test",
        required_features=features,
        prediction_target=target,
        prediction_datetime=PREDICTION_AT,
    )

    assert len(result) == len(frame)
    assert result.NA_COLUMNS_COUNT.eq(1).all()
    assert result.NA_COLUMNS_NAMES.eq(missing_feature).all()


def test_snapshot_marks_reach_the_output():
    """An intermediate row's snapshot time and passed flag survive cleaning
    into the summary."""
    frame = _frame()
    snapshot_ts = pd.Timestamp("2026-09-21 22:30", tz="UTC")
    frame["SNAPSHOT_TS_UTC"] = snapshot_ts
    frame["SNAPSHOT_PASSED"] = [True, False] * 3
    result = load_and_predict_model_for_nba_games(
        df=frame,
        regressor=_model(),
        model_name="line_error_t0030_main_2_5_21_09_26",
        model_type="line_error_t0030",
        model_version="21_09_26",
        required_features=FEATURES,
        prediction_target=PREDICTION_TARGET_LINE_ERROR,
        prediction_datetime=PREDICTION_AT,
        passthrough_columns=("SNAPSHOT_TS_UTC", "SNAPSHOT_PASSED", "NOT_THERE"),
    )

    assert (result["SNAPSHOT_TS_UTC"] == snapshot_ts).all()
    assert result["SNAPSHOT_PASSED"].tolist() == [True, False] * 3
    assert "NOT_THERE" not in result.columns


def test_a_row_over_the_training_nan_limit_is_not_predicted():
    """The spec's max_na_per_row, counted over the model's features."""
    frame = _frame()
    frame.loc[frame.index[0], FEATURES] = np.nan
    kwargs = dict(
        regressor=_model(),
        model_name="line_error_t0000_main_2_5_21_09_26",
        model_type="line_error_t0000",
        model_version="21_09_26",
        required_features=FEATURES,
        prediction_target=PREDICTION_TARGET_LINE_ERROR,
        prediction_datetime=PREDICTION_AT,
    )

    limited = load_and_predict_model_for_nba_games(df=frame, max_na_per_row=1, **kwargs)
    assert frame["GAME_ID"].iloc[0] not in set(limited["GAME_ID"])
    assert len(limited) == len(frame) - 1

    # At the limit the row is kept, as training's `<=` keeps it.
    kept = load_and_predict_model_for_nba_games(df=frame, max_na_per_row=2, **kwargs)
    assert len(kept) == len(frame)

    nothing_left = load_and_predict_model_for_nba_games(
        df=frame.iloc[:1], max_na_per_row=0, **kwargs
    )
    assert nothing_left.empty


def test_rows_of_the_frame_cleaned_once_predict_as_when_cleaned_alone():
    """What lets the daily job clean each frame once for its 51 slots."""
    frame = _frame()
    # One row cleaning drops (no totals line), one display column it drops.
    frame.loc[frame.index[2], ODDS_TOTAL_COL] = np.nan
    frame["DISPLAY_NOTE"] = "text"
    rows = frame.index[1:5]
    kwargs = dict(
        regressor=_model(),
        model_name="line_error_t0000_main_2_5_21_09_26",
        model_type="line_error_t0000",
        model_version="21_09_26",
        required_features=FEATURES,
        prediction_target=PREDICTION_TARGET_LINE_ERROR,
        prediction_datetime=PREDICTION_AT,
    )

    alone = load_and_predict_model_for_nba_games(df=frame.loc[rows], **kwargs)
    clean = clean_prediction_frame(frame)
    once = load_and_predict_model_for_nba_games(
        df=clean.loc[rows[rows.isin(clean.index)]], cleaned=True, **kwargs
    )

    assert len(once) == 3
    pd.testing.assert_frame_equal(
        once.drop(columns="PREDICTED_AT_UTC"), alone.drop(columns="PREDICTED_AT_UTC")
    )


def test_an_xgboost_model_reads_an_array_with_the_dataframe_s_predictions():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(
        {
            "FEAT_A": rng.normal(size=40),
            "FEAT_COUNT": rng.integers(0, 9, size=40),
            "FEAT_FLAG": rng.integers(0, 2, size=40).astype(bool),
        }
    )
    X.loc[::5, "FEAT_A"] = np.nan
    model = XGBRegressor(n_estimators=8, max_depth=3, verbosity=0)
    model.fit(X, X["FEAT_A"].fillna(0) + X["FEAT_COUNT"] - X["FEAT_FLAG"])

    np.testing.assert_array_equal(
        prediction_module._predict(model, X), model.predict(X)
    )


def test_features_in_another_order_than_the_booster_s_raise():
    """An array carries no names: the order is checked before it is built."""
    frame = _frame()
    with pytest.raises(ValueError, match="feature_names mismatch"):
        prediction_module._predict(_model(), frame[FEATURES[::-1]])
