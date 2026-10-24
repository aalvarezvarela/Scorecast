import numpy as np
import pandas as pd
from nba_ou.config.odds_columns import total_line_col

from lab.meta_learner.meta_learner_feature_utils import build_meta_learner_feature_frame


def _make_meta_feature_frame_input() -> pd.DataFrame:
    line_col = total_line_col()
    lines = [226.0, 227.0, 228.0, 229.0, 230.0, 231.0, 232.0, 233.0]
    true_errors = [10.0, -5.0, 6.0, 0.0, -4.0, -2.0, 8.0, 4.0]

    pred_total_full_errors = [7.0, -2.0, 4.0, 0.0, -1.0, -2.0, 6.0, 3.0]
    pred_total_last_5_errors = [11.0, -4.0, 8.0, -1.0, -6.0, -4.0, 10.0, 4.0]
    pred_total_last_3_errors = [10.0, -5.0, 5.0, 1.0, -3.0, -1.0, 7.0, 1.0]

    return pd.DataFrame(
        {
            "GAME_ID": [str(idx) for idx in range(1, 9)],
            "GAME_DATE": pd.to_datetime(
                [
                    "2024-10-01",
                    "2024-10-01",
                    "2024-10-03",
                    "2024-10-03",
                    "2024-10-05",
                    "2024-10-05",
                    "2025-10-01",
                    "2025-10-01",
                ]
            ),
            "SEASON_YEAR": [2024, 2024, 2024, 2024, 2024, 2024, 2025, 2025],
            line_col: lines,
            "LINE_ERROR": true_errors,
            "TOTAL_POINTS": [line + error for line, error in zip(lines, true_errors, strict=True)],
            "PRED_TOTAL_POINTS_FULL_DATASET": [
                line + error
                for line, error in zip(lines, pred_total_full_errors, strict=True)
            ],
            "PRED_TOTAL_POINTS_LAST_5_SEASONS": [
                line + error
                for line, error in zip(lines, pred_total_last_5_errors, strict=True)
            ],
            "PRED_TOTAL_POINTS_LAST_3_SEASONS": [
                line + error
                for line, error in zip(lines, pred_total_last_3_errors, strict=True)
            ],
            "PRED_LINE_ERROR_FULL_DATASET": [8.0, -3.0, 5.0, 1.0, -3.0, -1.0, 7.0, 5.0],
            "PRED_LINE_ERROR_LAST_5_SEASONS": [12.0, -4.0, 7.0, -1.0, -5.0, -3.0, 9.0, 3.0],
            "PRED_LINE_ERROR_LAST_3_SEASONS": [9.0, -6.0, 4.0, 2.0, -2.0, -4.0, 6.0, 2.0],
        }
    )


def test_build_meta_learner_feature_frame_adds_market_and_season_context_without_leakage() -> None:
    feature_result = build_meta_learner_feature_frame(_make_meta_feature_frame_input())
    df_features = feature_result.dataframe

    day1 = df_features.loc[df_features["GAME_DATE"] == pd.Timestamp("2024-10-01")]
    day2 = df_features.loc[df_features["GAME_DATE"] == pd.Timestamp("2024-10-03")]
    day3 = df_features.loc[df_features["GAME_DATE"] == pd.Timestamp("2024-10-05")]
    day4 = df_features.loc[df_features["GAME_DATE"] == pd.Timestamp("2025-10-01")]

    assert day1["PREV_DAY_MEAN_LINE_ERROR"].isna().all()
    np.testing.assert_allclose(day2["PREV_DAY_MEAN_LINE_ERROR"], [2.5, 2.5])
    np.testing.assert_allclose(day3["PREV_2DAY_AVG_MEAN_LINE_ERROR"], [2.75, 2.75])
    np.testing.assert_allclose(day3["SEASON_TO_DATE_GAME_COUNT_PRIOR"], [4.0, 4.0])
    np.testing.assert_allclose(day3["SEASON_TO_DATE_MEAN_LINE_ERROR_PRIOR"], [2.75, 2.75])
    np.testing.assert_allclose(day3["SEASON_TO_DATE_OVER_RATE_PRIOR"], [0.5, 0.5])
    np.testing.assert_allclose(day3["PREV_DAY_MEAN_LINE_ERROR_MINUS_SEASON_MEAN"], [0.25, 0.25])
    assert day4["SEASON_TO_DATE_GAME_COUNT_PRIOR"].isna().all()

    np.testing.assert_allclose(day3["CUR_DAY_GAME_COUNT"], [2.0, 2.0])
    np.testing.assert_allclose(day3["CUR_DAY_MEAN_TOTAL_LINE"], [230.5, 230.5])
    np.testing.assert_allclose(day3["MARKET_LINE_BUCKET_HISTORY_N_10D"], [4.0, 4.0])
    np.testing.assert_allclose(
        day3["MARKET_LINE_BUCKET_MEAN_LINE_ERROR_10D"],
        [2.75, 2.75],
    )
    np.testing.assert_allclose(
        day3["TOTAL_LINE_MINUS_PREV_DAY_AVG_LINE"],
        [1.5, 2.5],
    )


def test_build_meta_learner_feature_frame_adds_model_and_consensus_structure_features() -> None:
    feature_result = build_meta_learner_feature_frame(_make_meta_feature_frame_input())
    df_features = feature_result.dataframe
    day2 = df_features.loc[df_features["GAME_DATE"] == pd.Timestamp("2024-10-03")]
    day3 = df_features.loc[df_features["GAME_DATE"] == pd.Timestamp("2024-10-05")]
    first_day2 = day2.iloc[0]
    first_day3 = day3.iloc[0]
    second_day3 = day3.iloc[1]

    model_error_col = "PRED_LINE_ERROR_FULL_DATASET__ERR"
    np.testing.assert_allclose(day2[f"{model_error_col}_PREV_DAY_MAE"], [2.0, 2.0])
    np.testing.assert_allclose(day2[f"{model_error_col}_PREV_DAY_ACC"], [1.0, 1.0])
    np.testing.assert_allclose(day2[f"{model_error_col}_PREV_DAY_MEAN_ABS_EDGE"], [5.5, 5.5])
    np.testing.assert_allclose(day2[f"{model_error_col}_ROLL_OVER_CALL_RATE_10D"], [0.5, 0.5])
    assert day2[f"{model_error_col}_ROLLING_CONF_BUCKET_SIGNED_RETURN_10D"].isna().all()
    assert np.isclose(first_day3[f"{model_error_col}_ROLLING_CONF_BUCKET_SIGNED_RETURN_10D"], 5.0)
    assert np.isclose(second_day3[f"{model_error_col}_ROLLING_CONF_BUCKET_SIGNED_RETURN_10D"], 0.0)
    assert np.isclose(first_day3[f"{model_error_col}_CURRENT_SIDE_EDGE_BUCKET_ACC_20D"], 1.0)
    assert np.isclose(first_day3[f"{model_error_col}_CURRENT_SIDE_EDGE_BUCKET_SIGNED_RETURN_20D"], 5.0)

    assert np.isclose(first_day2["CUR_DAY_PRED_LINE_ERROR_FULL_DATASET__ERR_MEAN_EDGE"], 3.0)
    assert np.isclose(
        first_day2["CUR_DAY_BOARD_OVER_RATE_BY_PRED_LINE_ERROR_FULL_DATASET__ERR"],
        1.0,
    )
    assert np.isclose(first_day2["PRED_LINE_ERROR_FULL_DATASET__ERR_MINUS_META_AVG_ERR"], -0.5)
    assert np.isclose(first_day2["PRED_LINE_ERROR_FULL_DATASET__ERR_EDGE_MAGNITUDE_RANK"], 3.5)
    assert np.isclose(first_day2["META_MEAN_PAIRWISE_ABS_DIFF_ERR"], 1.9333333333333333)
