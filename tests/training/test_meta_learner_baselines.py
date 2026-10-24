import pandas as pd

from lab.meta_learner.meta_learner_baselines import (
    BASE_AVG_ALL_6_ERR_COL,
    BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL,
    BASE_MAJORITY_LINE_ERROR_ONLY_ERR_COL,
    BASE_MAJORITY_TOTAL_ONLY_ERR_COL,
    add_default_meta_learner_baselines,
    build_base_avg_all_6_error,
    build_base_majority_all_6_tie_line_error_full_dataset_error,
    build_base_majority_line_error_only_error,
    build_base_majority_total_only_error,
    build_total_points_error_space_predictions,
)


def _make_meta_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ODDS_TOTAL_LINE_bet365": [220.0, 220.0, 220.0],
            "PRED_TOTAL_POINTS_FULL_DATASET": [222.0, 222.0, 221.0],
            "PRED_TOTAL_POINTS_LAST_5_SEASONS": [223.0, 223.0, 219.0],
            "PRED_TOTAL_POINTS_LAST_3_SEASONS": [224.0, 218.0, 219.0],
            "PRED_LINE_ERROR_FULL_DATASET": [-1.0, 2.0, 0.0],
            "PRED_LINE_ERROR_LAST_5_SEASONS": [-2.0, -1.0, -1.0],
            "PRED_LINE_ERROR_LAST_3_SEASONS": [-3.0, -2.0, 1.0],
        }
    )


def test_build_total_points_error_space_predictions() -> None:
    df = _make_meta_df()

    converted = build_total_points_error_space_predictions(df)

    assert list(converted.columns) == [
        "PRED_TOTAL_POINTS_FULL_DATASET__ERR",
        "PRED_TOTAL_POINTS_LAST_5_SEASONS__ERR",
        "PRED_TOTAL_POINTS_LAST_3_SEASONS__ERR",
    ]
    assert list(converted.iloc[0]) == [2.0, 3.0, 4.0]


def test_build_base_avg_all_6_error() -> None:
    df = _make_meta_df()

    baseline = build_base_avg_all_6_error(df)

    assert baseline.name == BASE_AVG_ALL_6_ERR_COL
    assert baseline.round(6).tolist() == [0.5, 0.333333, -0.166667]


def test_build_majority_all_6_tie_line_error_full_dataset_error() -> None:
    df = _make_meta_df()

    baseline = build_base_majority_all_6_tie_line_error_full_dataset_error(df)

    assert (
        baseline.name
        == BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL
    )
    assert baseline.tolist() == [-1.0, 1.0, -1.0]


def test_build_base_majority_total_only_error() -> None:
    df = _make_meta_df()

    baseline = build_base_majority_total_only_error(df)

    assert baseline.name == BASE_MAJORITY_TOTAL_ONLY_ERR_COL
    assert baseline.tolist() == [1.0, 1.0, -1.0]


def test_build_base_majority_line_error_only_error() -> None:
    df = _make_meta_df()

    baseline = build_base_majority_line_error_only_error(df)

    assert baseline.name == BASE_MAJORITY_LINE_ERROR_ONLY_ERR_COL
    assert baseline.tolist() == [-1.0, -1.0, 0.0]


def test_add_default_meta_learner_baselines() -> None:
    df = _make_meta_df()

    enriched = add_default_meta_learner_baselines(df)

    assert BASE_AVG_ALL_6_ERR_COL in enriched.columns
    assert BASE_MAJORITY_TOTAL_ONLY_ERR_COL in enriched.columns
    assert BASE_MAJORITY_LINE_ERROR_ONLY_ERR_COL in enriched.columns
    assert (
        BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL
        in enriched.columns
    )
    assert enriched[BASE_AVG_ALL_6_ERR_COL].round(6).tolist() == [
        0.5,
        0.333333,
        -0.166667,
    ]
    assert enriched[BASE_MAJORITY_TOTAL_ONLY_ERR_COL].tolist() == [1.0, 1.0, -1.0]
    assert enriched[BASE_MAJORITY_LINE_ERROR_ONLY_ERR_COL].tolist() == [-1.0, -1.0, 0.0]
    assert enriched[
        BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL
    ].tolist() == [-1.0, 1.0, -1.0]
