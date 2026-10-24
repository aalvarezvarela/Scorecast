import math

import numpy as np
import pandas as pd
import pytest

from training_pipeline.baseline import (
    compute_baseline_metrics,
    compute_baseline_metrics_across_folds,
    compute_baseline_metrics_for_rows,
    compute_bias_corrected_baseline_metrics,
    compute_line_error_bias,
)


def test_baseline_metrics_match_hand_computed_values_for_constant_offset():
    y_true = pd.Series([210.0, 220.0, 200.0, 230.0])
    baseline_line = pd.Series([205.0, 215.0, 195.0, 225.0])  # always 5 points low

    metrics = compute_baseline_metrics(
        y_true_total_points=y_true, baseline_line=baseline_line, line_col="TEST_LINE"
    )

    assert metrics.n_games == 4
    assert metrics.mae == pytest.approx(5.0)
    assert metrics.rmse == pytest.approx(5.0)
    assert metrics.r2 == pytest.approx(1.0 - (4 * 25.0) / np.sum((y_true - y_true.mean()) ** 2))


def test_baseline_ou_accuracy_is_nan_not_zero():
    y_true = pd.Series([210.0, 220.0])
    baseline_line = pd.Series([205.0, 215.0])

    metrics = compute_baseline_metrics(
        y_true_total_points=y_true, baseline_line=baseline_line, line_col="TEST_LINE"
    )

    assert metrics.ou_accuracy is None


def test_baseline_metrics_drops_non_finite_rows():
    y_true = pd.Series([210.0, np.nan, 200.0])
    baseline_line = pd.Series([205.0, 215.0, np.nan])

    metrics = compute_baseline_metrics(
        y_true_total_points=y_true, baseline_line=baseline_line, line_col="TEST_LINE"
    )

    assert metrics.n_games == 1


def test_baseline_metrics_raises_when_no_valid_rows():
    y_true = pd.Series([np.nan, np.nan])
    baseline_line = pd.Series([1.0, 2.0])

    with pytest.raises(ValueError):
        compute_baseline_metrics(
            y_true_total_points=y_true, baseline_line=baseline_line, line_col="TEST_LINE"
        )


def test_compute_baseline_metrics_for_rows_slices_by_position():
    df = pd.DataFrame(
        {
            "TOTAL_POINTS": [210.0, 220.0, 200.0, 230.0],
            "BASELINE_LINE": [205.0, 215.0, 195.0, 225.0],
        }
    )
    metrics = compute_baseline_metrics_for_rows(
        df, np.array([0, 2]), baseline_line_col="BASELINE_LINE"
    )
    assert metrics.n_games == 2
    assert metrics.mae == pytest.approx(5.0)


def test_compute_line_error_bias_is_mean_signed_error():
    df = pd.DataFrame(
        {
            "TOTAL_POINTS": [215.0, 225.0, 205.0],
            "LINE": [210.0, 210.0, 210.0],
        }
    )
    assert compute_line_error_bias(df, baseline_line_col="LINE") == pytest.approx(5.0)


def test_compute_line_error_bias_rejects_all_missing_rows():
    df = pd.DataFrame({"TOTAL_POINTS": [np.nan], "LINE": [210.0]})
    with pytest.raises(ValueError):
        compute_line_error_bias(df, baseline_line_col="LINE")


def test_bias_corrected_baseline_is_a_harder_null_than_the_raw_line():
    """Shifting the line by its historical drift must reduce MAE whenever the
    line is systematically biased -- that is the point of the stronger null.
    """
    df = pd.DataFrame(
        {
            "TOTAL_POINTS": [215.0, 216.0, 214.0, 215.0],
            "LINE": [210.0, 210.0, 210.0, 210.0],
        }
    )
    raw = compute_baseline_metrics(
        y_true_total_points=df["TOTAL_POINTS"],
        baseline_line=df["LINE"],
        line_col="LINE",
    )
    bias = compute_line_error_bias(df, baseline_line_col="LINE")
    corrected = compute_bias_corrected_baseline_metrics(
        df, baseline_line_col="LINE", bias=bias
    )

    assert bias == pytest.approx(5.0)
    assert raw.mae == pytest.approx(5.0)
    assert corrected.mae == pytest.approx(0.5)
    assert corrected.mae < raw.mae
    assert "bias" in corrected.line_col


def test_compute_baseline_metrics_across_folds_uses_validation_rows_only():
    df = pd.DataFrame(
        {
            "TOTAL_POINTS": [210.0, 220.0, 200.0, 230.0, 240.0, 250.0],
            "BASELINE_LINE": [205.0, 215.0, 195.0, 220.0, 230.0, 240.0],
        }
    )
    splits = [
        (np.array([0, 1]), np.array([2, 3])),
        (np.array([0, 1, 2, 3]), np.array([4, 5])),
    ]

    fold_df, aggregate = compute_baseline_metrics_across_folds(
        df, splits, baseline_line_col="BASELINE_LINE", pooled=False
    )

    assert len(fold_df) == 2
    assert fold_df.loc[0, "mae"] == pytest.approx(7.5)  # |200-195|=5, |230-220|=10 -> mean 7.5
    assert fold_df.loc[1, "mae"] == pytest.approx(10.0)  # |240-230|=10, |250-240|=10
    assert aggregate.mae == pytest.approx((7.5 + 10.0) / 2)
    assert aggregate.ou_accuracy is None
    assert not math.isnan(aggregate.mae)


# ---------------------------------------------------------------------------
# the baseline must be aggregated the same way the model's objective is
# ---------------------------------------------------------------------------


def _uneven_folds():
    """Deliberately unequal fold sizes -- the only case where the two
    aggregations differ. With equal folds the bug is invisible, which is
    exactly why it survived: rolling_origin fold sizes swing with the NBA
    schedule (measured 2 to 36 games on cell A), while the fold layouts this
    pipeline was built on were 12 blocks of 50."""
    df = pd.DataFrame(
        {
            #                 fold 1 (4 games, error 1)   fold 2 (1 game, error 21)
            "TOTAL_POINTS": [201.0, 201.0, 201.0, 201.0, 221.0],
            "BASELINE_LINE": [200.0, 200.0, 200.0, 200.0, 200.0],
        }
    )
    splits = [
        (np.array([4]), np.array([0, 1, 2, 3])),
        (np.array([0, 1, 2, 3]), np.array([4])),
    ]
    return df, splits


def test_pooled_baseline_weights_every_game_equally():
    """Pooled means one metric over the concatenated validation games, exactly
    as _PooledCollector does for the model: (1+1+1+1+21)/5 = 5.0."""
    df, splits = _uneven_folds()

    _, aggregate = compute_baseline_metrics_across_folds(
        df, splits, baseline_line_col="BASELINE_LINE", pooled=True
    )

    assert aggregate.mae == pytest.approx(5.0)
    assert aggregate.n_games == 5


def test_mean_baseline_weights_every_fold_equally():
    """The other aggregation, unchanged: (1 + 21)/2 = 11.0. A 1-game fold
    carries the same weight as a 4-game one."""
    df, splits = _uneven_folds()

    _, aggregate = compute_baseline_metrics_across_folds(
        df, splits, baseline_line_col="BASELINE_LINE", pooled=False
    )

    assert aggregate.mae == pytest.approx(11.0)
    assert aggregate.n_games == 5


def test_the_two_aggregations_actually_disagree():
    """Guards the guard: if these ever coincide the two tests above would both
    pass under a hardcoded aggregation and prove nothing."""
    df, splits = _uneven_folds()

    _, pooled = compute_baseline_metrics_across_folds(
        df, splits, baseline_line_col="BASELINE_LINE", pooled=True
    )
    _, mean = compute_baseline_metrics_across_folds(
        df, splits, baseline_line_col="BASELINE_LINE", pooled=False
    )

    assert pooled.mae != pytest.approx(mean.mae)
    assert pooled.rmse != pytest.approx(mean.rmse)


def test_pooled_baseline_scores_exactly_the_model_s_validation_rows():
    """Overlapping folds are not deduplicated: the model's pooled metric counts
    a repeated game twice, so the baseline has to as well. Matching the model
    beats being tidy."""
    df = pd.DataFrame(
        {
            "TOTAL_POINTS": [210.0, 220.0, 260.0],
            "BASELINE_LINE": [200.0, 200.0, 200.0],
        }
    )
    # Game 2 appears in both folds, as an overlapping splitter would produce.
    splits = [(np.array([0]), np.array([1, 2])), (np.array([1]), np.array([2]))]

    _, aggregate = compute_baseline_metrics_across_folds(
        df, splits, baseline_line_col="BASELINE_LINE", pooled=True
    )

    # (20 + 60 + 60) / 3, not (20 + 60) / 2.
    assert aggregate.n_games == 3
    assert aggregate.mae == pytest.approx(140.0 / 3)


def test_pooled_flag_is_required():
    """No default. A caller that does not say which aggregation it wants is the
    bug this whole module was fixed for, and it must not be expressible."""
    df, splits = _uneven_folds()

    with pytest.raises(TypeError):
        compute_baseline_metrics_across_folds(
            df, splits, baseline_line_col="BASELINE_LINE"
        )
