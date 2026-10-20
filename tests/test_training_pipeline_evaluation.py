import numpy as np
import pandas as pd
import pytest

from training_pipeline.config import DataConfig, ExperimentConfig, TargetFamily
from training_pipeline.evaluation import evaluate_on_holdout


class _StubModel:
    def __init__(self, predictions):
        self._predictions = np.asarray(predictions, dtype=float)

    def predict(self, X):
        return self._predictions


class _StubStrategy:
    """Returns a threshold table shaped like nba_ou.modeling.scorers output."""

    def __init__(self, *, accuracy_col: str, threshold_col: str):
        self.accuracy_col = accuracy_col
        self.threshold_col = threshold_col

    def evaluate_holdout(self, model, X_test, y_test, config):
        table = pd.DataFrame(
            {
                self.threshold_col: [0, 1, 2],
                "n_games": [4, 3, 2],
                self.accuracy_col: [0.55, 0.60, 0.65],
            }
        )
        return table, np.zeros(len(X_test))


def _df_test_full(line_values, total_points):
    return pd.DataFrame(
        {
            "TOTAL_POINTS": total_points,
            "ODDS_TOTAL_LINE_bet365": line_values,
            "GAME_DATE": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]
            ),
        }
    )


def _total_points_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="eval_test",
        target_family=TargetFamily.TOTAL_POINTS,
        line_col="ODDS_TOTAL_LINE_bet365",
        data=DataConfig(csv_path="x.csv"),
    )


def _line_error_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="eval_test_le",
        target_family=TargetFamily.LINE_ERROR,
        data=DataConfig(csv_path="x.csv"),
    )


def test_total_points_predictions_df_baseline_pred_is_the_line():
    total_points = [210.0, 220.0, 200.0, 230.0]
    lines = [205.0, 215.0, 195.0, 225.0]
    df_test = _df_test_full(lines, total_points)

    result = evaluate_on_holdout(
        _StubStrategy(
            accuracy_col="ou_betting_accuracy", threshold_col="threshold_abs_pred_edge_gt"
        ),
        _StubModel([208.0, 218.0, 198.0, 228.0]),
        X_test=df_test[["ODDS_TOTAL_LINE_bet365"]],
        y_test=pd.Series(total_points),
        df_test_full=df_test,
        baseline_line_col="ODDS_TOTAL_LINE_bet365",
        target_line_col="ODDS_TOTAL_LINE_bet365",
        dev_line_error_bias=0.0,
        config=_total_points_config(),
    )

    preds = result.predictions_df
    assert preds["baseline_pred"].tolist() == pytest.approx(lines)
    assert preds["baseline_line"].tolist() == pytest.approx(lines)
    assert result.ou_accuracy == pytest.approx(0.55)


def test_line_error_predictions_df_baseline_pred_is_zero_not_the_raw_line():
    """Regression: baseline_pred used to hold the raw line (~220) while
    y_true/y_pred were in error space (~5), making the columns in the saved
    parquet incomparable. "Trust the line" means predicting exactly 0 error.
    """
    total_points = [210.0, 220.0, 200.0, 230.0]
    lines = [205.0, 215.0, 195.0, 225.0]
    line_errors = [5.0, 5.0, 5.0, 5.0]
    df_test = _df_test_full(lines, total_points)

    result = evaluate_on_holdout(
        _StubStrategy(
            accuracy_col="directional_accuracy",
            threshold_col="threshold_abs_pred_error_gt",
        ),
        _StubModel([4.0, 6.0, 3.0, 7.0]),
        X_test=df_test[["ODDS_TOTAL_LINE_bet365"]],
        y_test=pd.Series(line_errors),
        df_test_full=df_test,
        baseline_line_col="ODDS_TOTAL_LINE_bet365",
        target_line_col="ODDS_TOTAL_LINE_bet365",
        dev_line_error_bias=0.0,
        config=_line_error_config(),
    )

    preds = result.predictions_df
    assert preds["baseline_pred"].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert preds["baseline_line"].tolist() == pytest.approx(lines)
    # y_pred stays in error space, directly comparable to baseline_pred.
    assert preds["y_pred"].tolist() == pytest.approx([4.0, 6.0, 3.0, 7.0])


def test_bias_corrected_null_still_bets_when_drift_is_below_the_edge_threshold():
    """Regression: the bias-corrected null has a *constant* edge, so scoring it
    under the model's min-edge filter made it place either every bet or none.
    With a small drift it placed none, and the comparison column was empty. It
    must be scored on every candidate game instead.
    """
    total_points = [220.0, 200.0, 225.0, 195.0]
    lines = [210.0] * 4
    df_test = _df_test_full(lines, total_points)

    result = evaluate_on_holdout(
        _StubStrategy(
            accuracy_col="ou_betting_accuracy", threshold_col="threshold_abs_pred_edge_gt"
        ),
        _StubModel([215.0, 205.0, 220.0, 200.0]),
        X_test=df_test[["ODDS_TOTAL_LINE_bet365"]],
        y_test=pd.Series(total_points),
        df_test_full=df_test,
        baseline_line_col="ODDS_TOTAL_LINE_bet365",
        target_line_col="ODDS_TOTAL_LINE_bet365",
        # Drift far smaller than the 2.0 primary edge threshold.
        dev_line_error_bias=-0.67,
        config=_total_points_config(),
    )

    null_betting = result.baseline_bias_corrected_betting
    assert null_betting.n_bets == 4
    assert null_betting.roi is not None


def test_baseline_metrics_are_scored_in_points_space_for_both_targets():
    """The baseline is always scored TOTAL_POINTS vs line, so the number is
    comparable across target families.
    """
    total_points = [210.0, 220.0, 200.0, 230.0]
    lines = [205.0, 215.0, 195.0, 225.0]
    df_test = _df_test_full(lines, total_points)

    result = evaluate_on_holdout(
        _StubStrategy(
            accuracy_col="directional_accuracy",
            threshold_col="threshold_abs_pred_error_gt",
        ),
        _StubModel([4.0, 6.0, 3.0, 7.0]),
        X_test=df_test[["ODDS_TOTAL_LINE_bet365"]],
        y_test=pd.Series([5.0, 5.0, 5.0, 5.0]),
        df_test_full=df_test,
        baseline_line_col="ODDS_TOTAL_LINE_bet365",
        target_line_col="ODDS_TOTAL_LINE_bet365",
        dev_line_error_bias=0.0,
        config=_line_error_config(),
    )

    assert result.baseline_holdout.mae == pytest.approx(5.0)
    assert result.baseline_holdout.ou_accuracy is None
