"""Refit strategy + final holdout scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from training_pipeline.baseline import (
    BaselineMetrics,
    compute_baseline_metrics_for_rows,
    compute_bias_corrected_baseline_metrics,
)
from training_pipeline.betting import (
    OUTCOME_COLUMN,
    BettingMetrics,
    betting_threshold_sweep,
    evaluate_betting,
)
from training_pipeline.calibration import (
    CalibrationSummary,
    calibration_summary,
    calibration_table,
)
from training_pipeline.config import (
    RESIDUAL_TARGET_COLUMNS,
    ExperimentConfig,
    RefitStrategy,
)
from training_pipeline.data import rolling_window_index
from training_pipeline.decisions import (
    collect_prices,
    predict_decisions,
    primary_threshold,
    threshold_sweep_values,
)
from training_pipeline.line_scoring import (
    build_line_comparison,
    collect_comparison_lines,
)
from training_pipeline.tuning import TargetFamilyStrategy


def train_production_model(
    strategy: TargetFamilyStrategy,
    *,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    dates_dev: pd.Series | None,
    study: optuna.Study,
    config: ExperimentConfig,
    game_ids: pd.Series | None = None,
) -> tuple[XGBRegressor, optuna.trial.FrozenTrial | None]:
    """Fit the model that would actually be shipped.

    Called with the FULL dataset (dev + test), not just dev: nothing is held
    back in production, so the shipped model should see the freshest games.
    That also means this model is never itself scored -- the walk-forward
    evaluation measures the *procedure* that produces it.

    RefitStrategy.FULL_DATASET fits on everything using study.best_trial
    directly. ROLLING_WINDOW (default) optionally re-selects the trial
    lexicographically, then fits on the most recent
    ``walk_forward.train_games`` GAMES.

    ``game_ids`` is a per-row game identifier sharing X_dev's index, needed only
    on a frame holding several rows per game (the pooled intermediate-line
    dataset). Without it the window counts rows, which on that dataset is a
    tenth of the history the hyperparameters were selected under. None on every
    one-row-per-game frame, where the two are the same number.
    """
    if config.refit.strategy == RefitStrategy.FULL_DATASET:
        model = strategy.fit_best(
            X_dev=X_dev,
            y_dev=y_dev,
            study=study,
            trial=None,
            config=config,
            dates_dev=dates_dev,
        )
        return model, None

    # Same window the CV folds used, so the final model is fitted on the
    # amount of history the hyperparameters were selected for.
    train_games = config.walk_forward.train_games

    selected_trial: optuna.trial.FrozenTrial | None = None
    if config.refit.use_lexicographic_selection:
        selected_trial = strategy.select_best_trial(
            study,
            mae_tolerance_abs=config.optuna.mae_tolerance_abs,
            mae_tolerance_pct=config.optuna.mae_tolerance_pct,
        )

    if train_games is not None:
        window = rolling_window_index(X_dev.index, train_games, game_ids=game_ids)
        X_dev = X_dev.loc[window]
        y_dev = y_dev.loc[window]
        dates_dev = dates_dev.loc[window] if dates_dev is not None else None

    if selected_trial is not None:
        model = strategy.fit_best(
            X_dev=X_dev,
            y_dev=y_dev,
            study=None,
            trial=selected_trial,
            config=config,
            dates_dev=dates_dev,
        )
    else:
        model = strategy.fit_best(
            X_dev=X_dev,
            y_dev=y_dev,
            study=study,
            trial=None,
            config=config,
            dates_dev=dates_dev,
        )

    return model, selected_trial


@dataclass
class HoldoutEvaluationResult:
    mae: float
    rmse: float
    r2: float
    ou_accuracy: float
    baseline_holdout: BaselineMetrics
    threshold_results: pd.DataFrame
    predictions_df: pd.DataFrame
    #: Profit-oriented metrics across every configured edge threshold.
    betting_sweep: pd.DataFrame
    #: The row of ``betting_sweep`` at betting.primary_edge_threshold -- the
    #: number that answers "would this have made money".
    betting_primary: BettingMetrics
    #: "Trust the line + its historical drift" null, fitted on dev rows only.
    baseline_bias_corrected: BaselineMetrics
    baseline_bias_corrected_betting: BettingMetrics
    dev_line_error_bias: float
    #: The same predictions re-scored against betting.comparison_line_cols.
    #: None when none were configured, none survived into the data, or the
    #: strategy is the classifier (no predicted total to re-express).
    line_comparison: pd.DataFrame | None = None
    #: Probability quality. Classifier only.
    calibration: CalibrationSummary | None = None
    calibration_buckets: pd.DataFrame | None = None


def _extract_zero_threshold_accuracy(threshold_results: pd.DataFrame) -> float:
    """Directional accuracy with no selection filter applied.

    Each strategy names its own threshold column: points for the two
    regressors, expected value for the classifier. Reading the zero row keeps
    "accuracy over every candidate game" comparable across all three.
    """
    if "threshold_min_ev" in threshold_results.columns:
        accuracy_col, threshold_col = "ou_betting_accuracy", "threshold_min_ev"
    elif "ou_betting_accuracy" in threshold_results.columns:
        accuracy_col, threshold_col = "ou_betting_accuracy", "threshold_abs_pred_edge_gt"
    else:
        accuracy_col, threshold_col = "directional_accuracy", "threshold_abs_pred_error_gt"

    zero_row = threshold_results.loc[threshold_results[threshold_col] == 0]
    if zero_row.empty:
        return float("nan")
    return float(zero_row.iloc[0][accuracy_col])


def evaluate_on_holdout(
    strategy: TargetFamilyStrategy,
    model: XGBRegressor,
    *,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_test_full: pd.DataFrame,
    baseline_line_col: str,
    target_line_col: str,
    dev_line_error_bias: float,
    config: ExperimentConfig,
) -> HoldoutEvaluationResult:
    y_true = pd.to_numeric(y_test, errors="coerce").to_numpy(dtype=float)
    target_line_values = pd.to_numeric(
        df_test_full[target_line_col], errors="coerce"
    ).to_numpy(dtype=float)

    # Prices are resolved BEFORE the decision: for a classifier the side is
    # chosen on expected value, so an asymmetric price can flip which side is
    # the better bet. Selecting on flat odds and settling on real ones would
    # score a bet the model would never have placed.
    over_prices, under_prices = collect_prices(df_test_full, config)

    decisions = predict_decisions(
        model,
        X_test,
        config=config,
        target_line=target_line_values,
        decimal_odds_over=over_prices,
        decimal_odds_under=under_prices,
    )
    y_pred = decisions.raw_prediction

    # Point-error metrics do not apply to a probability scored against a 0/1
    # label; NaN keeps them from being compared to a regressor's points MAE.
    if config.is_classifier:
        mae = rmse = r2 = float("nan")
    else:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))

    threshold_results, _ = strategy.evaluate_holdout(model, X_test, y_test, config)
    ou_accuracy = _extract_zero_threshold_accuracy(threshold_results)

    baseline_holdout = compute_baseline_metrics_for_rows(
        df_test_full,
        np.arange(len(df_test_full)),
        baseline_line_col=baseline_line_col,
        outcome_col=config.outcome_col,
    )

    baseline_line = pd.to_numeric(
        df_test_full[baseline_line_col], errors="coerce"
    ).to_numpy(dtype=float)

    # baseline_pred must live in the SAME space as y_true/y_pred so the three
    # columns are directly comparable in the saved parquet. For a TOTAL_POINTS
    # run that space is points (the line itself); for a RESIDUAL target it is
    # error-vs-line, where "trust the line" means predicting exactly 0. The
    # raw line is kept separately as baseline_line.
    #
    # SPREAD_ERROR is a residual target exactly as LINE_ERROR is, so it belongs
    # in the same branch. Falling through to `baseline_line` would put a spread
    # line (~0 +/- 7 points of handicap) beside a target that is already a
    # residual -- two different spaces in adjacent columns, with no NaN and no
    # error to reveal it.
    if config.is_classifier:
        # There is no prediction in points space to place beside y_true here;
        # the comparable "trust the line" null lives in the betting metrics.
        baseline_pred = np.full_like(y_pred, np.nan, dtype=float)
    elif config.target_col in RESIDUAL_TARGET_COLUMNS:
        baseline_pred = np.zeros_like(y_pred)
    else:
        baseline_pred = baseline_line

    # --- Betting evaluation -------------------------------------------------
    # Bets settle against the line the target is defined relative to, which is
    # not necessarily baseline_line_col (that one may point at an alternative
    # consensus line purely for the MAE comparison).
    actual_total = pd.to_numeric(
        df_test_full[config.outcome_col], errors="coerce"
    ).to_numpy(dtype=float)
    target_line = target_line_values
    predicted_edge = decisions.predicted_edge

    betting_kwargs = {
        "actual_total": actual_total,
        "line": target_line,
        "flat_decimal_odds": config.betting.flat_decimal_odds,
        "decimal_odds_over": over_prices,
        "decimal_odds_under": under_prices,
    }

    betting_sweep = betting_threshold_sweep(
        predicted_edge=predicted_edge,
        thresholds=threshold_sweep_values(config),
        selection_score=decisions.selection_score,
        **betting_kwargs,
    )
    betting_primary = evaluate_betting(
        predicted_edge=predicted_edge,
        min_edge=primary_threshold(config),
        selection_score=decisions.selection_score,
        **betting_kwargs,
    )

    baseline_bias_corrected = compute_bias_corrected_baseline_metrics(
        df_test_full,
        baseline_line_col=baseline_line_col,
        bias=dev_line_error_bias,
        outcome_col=config.outcome_col,
    )
    # The bias-corrected null bets the same side on every game with a constant
    # edge, which makes it all-or-nothing under a min-edge filter: it would
    # place either every bet or none, depending purely on whether the drift
    # happens to exceed the threshold. It is a fixed (non-selective) strategy,
    # so it is scored on every candidate game and compared against the model's
    # selective ROI.
    baseline_bias_corrected_betting = evaluate_betting(
        predicted_edge=np.full_like(actual_total, dev_line_error_bias),
        min_edge=0.0,
        **betting_kwargs,
    )

    line_comparison = None if config.is_classifier else build_line_comparison(
        y_pred=y_pred,
        target_line=target_line,
        actual_total=actual_total,
        lines=collect_comparison_lines(
            df_test_full, config, target_line_col=target_line_col
        ),
        config=config,
    )

    calibration: CalibrationSummary | None = None
    calibration_buckets: pd.DataFrame | None = None
    if decisions.p_over is not None:
        calibration = calibration_summary(
            y_true, decisions.p_over, n_buckets=config.betting.calibration_buckets
        )
        calibration_buckets = calibration_table(
            y_true, decisions.p_over, n_buckets=config.betting.calibration_buckets
        )

    predictions_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "baseline_pred": baseline_pred,
            "baseline_line": baseline_line,
            "target_line": target_line,
            "predicted_edge": predicted_edge,
            "selection_score": decisions.selection_score,
            OUTCOME_COLUMN: actual_total,
            # Kept under its historical name too when it really is the total, so
            # archived readers and notebooks that select "TOTAL_POINTS" keep
            # working on totals runs. Absent on spread runs by design.
            **(
                {"TOTAL_POINTS": actual_total}
                if config.outcome_col == "TOTAL_POINTS"
                else {}
            ),
            config.data.date_col: df_test_full[config.data.date_col].to_numpy(),
            **(
                {}
                if decisions.p_over is None
                else {
                    "p_over": decisions.p_over,
                    "expected_value": decisions.expected_value,
                    "bets_over": decisions.bets_over,
                }
            ),
        }
    )

    return HoldoutEvaluationResult(
        mae=mae,
        rmse=rmse,
        r2=r2,
        ou_accuracy=ou_accuracy,
        baseline_holdout=baseline_holdout,
        threshold_results=threshold_results,
        predictions_df=predictions_df,
        betting_sweep=betting_sweep,
        betting_primary=betting_primary,
        baseline_bias_corrected=baseline_bias_corrected,
        baseline_bias_corrected_betting=baseline_bias_corrected_betting,
        dev_line_error_bias=dev_line_error_bias,
        line_comparison=line_comparison,
        calibration=calibration,
        calibration_buckets=calibration_buckets,
    )
