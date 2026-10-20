"""Daily walk-forward backtest: the closest thing to a production dry run.

For every game-day in the backtest window the model is refitted on all games
that finished strictly before that day -- including earlier backtest days,
which become training data once played -- and then predicts only that day's
games. No result is visible to the model before it would have been in real
life, so the resulting predictions are genuinely out-of-sample in a way that a
single train/test split is not.

The day-by-day loop itself is
``nba_ou.modeling.modeling.evaluate_day_by_day_walk_forward``; this module
supplies the fit/predict closure, sizes the window, and scores the pooled
predictions with the same betting layer used elsewhere.

Hyperparameters are fixed across all days. Re-tuning daily would be
unrealistic and prohibitively slow (one Optuna study per day). The cost of
this choice: if those hyperparameters were selected on data overlapping the
backtest window, the backtest inherits that selection bias. Tune on data
preceding the window to keep it honest.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from nba_ou.modeling.modeling import (
    build_recency_sample_weights,
    evaluate_day_by_day_walk_forward,
    split_latest_dates_holdout,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBClassifier, XGBRegressor

from training_pipeline.baseline import (
    BaselineMetrics,
    compute_baseline_metrics,
    compute_bias_corrected_baseline_metrics,
    compute_line_error_bias,
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
from training_pipeline.config import ExperimentConfig
from training_pipeline.data import (
    PreparedDataset,
    build_feature_matrix,
    prepare_dataset,
    training_eligible_mask,
)
from training_pipeline.decisions import (
    collect_prices,
    decisions_from_pooled_predictions,
    primary_threshold,
    threshold_sweep_values,
)
from training_pipeline.line_scoring import (
    build_line_comparison,
    collect_comparison_lines,
)
from training_pipeline.tuning import (  # noqa: E402
    NON_XGB_TRIAL_PARAMS as _NON_XGB_TRIAL_PARAMS,
)
from training_pipeline.tuning import (  # noqa: E402
    USE_SAMPLE_WEIGHT_PARAM,
    resolve_final_params,
)


def xgb_params_from_trial(
    trial: optuna.trial.FrozenTrial,
    *,
    config: ExperimentConfig | None = None,
) -> tuple[dict[str, Any], int, float | None]:
    """Extract fixed hyperparameters from a tuned Optuna trial.

    Returns ``(xgb_params, n_estimators, sample_weight_lambda)``.
    ``sample_weight_lambda`` is separated out because it is a training-protocol
    parameter, not an XGBoost one -- passing it through to ``XGBRegressor``
    would be silently accepted and then ignored.

    **Pass ``config`` whenever you have one.** Without it the returned lambda is
    ambiguous: ``None`` means both "this trial chose not to weight" and "this
    trial never sampled a lambda because the config pins one". Only the config
    can tell those apart, and getting it wrong silently reinstates weighting a
    trial explicitly rejected. With a config this delegates to the single
    authoritative implementation in ``tuning.resolve_final_params``.
    """
    if config is not None:
        return resolve_final_params(trial, config)

    from nba_ou.modeling.optuna_total_points import get_trial_n_estimators

    params = {k: v for k, v in trial.params.items() if k not in _NON_XGB_TRIAL_PARAMS}
    lambda_ = trial.params.get("sample_weight_lambda")
    if trial.params.get(USE_SAMPLE_WEIGHT_PARAM) is False:
        lambda_ = None
    return params, get_trial_n_estimators(trial), lambda_


def _window_group_col(
    config: ExperimentConfig, prepared: PreparedDataset
) -> str | None:
    """The column ``train_games`` must be counted over, or None for row counts.

    None whenever the frame holds one row per game, which keeps every
    closing-line run on exactly the arithmetic it has always used.
    """
    if not prepared.is_pooled_snapshots:
        return None
    return config.data.game_id_col


def _build_static_params(
    config: ExperimentConfig, *, random_state: int | None = None
) -> dict[str, Any]:
    return {
        "booster": "gbtree",
        "tree_method": "hist",
        "device": config.device,
        "objective": config.optuna.objective_name,
        "eval_metric": "logloss" if config.is_classifier else "mae",
        "random_state": config.random_state if random_state is None else random_state,
        "n_jobs": -1,
        "verbosity": 0,
    }


def _make_fit_and_predict(
    config: ExperimentConfig,
    *,
    target_col: str,
    xgb_params: dict[str, Any],
    n_estimators: int,
    sample_weight_lambda: float | None,
    feature_names: list[str],
    random_state: int | None = None,
) -> Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]:
    """Closure fitting one model per day and predicting that day's games."""
    final_params = {
        **_build_static_params(config),
        **xgb_params,
        "n_estimators": n_estimators,
    }
    # Applied last on purpose: _build_static_params has already seeded from the
    # config, so without this every "different" evaluation seed would fit
    # identically (a test asserts exactly that). A user-supplied
    # optuna.fixed_params may also carry its own random_state.
    if random_state is not None:
        final_params["random_state"] = random_state
    # No eval_set exists inside the daily loop, so early stopping cannot apply.
    final_params.pop("early_stopping_rounds", None)
    date_col = config.data.date_col

    def fit_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        # Filter the day's TRAINING history only. test_df is never touched, so
        # every game of the evaluation period is still predicted and scored.
        eligible = training_eligible_mask(train_df, config)
        if not eligible.all():
            train_df = train_df.loc[eligible].copy()

        # By allow-list: these frames carry GAME_ID (the daily window counts
        # games with it) and may carry joined closing lines, and a deny-list
        # would hand both to the model.
        X_train, y_train = build_feature_matrix(
            train_df, target_col=target_col, feature_names=feature_names
        )
        X_test, _ = build_feature_matrix(
            test_df, target_col=target_col, feature_names=feature_names
        )

        sample_weight = None
        if sample_weight_lambda is not None:
            sample_weight = build_recency_sample_weights(
                train_df[date_col], lambda_=float(sample_weight_lambda)
            ).to_numpy(dtype=float)

        estimator_cls = XGBClassifier if config.is_classifier else XGBRegressor
        model = estimator_cls(**final_params)
        model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
        if config.is_classifier:
            # P(OVER). The day-by-day loop only knows how to carry one number
            # per game, so the probability travels as "y_pred" and is turned
            # back into a side and an EV once the days are pooled.
            return np.asarray(model.predict_proba(X_test), dtype=float)[:, 1]
        return np.asarray(model.predict(X_test), dtype=float)

    return fit_and_predict


@dataclass
class DailyBacktestResult:
    """Pooled outcome of a daily walk-forward backtest."""

    n_days: int
    n_games: int
    backtest_start: pd.Timestamp
    backtest_end: pd.Timestamp
    mae: float
    rmse: float
    r2: float
    mean_daily_mae: float
    baseline: BaselineMetrics
    betting_primary: BettingMetrics
    betting_sweep: pd.DataFrame
    #: "Trust the line + its historical drift" null, fitted on the history rows
    #: only and applied to the evaluated ones. The same pair of fields
    #: HoldoutEvaluationResult carries, and for the same reason: without them
    #: this path has no harder null than the raw line, and anything reading
    #: "model versus drift" off a daily_walk_forward run would be comparing the
    #: model against itself.
    baseline_bias_corrected: BaselineMetrics
    baseline_bias_corrected_betting: BettingMetrics
    dev_line_error_bias: float
    #: One row per game-day: metric, training-set size and its date range.
    daily_results: pd.DataFrame
    #: One row per game with y_true/y_pred/line/edge, in chronological order.
    predictions: pd.DataFrame
    xgb_params: dict[str, Any]
    n_estimators: int
    sample_weight_lambda: float | None
    #: The seed every daily fit used, so a repeat under another seed is
    #: identifiable in the saved artifacts.
    random_state: int
    #: The same predictions re-scored against betting.comparison_line_cols.
    #: None when none were configured or none survived into the data, and
    #: always None for the classifier, which has no predicted total to
    #: re-express against another line.
    line_comparison: pd.DataFrame | None = None
    #: Probability quality. Classifier only.
    calibration: CalibrationSummary | None = None
    calibration_buckets: pd.DataFrame | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "n_days": self.n_days,
            "n_games": self.n_games,
            "backtest_start": self.backtest_start.isoformat(),
            "backtest_end": self.backtest_end.isoformat(),
            "mae": self.mae,
            "rmse": self.rmse,
            "r2": self.r2,
            "mean_daily_mae": self.mean_daily_mae,
            "baseline_mae": self.baseline.mae,
            "baseline_rmse": self.baseline.rmse,
            "mae_improvement_over_baseline_pct": (
                (self.baseline.mae - self.mae) / self.baseline.mae
                if self.baseline.mae
                else None
            ),
            "roi": self.betting_primary.roi,
            "n_bets": self.betting_primary.n_bets,
            "win_rate": self.betting_primary.win_rate,
            "break_even_rate": self.betting_primary.break_even_rate,
            "is_significant": self.betting_primary.is_significant,
            "profit_units": self.betting_primary.profit_units,
            "xgb_params": self.xgb_params,
            "n_estimators": self.n_estimators,
            "sample_weight_lambda": self.sample_weight_lambda,
            "random_state": self.random_state,
            **(
                {}
                if self.calibration is None
                else {
                    "log_loss": self.calibration.log_loss,
                    "brier": self.calibration.brier,
                    "log_loss_improvement": self.calibration.log_loss_improvement,
                    "expected_calibration_error": (
                        self.calibration.expected_calibration_error
                    ),
                    "mean_bias": self.calibration.mean_bias,
                }
            ),
        }


def run_daily_backtest(
    config: ExperimentConfig,
    *,
    prepared: PreparedDataset | None = None,
    xgb_params: dict[str, Any] | None = None,
    n_estimators: int | None = None,
    sample_weight_lambda: float | None = None,
) -> DailyBacktestResult:
    """Run the daily walk-forward backtest described in BacktestConfig.

    ``prepared`` may be passed to reuse an already-loaded dataset; otherwise it
    is built from ``config`` (which excludes playoffs by default). Explicit
    ``xgb_params``/``n_estimators`` override BacktestConfig; pass the output of
    :func:`xgb_params_from_trial` to backtest a tuned configuration.
    """
    if prepared is None:
        prepared = prepare_dataset(config)

    # No trial exists here to have made a weighting decision, so an explicitly
    # configured lambda is the only signal available and must be honoured.
    # run_walk_forward_evaluation deliberately does not do this for callers that
    # DO have a trial -- see the note there.
    if sample_weight_lambda is None and config.sample_weight.enabled:
        sample_weight_lambda = config.sample_weight.lambda_

    df_history, df_backtest = split_latest_dates_holdout(
        df=prepared.df_full,
        date_col=config.data.date_col,
        test_size=None,
        test_games=config.backtest.test_games,
    )
    return run_walk_forward_evaluation(
        config,
        prepared=prepared,
        df_history=df_history,
        df_evaluation=df_backtest,
        train_games=config.backtest.train_games,
        xgb_params=xgb_params or config.backtest.resolved_xgb_params(),
        n_estimators=n_estimators or config.backtest.resolved_n_estimators(),
        sample_weight_lambda=sample_weight_lambda,
        show_progress=config.backtest.show_progress,
    )


def run_walk_forward_evaluation(
    config: ExperimentConfig,
    *,
    prepared: PreparedDataset,
    df_history: pd.DataFrame,
    df_evaluation: pd.DataFrame,
    train_games: int | None,
    xgb_params: dict[str, Any],
    n_estimators: int,
    sample_weight_lambda: float | None = None,
    show_progress: bool = True,
    random_state: int | None = None,
    dev_line_error_bias: float | None = None,
) -> DailyBacktestResult:
    """Score ``df_evaluation`` one game-day at a time, retraining each day.

    The caller supplies the split, so this serves both the standalone backtest
    and ``run_experiment``'s evaluation of its own holdout period with
    Optuna-tuned hyperparameters.

    ``random_state`` overrides ``config.random_state`` for every daily fit,
    which is how the same evaluation gets repeated under several seeds to
    measure how much of a result is just fit noise.

    ``dev_line_error_bias`` is the drift the bias-corrected null adds to the
    line. It must be measured on data the evaluation window cannot see; the
    caller passes the value it already computed on dev, and when it is omitted
    this falls back to ``df_history``, which is by construction everything
    before the evaluation period. Never derived from ``df_evaluation``: a null
    fitted on the rows it is scored on is not a null.
    """
    resolved_random_state = (
        config.random_state if random_state is None else random_state
    )
    target_col = config.target_col
    resolved_params = xgb_params
    resolved_n_estimators = n_estimators
    # Deliberately NO config fallback here. The caller has already decided --
    # run_experiment resolves the lambda from the selected trial, and a trial
    # that chose not to weight reports None. Treating that None as "unset" and
    # substituting config.sample_weight.lambda_ would re-enable exactly the
    # weighting the trial rejected, and score the model under a regime no trial
    # measured. run_daily_backtest, which has no trial to consult, applies the
    # config fallback itself before calling in.
    resolved_lambda = sample_weight_lambda

    df_backtest = df_evaluation

    walk_forward = evaluate_day_by_day_walk_forward(
        df_history,
        df_backtest,
        fit_and_predict=_make_fit_and_predict(
            config,
            target_col=target_col,
            xgb_params=resolved_params,
            n_estimators=resolved_n_estimators,
            sample_weight_lambda=resolved_lambda,
            feature_names=prepared.feature_names,
            random_state=resolved_random_state,
        ),
        metric_fn=lambda y_true, y_pred: float(mean_absolute_error(y_true, y_pred)),
        target_col=target_col,
        date_col=config.data.date_col,
        max_games=train_games,
        # So train_games means GAMES here too. Without it the daily retrain
        # would keep train_games ROWS while CV kept train_games games, and the
        # holdout would be scoring a model trained on a tenth of the history
        # the tuning was done under -- the two measurements would no longer be
        # about the same model. None on any one-row-per-game frame, where it
        # changes nothing.
        max_games_group_col=_window_group_col(config, prepared),
        metric_name="mae",
        show_progress=show_progress,
        progress_desc=f"Daily backtest ({config.experiment_name})",
    )

    # Re-attach the line and actual total for each predicted game.
    # ``row_in_test_final`` is a positional index into df_backtest.
    positions = walk_forward.predictions["row_in_test_final"].to_numpy()
    target_line = pd.to_numeric(
        df_backtest[prepared.target_line_col], errors="coerce"
    ).to_numpy(dtype=float)[positions]
    baseline_line = pd.to_numeric(
        df_backtest[prepared.baseline_line_col], errors="coerce"
    ).to_numpy(dtype=float)[positions]
    actual_total = pd.to_numeric(
        df_backtest[config.outcome_col], errors="coerce"
    ).to_numpy(dtype=float)[positions]

    y_true = walk_forward.predictions["y_true"].to_numpy(dtype=float)
    y_pred = walk_forward.predictions["y_pred"].to_numpy(dtype=float)

    over_prices, under_prices = collect_prices(
        df_backtest, config, positions=positions
    )
    decisions = decisions_from_pooled_predictions(
        y_pred,
        target_line=target_line,
        config=config,
        decimal_odds_over=over_prices,
        decimal_odds_under=under_prices,
    )
    predicted_edge = decisions.predicted_edge

    betting_kwargs: dict[str, Any] = {
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

    baseline = compute_baseline_metrics(
        y_true_total_points=pd.Series(actual_total),
        baseline_line=pd.Series(baseline_line),
        line_col=prepared.baseline_line_col,
    )

    resolved_bias = (
        compute_line_error_bias(
            df_history,
            baseline_line_col=prepared.baseline_line_col,
            outcome_col=config.outcome_col,
        )
        if dev_line_error_bias is None
        else dev_line_error_bias
    )
    # Scored on exactly the games the model was scored on -- df_backtest rows
    # the walk-forward actually predicted, in its order -- so the two are
    # comparable game for game.
    baseline_bias_corrected = compute_bias_corrected_baseline_metrics(
        df_backtest.iloc[positions],
        baseline_line_col=prepared.baseline_line_col,
        bias=resolved_bias,
        outcome_col=config.outcome_col,
    )
    # A constant edge on every game, so a min-edge filter would take either all
    # of them or none. It is a fixed, non-selective strategy: score every
    # candidate game and let the model's selectivity be the thing that has to
    # earn its keep.
    baseline_bias_corrected_betting = evaluate_betting(
        predicted_edge=np.full_like(actual_total, resolved_bias),
        min_edge=0.0,
        **betting_kwargs,
    )

    # A classifier has no view on the total, so there is nothing to re-score
    # against a different line: its label was defined relative to THIS one.
    line_comparison = (
        None
        if decisions.predicted_total is None
        else build_line_comparison(
            y_pred=y_pred,
            target_line=target_line,
            actual_total=actual_total,
            lines=collect_comparison_lines(
                df_backtest,
                config,
                target_line_col=prepared.target_line_col,
                positions=positions,
            ),
            config=config,
        )
    )

    predictions = walk_forward.predictions.assign(
        target_line=target_line,
        baseline_line=baseline_line,
        **{OUTCOME_COLUMN: actual_total},
        **(
            {"TOTAL_POINTS": actual_total}
            if config.outcome_col == "TOTAL_POINTS"
            else {}
        ),
        predicted_edge=predicted_edge,
        selection_score=decisions.selection_score,
    )
    calibration: CalibrationSummary | None = None
    calibration_buckets: pd.DataFrame | None = None
    if decisions.p_over is not None:
        predictions = predictions.assign(
            p_over=decisions.p_over,
            expected_value=decisions.expected_value,
            bets_over=decisions.bets_over,
        )
        calibration = calibration_summary(
            y_true, decisions.p_over, n_buckets=config.betting.calibration_buckets
        )
        calibration_buckets = calibration_table(
            y_true, decisions.p_over, n_buckets=config.betting.calibration_buckets
        )

    return DailyBacktestResult(
        n_days=int(len(walk_forward.daily_results)),
        n_games=int(len(predictions)),
        backtest_start=pd.Timestamp(walk_forward.daily_results["date"].min()),
        backtest_end=pd.Timestamp(walk_forward.daily_results["date"].max()),
        # Point-error metrics are meaningless against a 0/1 label: the "error"
        # of a probability is not in points and is not comparable to a
        # regressor's MAE. NaN rather than a number that invites the comparison.
        mae=float("nan") if config.is_classifier
        else float(mean_absolute_error(y_true, y_pred)),
        rmse=float("nan") if config.is_classifier
        else float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float("nan") if config.is_classifier
        else float(r2_score(y_true, y_pred)),
        mean_daily_mae=float(walk_forward.mean_metric),
        baseline=baseline,
        betting_primary=betting_primary,
        betting_sweep=betting_sweep,
        baseline_bias_corrected=baseline_bias_corrected,
        baseline_bias_corrected_betting=baseline_bias_corrected_betting,
        dev_line_error_bias=resolved_bias,
        daily_results=walk_forward.daily_results,
        predictions=predictions,
        xgb_params=resolved_params,
        n_estimators=resolved_n_estimators,
        sample_weight_lambda=resolved_lambda,
        random_state=resolved_random_state,
        line_comparison=line_comparison,
        calibration=calibration,
        calibration_buckets=calibration_buckets,
    )
