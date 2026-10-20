"""run_experiment(): the single high-level training method.

Composes every lower-level step in order. Every intermediate object remains
independently callable/inspectable -- a notebook can call run_experiment(config)
and still display() any intermediate table afterward (result.fold_info,
result.trials_df, result.holdout_result.threshold_results, ...), or skip
run_experiment entirely and call the lower-level functions one at a time, the
same way the repo's example notebooks already do.
"""

from __future__ import annotations

import uuid
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from nba_ou.modeling.modeling import save_model_bundle
from xgboost import XGBRegressor

from training_pipeline import naming, tracking
from training_pipeline.backtest import (
    DailyBacktestResult,
    run_walk_forward_evaluation,
    xgb_params_from_trial,
)
from training_pipeline.baseline import (
    BaselineMetrics,
    compute_baseline_metrics_across_folds,
    compute_line_error_bias,
)
from training_pipeline.betting import BettingMetrics
from training_pipeline.calibration import CalibrationSummary
from training_pipeline.config import (
    ExperimentConfig,
    HoldoutEvaluation,
    RefitStrategy,
)
from training_pipeline.cv_betting import (
    CrossValidationBettingResult,
    evaluate_cv_betting,
)
from training_pipeline.data import (
    PreparedDataset,
    build_feature_matrix,
    prepare_dataset,
    rolling_window_index,
    training_eligible_mask,
)
from training_pipeline.evaluation import (
    HoldoutEvaluationResult,
    evaluate_on_holdout,
)
from training_pipeline.season_phase import describe_phases, phases_present
from training_pipeline.snapshot_scoring import (
    build_snapshot_report,
    save_snapshot_report,
)
from training_pipeline.splits import (
    TRAIN_GAMES_PARAM,
    build_holdout_split,
    build_split_provider,
)
from training_pipeline.tuning import (
    TieToleranceResult,
    completed_primary_values,
    fit_final_model,
    get_strategy,
    resolve_tie_tolerance,
)


def _window_game_ids(
    df: pd.DataFrame, config: ExperimentConfig, prepared: PreparedDataset
) -> pd.Series | None:
    """Per-row game identifier for window arithmetic, or None for row counts.

    None on every one-row-per-game frame, which is what keeps the closing-line
    path on the tail slice it has always used.
    """
    if not prepared.is_pooled_snapshots:
        return None
    if config.data.game_id_col not in df.columns:
        return None
    return df[config.data.game_id_col]


def _feature_target_subset(
    df_subset: pd.DataFrame, config: ExperimentConfig, prepared: PreparedDataset
) -> tuple[pd.DataFrame, pd.Series]:
    """Rebuild X/y directly from a dev/test subset rather than slicing
    PreparedDataset.X/.y by index -- split_latest_dates_holdout resets the
    index on both halves it returns, so index-based alignment back to the
    full prepared frame would silently misalign rows.

    Selected by ``prepared.feature_names`` rather than by dropping the
    configured exclusions, so a frame carrying non-feature columns (GAME_ID for
    the window arithmetic, joined closing lines for re-scoring) yields the same
    matrix prepare_dataset settled on.
    """
    return build_feature_matrix(
        df_subset,
        target_col=config.target_col,
        feature_names=prepared.feature_names,
    )


@dataclass
class _EvaluationMetrics:
    """One evaluation's numbers, normalized across both holdout modes.

    daily_walk_forward and single_shot produce different result objects; every
    consumer downstream (artifacts, leaderboard, seed table) wants the same
    handful of numbers, so the mode is absorbed exactly once, here.
    """

    mae: float
    rmse: float
    r2: float
    ou_acc: float
    baseline: BaselineMetrics
    betting_primary: BettingMetrics
    betting_sweep: pd.DataFrame
    #: The harder "line + its historical drift" null. Both modes compute it;
    #: it used to be read off holdout_result alone, with the walk-forward
    #: branch falling back to the MODEL's own metrics -- which made
    #: roi_vs_bias_baseline exactly 0.00 on every daily_walk_forward run,
    #: silently, because a model always ties itself.
    baseline_bias_corrected: BaselineMetrics
    baseline_bias_corrected_betting: BettingMetrics
    predictions: pd.DataFrame
    line_comparison: pd.DataFrame | None
    calibration: CalibrationSummary | None
    calibration_buckets: pd.DataFrame | None


def _normalize_evaluation(
    holdout_result: HoldoutEvaluationResult | None,
    walk_forward_result: DailyBacktestResult | None,
) -> _EvaluationMetrics:
    if holdout_result is not None:
        return _EvaluationMetrics(
            mae=holdout_result.mae,
            rmse=holdout_result.rmse,
            r2=holdout_result.r2,
            ou_acc=holdout_result.ou_accuracy,
            baseline=holdout_result.baseline_holdout,
            betting_primary=holdout_result.betting_primary,
            betting_sweep=holdout_result.betting_sweep,
            baseline_bias_corrected=holdout_result.baseline_bias_corrected,
            baseline_bias_corrected_betting=(
                holdout_result.baseline_bias_corrected_betting
            ),
            predictions=holdout_result.predictions_df,
            line_comparison=holdout_result.line_comparison,
            calibration=holdout_result.calibration,
            calibration_buckets=holdout_result.calibration_buckets,
        )

    assert walk_forward_result is not None
    return _EvaluationMetrics(
        mae=walk_forward_result.mae,
        rmse=walk_forward_result.rmse,
        r2=walk_forward_result.r2,
        # The walk-forward's directional accuracy at the primary edge; there is
        # no single-model threshold sweep to read a zero-edge accuracy from.
        ou_acc=walk_forward_result.betting_primary.win_rate or float("nan"),
        baseline=walk_forward_result.baseline,
        betting_primary=walk_forward_result.betting_primary,
        betting_sweep=walk_forward_result.betting_sweep,
        baseline_bias_corrected=walk_forward_result.baseline_bias_corrected,
        baseline_bias_corrected_betting=(
            walk_forward_result.baseline_bias_corrected_betting
        ),
        predictions=walk_forward_result.predictions,
        line_comparison=walk_forward_result.line_comparison,
        calibration=walk_forward_result.calibration,
        calibration_buckets=walk_forward_result.calibration_buckets,
    )


def _seed_metrics_row(
    seed: int, evaluation: _EvaluationMetrics, *, primary: bool = False
) -> dict[str, Any]:
    """One row of the seed-stability table."""
    return {
        "random_state": seed,
        # Marks the seed whose numbers are reported everywhere else, so the
        # headline is never mistaken for the mean across seeds.
        "is_primary": primary,
        "mae": evaluation.mae,
        "rmse": evaluation.rmse,
        "r2": evaluation.r2,
        "ou_acc": evaluation.ou_acc,
        "baseline_mae": evaluation.baseline.mae,
        "mae_edge_vs_line": evaluation.baseline.mae - evaluation.mae,
        "roi": evaluation.betting_primary.roi,
        "n_bets": evaluation.betting_primary.n_bets,
        "win_rate": evaluation.betting_primary.win_rate,
        "profit_units": evaluation.betting_primary.profit_units,
        "is_significant": evaluation.betting_primary.is_significant,
        "log_loss": (
            None if evaluation.calibration is None else evaluation.calibration.log_loss
        ),
        "brier": (
            None if evaluation.calibration is None else evaluation.calibration.brier
        ),
    }


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    prepared: PreparedDataset
    df_dev: pd.DataFrame
    df_test: pd.DataFrame
    splits: list[tuple[np.ndarray, np.ndarray]]
    fold_info: pd.DataFrame
    #: None when optuna.fixed_params skipped tuning.
    study: optuna.Study | None
    trials_df: pd.DataFrame | None
    candidates_df: pd.DataFrame | None
    selected_trial: optuna.trial.FrozenTrial | None
    #: The shipped model. None unless refit.train_production_model is on.
    model: XGBRegressor | None
    #: Set when holdout_evaluation == single_shot.
    holdout_result: HoldoutEvaluationResult | None
    #: Set when holdout_evaluation == daily_walk_forward (the default): the
    #: test period scored one day at a time, retraining each day.
    walk_forward_result: DailyBacktestResult | None
    baseline_cv: BaselineMetrics
    baseline_fold_df: pd.DataFrame
    #: Profit metrics pooled over the CV validation rows (~5x the holdout's bet
    #: volume). None when betting.evaluate_cv_folds is off or no
    #: hyperparameters were available. Biased by hyperparameter selection --
    #: read it to compare configurations, not to estimate live ROI.
    cv_betting: CrossValidationBettingResult | None
    #: One row per seed when evaluation_seeds is set: the error bar on every
    #: between-experiment comparison. None when no extra seeds were requested.
    seed_stability: pd.DataFrame | None
    run_dir: Path | None
    model_path: Path | None
    meta_path: Path | None
    #: Per-horizon betting tables, keyed "cv"/"holdout". Populated automatically
    #: whenever the dataset holds several rows per game, because on such a frame
    #: the headline betting numbers count one game once per snapshot and are the
    #: ONLY numbers in the run that cannot be read at face value. Building it
    #: here rather than in a wrapper script is the point: it used to require
    #: remembering to launch a different entry point, and the plain CLI happily
    #: printed the inflated pooled row with nothing saying so.
    snapshot_report: dict[str, pd.DataFrame] | None = None


def _save_planted_signal_artifact(
    run_dir: Path,
    *,
    config: ExperimentConfig,
    prepared: PreparedDataset,
    cv_betting: CrossValidationBettingResult | None,
) -> None:
    """Write planted_signal.json: what was requested, what landed, what was used.

    The importance columns are aggregated across CV folds rather than reported
    per fold (cv_fold_betting.csv already carries those) because the question at
    this level is "did the pipeline USE the feature at all", and a single fold
    splitting on it once is not that.
    """
    if prepared.planted_signal is None:
        return

    payload: dict[str, Any] = {
        **prepared.planted_signal.summary(),
        "is_diagnostic": True,
        "experiment_name": config.experiment_name,
        "target_col": config.target_col,
    }

    if cv_betting is not None:
        folds = cv_betting.fold_metrics
        used = [c for c in folds.columns if c.startswith("planted_")]
        if used:
            payload["n_folds"] = int(len(folds))
            # Share of folds where the tree builder chose it at least once. The
            # headline diagnostic number: 0.0 means the pipeline was handed a
            # known signal and never once split on it.
            if "planted_gain_used" in folds.columns:
                payload["fold_use_rate"] = float(folds["planted_gain_used"].mean())
            for column in used:
                payload[f"mean_{column}"] = float(
                    pd.to_numeric(folds[column], errors="coerce").mean()
                )

    tracking.save_planted_signal(run_dir, payload=payload)


def _baseline_window(config: ExperimentConfig) -> int | None:
    """A window that is certain to exist, for the pre-tuning baselines.

    Those baselines score the LINE on each fold's validation rows, which are
    identical whatever the training window is, so the choice is arbitrary -- but
    it must be a window every fold can actually supply, or building the splits
    raises. The smallest candidate always can.
    """
    choices = config.walk_forward.train_games_choices
    return min(choices) if choices else config.walk_forward.train_games


def resolve_selected_train_games(
    trial: optuna.trial.FrozenTrial | None, config: ExperimentConfig
) -> int | None:
    """The training window the run must use everywhere after selection.

    When Optuna tuned the window, the selected trial's value is authoritative and
    has to reach the holdout walk-forward, the CV betting refits and the
    production model -- otherwise hyperparameters would be chosen for one amount
    of history and then applied to another, which is the exact failure
    ``RefitConfig`` warns about for the fixed case.

    Falls back to the configured value when there is no trial (``skip_tuning``)
    or the trial did not sample a window.
    """
    if trial is not None and TRAIN_GAMES_PARAM in trial.params:
        return int(trial.params[TRAIN_GAMES_PARAM])
    return config.walk_forward.train_games


def run_experiment(
    config: ExperimentConfig, *, save_model: bool | None = None
) -> ExperimentResult:
    """Run one experiment end to end.

    ``save_model`` overrides ``refit.train_production_model`` when given;
    leave it None to let the config decide.
    """
    train_production = (
        config.refit.train_production_model if save_model is None else save_model
    )
    # ExperimentConfig already refuses refit.train_production_model on a
    # diagnostic run; this catches the other door into the same room, where
    # save_model=True is passed at the call site and the config never sees it.
    if train_production and config.is_diagnostic:
        raise ValueError(
            "Refusing to train a production model on a diagnostic run: "
            f"{config.experiment_name!r} carries the target-derived feature "
            f"{config.diagnostics.planted_signal.column!r}. The resulting model "
            "would score spectacularly and predict nothing."
        )
    prepared = prepare_dataset(config)
    df_dev, df_test = build_holdout_split(prepared.df_full, config)

    X_dev, y_dev = _feature_target_subset(df_dev, config, prepared)
    X_test, y_test = _feature_target_subset(df_test, config, prepared)
    dates_dev = df_dev[config.data.date_col]

    # Resolve the output bundle path and check writability *before* tuning:
    # the name depends only on the training window's end date, which is known
    # now, so a refusal costs seconds instead of discarding a full Optuna run.
    train_date_max = df_dev[config.data.date_col].max()
    model_name: str | None = None
    model_out_dir: Path | None = None
    if train_production:
        model_name = naming.build_model_name(
            config, as_of=pd.Timestamp(train_date_max).date()
        )
        model_out_dir = naming.resolve_model_output_dir(config)
        naming.assert_model_bundle_is_writable(
            model_out_dir,
            model_name=model_name,
            overwrite_existing_model=config.overwrite_existing_model,
        )

    split_provider = build_split_provider(df_dev, config)
    fold_info = split_provider.fold_info
    # The fold set at the CONFIGURED window. When the window is tuned this is the
    # layout used for the pre-tuning baselines only; everything scored after
    # selection is re-materialised at the window the selected trial chose.
    splits = split_provider.splits_for(_baseline_window(config))

    strategy = get_strategy(config)

    study: optuna.Study | None = None
    trials_df: pd.DataFrame | None = None
    candidates_df: pd.DataFrame | None = None
    tie_tolerance: TieToleranceResult | None = None
    if not config.optuna.skip_tuning:
        study = strategy.tune(
            X=X_dev,
            y=y_dev,
            split_provider=split_provider,
            config=config,
            dates=dates_dev,
        )
        trials_df = strategy.summarize_trials(study)
        # Resolve the tie band HERE rather than inside the selectors: every
        # strategy then gets the same rule for free, the saved candidate table
        # and the selector are guaranteed to use the SAME band, and the
        # diagnostics (how wide, how many trials it admitted) become recordable.
        # A band that swallows the whole study means the secondary metric picked
        # the model -- a fact about the run, which belongs in its metadata
        # rather than in a log line nobody reads.
        tie_tolerance = resolve_tie_tolerance(
            completed_primary_values(study, primary_key=strategy.primary_metric_key),
            policy=config.optuna.tie_tolerance,
            fixed_abs=config.optuna.mae_tolerance_abs,
            fixed_pct=config.optuna.mae_tolerance_pct,
            max_fraction=config.optuna.tie_max_fraction,
            floor=config.optuna.tie_tolerance_floor,
            cap=config.optuna.tie_tolerance_cap,
            warn_fraction=config.optuna.tie_warn_fraction,
        )
        if tie_tolerance.warning is not None:
            warnings.warn(tie_tolerance.warning, RuntimeWarning, stacklevel=2)
        # Same band the selector used, so the saved candidate table is exactly
        # the set the tie-break ran over rather than a differently-derived one.
        candidates_df = strategy.summarize_candidates(
            study,
            mae_tolerance_abs=(
                config.optuna.mae_tolerance_abs
                if tie_tolerance is None
                else tie_tolerance.tolerance
            ),
            mae_tolerance_pct=(
                config.optuna.mae_tolerance_pct if tie_tolerance is None else None
            ),
        )

    # pooled= mirrors optuna.objective_aggregation so the baseline is the same
    # KIND of number as cv_mae below, which reads pooled_mae or mean_mae by the
    # same flag. Anything else compares a pooled model MAE against a fold-mean
    # baseline and reports the difference as if it meant something.
    baseline_fold_df, baseline_cv = compute_baseline_metrics_across_folds(
        df_dev,
        splits,
        baseline_line_col=prepared.baseline_line_col,
        pooled=config.pools_objective,
        outcome_col=config.outcome_col,
    )

    # Fitted on dev rows only, then applied to the test period, so the
    # bias-corrected null never sees it.
    dev_line_error_bias = compute_line_error_bias(
        df_dev,
        baseline_line_col=prepared.baseline_line_col,
        outcome_col=config.outcome_col,
    )

    # The hyperparameters to hold fixed for evaluation: either recovered from
    # a config that pinned them, or the ones Optuna just chose.
    selected_trial: optuna.trial.FrozenTrial | None = None
    reporting_trial: optuna.trial.FrozenTrial | None = None
    if config.optuna.skip_tuning:
        tuned_params = dict(config.optuna.fixed_params or {})
        tuned_n_estimators = int(config.optuna.fixed_n_estimators or 0)
        tuned_lambda = config.optuna.fixed_sample_weight_lambda
        if tuned_lambda is None and config.sample_weight.enabled:
            tuned_lambda = config.sample_weight.lambda_
    else:
        assert study is not None
        if config.refit.use_lexicographic_selection:
            assert tie_tolerance is not None
            selected_trial = strategy.select_best_trial(
                study,
                # Already an absolute width in primary-metric units, whichever
                # policy produced it, so the pct form is deliberately dropped:
                # passing both would let the selector re-derive a cutoff that
                # disagrees with the one just recorded.
                mae_tolerance_abs=tie_tolerance.tolerance,
                mae_tolerance_pct=None,
            )
        reporting_trial = (
            selected_trial if selected_trial is not None else study.best_trial
        )
        # config= is required, not cosmetic: without it a trial that declined
        # weighting is indistinguishable from one that never sampled a lambda,
        # and the two need opposite handling downstream.
        tuned_params, tuned_n_estimators, tuned_lambda = xgb_params_from_trial(
            reporting_trial, config=config
        )

    # The window the selected trial actually chose. Everything scored from here
    # down uses THIS value, not config.walk_forward.train_games, so the model
    # being evaluated is the model the tuning selected.
    selected_train_games = resolve_selected_train_games(reporting_trial, config)
    if selected_train_games != _baseline_window(config):
        splits = split_provider.splits_for(selected_train_games)

    def evaluate_under_seed(
        seed: int,
    ) -> tuple[HoldoutEvaluationResult | None, DailyBacktestResult | None]:
        """Score the held-out test period with every fit using ``seed``.

        Data, split and hyperparameters are all held fixed, so repeating this
        under different seeds isolates XGBoost's own randomness -- the error
        bar that says whether a gap between two experiments means anything.
        """
        if config.holdout_evaluation == HoldoutEvaluation.DAILY_WALK_FORWARD:
            # Score the test period the way production runs: retrain once per
            # game day on everything available strictly before it (dev, plus
            # test days already played), predict only that day, then pool.
            return None, run_walk_forward_evaluation(
                config,
                prepared=prepared,
                df_history=df_dev,
                df_evaluation=df_test,
                train_games=selected_train_games,
                xgb_params=tuned_params,
                n_estimators=tuned_n_estimators,
                sample_weight_lambda=tuned_lambda,
                show_progress=config.backtest.show_progress,
                random_state=seed,
                dev_line_error_bias=dev_line_error_bias,
            )

        # One model fitted on the dev window, predicting the whole test period
        # at once. Cheaper, but never absorbs completed test days.
        #
        # Same training filter the CV folds and the production refit apply.
        # Without it this path would silently ignore
        # data.exclude_overtime_from_training and fit on rows the rest of the
        # pipeline excluded -- a config that quietly means nothing on one code
        # path is worse than one that errors. X_test/y_test are untouched:
        # the filter changes what is learned from, never what is scored.
        eligible = training_eligible_mask(df_dev, config)
        X_fit, y_fit, dates_fit = X_dev, y_dev, dates_dev
        if not eligible.all():
            X_fit = X_dev.loc[eligible]
            y_fit = y_dev.loc[X_fit.index]
            dates_fit = dates_dev.loc[X_fit.index]

        single_shot_model = fit_final_model(
            X_dev=X_fit,
            y_dev=y_fit,
            params=tuned_params,
            n_estimators=tuned_n_estimators,
            config=config,
            dates_dev=dates_fit,
            sample_weight_lambda=tuned_lambda,
            random_state=seed,
        )
        return (
            evaluate_on_holdout(
                strategy,
                single_shot_model,
                X_test=X_test,
                y_test=y_test,
                df_test_full=df_test,
                baseline_line_col=prepared.baseline_line_col,
                target_line_col=prepared.target_line_col,
                dev_line_error_bias=dev_line_error_bias,
                config=config,
            ),
            None,
        )

    holdout_result, walk_forward_result = evaluate_under_seed(config.random_state)
    evaluation = _normalize_evaluation(holdout_result, walk_forward_result)

    # --- profit across the CV folds ----------------------------------------
    # ~5x the bet volume of the holdout, which is the binding constraint on
    # telling a real edge from a lucky one at these sample sizes. Runs before
    # the seed loop below because it is much cheaper (one fit per fold, versus
    # a whole re-evaluation per seed), so a failure here surfaces early.
    cv_betting: CrossValidationBettingResult | None = None
    if config.betting.evaluate_cv_folds and tuned_params:
        cv_betting = evaluate_cv_betting(
            config,
            df_dev=df_dev,
            X_dev=X_dev,
            y_dev=y_dev,
            dates_dev=dates_dev,
            splits=splits,
            holdout_phases=phases_present(df_test[config.data.date_col]),
            params=tuned_params,
            n_estimators=tuned_n_estimators,
            target_line_col=prepared.target_line_col,
            sample_weight_lambda=tuned_lambda,
        )

    # --- seed stability ----------------------------------------------------
    # The headline seed is always row 0, so the table reads as "the number I
    # reported, plus what it would have been under other seeds".
    seed_stability: pd.DataFrame | None = None
    if config.evaluation_seeds:
        seed_rows = [_seed_metrics_row(config.random_state, evaluation, primary=True)]
        for seed in config.evaluation_seeds:
            seed_rows.append(
                _seed_metrics_row(
                    seed, _normalize_evaluation(*evaluate_under_seed(seed))
                )
            )
        seed_stability = pd.DataFrame(seed_rows)

    # --- production model -------------------------------------------------
    # Fitted on the FULL dataset (dev + test) because production holds nothing
    # back. Never scored here: the walk-forward above measures the procedure
    # that produces it.
    model: XGBRegressor | None = None
    if train_production:
        # Same training regime the hyperparameters were selected under, so the
        # shipped model matches what was evaluated.
        df_production = prepared.df_full
        eligible = training_eligible_mask(df_production, config)
        if not eligible.all():
            df_production = df_production.loc[eligible].copy()

        X_full, y_full = _feature_target_subset(df_production, config, prepared)
        dates_full = df_production[config.data.date_col]
        # The SELECTED window, not the configured one: when Optuna tuned it, the
        # shipped model must be fitted on the amount of history the chosen
        # hyperparameters were actually scored under.
        train_games = selected_train_games
        if config.refit.strategy == RefitStrategy.ROLLING_WINDOW and train_games:
            window = rolling_window_index(
                df_production.index,
                train_games,
                game_ids=_window_game_ids(df_production, config, prepared),
            )
            X_full = X_full.loc[window]
            y_full = y_full.loc[window]
            dates_full = dates_full.loc[window]
        model = fit_final_model(
            X_dev=X_full,
            y_dev=y_full,
            params=tuned_params,
            n_estimators=tuned_n_estimators,
            config=config,
            dates_dev=dates_full,
            sample_weight_lambda=tuned_lambda,
        )

    # No CV metrics exist when tuning was skipped -- nothing was cross-validated.
    cv_mae: float | None = None
    cv_rmse = None
    cv_ou_acc = None
    cv_extra: dict[str, Any] = {}
    if reporting_trial is not None:
        attrs = reporting_trial.user_attrs
        # Pooled first when the run pooled, so the reported cv_* numbers are the
        # ones selection actually ranked on. Falling back to the means would
        # report a different quantity than the one that chose the model.
        cv_ou_acc = attrs.get("pooled_ou_acc", attrs.get("mean_ou_acc"))
        if config.is_classifier:
            # The classifier's trial value is LOG LOSS. Falling back to
            # trial.value for a missing "mean_mae" would file it under cv_mae
            # and report 0.69 as though it were a points error.
            cv_extra = {
                "log_loss": attrs.get(
                    "pooled_log_loss",
                    attrs.get("mean_logloss", reporting_trial.value),
                ),
                "brier": attrs.get("pooled_brier", attrs.get("mean_brier")),
            }
        else:
            reported_mae = attrs.get(
                "pooled_mae", attrs.get("mean_mae", reporting_trial.value)
            )
            cv_mae = None if reported_mae is None else float(reported_mae)
            cv_rmse = attrs.get("pooled_rmse", attrs.get("mean_rmse"))

    # One metric surface regardless of evaluation mode, so artifacts and the
    # leaderboard read the same either way (see _normalize_evaluation).
    test_mae = evaluation.mae
    test_rmse = evaluation.rmse
    test_r2 = evaluation.r2
    test_ou_acc = evaluation.ou_acc
    test_baseline = evaluation.baseline
    test_betting_primary = evaluation.betting_primary
    test_betting_sweep = evaluation.betting_sweep
    test_predictions = evaluation.predictions

    run_dir: Path | None = None
    model_path: Path | None = None
    meta_path: Path | None = None

    if config.save_experiment_artifacts:
        created_at = datetime.now(tz=UTC)
        # Short, citable identity for this run. The config fingerprint already
        # answers "was this the same setup"; this answers "which run was it".
        experiment_id = uuid.uuid4().hex[:12]
        run_dir = tracking.create_experiment_run_dir(
            config.experiment_name,
            root_dir=config.experiment_root_dir,
            timestamp=created_at,
        )
        holdout_dates = pd.to_datetime(df_test[config.data.date_col])
        tracking.save_experiment_metadata(
            run_dir,
            {
                "experiment_id": experiment_id,
                "experiment_name": config.experiment_name,
                "training_version": config.training_version,
                # Research log: why this run exists, and what it should be read
                # alongside.
                "comparison_group": config.comparison_group,
                "hypothesis": config.hypothesis,
                "tags": list(config.tags),
                # Provenance of the bytes actually trained on.
                "data_version": config.data.data_version,
                "dataset_checksum": prepared.dataset_checksum,
                "csv_path": str(config.data.csv_path),
                "created_at": created_at.isoformat(),
                "target_family": config.family.value,
                "prediction_strategy": config.strategy.value,
                # Games with no OVER/UNDER answer, removed before training the
                # classifier. 0 for the regressors, which keep them.
                "n_pushes_excluded": prepared.n_pushes_excluded,
                # Resolved (not raw) labels, so the leaderboard can group runs
                # even when these were auto-derived rather than set explicitly.
                "window_dir_label": config.resolved_window_dir_label,
                "window_name_label": config.resolved_window_name_label,
                "config_fingerprint": config.fingerprint(),
                # The evaluation cohort. Two runs scored on different holdout
                # windows are not directly comparable, so the leaderboard
                # surfaces this rather than silently ranking them together.
                "holdout_start": pd.Timestamp(holdout_dates.min()).isoformat(),
                "holdout_end": pd.Timestamp(holdout_dates.max()).isoformat(),
                "holdout_n_games": int(len(df_test)),
                "holdout_evaluation": config.holdout_evaluation.value,
                "trained_production_model": bool(train_production),
                # --- what the CV actually produced ---------------------------
                # eval_span_games is a target and whole game-days are
                # indivisible, so the realised volume is never assumed: it is
                # recorded here and can be checked against the config.
                "cv_strategy": config.walk_forward.strategy.value,
                "cv_n_folds": int(split_provider.n_folds),
                "cv_n_validation_games": int(split_provider.n_validation_games),
                "cv_retrain_every_days": config.walk_forward.retrain_every_days,
                "cv_eval_span_games": config.walk_forward.eval_span_games,
                "cv_objective_aggregation": (
                    config.optuna.objective_aggregation.value
                ),
                "cv_pruner_warmup_steps": config.resolve_pruner_warmup_steps(
                    split_provider.n_folds
                ),
                # Realised fold sizes, not the configured intent. A fold is
                # whole game-days, so its game count is whatever the schedule
                # gave it; min_validation_games raises the floor but a season
                # boundary can still close a fold early.
                "cv_min_validation_games": config.walk_forward.min_validation_games,
                "cv_fold_game_counts": split_provider.fold_game_counts,
                # --- dataset grain -------------------------------------------
                # What one row IS. On the closing-line dataset a row is a game
                # and these are all trivial; on a pooled intermediate-line frame
                # rows_per_game is the factor separating a bet count from a game
                # count, and the factor every *_games knob is defined against.
                "dataset_type": config.data.dataset_type.value,
                "n_games": prepared.n_games,
                "n_rows": len(prepared.df_full),
                "n_snapshots": prepared.n_snapshots,
                "rows_per_game": round(prepared.rows_per_game, 4),
                "snapshot_minutes": prepared.snapshot_minutes,
                # How the tie band that ran the lexicographic selection was
                # derived, and how much of the study it admitted. A fraction
                # near 1.0 means the SECONDARY metric chose this model.
                **({} if tie_tolerance is None else tie_tolerance.summary()),
                # The window in force for every number in this run. When it was
                # tuned this is the SELECTED value, not the configured fallback.
                "train_games": selected_train_games,
                "train_games_tuned": config.tunes_train_games,
                "train_games_choices": list(
                    config.walk_forward.train_games_choices or ()
                ),
                "n_estimators": int(tuned_n_estimators),
                "n_estimators_tuned": config.tunes_n_estimators,
                "holdout_phases": describe_phases(
                    phases_present(df_test[config.data.date_col])
                ),
                # --- diagnostics ---------------------------------------------
                # Present and true ONLY on a deliberately corrupted run. The
                # leaderboard and any future reader should treat is_diagnostic
                # as disqualifying for anything except pipeline questions.
                "is_diagnostic": config.is_diagnostic,
                **(
                    {}
                    if prepared.planted_signal is None
                    else prepared.planted_signal.summary()
                ),
            },
        )
        tracking.save_config_snapshot(run_dir, config)
        tracking.save_feature_schema(run_dir, prepared.feature_names)
        if prepared.cleaning_report is not None:
            tracking.save_cleaning_report(run_dir, prepared.cleaning_report)
        if study is not None:
            tracking.save_optuna_artifacts(
                run_dir=run_dir,
                study=study,
                trials_df=trials_df,
                selected_trial=selected_trial,
                candidates_df=candidates_df,
            )
        tracking.save_final_test_artifacts(
            run_dir=run_dir,
            metrics={
                "cv": {
                    "mae": cv_mae,
                    "rmse": cv_rmse,
                    "ou_acc": cv_ou_acc,
                    **cv_extra,
                },
                "holdout": {
                    "mae": test_mae,
                    "rmse": test_rmse,
                    "r2": test_r2,
                    "ou_acc": test_ou_acc,
                    **(
                        {}
                        if evaluation.calibration is None
                        else evaluation.calibration.model_dump()
                    ),
                },
            },
            predictions_df=test_predictions,
            threshold_results_df=(
                holdout_result.threshold_results if holdout_result is not None else None
            ),
        )
        tracking.save_baseline_metrics(
            run_dir,
            baseline_cv=baseline_cv,
            baseline_fold_df=baseline_fold_df,
            baseline_holdout=test_baseline,
        )
        tracking.save_betting_metrics(
            run_dir,
            betting_sweep=test_betting_sweep,
            betting_primary=test_betting_primary,
            baseline_bias_corrected=evaluation.baseline_bias_corrected,
            baseline_bias_corrected_betting=(
                evaluation.baseline_bias_corrected_betting
            ),
            dev_line_error_bias=dev_line_error_bias,
        )
        if cv_betting is not None:
            _save_planted_signal_artifact(
                run_dir, config=config, prepared=prepared, cv_betting=cv_betting
            )
            tracking.save_cv_betting_artifacts(
                run_dir,
                summary=cv_betting.summary(),
                fold_metrics_df=cv_betting.fold_metrics,
                betting_sweep=cv_betting.betting_sweep,
                predictions_df=cv_betting.predictions,
                line_comparison_df=cv_betting.line_comparison,
            )
        if seed_stability is not None:
            tracking.save_seed_stability(run_dir, seed_stability)
        if evaluation.line_comparison is not None:
            tracking.save_line_comparison(run_dir, evaluation.line_comparison)
        if evaluation.calibration is not None:
            tracking.save_calibration(
                run_dir,
                summary=evaluation.calibration,
                buckets_df=evaluation.calibration_buckets,
                cv_summary=None if cv_betting is None else cv_betting.calibration,
                cv_buckets_df=(
                    None if cv_betting is None else cv_betting.calibration_buckets
                ),
            )
        if walk_forward_result is not None:
            tracking.save_backtest_artifacts(
                run_dir,
                summary=walk_forward_result.summary(),
                daily_results_df=walk_forward_result.daily_results,
                predictions_df=walk_forward_result.predictions,
                betting_sweep=walk_forward_result.betting_sweep,
            )

    if train_production:
        assert model_name is not None and model_out_dir is not None
        metadata = naming.build_model_bundle_metadata(
            config,
            model_name=model_name,
            best_params=tuned_params,
            selected_trial_number=(
                reporting_trial.number if reporting_trial is not None else None
            ),
            mean_best_iteration=tuned_n_estimators,
            median_best_iteration=tuned_n_estimators,
            # selected_train_games, NOT config.walk_forward.train_games. The
            # config value is only the fallback for when tuning is skipped; when
            # the window is tuned it is whatever _base.yaml happened to set, and
            # the selected trial may have chosen something else entirely.
            #
            # This field is not decoration. retraining_utils.
            # build_retraining_settings_from_production_metadata reads it off the
            # shipped bundle and makes it THE window for every daily retrain, so
            # writing the fallback here meant: Optuna selects 2500, evaluation
            # and this very refit use 2500, and from tomorrow production trains
            # on 3500 forever. The model in service stops being the model that
            # was chosen, silently, with every recorded metric still describing
            # the 2500-game one.
            train_games=selected_train_games,
            cv_mae=cv_mae if cv_mae is not None else float("nan"),
            cv_rmse=cv_rmse,
            cv_ou_acc=cv_ou_acc,
            final_test_mae=test_mae,
            final_test_rmse=test_rmse,
            final_test_ou_acc=test_ou_acc,
            train_date_min=df_dev[config.data.date_col].min(),
            train_date_max=train_date_max,
        )
        model_path, meta_path = save_model_bundle(
            model=model,
            feature_names=prepared.feature_names,
            out_dir=model_out_dir,
            metadata=metadata,
        )

    result = ExperimentResult(
        config=config,
        prepared=prepared,
        df_dev=df_dev,
        df_test=df_test,
        splits=splits,
        fold_info=fold_info,
        study=study,
        trials_df=trials_df,
        candidates_df=candidates_df,
        selected_trial=selected_trial,
        model=model,
        holdout_result=holdout_result,
        walk_forward_result=walk_forward_result,
        baseline_cv=baseline_cv,
        baseline_fold_df=baseline_fold_df,
        cv_betting=cv_betting,
        seed_stability=seed_stability,
        run_dir=run_dir,
        model_path=model_path,
        meta_path=meta_path,
    )

    # Built from the finished result, so it can reuse the same evaluate_betting
    # and the same threshold the run already applied -- it regroups predictions
    # and fits nothing, and therefore cannot introduce look-ahead.
    if prepared.is_pooled_snapshots:
        result.snapshot_report = build_snapshot_report(
            result, snapshot_col=config.data.snapshot_col
        )
        if run_dir is not None and result.snapshot_report:
            save_snapshot_report(result.snapshot_report, run_dir)

    return result
