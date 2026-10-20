"""Betting metrics across the cross-validation folds, not just the holdout.

The motivation is statistical power. At -110 the break-even win rate is 52.38%,
and a Wilson interval around a genuinely good 55% win rate does not clear that
bar until well over a thousand bets. A 5% holdout of this dataset yields ~290
candidate games and ~115 actual bets, which cannot distinguish a real edge from
a lucky one -- any ranking decided on holdout ROI alone at that volume is close
to a coin flip.

The CV folds already exist and already cover far more games: 12 folds of ~50
validation games is ~600, roughly 5x the holdout. Scoring them for profit costs
one extra fit per fold at the already-chosen hyperparameters -- about the same
as a single Optuna trial -- and turns the split that actually drives decisions
into the split with the most evidence behind it.

The honest caveat, which is why this does not replace the holdout: these folds
selected the hyperparameters, so their betting metrics are optimistically
biased. Use them to COMPARE configurations (the bias applies roughly equally to
each) and the holdout to ESTIMATE out-of-sample performance. The gap between
the two is itself informative -- a large one is the signature of selection
overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from training_pipeline.betting import (
    OUTCOME_COLUMN,
    BettingMetrics,
    betting_threshold_sweep,
    evaluate_betting,
    outcome_from_predictions,
)
from training_pipeline.calibration import (
    CalibrationSummary,
    calibration_summary,
    calibration_table,
)
from training_pipeline.config import ExperimentConfig
from training_pipeline.decisions import (
    collect_prices,
    predict_decisions,
    primary_threshold,
    threshold_sweep_values,
)
from training_pipeline.diagnostics import planted_feature_importance
from training_pipeline.line_scoring import (
    build_line_comparison,
    collect_comparison_lines,
)
from training_pipeline.season_phase import (
    annotate,
    describe_phases,
    game_months,
    season_phases,
)
from training_pipeline.tuning import fit_final_model


@dataclass
class CrossValidationBettingResult:
    """Profit metrics over the pooled CV validation rows."""

    n_folds: int
    #: Pooled validation rows. Equals n_unique_games unless the fold layout
    #: overlaps (step_games_between_tests < test_games), in which case some
    #: games are counted more than once -- which is why both are reported.
    n_games: int
    n_unique_games: int
    mae: float
    rmse: float
    #: Mean of the per-fold MAEs, matching how the Optuna objective aggregates
    #: mean_mae -- so this is directly comparable to the tuned cv_mae, while
    #: ``mae`` above is pooled over games and weights larger folds more.
    mean_fold_mae: float
    betting_primary: BettingMetrics
    betting_sweep: pd.DataFrame
    #: One row per fold: size, error and profit. Read this before the pooled
    #: number -- an ROI carried by one fold is a different claim from one that
    #: held across twelve.
    fold_metrics: pd.DataFrame
    #: One row per pooled validation game.
    predictions: pd.DataFrame
    line_comparison: pd.DataFrame | None
    n_profitable_folds: int
    random_state: int
    #: Betting metrics over the subset of CV games whose season phase also
    #: appears in the holdout. Every holdout to date is the tail of the season,
    #: so pooled CV covers Oct-Apr while the holdout covers only Feb-Apr, and the
    #: two numbers are computed on different populations. Measured across 36
    #: runs, Oct-Dec games score 52.1% and Jan-Apr 53.5%, so roughly half of the
    #: usual "CV looks worse than holdout" gap is this mixture. None when the
    #: holdout's phases were not supplied.
    betting_phase_matched: BettingMetrics | None = None
    #: MAE over the same phase-matched subset. NaN for the classifier.
    mae_phase_matched: float = float("nan")
    #: The phases the match was made against, e.g. "late". Recorded so the
    #: subset can be re-derived from cv_predictions.parquet afterwards.
    holdout_phases: str = ""
    n_games_phase_matched: int = 0
    #: Probability quality over the pooled folds. Classifier only, and the more
    #: readable of the two measurements: ~5x the games of the holdout.
    calibration: CalibrationSummary | None = None
    calibration_buckets: pd.DataFrame | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "n_games": self.n_games,
            "n_unique_games": self.n_unique_games,
            "mae": self.mae,
            "rmse": self.rmse,
            "mean_fold_mae": self.mean_fold_mae,
            "roi": self.betting_primary.roi,
            "n_bets": self.betting_primary.n_bets,
            "bet_rate": self.betting_primary.bet_rate,
            "win_rate": self.betting_primary.win_rate,
            "win_rate_ci_low": self.betting_primary.win_rate_ci_low,
            "win_rate_ci_high": self.betting_primary.win_rate_ci_high,
            "break_even_rate": self.betting_primary.break_even_rate,
            "is_significant": self.betting_primary.is_significant,
            "profit_units": self.betting_primary.profit_units,
            "n_profitable_folds": self.n_profitable_folds,
            "random_state": self.random_state,
            "holdout_phases": self.holdout_phases,
            "n_games_phase_matched": self.n_games_phase_matched,
            "mae_phase_matched": self.mae_phase_matched,
            **(
                {}
                if self.betting_phase_matched is None
                else {
                    "roi_phase_matched": self.betting_phase_matched.roi,
                    "win_rate_phase_matched": self.betting_phase_matched.win_rate,
                    "n_bets_phase_matched": self.betting_phase_matched.n_bets,
                }
            ),
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


def _distinct(values: pd.Series, positions: np.ndarray | None) -> str:
    """The distinct values in a fold's slice, rendered stably.

    A single value is reported as itself; several are joined, so a fold that
    straddles a month or season boundary says so instead of reporting whichever
    row happened to come first.
    """
    series = values if positions is None else values.iloc[positions]
    unique = pd.Series(series).dropna().unique().tolist()
    return "+".join(str(value) for value in sorted(unique, key=str))


def _planted_importance(
    model: Any, config: ExperimentConfig, *, n_features: int
) -> dict[str, float]:
    """Importance of the planted diagnostic feature in one fold's model.

    Empty on any normal run, so the extra columns exist only where they mean
    something. Three importance types because they answer different questions:
    ``weight`` is how often the tree builder chose it, ``gain`` how much it
    helped on average when chosen, ``total_gain`` the product -- and a weak
    feature can rank well on one and poorly on another.
    """
    planted = config.diagnostics.planted_signal
    if not planted.enabled:
        return {}

    booster = model.get_booster()
    scores = {
        kind: booster.get_score(importance_type=kind)
        for kind in ("weight", "gain", "total_gain")
    }
    return planted_feature_importance(
        scores, column=planted.column, n_features=n_features
    )


def evaluate_cv_betting(
    config: ExperimentConfig,
    *,
    df_dev: pd.DataFrame,
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    dates_dev: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    params: dict[str, Any],
    n_estimators: int,
    target_line_col: str,
    sample_weight_lambda: float | None = None,
    random_state: int | None = None,
    holdout_phases: frozenset[str] | None = None,
) -> CrossValidationBettingResult:
    """Refit at the chosen hyperparameters on each fold and score for profit.

    One fit per fold, on that fold's training rows only, predicting only its
    validation rows -- the same time-aware discipline the tuning objective
    used, so no fold ever sees its own future.
    """
    resolved_random_state = (
        config.random_state if random_state is None else random_state
    )
    ev_threshold = primary_threshold(config)

    target_line_all = pd.to_numeric(
        df_dev[target_line_col], errors="coerce"
    ).to_numpy(dtype=float)
    actual_total_all = pd.to_numeric(
        df_dev[config.outcome_col], errors="coerce"
    ).to_numpy(dtype=float)

    fold_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for fold_num, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = fit_final_model(
            X_dev=X_dev.iloc[train_idx],
            y_dev=y_dev.iloc[train_idx],
            params=params,
            n_estimators=n_estimators,
            config=config,
            dates_dev=dates_dev.iloc[train_idx],
            sample_weight_lambda=sample_weight_lambda,
            random_state=resolved_random_state,
        )

        y_true = pd.to_numeric(y_dev.iloc[valid_idx], errors="coerce").to_numpy(
            dtype=float
        )
        fold_line = target_line_all[valid_idx]
        fold_actual = actual_total_all[valid_idx]

        fold_over, fold_under = collect_prices(df_dev, config, positions=valid_idx)
        fold_decisions = predict_decisions(
            model,
            X_dev.iloc[valid_idx],
            config=config,
            target_line=fold_line,
            decimal_odds_over=fold_over,
            decimal_odds_under=fold_under,
        )
        y_pred = fold_decisions.raw_prediction
        fold_edge = fold_decisions.predicted_edge

        fold_betting = evaluate_betting(
            predicted_edge=fold_edge,
            actual_total=fold_actual,
            line=fold_line,
            min_edge=ev_threshold,
            selection_score=fold_decisions.selection_score,
            flat_decimal_odds=config.betting.flat_decimal_odds,
            decimal_odds_over=fold_over,
            decimal_odds_under=fold_under,
        )

        fold_rows.append(
            {
                "fold": fold_num,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                # Read off the fold model that was fitted anyway, so the
                # diagnostic costs nothing extra. Empty dict on a normal run.
                **_planted_importance(model, config, n_features=X_dev.shape[1]),
                "valid_start": pd.Timestamp(dates_dev.iloc[valid_idx].min()),
                "valid_end": pd.Timestamp(dates_dev.iloc[valid_idx].max()),
                # Where in the season this fold sits. A rolling-origin fold spans
                # a handful of game-days so these are usually single-valued;
                # "+"-joined when a fold straddles a boundary, rather than
                # silently reporting only the first.
                "season": _distinct(df_dev[config.data.season_col], valid_idx),
                "game_month": _distinct(
                    game_months(dates_dev.iloc[valid_idx]).astype("object"),
                    None,
                ),
                "season_phase": _distinct(
                    season_phases(dates_dev.iloc[valid_idx]).astype("object"), None
                ),
                # NaN for the classifier: the "error" of a probability against
                # a 0/1 label is not in points and must not be read next to a
                # regressor's MAE.
                "mae": float("nan") if config.is_classifier
                else float(mean_absolute_error(y_true, y_pred)),
                "rmse": float("nan") if config.is_classifier
                else float(np.sqrt(mean_squared_error(y_true, y_pred))),
                # The line's own error on the same rows, so each fold shows
                # whether the model beat the market there or merely tracked it.
                "line_mae": float(np.mean(np.abs(fold_actual - fold_line))),
                "n_bets": fold_betting.n_bets,
                "win_rate": fold_betting.win_rate,
                "roi": fold_betting.roi,
                "profit_units": fold_betting.profit_units,
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "fold": fold_num,
                    "row_in_dev": valid_idx,
                    config.data.date_col: dates_dev.iloc[valid_idx].to_numpy(),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "target_line": fold_line,
                    OUTCOME_COLUMN: fold_actual,
                    **(
                        {"TOTAL_POINTS": fold_actual}
                        if config.outcome_col == "TOTAL_POINTS"
                        else {}
                    ),
                    "predicted_edge": fold_edge,
                    "selection_score": fold_decisions.selection_score,
                    **(
                        {}
                        if fold_decisions.p_over is None
                        else {
                            "p_over": fold_decisions.p_over,
                            "expected_value": fold_decisions.expected_value,
                        }
                    ),
                }
            )
        )

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    # Per-GAME month and phase, which is what makes the phase-matched number
    # auditable: the subset can be rebuilt from cv_predictions.parquet without
    # trusting the summary.
    predictions = annotate(predictions, predictions[config.data.date_col])
    predictions[config.data.season_col] = (
        df_dev[config.data.season_col]
        .to_numpy()[predictions["row_in_dev"].to_numpy()]
    )

    pooled_edge = predictions["predicted_edge"].to_numpy(dtype=float)
    pooled_line = predictions["target_line"].to_numpy(dtype=float)
    pooled_actual = outcome_from_predictions(predictions)

    pooled_over, pooled_under = collect_prices(
        df_dev, config, positions=predictions["row_in_dev"].to_numpy()
    )
    betting_kwargs: dict[str, Any] = {
        "actual_total": pooled_actual,
        "line": pooled_line,
        "flat_decimal_odds": config.betting.flat_decimal_odds,
        "decimal_odds_over": pooled_over,
        "decimal_odds_under": pooled_under,
    }
    pooled_score = predictions["selection_score"].to_numpy(dtype=float)
    betting_sweep = betting_threshold_sweep(
        predicted_edge=pooled_edge,
        thresholds=threshold_sweep_values(config),
        selection_score=pooled_score,
        **betting_kwargs,
    )
    betting_primary = evaluate_betting(
        predicted_edge=pooled_edge,
        min_edge=ev_threshold,
        selection_score=pooled_score,
        **betting_kwargs,
    )

    calibration: CalibrationSummary | None = None
    calibration_buckets: pd.DataFrame | None = None
    if config.is_classifier:
        pooled_p = predictions["p_over"].to_numpy(dtype=float)
        pooled_y = predictions["y_true"].to_numpy(dtype=float)
        calibration = calibration_summary(
            pooled_y, pooled_p, n_buckets=config.betting.calibration_buckets
        )
        calibration_buckets = calibration_table(
            pooled_y, pooled_p, n_buckets=config.betting.calibration_buckets
        )

    line_comparison = None if config.is_classifier else build_line_comparison(
        y_pred=predictions["y_pred"].to_numpy(dtype=float),
        target_line=pooled_line,
        actual_total=pooled_actual,
        lines=collect_comparison_lines(
            df_dev,
            config,
            target_line_col=target_line_col,
            positions=predictions["row_in_dev"].to_numpy(),
        ),
        config=config,
    )

    # --- phase-matched view -------------------------------------------------
    # Same predictions, restricted to the season phases the holdout covers, so
    # the CV number and the holdout number describe the same population. The
    # pooled number above stays the headline: production bets all season, and
    # restricting to the holdout's phases would throw away most of the volume
    # that makes CV worth reading at all.
    betting_phase_matched: BettingMetrics | None = None
    mae_phase_matched = float("nan")
    n_games_phase_matched = 0
    if holdout_phases:
        matched = (
            predictions["season_phase"].astype("string").isin(holdout_phases).to_numpy()
        )
        n_games_phase_matched = int(matched.sum())
        if n_games_phase_matched:
            matched_positions = predictions["row_in_dev"].to_numpy()[matched]
            matched_over, matched_under = collect_prices(
                df_dev, config, positions=matched_positions
            )
            betting_phase_matched = evaluate_betting(
                predicted_edge=pooled_edge[matched],
                min_edge=ev_threshold,
                selection_score=pooled_score[matched],
                actual_total=pooled_actual[matched],
                line=pooled_line[matched],
                flat_decimal_odds=config.betting.flat_decimal_odds,
                decimal_odds_over=matched_over,
                decimal_odds_under=matched_under,
            )
            if not config.is_classifier:
                mae_phase_matched = float(
                    mean_absolute_error(
                        predictions["y_true"].to_numpy(dtype=float)[matched],
                        predictions["y_pred"].to_numpy(dtype=float)[matched],
                    )
                )

    return CrossValidationBettingResult(
        n_folds=len(splits),
        n_games=int(len(predictions)),
        n_unique_games=int(predictions["row_in_dev"].nunique()),
        mae=float("nan") if config.is_classifier else float(
            mean_absolute_error(predictions["y_true"], predictions["y_pred"])
        ),
        rmse=float("nan") if config.is_classifier else float(
            np.sqrt(mean_squared_error(predictions["y_true"], predictions["y_pred"]))
        ),
        mean_fold_mae=float(fold_metrics["mae"].mean()),
        betting_primary=betting_primary,
        betting_sweep=betting_sweep,
        fold_metrics=fold_metrics,
        predictions=predictions,
        line_comparison=line_comparison,
        # to_numeric first: a fold that placed no bets reports roi=None, which
        # makes the column object-dtype and fillna's downcast deprecated.
        n_profitable_folds=int(
            (pd.to_numeric(fold_metrics["roi"], errors="coerce").fillna(0.0) > 0).sum()
        ),
        random_state=resolved_random_state,
        calibration=calibration,
        calibration_buckets=calibration_buckets,
        betting_phase_matched=betting_phase_matched,
        mae_phase_matched=mae_phase_matched,
        holdout_phases=describe_phases(holdout_phases),
        n_games_phase_matched=n_games_phase_matched,
    )
