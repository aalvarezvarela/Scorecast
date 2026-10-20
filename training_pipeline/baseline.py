"""The "trust the bookmaker's line" naive baseline.

No such baseline exists anywhere else in the repo (confirmed by a full-repo
search this session): nba_ou.prediction.baseline_predictions only has a
random +/-1 point baseline and a historical-mean-TOTAL_POINTS baseline --
neither treats the betting line itself as the forecast.

Formula: for a TOTAL_POINTS run, PRED = df[baseline_line_col]; for a
LINE_ERROR run, PRED = 0 for every row. Both express the same underlying
claim ("the bookmaker line is unbiased"), so compute_baseline_metrics always
scores in TOTAL_POINTS space regardless of which target the run trains on --
this keeps baseline numbers numerically comparable across both target
families and directly comparable to TrainingMetrics.final_test_mae/cv_mae for
TOTAL_POINTS runs.

For a SPREAD_ERROR run the same argument holds one market over: the outcome is
HOME_MARGIN and the line is the anchor spread, so the baseline is scored in
HOME_MARGIN space. Callers select this with ``outcome_col``; it defaults to
TOTAL_POINTS so nothing that predates the spread market changes behaviour.

ou_accuracy is always reported as None (serialized as NaN), never 0.0:
predicted edge (pred - line) is exactly 0 for every row, so every row is a
"push" under the existing scorers' push-exclusion convention in
nba_ou.modeling.scorers (over_under_betting_accuracy_total_points /
over_under_betting_accuracy_error_line both return 0.0 when no non-push rows
remain). Reporting 0.0 here would misleadingly read as "the baseline is
always wrong"; None/NaN correctly reads as "no defined bet exists for this
baseline."
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class BaselineMetrics(BaseModel):
    line_col: str
    n_games: int
    mae: float
    rmse: float
    r2: float
    ou_accuracy: float | None = None


def compute_baseline_metrics(
    *,
    y_true_total_points: pd.Series,
    baseline_line: pd.Series,
    line_col: str,
) -> BaselineMetrics:
    y_true = pd.to_numeric(y_true_total_points, errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(baseline_line, errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(valid):
        raise ValueError(
            f"No valid (finite) rows to score the baseline against column {line_col!r}."
        )

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    return BaselineMetrics(
        line_col=line_col,
        n_games=int(valid.sum()),
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        ou_accuracy=None,
    )


def compute_line_error_bias(
    df: pd.DataFrame, *, baseline_line_col: str, outcome_col: str = "TOTAL_POINTS"
) -> float:
    """Mean signed line error (actual outcome minus line) over the given rows.

    Must be computed on development rows only and then applied to the holdout,
    otherwise the "bias-corrected" baseline peeks at the test period.

    ``outcome_col`` is TOTAL_POINTS for the totals market and HOME_MARGIN for the
    spread. It defaults to the totals column so every existing caller keeps its
    exact behaviour.
    """
    actual = pd.to_numeric(df[outcome_col], errors="coerce")
    line = pd.to_numeric(df[baseline_line_col], errors="coerce")
    bias = (actual - line).mean()
    if not np.isfinite(bias):
        raise ValueError("Could not compute a finite line-error bias.")
    return float(bias)


def compute_bias_corrected_baseline_metrics(
    df: pd.DataFrame,
    *,
    baseline_line_col: str,
    bias: float,
    outcome_col: str = "TOTAL_POINTS",
) -> BaselineMetrics:
    """A deliberately harder null than "trust the line".

    Predicts ``line + bias``, where ``bias`` is the average amount by which
    games historically exceeded the line. If a model cannot beat this, it has
    only rediscovered a league-wide over/under drift rather than learned
    anything game-specific. Unlike the pure line baseline this one has a
    non-zero edge on every row, so it does place bets and has a defined win
    rate.
    """
    actual = pd.to_numeric(df[outcome_col], errors="coerce")
    line = pd.to_numeric(df[baseline_line_col], errors="coerce")
    return compute_baseline_metrics(
        y_true_total_points=actual,
        baseline_line=line + bias,
        line_col=f"{baseline_line_col}+bias({bias:+.3f})",
    )


def compute_baseline_metrics_for_rows(
    df_full: pd.DataFrame,
    idx: np.ndarray,
    *,
    baseline_line_col: str,
    outcome_col: str = "TOTAL_POINTS",
) -> BaselineMetrics:
    rows = df_full.iloc[idx]
    return compute_baseline_metrics(
        y_true_total_points=rows[outcome_col],
        baseline_line=rows[baseline_line_col],
        line_col=baseline_line_col,
    )


def compute_baseline_metrics_across_folds(
    df_dev_full: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    baseline_line_col: str,
    pooled: bool,
    outcome_col: str = "TOTAL_POINTS",
) -> tuple[pd.DataFrame, BaselineMetrics]:
    """Compute the baseline on each CV fold's *validation* rows (never train rows).

    Returns a per-fold table (to sit next to the Optuna trial summary's per-fold
    MAE/RMSE) and one aggregate BaselineMetrics.

    ``pooled`` MUST mirror ``optuna.objective_aggregation``, because the
    aggregate is subtracted from the model's ``cv_mae`` and the two have to be
    the same kind of number:

    * ``pooled=True`` concatenates every fold's validation rows and scores them
      once, exactly as ``_PooledCollector`` does for the model. Each GAME counts
      equally, so a 2-game fold contributes 2/855 rather than 1/30.
    * ``pooled=False`` averages the folds' own metrics, which is what
      ``mean_mae`` does.

    This used to be hardcoded to the fold-mean. Under
    ``objective_aggregation: pooled`` that made every model-vs-line CV number a
    comparison between a pooled MAE and a fold-mean MAE -- on cell A of
    public_betting_tradeoff_2026_08 it turned a real +0.02 edge into a reported
    -0.19 deficit, purely from the aggregation mismatch.

    Concatenation is deliberately NOT deduplicated: if a splitter lets two folds
    share a game, the model's pooled metric counts it twice, so the baseline
    must too. Matching the model beats being tidy.
    """
    fold_rows: list[dict] = []
    for fold_num, (_, val_idx) in enumerate(splits, start=1):
        fold_metrics = compute_baseline_metrics_for_rows(
            df_dev_full,
            val_idx,
            baseline_line_col=baseline_line_col,
            outcome_col=outcome_col,
        )
        fold_rows.append(
            {
                "fold": fold_num,
                "n_games": fold_metrics.n_games,
                "mae": fold_metrics.mae,
                "rmse": fold_metrics.rmse,
                "r2": fold_metrics.r2,
            }
        )

    fold_df = pd.DataFrame(fold_rows)

    if pooled:
        pooled_idx = np.concatenate([val_idx for _, val_idx in splits])
        aggregate = compute_baseline_metrics_for_rows(
            df_dev_full,
            pooled_idx,
            baseline_line_col=baseline_line_col,
            outcome_col=outcome_col,
        )
    else:
        aggregate = BaselineMetrics(
            line_col=baseline_line_col,
            n_games=int(fold_df["n_games"].sum()),
            mae=float(fold_df["mae"].mean()),
            rmse=float(fold_df["rmse"].mean()),
            r2=float(fold_df["r2"].mean()),
            ou_accuracy=None,
        )
    return fold_df, aggregate
