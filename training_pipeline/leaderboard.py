"""Cross-run comparison: reads every saved experiment run and builds one
leaderboard row per run, always framed relative to the "trust the bookmaker
line" baseline (never in isolation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from training_pipeline.config import TargetFamily
from training_pipeline.tracking import parse_run_dir_timestamp

DEFAULT_EXPERIMENT_ROOT = Path("artifacts") / "experiments"


def discover_run_dirs(root_dir: str | Path = DEFAULT_EXPERIMENT_ROOT) -> list[Path]:
    root = Path(root_dir)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _read_seed_stability(run_dir: Path) -> dict[str, Any]:
    """Spread of the headline metrics across evaluation seeds.

    Collapsed to a std and a range here because that is what a leaderboard row
    can usefully carry: the question it answers is "is the gap between two runs
    bigger than the noise within one of them?". The full per-seed table stays
    in seed_stability.csv.
    """
    path = run_dir / "seed_stability.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return {}
    if df.empty or len(df) < 2:
        return {}

    def _spread(column: str) -> tuple[float | None, float | None]:
        if column not in df.columns:
            return None, None
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(values) < 2:
            return None, None
        return float(values.std(ddof=1)), float(values.max() - values.min())

    roi_std, roi_range = _spread("roi")
    mae_std, mae_range = _spread("mae")
    return {
        "n_seeds": int(len(df)),
        "seed_roi_std": roi_std,
        "seed_roi_range": roi_range,
        "seed_mae_std": mae_std,
        "seed_mae_range": mae_range,
    }


def load_run_summary(run_dir: str | Path) -> dict[str, Any]:
    """Load whatever artifacts exist for one run directory into a flat summary
    row. Missing files degrade to None rather than raising, so a partially
    populated run (e.g. one that skipped experiment tracking) still shows up.
    """
    run_dir = Path(run_dir)
    metadata = _read_json(run_dir / "metadata.json") or {}
    config = _read_json(run_dir / "config.json") or {}
    final_metrics = _read_json(run_dir / "final_test_metrics.json") or {}
    baseline_metrics = _read_json(run_dir / "baseline_metrics.json") or {}
    betting_metrics = _read_json(run_dir / "betting_metrics.json") or {}
    calibration = _read_json(run_dir / "calibration.json") or {}
    cv_betting = _read_json(run_dir / "cv_betting_summary.json") or {}
    seed_stability = _read_seed_stability(run_dir)
    selected_trial_payload = _read_json(run_dir / "optuna_selected_trial.json")
    best_trial_payload = _read_json(run_dir / "optuna_best_trial.json")

    trial: dict[str, Any] | None = None
    if selected_trial_payload:
        trial = selected_trial_payload.get("selected_trial")
    elif best_trial_payload:
        trial = best_trial_payload.get("best_trial")

    cv = final_metrics.get("cv") or {}
    holdout = final_metrics.get("holdout") or {}
    baseline_cv = baseline_metrics.get("cv") or {}
    baseline_holdout = baseline_metrics.get("holdout") or {}
    optuna_cfg = config.get("optuna") or {}
    walk_forward_cfg = config.get("walk_forward") or {}
    refit_cfg = config.get("refit") or {}
    betting_primary = betting_metrics.get("primary") or {}
    bias_baseline = betting_metrics.get("baseline_bias_corrected") or {}
    bias_baseline_betting = betting_metrics.get("baseline_bias_corrected_betting") or {}

    # metadata.json holds *resolved* labels and the run timestamp; config.json
    # holds the raw (possibly None) user-supplied values. Prefer metadata, fall
    # back to config, and finally to the timestamp embedded in the directory
    # name so runs written before created_at was recorded still sort correctly.
    created_at = metadata.get("created_at")
    if not created_at:
        parsed = parse_run_dir_timestamp(run_dir)
        created_at = parsed.isoformat() if parsed is not None else None

    return {
        "run_name": run_dir.name,
        "experiment_id": metadata.get("experiment_id"),
        "experiment_name": metadata.get("experiment_name"),
        # Runs sharing a comparison_group are the ones actually meant to be
        # read against each other.
        "comparison_group": (
            metadata.get("comparison_group") or config.get("comparison_group")
        ),
        "hypothesis": metadata.get("hypothesis") or config.get("hypothesis"),
        "tags": metadata.get("tags") or config.get("tags"),
        "data_version": metadata.get("data_version"),
        "dataset_checksum": metadata.get("dataset_checksum"),
        # Manually curated label; falls back to config.json for runs whose
        # metadata predates it being recorded there.
        "training_version": (
            metadata.get("training_version") or config.get("training_version")
        ),
        "created_at": created_at,
        "target_family": metadata.get("target_family") or config.get("target_family"),
        # Names the model class as well as the target. Older runs predate it and
        # fall back to the target family, which for them was unambiguous.
        "prediction_strategy": (
            metadata.get("prediction_strategy")
            or config.get("prediction_strategy")
            or (
                f"{metadata.get('target_family')}_regressor"
                if metadata.get("target_family") else None
            )
        ),
        "n_pushes_excluded": metadata.get("n_pushes_excluded"),
        "window_dir_label": (
            metadata.get("window_dir_label") or config.get("window_dir_label")
        ),
        "config_fingerprint": metadata.get("config_fingerprint"),
        # --- evaluation cohort: what this run's numbers were measured on ---
        # ROI and MAE are only comparable between runs scored on the same
        # holdout window and a similar number of games, so these travel next
        # to the metrics instead of being left implicit.
        "holdout_start": metadata.get("holdout_start"),
        "holdout_end": metadata.get("holdout_end"),
        "holdout_n_games": metadata.get("holdout_n_games"),
        # How the holdout was scored: daily_walk_forward retrains once per game
        # day, single_shot fits once. Surfaced so "the walk-forward result" is
        # only called that when it actually is one.
        "holdout_evaluation": metadata.get("holdout_evaluation"),
        "n_candidates": betting_primary.get("n_candidates"),
        # metadata FIRST. config.walk_forward.train_games is only the fallback
        # for when tuning is skipped; on a tuned-window run it is whatever
        # _base.yaml happened to set, while metadata.json records the window the
        # selected trial actually chose and every fit site actually used.
        #
        # This is a matching factor (reporting.factors), so reading the config
        # value meant two tuned runs that selected DIFFERENT windows were
        # compared as though they were the same experiment whenever their
        # configs shared a fallback -- and the run label said the wrong number
        # too. The same defect as the one that shipped the fallback window into
        # the model bundle; this is its reporting-side twin.
        #
        # refit_cfg is read last, for runs saved before the two were unified.
        "train_games": (
            metadata.get("train_games")
            or walk_forward_cfg.get("train_games")
            or refit_cfg.get("train_games")
        ),
        "n_trials": optuna_cfg.get("n_trials"),
        "objective_name": optuna_cfg.get("objective_name"),
        "cv_mae": cv.get("mae"),
        "cv_rmse": cv.get("rmse"),
        "cv_ou_acc": cv.get("ou_acc"),
        "final_test_mae": holdout.get("mae"),
        "final_test_rmse": holdout.get("rmse"),
        "final_test_ou_acc": holdout.get("ou_acc"),
        "baseline_cv_mae": baseline_cv.get("mae"),
        "baseline_cv_rmse": baseline_cv.get("rmse"),
        "baseline_holdout_mae": baseline_holdout.get("mae"),
        "baseline_holdout_rmse": baseline_holdout.get("rmse"),
        # --- directional results at the primary edge threshold ---
        "bet_min_edge": betting_primary.get("min_edge"),
        "n_bets": betting_primary.get("n_bets"),
        "n_pushes": betting_primary.get("n_pushes"),
        "bet_rate": betting_primary.get("bet_rate"),
        "win_rate": betting_primary.get("win_rate"),
        "win_rate_ci_low": betting_primary.get("win_rate_ci_low"),
        "win_rate_ci_high": betting_primary.get("win_rate_ci_high"),
        "break_even_rate": betting_primary.get("break_even_rate"),
        "edge_vs_break_even": betting_primary.get("edge_vs_break_even"),
        "roi": betting_primary.get("roi"),
        "profit_units": betting_primary.get("profit_units"),
        "is_significant": betting_primary.get("is_significant"),
        # --- directional results across the CV folds ---
        # The ROI fields remain available as settlement diagnostics, but are not
        # promoted to the headline or used for the default ranking.
        "cv_roi": cv_betting.get("roi"),
        "cv_n_bets": cv_betting.get("n_bets"),
        "cv_win_rate": cv_betting.get("win_rate"),
        "cv_win_rate_ci_low": cv_betting.get("win_rate_ci_low"),
        "cv_is_significant": cv_betting.get("is_significant"),
        "cv_profit_units": cv_betting.get("profit_units"),
        "cv_n_profitable_folds": cv_betting.get("n_profitable_folds"),
        "cv_n_folds": cv_betting.get("n_folds"),
        # Follows objective_aggregation, exactly as baseline_cv_mae and cv_mae
        # do. cv_betting records BOTH aggregations; taking the pooled one
        # unconditionally would put a pooled number beside a fold-mean baseline
        # on any run that aggregated by mean -- the same mismatch, mirrored.
        "cv_betting_mae": (
            cv_betting.get("mae")
            if (optuna_cfg.get("objective_aggregation") or "mean") == "pooled"
            else cv_betting.get("mean_fold_mae", cv_betting.get("mae"))
        ),
        # --- probability quality (classifier only; None for regressors) ---
        # log_loss_improvement is the one to read: raw log loss looks flat
        # because a coin flip already scores 0.693, so the meaningful quantity
        # is how far it beats predicting the base rate for every game.
        "log_loss": (calibration.get("holdout") or {}).get("log_loss"),
        "brier": (calibration.get("holdout") or {}).get("brier"),
        "log_loss_improvement": (
            (calibration.get("holdout") or {}).get("log_loss_improvement")
        ),
        "cv_log_loss": (calibration.get("cv") or {}).get("log_loss"),
        "cv_log_loss_improvement": (
            (calibration.get("cv") or {}).get("log_loss_improvement")
        ),
        "expected_calibration_error": (
            (calibration.get("holdout") or {}).get("expected_calibration_error")
        ),
        # --- how much of this is just the seed? ---
        **seed_stability,
        # --- the harder "line + historical drift" null ---
        "bias_baseline_mae": bias_baseline.get("mae"),
        "bias_baseline_roi": bias_baseline_betting.get("roi"),
        "dev_line_error_bias": betting_metrics.get("dev_line_error_bias"),
        "selected_trial_number": (trial or {}).get("number"),
    }


#: Columns worth looking at first when comparing runs. Directional win rate is
#: paired with its Wilson interval and volume; the lower confidence bound is the
#: default ranking so a tiny lucky cohort does not outrank a larger one merely
#: because its point estimate is higher. Flat-odds ROI remains in the full table
#: as an optional settlement diagnostic.
HEADLINE_COLUMNS: tuple[str, ...] = (
    "experiment_id",
    "comparison_group",
    "training_version",
    "prediction_strategy",
    "window_dir_label",
    "win_rate",
    "win_rate_ci_low",
    "win_rate_ci_high",
    "n_bets",
    "n_pushes",
    "bet_rate",
    "cv_win_rate",
    "cv_win_rate_ci_low",
    "cv_n_bets",
    "n_seeds",
    "holdout_start",
    "holdout_end",
    "n_candidates",
    "config_fingerprint",
    "final_test_mae",
    "baseline_holdout_mae",
    "mae_improvement_over_baseline_pct",
    # Blank for regressors, populated for classifiers. Their MAE columns are
    # blank in turn -- point error against a 0/1 label is not a points error and
    # must not be ranked beside one.
    "cv_log_loss_improvement",
    "expected_calibration_error",
    "created_at",
)


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the comparison columns that are functions of the loaded metrics.

    Split out from ``build_leaderboard`` so a frame assembled from an explicit
    list of run directories (see ``training_pipeline.reporting.discovery``) gets
    exactly the same derived columns, rather than a near-copy that drifts.
    """
    if df.empty:
        return df

    df = df.copy()
    df["mae_improvement_over_baseline_pct"] = (
        df["baseline_holdout_mae"] - df["final_test_mae"]
    ) / df["baseline_holdout_mae"]
    df["rmse_improvement_over_baseline_pct"] = (
        df["baseline_holdout_rmse"] - df["final_test_rmse"]
    ) / df["baseline_holdout_rmse"]
    df["beats_baseline_mae"] = df["mae_improvement_over_baseline_pct"] > 0
    df["beats_baseline_rmse"] = df["rmse_improvement_over_baseline_pct"] > 0
    df["roi_vs_bias_baseline"] = df["roi"] - df["bias_baseline_roi"]
    if "cv_roi" in df.columns:
        # Positive = the holdout underperformed the folds the hyperparameters
        # were chosen on, i.e. some of the CV edge was selection, not signal.
        df["cv_minus_holdout_roi"] = df["cv_roi"] - df["roi"]
    return df


def build_leaderboard(
    root_dir: str | Path = DEFAULT_EXPERIMENT_ROOT,
    *,
    target_family: TargetFamily | str | None = None,
    sort_by: str = "win_rate_ci_low",
) -> pd.DataFrame:
    """One row per saved run, ranked by conservative directional accuracy.

    The default is the holdout win rate's Wilson lower bound, not flat-odds ROI
    or raw win rate. It rewards correct OVER/UNDER calls while accounting for
    the fact that a high percentage on very few bets is weak evidence. Callers
    can still explicitly sort by any diagnostic column in the full table.
    """
    rows = [load_run_summary(run_dir) for run_dir in discover_run_dirs(root_dir)]

    if target_family is not None:
        target_value = (
            target_family.value if isinstance(target_family, TargetFamily) else target_family
        )
        rows = [row for row in rows if row.get("target_family") == target_value]

    df = add_derived_columns(pd.DataFrame(rows))
    if df.empty:
        return df

    if sort_by not in df.columns:
        raise KeyError(
            f"sort_by={sort_by!r} is not a leaderboard column. Available: "
            f"{sorted(df.columns)}"
        )

    return df.sort_values(sort_by, ascending=False, na_position="last").reset_index(
        drop=True
    )


def headline_leaderboard(
    root_dir: str | Path = DEFAULT_EXPERIMENT_ROOT,
    *,
    target_family: TargetFamily | str | None = None,
) -> pd.DataFrame:
    """``build_leaderboard`` trimmed to HEADLINE_COLUMNS for quick display."""
    df = build_leaderboard(root_dir, target_family=target_family)
    if df.empty:
        return df
    return df[[column for column in HEADLINE_COLUMNS if column in df.columns]]
