"""Self-contained experiment-run persistence.

A near-verbatim port of lab/utils/experiment_tracking.py's helpers -- copied,
not imported, so training_pipeline has zero dependency on lab/'s fragile
import mechanics (lab/ only resolves via pytest's cwd-on-path behavior or a
manual sys.path.insert in one notebook; training_pipeline is meant to work
identically from any notebook kernel or script). Two additions over the
lab/ original: save_config_snapshot and save_baseline_metrics.

tune_xgb_total_points_optuna_persistent is deliberately NOT ported here --
that logic now lives target-family-agnostically inside
training_pipeline.tuning's strategies, so this module has zero Optuna
objective-function knowledge and zero target-family knowledge; it only
persists artifacts.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nba_ou.data_processing.missing_data.cleaning_report import CleaningReport

from training_pipeline.baseline import BaselineMetrics
from training_pipeline.betting import BettingMetrics
from training_pipeline.config import ExperimentConfig

DEFAULT_EXPERIMENT_ROOT = Path("artifacts") / "experiments"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("_")
    return slug or "experiment"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    return path


RUN_DIR_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def create_experiment_run_dir(
    experiment_name: str,
    *,
    root_dir: str | Path = DEFAULT_EXPERIMENT_ROOT,
    timestamp: datetime | None = None,
) -> Path:
    timestamp = timestamp or datetime.now(tz=UTC)
    run_name = (
        f"{_slugify(experiment_name)}_{timestamp.strftime(RUN_DIR_TIMESTAMP_FORMAT)}"
    )
    run_dir = Path(root_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_run_dir_timestamp(run_dir: str | Path) -> datetime | None:
    """Recover a run's creation time from its directory name.

    Used as a fallback by the leaderboard so runs written before ``created_at``
    was recorded in metadata.json still sort chronologically.
    """
    name = Path(run_dir).name
    parts = name.rsplit("_", 2)
    if len(parts) < 3:
        return None
    candidate = f"{parts[-2]}_{parts[-1]}"
    try:
        return datetime.strptime(candidate, RUN_DIR_TIMESTAMP_FORMAT).replace(
            tzinfo=UTC
        )
    except ValueError:
        return None


def optuna_sqlite_storage_url(
    run_dir: str | Path, filename: str = "optuna_study.db"
) -> str:
    sqlite_path = Path(run_dir) / filename
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{sqlite_path.resolve().as_posix()}"


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".json":
        df.to_json(path, orient="records", date_format="iso", indent=2)
    else:
        raise ValueError(f"Unsupported dataframe artifact extension: {path.suffix}")

    return path


def _trial_payload(trial: Any) -> dict[str, Any] | None:
    if trial is None:
        return None

    return {
        "number": trial.number,
        "value": trial.value,
        "state": trial.state.name,
        "params": dict(trial.params),
        "user_attrs": dict(trial.user_attrs),
    }


def save_experiment_metadata(run_dir: str | Path, metadata: dict[str, Any]) -> Path:
    return write_json(Path(run_dir) / "metadata.json", metadata)


def save_feature_schema(run_dir: str | Path, feature_names: list[str]) -> Path:
    return write_json(
        Path(run_dir) / "feature_schema.json",
        {"feature_names": feature_names, "n_features": len(feature_names)},
    )


def save_cleaning_report(run_dir: str | Path, report: CleaningReport) -> Path:
    """Persist which cleaning step dropped each column and row, and why.

    Written for every run because it is the only record of the ~350 columns
    cleaning removes. Without it, "why is this feature not in the model?" is
    answerable only by re-running cleaning at verbose=2 against the same data
    and settings, and reading several hundred lines of print.

    Pairs with feature_schema.json, which lists what survived: together they
    account for every column the CSV started with.
    """
    return write_json(Path(run_dir) / "cleaning_report.json", report.to_dict())


def save_config_snapshot(run_dir: str | Path, config: ExperimentConfig) -> Path:
    """Write config.json so every run directory is fully reproducible from
    its own snapshot alone.
    """
    path = Path(run_dir) / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2) + "\n")
    return path


def save_optuna_artifacts(
    *,
    run_dir: str | Path,
    study: Any,
    trials_df: pd.DataFrame | None = None,
    selected_trial: Any | None = None,
    candidates_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    artifacts: dict[str, Path] = {}

    if trials_df is None:
        trials_df = study.trials_dataframe(
            attrs=("number", "value", "state", "params", "user_attrs")
        )

    artifacts["trials"] = save_dataframe(trials_df, run_dir / "optuna_trials.csv")
    artifacts["best_trial"] = write_json(
        run_dir / "optuna_best_trial.json",
        {"best_trial": _trial_payload(study.best_trial)},
    )

    if selected_trial is not None:
        artifacts["selected_trial"] = write_json(
            run_dir / "optuna_selected_trial.json",
            {"selected_trial": _trial_payload(selected_trial)},
        )

    if candidates_df is not None:
        artifacts["candidates"] = save_dataframe(
            candidates_df, run_dir / "optuna_lexicographic_candidates.csv"
        )

    return artifacts


def save_final_test_artifacts(
    *,
    run_dir: str | Path,
    metrics: dict[str, Any],
    predictions_df: pd.DataFrame,
    threshold_results_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    artifacts = {
        "metrics": write_json(run_dir / "final_test_metrics.json", metrics),
        "predictions": save_dataframe(
            predictions_df, run_dir / "final_test_predictions.parquet"
        ),
    }

    if threshold_results_df is not None:
        artifacts["thresholds"] = save_dataframe(
            threshold_results_df, run_dir / "threshold_results.csv"
        )

    return artifacts


def save_backtest_artifacts(
    run_dir: str | Path,
    *,
    summary: dict[str, Any],
    daily_results_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    betting_sweep: pd.DataFrame,
) -> dict[str, Path]:
    """Persist a daily walk-forward backtest.

    ``backtest_daily.csv`` is the per-day audit trail (training-set size and
    date range for every retrain), which is what you inspect to confirm the
    simulation never trained on a future game.
    """
    run_dir = Path(run_dir)
    return {
        "summary": write_json(run_dir / "backtest_summary.json", summary),
        "daily": save_dataframe(daily_results_df, run_dir / "backtest_daily.csv"),
        "predictions": save_dataframe(
            predictions_df, run_dir / "backtest_predictions.parquet"
        ),
        "betting_sweep": save_dataframe(
            betting_sweep, run_dir / "backtest_betting_sweep.csv"
        ),
    }


def save_betting_metrics(
    run_dir: str | Path,
    *,
    betting_sweep: pd.DataFrame,
    betting_primary: BettingMetrics,
    baseline_bias_corrected: BaselineMetrics,
    baseline_bias_corrected_betting: BettingMetrics,
    dev_line_error_bias: float,
) -> dict[str, Path]:
    """Persist the profit-oriented metrics.

    ``betting_metrics.json`` holds the headline numbers the leaderboard reads;
    ``betting_sweep.csv`` holds the full edge-threshold table (win rate, bet
    volume and CI at each threshold), which is what you actually inspect to
    decide whether an apparent edge is real or a small-sample artefact.
    """
    run_dir = Path(run_dir)
    return {
        "metrics": write_json(
            run_dir / "betting_metrics.json",
            {
                "primary": betting_primary.model_dump(),
                "baseline_bias_corrected": baseline_bias_corrected.model_dump(),
                "baseline_bias_corrected_betting": (
                    baseline_bias_corrected_betting.model_dump()
                ),
                "dev_line_error_bias": dev_line_error_bias,
            },
        ),
        "sweep": save_dataframe(betting_sweep, run_dir / "betting_sweep.csv"),
    }


def save_cv_betting_artifacts(
    run_dir: str | Path,
    *,
    summary: dict[str, Any],
    fold_metrics_df: pd.DataFrame,
    betting_sweep: pd.DataFrame,
    predictions_df: pd.DataFrame,
    line_comparison_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Persist profit metrics measured across the CV folds.

    ``cv_fold_betting.csv`` is the one to read first: a pooled ROI carried by a
    single fold is a different claim from one that held across all of them, and
    only the per-fold table can tell those apart.
    """
    run_dir = Path(run_dir)
    artifacts = {
        "summary": write_json(run_dir / "cv_betting_summary.json", summary),
        "folds": save_dataframe(fold_metrics_df, run_dir / "cv_fold_betting.csv"),
        "sweep": save_dataframe(betting_sweep, run_dir / "cv_betting_sweep.csv"),
        "predictions": save_dataframe(
            predictions_df, run_dir / "cv_predictions.parquet"
        ),
    }
    if line_comparison_df is not None:
        # Kept separate from the holdout's line_comparison.csv: this one has
        # several times the bet volume behind it, so it is the more readable of
        # the two even though it carries the same selection bias as cv_roi.
        artifacts["line_comparison"] = save_dataframe(
            line_comparison_df, run_dir / "cv_line_comparison.csv"
        )
    return artifacts


def save_planted_signal(run_dir: str | Path, *, payload: dict[str, Any]) -> Path:
    """Persist what the planted-signal diagnostic asked for and what it got.

    Written as its own file rather than folded into metadata.json so that the
    presence of ``planted_signal.json`` in a run directory is by itself the
    answer to "is this run real?".
    """
    return write_json(Path(run_dir) / "planted_signal.json", payload)


def save_seed_stability(run_dir: str | Path, seed_stability_df: pd.DataFrame) -> Path:
    """Persist the same evaluation repeated under several seeds.

    This is the error bar for every cross-experiment comparison: if two runs
    differ by less than the spread here, the difference is fit noise.
    """
    return save_dataframe(seed_stability_df, Path(run_dir) / "seed_stability.csv")


def save_line_comparison(run_dir: str | Path, line_comparison_df: pd.DataFrame) -> Path:
    """Persist the same predictions re-scored against alternative total lines.

    Informational: bets are settled against the closing line, which is not a
    price you can actually take. This table shows what the same model would
    have done at other lines (typically the consensus opener).
    """
    return save_dataframe(line_comparison_df, Path(run_dir) / "line_comparison.csv")


def save_calibration(
    run_dir: str | Path,
    *,
    summary: Any,
    buckets_df: pd.DataFrame | None,
    cv_summary: Any | None = None,
    cv_buckets_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Persist probability quality for a classifier run.

    Both splits are written where available. The CV table is the one to read:
    it pools ~5x the games, and a reliability curve built from ~115 holdout
    games is mostly noise.

    Phase 1 only measures this. Nothing here adjusts a probability -- the point
    is to learn whether raw XGBoost output is trustworthy enough to bet against
    an absolute threshold before paying to refit a calibrator inside the daily
    walk-forward.
    """
    run_dir = Path(run_dir)
    payload: dict[str, Any] = {"holdout": summary.model_dump()}
    if cv_summary is not None:
        payload["cv"] = cv_summary.model_dump()

    artifacts = {"metrics": write_json(run_dir / "calibration.json", payload)}
    if buckets_df is not None and not buckets_df.empty:
        artifacts["buckets"] = save_dataframe(
            buckets_df, run_dir / "calibration_buckets.csv"
        )
    if cv_buckets_df is not None and not cv_buckets_df.empty:
        artifacts["cv_buckets"] = save_dataframe(
            cv_buckets_df, run_dir / "cv_calibration_buckets.csv"
        )
    return artifacts


def save_baseline_metrics(
    run_dir: str | Path,
    *,
    baseline_cv: BaselineMetrics,
    baseline_fold_df: pd.DataFrame,
    baseline_holdout: BaselineMetrics,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    return {
        "metrics": write_json(
            run_dir / "baseline_metrics.json",
            {
                "cv": baseline_cv.model_dump(),
                "holdout": baseline_holdout.model_dump(),
            },
        ),
        "fold_metrics": save_dataframe(
            baseline_fold_df, run_dir / "baseline_fold_metrics.csv"
        ),
    }
