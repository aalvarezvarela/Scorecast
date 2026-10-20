"""Read the per-run artifact files the charts need.

Every loader returns ``None`` (or an empty frame) rather than raising when a
file is absent. Runs legitimately differ in what they wrote -- regressors have
no calibration buckets, classifiers have no MAE, older runs predate whole
artifact families -- so a missing file is normal and a section should skip a
run rather than fail the notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from training_pipeline.betting import OUTCOME_COLUMN, outcome_from_predictions


def run_dir_of(row: Any) -> Path:
    """The run's directory, from either the explicit column or its parts."""
    if "run_dir" in row and pd.notna(row["run_dir"]):
        return Path(row["run_dir"])
    return Path(row["source_path"]) / row["run_name"]


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _optional_csv(path: Path, **kwargs: Any) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except (OSError, pd.errors.ParserError, ValueError):
        return None


def load_fold_betting(row: Any) -> pd.DataFrame | None:
    return _optional_csv(
        run_dir_of(row) / "cv_fold_betting.csv", parse_dates=["valid_start", "valid_end"]
    )


def load_seed_stability(row: Any) -> pd.DataFrame | None:
    return _optional_csv(run_dir_of(row) / "seed_stability.csv")


def load_calibration_buckets(row: Any) -> list[tuple[str, pd.DataFrame]]:
    """Reliability buckets, CV first because it pools far more games."""
    found = []
    for filename, source in (("cv_calibration_buckets.csv", "CV"),
                             ("calibration_buckets.csv", "holdout")):
        frame = _optional_csv(run_dir_of(row) / filename)
        if frame is not None and not frame.empty:
            found.append((source, frame))
    return found


def load_line_comparison(row: Any) -> pd.DataFrame | None:
    """Alternative-line scoring, preferring the CV table for its bet volume."""
    for filename, source in (("cv_line_comparison.csv", "CV folds"),
                             ("line_comparison.csv", "holdout")):
        frame = _optional_csv(run_dir_of(row) / filename)
        if frame is not None and not frame.empty:
            frame = frame.copy()
            frame.insert(0, "measured_on", source)
            return frame
    return None


def load_feature_names(row: Any) -> set[str]:
    path = run_dir_of(row) / "feature_schema.json"
    if not path.exists():
        return set()
    return set(read_json(path).get("feature_names", []))


def load_metadata(row: Any) -> dict[str, Any]:
    """The run's ``metadata.json``, or an empty dict when it is absent.

    This is where the RESULT of tuning is recorded -- ``train_games`` here is
    the window Optuna actually selected, while the same key under
    ``walk_forward`` in ``config.json`` is only the fallback used when tuning
    is off. Reading the config for it on a tuned run is a silent wrong answer,
    not an error, so prefer this loader whenever the question is what ran.
    """
    path = run_dir_of(row) / "metadata.json"
    if not path.exists():
        return {}
    return read_json(path)


def tuned_window_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Per run: was the training window tuned, and what did tuning pick?

    ``at_grid_edge`` is the column to read. A run that selected the largest
    offered window did not find an optimum -- it ran out of grid, and the true
    best window may be larger than anything it was allowed to try. The same
    holds at the small end. Either way the reported window is censored by the
    search space rather than chosen within it.
    """
    rows: list[dict[str, Any]] = []
    for _, run in runs.iterrows():
        metadata = load_metadata(run)
        choices = [int(c) for c in metadata.get("train_games_choices") or []]
        selected = metadata.get("train_games")
        selected = None if selected is None else int(selected)
        tuned = bool(metadata.get("train_games_tuned", False))
        rows.append({
            "label": run.get("label", run.get("run_name")),
            "run_name": run.get("run_name"),
            "dataset_type": metadata.get("dataset_type"),
            "snapshot_minutes": metadata.get("snapshot_minutes"),
            "tuned": tuned,
            "choices": ", ".join(str(c) for c in choices) if choices else "—",
            "selected": selected,
            "at_grid_edge": bool(
                tuned and choices and selected in (min(choices), max(choices))
            ),
            "n_games": metadata.get("n_games"),
            "rows_per_game": metadata.get("rows_per_game", 1.0),
            "cv_n_validation_games": metadata.get("cv_n_validation_games"),
            "holdout_n_games": metadata.get("holdout_n_games"),
        })
    return pd.DataFrame(rows)


def tuned_hyperparameters_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Per run: the hyperparameters the selected trial actually used.

    Prefers ``optuna_selected_trial.json`` (a lexicographic pick can override
    Optuna's own best) and falls back to ``optuna_best_trial.json``. Runs
    without any trial artifact are dropped rather than kept as empty rows.

    ``train_games`` is included when Optuna sampled it, so this table doubles
    as a cross-check against :func:`tuned_window_table` -- if the two disagree
    the run's metadata was written before the trial payload was refreshed.
    """
    rows: list[dict[str, Any]] = []
    for _, run in runs.iterrows():
        run_dir = run_dir_of(run)
        payload: dict[str, Any] | None = None
        for filename, key in (
            ("optuna_selected_trial.json", "selected_trial"),
            ("optuna_best_trial.json", "best_trial"),
        ):
            path = run_dir / filename
            if path.exists():
                payload = read_json(path).get(key)
                if payload:
                    break
        if not payload:
            continue
        params = payload.get("params") or {}
        rows.append({
            "label": run.get("label", run.get("run_name")),
            "trial": payload.get("number"),
            **params,
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("label")


def load_config_flat(row: Any) -> dict[str, Any]:
    """The run's resolved config, flattened to dotted keys for diffing."""
    path = run_dir_of(row) / "config.json"
    if not path.exists():
        return {}

    def flatten(value: dict, prefix: str = "") -> dict:
        flat: dict[str, Any] = {}
        for key, item in value.items():
            full = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                flat.update(flatten(item, full))
            else:
                flat[full] = (
                    json.dumps(item, sort_keys=True)
                    if isinstance(item, (list, dict)) else item
                )
        return flat

    return flatten(read_json(path))


def load_walk_forward(row: Any) -> dict[str, Any] | None:
    """Per-day retrain log plus pooled predictions, for runs scored daily."""
    run_dir = run_dir_of(row)
    daily_path = run_dir / "backtest_daily.csv"
    predictions_path = run_dir / "backtest_predictions.parquet"
    if not daily_path.exists() or not predictions_path.exists():
        return None
    daily = pd.read_csv(
        daily_path, parse_dates=["date", "train_start_date", "train_end_date"]
    )
    predictions = pd.read_parquet(predictions_path)
    predictions["date"] = pd.to_datetime(predictions["date"])
    return {
        "daily": daily.sort_values("date").reset_index(drop=True),
        "predictions": predictions.sort_values("date").reset_index(drop=True),
    }


def settle_bets(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach win/push outcomes to a predictions frame.

    A bet is on OVER (totals) or HOME (spread) when the predicted edge is
    positive. A game landing exactly on the line is a push: stake returned, and
    excluded from the win rate rather than counted as a loss.

    Reads the outcome from ``OUTCOME_COLUMN``, which every loader below
    materialises on the way in -- see ``_normalise_outcome``.
    """
    # Resolved via the helper, not read by name: settle_bets is called from
    # charts and alt_line as well as from the loaders below, and those callers
    # may hand it a frame that never passed through _normalise_outcome.
    margin = outcome_from_predictions(frame) - frame["target_line"]
    bet_over = frame["predicted_edge"] > 0
    frame = frame.copy()
    frame["push"] = margin == 0
    frame["won"] = (
        np.where(bet_over, margin > 0, margin < 0) & ~frame["push"]
    ).astype(int)
    return frame


#: Columns any settled-bet analysis needs, present for every strategy.
#:
#: The outcome is required under its market-neutral name. Requiring
#: "TOTAL_POINTS" here instead would silently DROP every spread run: the check
#: below is a ``continue``, so a run whose parquet carries ``actual_outcome``
#: would vanish from every comparison without appearing in any error, any count,
#: or any list of skipped runs.
_PREDICTION_COLUMNS = {"predicted_edge", "target_line", OUTCOME_COLUMN}


def _normalise_outcome(frame: pd.DataFrame) -> pd.DataFrame | None:
    """Materialise ``OUTCOME_COLUMN``, whichever spelling the run wrote.

    Runs predating the spread market wrote the outcome as ``TOTAL_POINTS``;
    newer ones write ``actual_outcome`` (and ``TOTAL_POINTS`` too, when that is
    genuinely what it is). Normalising once, here, is what lets every consumer
    downstream read a single name instead of each deciding for itself.

    Returns None when the frame carries no outcome at all, so the caller skips
    it exactly as it would have before.
    """
    if OUTCOME_COLUMN in frame.columns:
        return frame
    try:
        return frame.assign(**{OUTCOME_COLUMN: outcome_from_predictions(frame)})
    except KeyError:
        return None

#: (filename, label) in the order they should be preferred/reported.
PREDICTION_SOURCES = (
    ("cv_predictions.parquet", "cross-validation"),
    ("final_test_predictions.parquet", "holdout"),
)


def load_all_predictions(
    row: Any, *, drop_pushes: bool = True
) -> list[tuple[str, pd.DataFrame]]:
    """Every settled prediction set this run wrote, labelled by source.

    Unlike :func:`load_predictions`, which picks one, this returns both so the
    cross-validation and holdout periods can be compared directly rather than
    one standing in for the other.

    ``drop_pushes`` removes games landing exactly on the line, which is right
    for win-rate work. Set it False when handing the frame to
    ``betting.evaluate_betting``, which counts pushes itself -- stripping them
    first would understate the candidate pool and misstate ROI, whose
    denominator includes staked-and-returned capital.
    """
    found: list[tuple[str, pd.DataFrame]] = []
    for filename, source in PREDICTION_SOURCES:
        path = run_dir_of(row) / filename
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        normalised = _normalise_outcome(frame)
        if normalised is None or not _PREDICTION_COLUMNS <= set(normalised.columns):
            continue
        frame = _ensure_selection_score(normalised, row)
        if frame is None:
            continue
        frame = frame.dropna(subset=[*_PREDICTION_COLUMNS, "selection_score"])
        if frame.empty:
            continue
        settled = settle_bets(frame)
        found.append((source, settled[~settled["push"]] if drop_pushes else settled))
    return found


def load_prediction_cache(
    row_frame: pd.DataFrame, *, drop_pushes: bool = False
) -> dict[str, dict[str, pd.DataFrame]]:
    """``label -> {source: settled predictions}`` for every run, read once.

    ``drop_pushes`` defaults to False here, the opposite of
    :func:`load_all_predictions`, because the cache feeds
    ``betting.evaluate_betting``: it counts pushes itself, and stripping them
    first would understate the candidate pool and misstate ROI, whose
    denominator includes staked-and-returned capital. A caller that wants win
    rates only can drop them per frame.
    """
    return {
        str(row["label"]): dict(
            load_all_predictions(row, drop_pushes=drop_pushes)
        )
        for _, row in row_frame.iterrows()
    }


def config_matrix(
    row_frame: pd.DataFrame, *, ignore: set[str] | None = None
) -> pd.DataFrame:
    """Config fields that actually differ across these runs, one column per run.

    ``ignore`` names fields that vary by construction -- an experiment's own
    name, its hypothesis prose, its output paths -- so they do not crowd out
    the settings that change a result. Returns an empty frame when the runs are
    configured identically.
    """
    # label when the frame has been through prepare_runs, run_name otherwise --
    # the same fallback tuned_window_table uses, so the two tables key alike.
    configs = {
        str(row.get("label", row.get("run_name"))): load_config_flat(row)
        for _, row in row_frame.iterrows()
    }
    configs = {label: config for label, config in configs.items() if config}
    if not configs:
        return pd.DataFrame()
    matrix = pd.DataFrame(configs).sort_index()
    varies = matrix.apply(
        lambda values: values.astype(str).nunique(dropna=False) > 1, axis=1
    )
    return matrix.loc[varies & ~matrix.index.isin(ignore or set())]


#: Config fields that differ between runs by design rather than by experiment.
#: Showing them in the sanity check would bury the settings that change a
#: result under the ones that only change a name.
LABEL_ONLY_CONFIG_KEYS: set[str] = {
    "experiment_name", "training_version", "comparison_group", "hypothesis",
    "tags", "data.data_version", "data.expected_checksum", "window_dir_label",
    "window_name_label", "experiment_root_dir", "model_output_root",
    "save_experiment_artifacts",
}


def _ensure_selection_score(frame: pd.DataFrame, row: Any) -> pd.DataFrame | None:
    """Reconstruct ``selection_score`` for runs that predate the column.

    For a regressor it is exactly ``abs(predicted_edge)`` by definition (see
    ``decisions.predict_decisions``), so recovering it is lossless and lets
    older runs stay in the comparison instead of vanishing from it silently --
    which is the worse failure, since a dropped run looks identical to a run
    that was never there.

    A classifier's is a maximum of two expected values, which needs the
    probabilities and prices, so it cannot be rebuilt from these columns. In
    practice the classifier postdates the column, so nothing is lost.
    """
    if "selection_score" in frame.columns:
        return frame
    if row.get("is_classifier", False):
        return None
    return frame.assign(selection_score=frame["predicted_edge"].abs())


def load_predictions(row: Any) -> tuple[pd.DataFrame, str] | None:
    """Settled predictions, preferring the pooled CV folds for their volume.

    Returns ``(frame, source)`` with pushes already removed, or None when the
    run wrote nothing usable.
    """
    required = {"selection_score", "predicted_edge", "target_line", OUTCOME_COLUMN}
    for filename, source in (("cv_predictions.parquet", "CV"),
                             ("final_test_predictions.parquet", "holdout")):
        path = run_dir_of(row) / filename
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        normalised = _normalise_outcome(frame)
        if normalised is None or not required <= set(normalised.columns):
            continue
        frame = normalised.dropna(subset=list(required))
        if frame.empty:
            continue
        settled = settle_bets(frame)
        return settled[~settled["push"]], source
    return None
