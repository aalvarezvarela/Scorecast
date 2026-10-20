#!/usr/bin/env python3
"""Compare the planted-signal diagnostic cells.

    poetry run python scripts/compare_planted_signal.py

The question this table answers is deliberately narrow:

    At what planted-signal strength does the current pipeline begin to detect
    and exploit a signal it is GIVEN?

It is NOT "is the model good". Every run here carries a feature derived from the
target, so its metrics are meaningless as evidence about live performance. They
are meaningful only in comparison with each other, because the four cells are
identical apart from how much target variance was planted.

What to read, in order
----------------------
1. **Did the pipeline use the feature at all?** ``fold_use_rate`` is the share of
   CV folds whose model split on ``PLANTED_SIGNAL`` at least once. If it is 0.00
   at 2% planted variance, the tree builder never chose a feature explaining 2%
   of the target across every fold -- and the search space, not the market, is
   the binding constraint.
2. **Did performance actually improve?** ``d_cv_mae`` versus the 0% control. Use
   of a feature is not the same as gain from it: a model can split on noise.
3. **Is the movement bigger than noise?** ``seed_roi_range`` and the control's
   own spread are the floor. A monotone-looking MAE curve across four cells is
   four measurements, not four independent confirmations.

Nothing here decides a threshold for you. The point is to make "the protocol
cannot see 1%" and "the market has no 1% to see" distinguishable, which they are
not from any real run.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from training_pipeline.reporting.discovery import (  # noqa: E402
    find_run_dirs,
    resolve_run_dirs,
)

DEFAULT_SOURCE = "experiments/diagnostics_planted_signal_2026_08"

#: Columns pulled straight out of each run's own artifacts. Read from files
#: rather than recomputed, so this script cannot disagree with the run it
#: describes.
_CV_KEYS = ("mae", "rmse", "win_rate", "roi", "n_bets", "n_games")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _selected_hyperparameters(run_dir: Path) -> dict[str, Any]:
    """The trial the run actually shipped its evaluation on."""
    payload = _read_json(run_dir / "optuna_selected_trial.json")
    trial = payload.get("selected_trial") or payload
    if not trial:
        trial = _read_json(run_dir / "optuna_best_trial.json")
    params = dict(trial.get("params") or {})
    attrs = dict(trial.get("user_attrs") or {})
    return {
        "train_games": params.get("train_games") or attrs.get("train_games"),
        "n_estimators": params.get("n_estimators") or attrs.get("n_estimators"),
        "learning_rate": params.get("learning_rate"),
        "max_depth": params.get("max_depth"),
        "colsample_bytree": params.get("colsample_bytree"),
        "min_child_weight": params.get("min_child_weight"),
        "gamma": params.get("gamma"),
        "cv_r2": attrs.get("pooled_r2"),
        "cv_ou_acc": attrs.get("pooled_ou_acc"),
        "n_folds": attrs.get("n_folds"),
    }


def collect(run_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        metadata = _read_json(run_dir / "metadata.json")
        planted = _read_json(run_dir / "planted_signal.json")
        if not planted and not metadata.get("is_diagnostic"):
            # Not a planted-signal run. Skipped by name rather than silently
            # averaged in -- a real run in this table would be the worst
            # possible error here.
            continue

        cv = _read_json(run_dir / "cv_betting_summary.json")
        row: dict[str, Any] = {
            "run": run_dir.name,
            "planted_variance": planted.get(
                "planted_requested_variance_explained"
            ),
            "measured_r2": planted.get("planted_measured_variance_explained"),
            "measured_corr": planted.get("planted_measured_correlation"),
            "fold_use_rate": planted.get("fold_use_rate"),
            "planted_gain": planted.get("mean_planted_gain"),
            "planted_gain_rank": planted.get("mean_planted_gain_rank"),
            "planted_weight": planted.get("mean_planted_weight"),
            "n_features": planted.get("mean_n_features"),
        }
        row.update({f"cv_{key}": cv.get(key) for key in _CV_KEYS})
        row.update(_selected_hyperparameters(run_dir))

        seeds_path = run_dir / "seed_stability.csv"
        if seeds_path.exists():
            seeds = pd.read_csv(seeds_path)
            if "roi" in seeds and len(seeds) > 1:
                row["seed_roi_range"] = float(
                    seeds["roi"].max() - seeds["roi"].min()
                )
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("planted_variance").reset_index(drop=True)


def add_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    """Differences against the 0% control, which is the only useful reading.

    An absolute CV MAE here is uninterpretable -- it is measured on a corrupted
    dataset. The *change* from the control is the whole result.
    """
    if frame.empty:
        return frame
    controls = frame[frame["planted_variance"] == 0.0]
    if controls.empty:
        print(
            "  ! No 0% control found. Deltas are omitted: without it there is "
            "nothing to attribute a difference to.\n"
        )
        return frame

    control = controls.iloc[0]
    out = frame.copy()
    for column in ("cv_mae", "cv_win_rate", "cv_roi", "cv_r2", "cv_ou_acc"):
        if column in out.columns and pd.notna(control.get(column)):
            out[f"d_{column}"] = pd.to_numeric(
                out[column], errors="coerce"
            ) - float(control[column])
    return out


def render(frame: pd.DataFrame) -> None:
    if frame.empty:
        print(
            "No planted-signal runs found. Run the campaign first:\n"
            "  bash experiments/runners/run_planted_signal_diagnostic.sh"
        )
        return

    headline = [
        "run", "planted_variance", "measured_r2", "cv_mae", "d_cv_mae",
        "cv_r2", "cv_ou_acc", "cv_win_rate", "cv_roi", "train_games",
        "n_estimators", "fold_use_rate", "planted_gain_rank",
    ]
    print("\n=== Planted-signal diagnostic ===")
    print("Every row carries a target-derived feature. Read the DELTAS, not the")
    print("levels: absolute metrics on a corrupted dataset mean nothing.\n")
    print(frame[[c for c in headline if c in frame.columns]].to_string(index=False))

    print("\n--- planted feature diagnostics ---")
    detail = [
        "run", "planted_variance", "measured_corr", "fold_use_rate",
        "planted_gain", "planted_gain_rank", "planted_weight", "n_features",
        "seed_roi_range",
    ]
    print(frame[[c for c in detail if c in frame.columns]].to_string(index=False))

    print("\n--- how to read it ---")
    use = frame.get("fold_use_rate")
    if use is not None and use.notna().any() and float(use.max()) == 0.0:
        print(
            "  fold_use_rate is 0.00 in EVERY cell: no fold's model ever split\n"
            "  on the planted feature, including where it explains 2% of the\n"
            "  target. That is a statement about the search space, not the market."
        )
    print(
        "  d_cv_mae should become more negative as planted_variance rises if the\n"
        "  pipeline is exploiting the signal. Compare its size against\n"
        "  seed_roi_range and against the control's own spread before calling\n"
        "  any particular strength a detection threshold."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "sources",
        nargs="*",
        default=[DEFAULT_SOURCE],
        help=(
            "Campaign config folder, artifacts folder, or run directories. "
            f"Defaults to {DEFAULT_SOURCE}."
        ),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Also write the comparison to this path.",
    )
    args = parser.parse_args()

    try:
        run_dirs = resolve_run_dirs(args.sources, project_root=REPO_ROOT)
    except (FileNotFoundError, ValueError):
        run_dirs = []
    if not run_dirs:
        # A campaign folder resolves to nothing until its runs exist; fall back
        # to scanning the artifacts tree so a partially finished campaign still
        # reports what it has.
        run_dirs = [
            path
            for path in find_run_dirs(REPO_ROOT / "artifacts" / "experiments")
            if path.name.startswith("diag_planted")
        ]

    frame = add_deltas(collect([Path(p) for p in run_dirs]))
    render(frame)

    if args.csv is not None and not frame.empty:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv, index=False)
        print(f"\nWritten to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
