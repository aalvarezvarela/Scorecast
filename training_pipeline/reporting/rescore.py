"""Re-score saved runs at one common bet threshold, at read time.

Why this exists: ``betting.primary_edge_threshold`` is an experiment-config
field, so the threshold in force *when a run executed* is frozen into that
run's ``betting_metrics.json`` and therefore into the leaderboard's ``roi`` /
``win_rate`` / ``n_bets`` columns. Change the config and old runs keep
reporting at the old threshold while new ones use the new one -- and the two
land in the same column, silently comparing different measurements.

The size of that mismatch is not subtle. One run, one model:

    threshold 2.0 -> 61.6% win rate, +17.5% ROI, 87 bets
    threshold 0.0 -> 55.7% win rate,  +6.3% ROI, 416 bets

Nothing about the model changed; only which bets were counted.

Every run also saves its raw predictions, so the fix is to ignore the frozen
headline and recompute from those at whichever threshold the analysis wants,
identically for every run. Scoring goes through
``betting.evaluate_betting`` -- the same function the pipeline itself uses --
so this is a re-application of the rule, not a second implementation of it.

Thresholds are per-strategy because their units differ: regressors select on
points away from the line, classifiers on expected value. One number cannot
serve both.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from training_pipeline.betting import (
    DECIMAL_ODDS_MINUS_110,
    evaluate_betting,
    outcome_from_predictions,
)
from training_pipeline.reporting import loaders

#: Which leaderboard columns each source's recomputed metrics overwrite.
_HOLDOUT_COLUMNS = {
    "n_bets": "n_bets",
    "bet_rate": "bet_rate",
    "win_rate": "win_rate",
    "win_rate_ci_low": "win_rate_ci_low",
    "win_rate_ci_high": "win_rate_ci_high",
    "roi": "roi",
    "profit_units": "profit_units",
    "is_significant": "is_significant",
    "n_candidates": "n_candidates",
    "min_edge": "bet_min_edge",
}
_CV_COLUMNS = {
    "n_bets": "cv_n_bets",
    "win_rate": "cv_win_rate",
    "win_rate_ci_low": "cv_win_rate_ci_low",
    "roi": "cv_roi",
    "profit_units": "cv_profit_units",
    "is_significant": "cv_is_significant",
}


def rescore_runs(
    runs: pd.DataFrame,
    *,
    edge_threshold: float | None = None,
    ev_threshold: float | None = None,
    flat_decimal_odds: float = DECIMAL_ODDS_MINUS_110,
) -> pd.DataFrame:
    """Recompute betting metrics for every run at a common threshold.

    ``edge_threshold`` applies to regressors (points away from the line) and
    ``ev_threshold`` to classifiers (expected value). Passing None for either
    leaves those runs at whatever threshold they were saved with, which is
    reported in the returned ``threshold`` column so a mixed comparison is at
    least visible rather than implied.

    Returns one row per run and prediction source.
    """
    rows: list[dict[str, Any]] = []

    for _, run in runs.iterrows():
        requested = ev_threshold if run["is_classifier"] else edge_threshold
        saved = run["bet_min_edge"] if pd.notna(run["bet_min_edge"]) else 0.0
        threshold = saved if requested is None else float(requested)

        # Pushes are kept: evaluate_betting counts them itself, and dropping
        # them first would understate the candidate pool and distort ROI.
        for source, frame in loaders.load_all_predictions(run, drop_pushes=False):
            metrics = evaluate_betting(
                predicted_edge=frame["predicted_edge"],
                actual_total=outcome_from_predictions(frame),
                line=frame["target_line"],
                selection_score=frame["selection_score"],
                min_edge=threshold,
                flat_decimal_odds=flat_decimal_odds,
            )
            rows.append({
                "label": run["label"],
                "prediction_strategy": run["prediction_strategy"],
                "source": source,
                "threshold": threshold,
                "threshold_units": "EV" if run["is_classifier"] else "points",
                "rescored": requested is not None,
                **metrics.model_dump(),
            })

    return pd.DataFrame(rows)


def apply_rescored(runs: pd.DataFrame, rescored: pd.DataFrame) -> pd.DataFrame:
    """Overwrite the runs frame's betting columns with the recomputed ones.

    Applied so that *every* downstream section reads the same threshold, not
    just the one table where the re-scoring was displayed. Runs missing a
    prediction file keep their saved values rather than being blanked.
    """
    if rescored.empty:
        return runs

    updated = runs.copy()
    by_source = {
        "holdout": _HOLDOUT_COLUMNS,
        "cross-validation": _CV_COLUMNS,
    }

    for source, mapping in by_source.items():
        subset = rescored[rescored["source"] == source].set_index("label")
        if subset.empty:
            continue
        for source_column, target_column in mapping.items():
            if source_column not in subset.columns:
                continue
            values = updated["label"].map(subset[source_column])
            if target_column not in updated.columns:
                updated[target_column] = values
            else:
                updated[target_column] = values.where(
                    values.notna(), updated[target_column]
                )

    # Derived columns are functions of the metrics just replaced, so they have
    # to be recomputed rather than left describing the superseded numbers.
    if {"cv_roi", "roi"} <= set(updated.columns):
        updated["cv_minus_holdout_roi"] = updated["cv_roi"] - updated["roi"]
    if {"roi", "bias_baseline_roi"} <= set(updated.columns):
        updated["roi_vs_bias_baseline"] = updated["roi"] - updated["bias_baseline_roi"]
    return updated
