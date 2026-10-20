"""Does trimming the low-margin predictions raise the win rate?

The betting rule this project ships is "bet when ``selection_score`` clears a
threshold". ``selection_score`` is the model's own margin: ``|predicted_edge|``
in points for a regressor, expected value for the classifier. A score near zero
means the model thinks the game lands on the line, so it has no opinion worth
acting on. The obvious hope is that discarding those raises the win rate on
what is left.

This module answers that question the way it has to be answered -- by
COVERAGE, not by a raw threshold. A cutoff of 2.0 points keeps 36% of one run's
games and 71% of another's, so a shared numeric threshold compares two different
selectivities and reads as a model difference. Coverage ("keep the top 60% by
margin") means the same thing for every run and every strategy, which is the
only way the curves below can sit on one axis.

Two questions, two modes, and they are NOT interchangeable
----------------------------------------------------------
``mode="self"`` ranks each source by its own margins. It answers *"is margin
ordered with winning at all?"* -- a description of the data. It is NOT a
strategy: choosing the cut that looks best on holdout, on holdout, is selecting
a threshold in-sample and the number it prints is not repeatable.

``mode="cv"`` derives the cutoff from cross-validation, freezes it, and applies
that same number to holdout. It answers *"what would this rule have paid?"* --
which is what you would actually have been able to run. Realised holdout
coverage then drifts from the target, and that drift is itself information: it
says the two periods' score distributions differ.

Read ``self`` first to see whether there is any signal to exploit; read ``cv``
to see what is left of it once the cutoff has to be chosen in advance.

Disjoint beats cumulative for the significance question
-------------------------------------------------------
The cumulative curve is hard to read for a real effect because every cut shares
most of its games with the one before it, so the whole curve moves together and
"top 60% beats top 100%" is largely a comparison of a set with itself.
:func:`margin_bucket_table` cuts the same ranking into disjoint equal-size
buckets instead. Those really are independent samples, so the trend across them
carries an honest p-value -- supplied here by :func:`margin_trend`.

The pooled-snapshot caveat
--------------------------
On the intermediate-line dataset one game appears once per pre-game snapshot,
so a prediction frame holds ~10 correlated rows per game that share a single
outcome. ``evaluate_betting`` counts each as an independent bet and its Wilson
interval is then far too narrow. ``training_pipeline.snapshot_scoring`` handles
this by suppressing the interval on its pooled row rather than printing a
corrected one, because an honest interval needs game-clustered inference. This
module follows that policy exactly: pass ``rows_per_game`` (from the run's
``metadata.json``) and any run above ``INDEPENDENCE_TOLERANCE`` gets its
interval and significance verdict blanked, and is marked ``independent=False``
so charts can draw it without an error band.
"""

from __future__ import annotations

import textwrap
from typing import Any, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

from training_pipeline.betting import (
    evaluate_betting,
    outcome_from_predictions,
    wilson_interval,
)
from training_pipeline.reporting.theme import (
    AXIS,
    BREAK_EVEN,
    CRITICAL,
    DECIMAL_ODDS,
    GOOD,
    INK,
    INK_2,
    LINE_REF,
    MUTED,
    STRATEGY_COLOR,
    SURFACE,
    rotate_xticks,
)

#: Coverage levels the notebook sweeps: keep the top 100%, 90%, ... 10% of
#: games by margin. A 10% step is fine enough to see where a curve turns and
#: coarse enough that the thinnest cut still holds ~40 holdout games. Nothing
#: below 10% is offered: at ~400 holdout games a 5% cut is 20 bets, where a
#: 60% win rate is two games away from a 50% one.
COVERAGE_GRID: tuple[float, ...] = (
    1.00, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10,
)

#: Rows-per-game above which a frame's rows are treated as correlated and its
#: binomial interval is suppressed. 1.0 is one row per game; the intermediate
#: dataset runs ~9.9 because a horizon is occasionally missing.
INDEPENDENCE_TOLERANCE = 1.05

#: Default y limits for every win-rate chart here. The interesting range for
#: this problem is narrow: break-even is 52.4%, nothing in this project has
#: come near 70%, and a 0-100% axis compresses the whole story into a band a
#: few pixels tall. The limits are a FLOOR, never a crop -- see
#: ``win_rate_limits``, which widens them rather than hide a point.
WIN_RATE_YLIM: tuple[float, float] = (0.35, 0.70)

#: The column carrying the model's margin. Regressors: |predicted edge| in
#: points. Classifier: expected value. Never comparable ACROSS those two as a
#: number, which is the whole reason this module works in coverage.
SCORE_COLUMN = "selection_score"


def cutoff_for_coverage(scores: Any, coverage: float) -> float:
    """Score cutoff that retains at least ``coverage`` of ``scores``.

    ``evaluate_betting`` selects on ``score > threshold``, strictly. Returning
    the empirical quantile itself would therefore drop every game sitting
    exactly on it -- which on a discrete score (rounded lines, tied classifier
    EVs) can be many games at once. Stepping one float below includes the
    boundary, so realised coverage lands at or just above the target rather
    than collapsing below it.
    """
    if not 0 < coverage <= 1:
        raise ValueError("coverage must be in (0, 1].")
    values = (
        pd.to_numeric(pd.Series(scores), errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if values.empty:
        return float("nan")
    quantile = float(values.quantile(1.0 - coverage, interpolation="higher"))
    return float(np.nextafter(quantile, -np.inf))


def _blank_if_correlated(row: dict[str, Any], independent: bool) -> dict[str, Any]:
    """Suppress binomial inference on a frame with repeated rows per game."""
    row["independent"] = independent
    if independent:
        return row
    for field in ("win_rate_ci_low", "win_rate_ci_high"):
        row[field] = float("nan")
    row["is_significant"] = pd.NA
    return row


def _score_at(
    frame: pd.DataFrame, cutoff: float, *, flat_decimal_odds: float
) -> dict[str, Any]:
    metrics = evaluate_betting(
        predicted_edge=frame["predicted_edge"],
        # Resolved per call rather than read by name: coverage is reached from
        # several places, not all of which come through loaders' normalisation,
        # and this helper accepts either spelling (and raises a named error if
        # neither is present) instead of a bare KeyError.
        actual_total=outcome_from_predictions(frame),
        line=frame["target_line"],
        selection_score=frame[SCORE_COLUMN],
        min_edge=cutoff,
        # Deliberately flat -110. The prediction artifacts do not retain
        # executable per-bet prices, so any other number would be invented.
        flat_decimal_odds=flat_decimal_odds,
    )
    return metrics.model_dump()


def coverage_table(
    frames: dict[str, pd.DataFrame],
    *,
    label: str = "",
    prediction_strategy: str = "",
    coverage_grid: tuple[float, ...] = COVERAGE_GRID,
    mode: str = "self",
    calibration_source: str = "cross-validation",
    rows_per_game: float = 1.0,
    flat_decimal_odds: float = DECIMAL_ODDS,
) -> pd.DataFrame:
    """Win rate and ROI for the top ``coverage`` fraction of games by margin.

    ``frames`` maps a source name ("cross-validation", "holdout") to that
    source's settled predictions, as returned by
    ``loaders.load_all_predictions(run, drop_pushes=False)``. Pushes must be
    KEPT: ``evaluate_betting`` counts them itself, and stripping them first
    would understate the candidate pool and misstate ROI, whose denominator
    includes staked-and-returned capital.

    ``mode="self"`` recomputes the cutoff inside each source (descriptive).
    ``mode="cv"`` computes it once on ``calibration_source`` and applies that
    frozen number everywhere (executable). See the module docstring: these
    answer different questions and the second is the honest one for a decision.
    """
    if mode not in {"self", "cv"}:
        raise ValueError("mode must be 'self' or 'cv'.")
    if mode == "cv" and calibration_source not in frames:
        raise ValueError(
            f"mode='cv' needs a {calibration_source!r} frame to learn the "
            f"cutoff from; got {sorted(frames)}."
        )

    independent = rows_per_game <= INDEPENDENCE_TOLERANCE
    rows: list[dict[str, Any]] = []
    for coverage in coverage_grid:
        frozen = (
            cutoff_for_coverage(frames[calibration_source][SCORE_COLUMN], coverage)
            if mode == "cv"
            else None
        )
        for source, frame in frames.items():
            cutoff = (
                frozen
                if frozen is not None
                else cutoff_for_coverage(frame[SCORE_COLUMN], coverage)
            )
            metrics = _score_at(frame, cutoff, flat_decimal_odds=flat_decimal_odds)
            row = {
                "label": label,
                "prediction_strategy": prediction_strategy,
                "source": source,
                "mode": mode,
                "target_coverage": coverage,
                "cutoff": cutoff,
                "cutoff_units": "EV"
                if prediction_strategy == "over_under_classifier"
                else "points",
                **metrics,
            }
            # bet_rate is realised coverage. Under mode="cv" it drifts away
            # from the target on holdout, and that drift is the point.
            row["realised_coverage"] = row.pop("bet_rate")
            rows.append(_blank_if_correlated(row, independent))

    table = pd.DataFrame(rows)
    # Lift against keeping everything, per source: the number the question
    # "does trimming help?" actually asks for.
    baseline = (
        table[table["target_coverage"] == max(coverage_grid)]
        .set_index("source")["win_rate"]
    )
    table["win_rate_vs_full"] = table["win_rate"] - table["source"].map(baseline)
    return table


def margin_bucket_table(
    frame: pd.DataFrame,
    *,
    label: str = "",
    source: str = "",
    n_buckets: int = 10,
    rows_per_game: float = 1.0,
    flat_decimal_odds: float = DECIMAL_ODDS,
) -> pd.DataFrame:
    """Win rate within DISJOINT equal-size margin buckets, highest first.

    The cumulative curve cannot separate a real effect from its own overlap:
    every cut contains the one above it. These buckets share no games, so their
    win rates are independent estimates and a monotone trend across them means
    something. Bucket 1 is the highest-margin tenth.

    Pushes are dropped here -- unlike the coverage table -- because a bucket's
    win rate is the only quantity being reported and a push is neither a win
    nor a loss. ``n_pushes`` is kept so the drop stays visible.
    """
    if n_buckets < 2:
        raise ValueError("n_buckets must be at least 2.")
    scores = pd.to_numeric(frame[SCORE_COLUMN], errors="coerce")
    usable = frame.loc[scores.notna()].copy()
    if usable.empty:
        return pd.DataFrame()

    # Rank descending so bucket 1 is the strongest margin. "first" breaks ties
    # by position rather than sharing a rank, which keeps the buckets equal in
    # size even when the score is heavily tied.
    order = scores.loc[usable.index].rank(method="first", ascending=False)
    usable["bucket"] = np.minimum(
        ((order - 1) * n_buckets // len(usable)).astype(int) + 1, n_buckets
    )

    independent = rows_per_game <= INDEPENDENCE_TOLERANCE
    rows: list[dict[str, Any]] = []
    for bucket, group in usable.groupby("bucket", sort=True):
        decided = group.loc[~group["push"].astype(bool)]
        n_wins = int(decided["won"].sum())
        n_decided = int(len(decided))
        win_rate = n_wins / n_decided if n_decided else float("nan")
        ci_low, ci_high = wilson_interval(n_wins, n_decided)
        profit = n_wins * (flat_decimal_odds - 1.0) - (n_decided - n_wins)
        rows.append(
            _blank_if_correlated(
                {
                    "label": label,
                    "source": source,
                    "bucket": int(bucket),
                    "pct_from_top": f"{(bucket - 1) * 100 // n_buckets}"
                    f"-{bucket * 100 // n_buckets}%",
                    "n_rows": int(len(group)),
                    "n_pushes": int(len(group) - n_decided),
                    "score_min": float(scores.loc[group.index].min()),
                    "score_max": float(scores.loc[group.index].max()),
                    "n_wins": n_wins,
                    "n_decided": n_decided,
                    "win_rate": win_rate,
                    "win_rate_ci_low": ci_low,
                    "win_rate_ci_high": ci_high,
                    "roi": profit / len(group) if len(group) else float("nan"),
                    "is_significant": bool(ci_low > BREAK_EVEN),
                },
                independent,
            )
        )
    return pd.DataFrame(rows)


def margin_trend(frame: pd.DataFrame, *, rows_per_game: float = 1.0) -> dict[str, Any]:
    """Is margin monotonically related to winning, and by how much?

    Two tests on the same ranking, because they fail differently:

    * **Rank correlation** between margin and the win indicator, over decided
      bets. Sensitive to a consistent ordering anywhere in the distribution.
    * **Top-half versus bottom-half** win rate as a two-proportion z-test. The
      halves are disjoint, so this is a legitimate comparison of two
      independent samples -- which "top 50% versus top 100%" is not.

    Both p-values assume independent rows. On a pooled-snapshot frame they are
    anti-conservative by an unknown factor, so ``independent`` is returned
    alongside and the caller must not report them as-is when it is False.
    """
    decided = frame.loc[~frame["push"].astype(bool)]
    scores = pd.to_numeric(decided[SCORE_COLUMN], errors="coerce")
    won = pd.to_numeric(decided["won"], errors="coerce")
    keep = scores.notna() & won.notna()
    scores, won = scores[keep], won[keep]

    result: dict[str, Any] = {
        "n_decided": int(len(scores)),
        "independent": rows_per_game <= INDEPENDENCE_TOLERANCE,
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
        "top_half_win_rate": float("nan"),
        "bottom_half_win_rate": float("nan"),
        "half_gap": float("nan"),
        "half_gap_p": float("nan"),
    }
    if len(scores) < 20 or won.nunique() < 2:
        return result

    rho, p_value = stats.spearmanr(scores, won)
    result["spearman_rho"] = float(rho)
    result["spearman_p"] = float(p_value)

    median = scores.median()
    top = won[scores > median]
    bottom = won[scores <= median]
    if len(top) and len(bottom):
        result["top_half_win_rate"] = float(top.mean())
        result["bottom_half_win_rate"] = float(bottom.mean())
        result["half_gap"] = result["top_half_win_rate"] - result["bottom_half_win_rate"]
        pooled = float(pd.concat([top, bottom]).mean())
        se = np.sqrt(pooled * (1 - pooled) * (1 / len(top) + 1 / len(bottom)))
        if se > 0:
            z = result["half_gap"] / se
            result["half_gap_p"] = float(2 * stats.norm.sf(abs(z)))
    return result


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

#: Source is encoded by colour and redundantly by marker and linestyle, so the
#: CV/holdout distinction survives a greyscale print and CVD alike.
SOURCE_STYLE: dict[str, dict[str, Any]] = {
    "cross-validation": {"color": "#244a6b", "marker": "o", "linestyle": "-"},
    "holdout": {"color": "#b0762f", "marker": "D", "linestyle": "--"},
}


def _style(source: str) -> dict[str, Any]:
    return SOURCE_STYLE.get(source, {"color": MUTED, "marker": "s", "linestyle": ":"})


def win_rate_limits(
    values: Any, ylim: tuple[float, float] = WIN_RATE_YLIM
) -> tuple[float, float]:
    """``ylim``, widened if a plotted win rate would fall outside it.

    Focusing the axis is a reading aid; hiding a data point is a lie. So the
    requested window is treated as a minimum extent and stretched whenever the
    series leaves it. Confidence BANDS are allowed to run off the top and
    bottom -- they are context around the point, and letting a 33%-to-80%
    interval set the scale is what made the axis useless in the first place.
    """
    finite = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if finite.empty:
        return ylim
    return (
        min(ylim[0], float(finite.min()) - 0.01),
        max(ylim[1], float(finite.max()) + 0.01),
    )


def scoreboard_table(
    table: pd.DataFrame,
    *,
    coverage_level: float = 1.0,
    cv_source: str = "cross-validation",
    holdout_source: str = "holdout",
) -> pd.DataFrame:
    """One row per run: CV and holdout win rate side by side, no threshold.

    Built from the ``coverage_level`` slice of :func:`coverage_table`, which at
    the default 1.0 means EVERY prediction the run made. That is deliberate for
    an overview: each run freezes its own bet threshold into
    ``betting_metrics.json``, so the leaderboard's ``win_rate`` columns can be
    measuring different selectivities for different runs and still line up in
    one column. Scoring every prediction removes the threshold from the
    comparison entirely.

    Sorted by holdout win rate, best last -- matplotlib's y axis grows upward,
    so the best run ends up at the top of the chart.
    """
    view = table[np.isclose(table["target_coverage"], coverage_level)]
    if view.empty:
        raise ValueError(
            f"No rows at coverage {coverage_level:.0%}; the grid holds "
            f"{sorted(table['target_coverage'].unique())}."
        )
    columns = ["win_rate", "win_rate_ci_low", "win_rate_ci_high", "n_bets"]
    parts = []
    for source, prefix in ((cv_source, "cv"), (holdout_source, "holdout")):
        side = view[view["source"] == source][["label", *columns]]
        parts.append(side.rename(columns={c: f"{prefix}_{c}" for c in columns}))

    merged = parts[0].merge(parts[1], on="label", how="outer")
    independence = view.groupby("label")["independent"].all()
    merged["independent"] = merged["label"].map(independence).fillna(True)
    merged["holdout_minus_cv"] = (
        merged["holdout_win_rate"] - merged["cv_win_rate"]
    )
    return merged.sort_values(
        "holdout_win_rate", ascending=True, na_position="first"
    ).reset_index(drop=True)


def operating_point_table(
    runs: pd.DataFrame,
    prediction_cache: dict[str, dict[str, pd.DataFrame]],
    *,
    coverage_level: float,
    calibration_source: str = "cross-validation",
) -> pd.DataFrame:
    """Every run scored at ONE cutoff, learned from CV and frozen.

    The operating point of the whole report: a cutoff derived from
    ``coverage_level`` of each run's cross-validation games, then applied
    unchanged to holdout. Expressed as coverage rather than as a raw number
    because a 0.5-point cutoff keeps a different share of each run's games and
    would read as a model difference.

    Runs with no usable CV predictions have no cutoff to freeze, so they are
    excluded -- and named in ``.attrs["missing_calibration"]``, because a run
    that vanishes from a comparison looks exactly like a run that was never
    there.
    """
    rows, missing = [], []
    for _, run in runs.iterrows():
        frames = prediction_cache.get(str(run["label"]), {})
        calibration = frames.get(calibration_source)
        if calibration is None or calibration.empty:
            missing.append(str(run["label"]))
            continue
        rows.append(
            coverage_table(
                frames,
                label=run["label"],
                prediction_strategy=run["prediction_strategy"],
                coverage_grid=(coverage_level,),
                mode="cv",
                calibration_source=calibration_source,
                rows_per_game=float(run["rows_per_game"]),
            )
        )
    if not rows:
        raise ValueError(
            "No run has usable cross-validation predictions, so no operating "
            "point can be derived."
        )
    table = pd.concat(rows, ignore_index=True)
    panel = dict(zip(runs["label"], runs.get("panel_label", runs["label"]), strict=True))
    table["panel"] = table["label"].map(panel)
    table.attrs["missing_calibration"] = missing
    return table


def headline_table(
    operating: pd.DataFrame,
    *,
    cv_source: str = "cross-validation",
    holdout_source: str = "holdout",
) -> pd.DataFrame:
    """The operating-point numbers as one row per run, CV beside holdout.

    Ordered by holdout win rate, best first -- the reading order of a table,
    which is the opposite of the plotting order a chart wants.
    """
    columns = {
        "realised_coverage": "kept",
        "n_candidates": "candidates",
        "n_bets": "bets",
        "win_rate": "win_rate",
        "win_rate_ci_low": "ci_low",
        "win_rate_ci_high": "ci_high",
        "roi": "flat_roi",
    }
    parts = []
    for source, prefix in ((cv_source, "cv"), (holdout_source, "holdout")):
        side = operating[operating["source"] == source]
        keep = ["label", *columns]
        if prefix == "cv":
            keep = ["label", "cutoff", "cutoff_units", *columns]
        parts.append(
            side[keep].rename(
                columns={old: f"{prefix}_{new}" for old, new in columns.items()}
            )
        )
    return (
        parts[0]
        .merge(parts[1], on="label", how="left")
        .sort_values("holdout_win_rate", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


class CoverageViews(NamedTuple):
    """The four tables the margin-threshold section reads, built in one pass.

    They are returned together because they must come from the same predictions
    and the same coverage grid: a bucket chart drawn from one grid beside a
    curve drawn from another would invite a comparison that is not valid.
    """

    #: Cumulative top-x%, cut inside each source. Descriptive.
    self_coverage: pd.DataFrame
    #: Cumulative top-x%, cutoff frozen from CV. Executable.
    cv_coverage: pd.DataFrame
    #: Disjoint equal-size margin bands, per run and source.
    buckets: pd.DataFrame
    #: Rank correlation and top-half/bottom-half test, per run and source.
    trends: pd.DataFrame


def build_views(
    runs: pd.DataFrame,
    prediction_cache: dict[str, dict[str, pd.DataFrame]],
    *,
    coverage_grid: tuple[float, ...] = COVERAGE_GRID,
    n_buckets: int = 10,
    calibration_source: str = "cross-validation",
) -> CoverageViews:
    """Every margin view for every run, from the shared prediction cache.

    ``runs`` must carry ``label``, ``prediction_strategy`` and
    ``rows_per_game`` -- the last decides whether an interval may be reported,
    and defaulting it to 1.0 here would silently hand a pooled-snapshot run the
    binomial maths it does not qualify for.
    """
    self_rows, cv_rows, bucket_rows, trend_rows = [], [], [], []
    for _, run in runs.iterrows():
        frames = prediction_cache.get(str(run["label"]), {})
        if not frames:
            continue
        rows_per_game = float(run["rows_per_game"])
        shared = {
            "label": run["label"],
            "prediction_strategy": run["prediction_strategy"],
            "coverage_grid": coverage_grid,
            "rows_per_game": rows_per_game,
        }
        self_rows.append(coverage_table(frames, mode="self", **shared))
        if calibration_source in frames:
            cv_rows.append(
                coverage_table(
                    frames, mode="cv",
                    calibration_source=calibration_source, **shared,
                )
            )
        for source, frame in frames.items():
            bucket_rows.append(
                margin_bucket_table(
                    frame, label=run["label"], source=source,
                    n_buckets=n_buckets, rows_per_game=rows_per_game,
                )
            )
            trend_rows.append({
                "label": run["label"], "source": source,
                **margin_trend(frame, rows_per_game=rows_per_game),
            })

    panel = dict(zip(runs["label"], runs.get("panel_label", runs["label"]), strict=True))
    views = []
    for rows in (self_rows, cv_rows, bucket_rows):
        frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if not frame.empty:
            frame["panel"] = frame["label"].map(panel)
        views.append(frame)
    self_view, cv_view, bucket_view = views
    return CoverageViews(
        self_coverage=self_view,
        cv_coverage=cv_view,
        buckets=bucket_view,
        trends=pd.DataFrame(trend_rows),
    )


def build_scoreboard(
    runs: pd.DataFrame,
    prediction_cache: dict[str, dict[str, pd.DataFrame]],
    *,
    coverage_level: float = 1.0,
) -> pd.DataFrame:
    """The overview table: one row per run, with its name and one-line spec.

    Scored at ``coverage_level`` -- 1.0, every prediction -- so no bet
    threshold enters the comparison. See :func:`scoreboard_table`.
    """
    from training_pipeline.reporting.theme import run_spec

    rows = [
        coverage_table(
            frames,
            label=run["label"],
            prediction_strategy=run["prediction_strategy"],
            coverage_grid=(coverage_level,),
            rows_per_game=float(run["rows_per_game"]),
        )
        for _, run in runs.iterrows()
        if (frames := prediction_cache.get(str(run["label"])))
    ]
    if not rows:
        raise ValueError("No cached predictions, so there is nothing to score.")

    board = scoreboard_table(pd.concat(rows, ignore_index=True),
                             coverage_level=coverage_level)
    by_label = runs.set_index("label")
    board["experiment_name"] = board["label"].map(by_label["experiment_name"])
    board["spec"] = board["label"].map(lambda label: run_spec(by_label.loc[label]))
    return board


def executable_table(cv_coverage: pd.DataFrame) -> pd.DataFrame:
    """CV cutoff and its frozen holdout result, one row per run and coverage.

    Merged on ``cutoff`` as well as coverage. Within a ``mode="cv"`` frame the
    two sides already share a cutoff, so that is redundant there -- it earns
    its place against the wrong input. Handed a ``mode="self"`` frame, where
    each source cut itself at its own cutoff, the extra key leaves the holdout
    columns empty instead of pairing two different rules under one heading and
    labelling the result "frozen from CV".
    """
    if cv_coverage.empty:
        return pd.DataFrame()
    cv_side = (
        cv_coverage[cv_coverage["source"] == "cross-validation"][[
            "label", "target_coverage", "cutoff", "cutoff_units",
            "realised_coverage", "n_bets", "win_rate", "roi",
        ]]
        .rename(columns={
            "realised_coverage": "cv_kept", "n_bets": "cv_bets",
            "win_rate": "cv_win", "roi": "cv_flat_roi",
        })
    )
    holdout_side = (
        cv_coverage[cv_coverage["source"] == "holdout"][[
            "label", "target_coverage", "cutoff", "realised_coverage", "n_bets",
            "win_rate", "win_rate_ci_low", "roi", "win_rate_vs_full", "independent",
        ]]
        .rename(columns={
            "realised_coverage": "holdout_kept", "n_bets": "holdout_bets",
            "win_rate": "holdout_win", "win_rate_ci_low": "holdout_ci_low",
            "roi": "holdout_flat_roi", "win_rate_vs_full": "holdout_win_vs_full",
        })
    )
    return (
        cv_side.merge(holdout_side, on=["label", "target_coverage", "cutoff"], how="left")
        .sort_values(["label", "target_coverage"], ascending=[True, False])
        .reset_index(drop=True)
    )


def plot_scoreboard(
    scoreboard: pd.DataFrame,
    *,
    name_col: str = "experiment_name",
    spec_col: str = "spec",
    ax: Any = None,
) -> Any:
    """The one-screen answer to "which run did best?".

    A paired dot plot, not bars. The values sit between 48% and 54% on a scale
    whose meaningful zero is 52.38%, so bars would need a truncated axis to
    show any difference at all -- and a truncated bar misreads as a ratio. A
    dot encodes position, where a non-zero axis is honest.

    CV and holdout are joined by a connector, so the out-of-sample movement is
    the shape of the row rather than a number to subtract. Wilson intervals are
    thin recessive rules behind the dots: at these sample sizes every one of
    them crosses break-even, and that overlap is the real finding -- but it
    should not shout louder than the estimates it surrounds. The x limits are
    set by the intervals rather than the dots, because an interval running off
    the edge of the panel would understate exactly the uncertainty it is there
    to show.

    Only the holdout value is labelled, and it is placed on the far side of the
    dot from its CV partner so it can never sit on top of it.
    """
    if ax is None:
        _, ax = plt.subplots(
            figsize=(9.6, 0.62 * len(scoreboard) + 1.9), constrained_layout=True
        )
    positions = np.arange(len(scoreboard))

    # Chance to break-even: the band a run can land in and still lose money.
    ax.axvspan(0.50, BREAK_EVEN, color=MUTED, alpha=0.07, zorder=0)

    cv_style, holdout_style = _style("cross-validation"), _style("holdout")
    for position, row in zip(positions, scoreboard.itertuples(), strict=True):
        if row.independent:
            for low, high, colour in (
                (row.cv_win_rate_ci_low, row.cv_win_rate_ci_high, cv_style["color"]),
                (row.holdout_win_rate_ci_low, row.holdout_win_rate_ci_high,
                 holdout_style["color"]),
            ):
                if pd.notna(low) and pd.notna(high):
                    ax.plot([low, high], [position, position], color=colour,
                            linewidth=1.1, alpha=0.28, solid_capstyle="butt", zorder=1)
        if pd.notna(row.cv_win_rate) and pd.notna(row.holdout_win_rate):
            ax.plot([row.cv_win_rate, row.holdout_win_rate], [position, position],
                    color=AXIS, linewidth=2.4, zorder=2)

    ax.scatter(scoreboard["cv_win_rate"], positions, s=68, zorder=3,
               color=cv_style["color"], marker=cv_style["marker"],
               edgecolor=SURFACE, linewidth=1.2, label="cross-validation")
    ax.scatter(scoreboard["holdout_win_rate"], positions, s=76, zorder=4,
               color=holdout_style["color"], marker=holdout_style["marker"],
               edgecolor=SURFACE, linewidth=1.2, label="holdout")

    for position, row in zip(positions, scoreboard.itertuples(), strict=True):
        if pd.isna(row.holdout_win_rate):
            continue
        # Away from the CV dot, so the label never lands on the connector.
        to_the_right = (
            pd.isna(row.cv_win_rate) or row.holdout_win_rate >= row.cv_win_rate
        )
        ax.annotate(
            f"{row.holdout_win_rate:.1%}", (row.holdout_win_rate, position),
            xytext=(10 if to_the_right else -10, -3), textcoords="offset points",
            ha="left" if to_the_right else "right",
            fontsize=8.5, fontweight="semibold", color=holdout_style["color"],
        )

    ax.axvline(BREAK_EVEN, color=CRITICAL, linewidth=1.4, linestyle=(0, (4, 3)), zorder=5)
    ax.annotate(f"break-even {BREAK_EVEN:.1%}", xy=(BREAK_EVEN, 1.0),
                xycoords=("data", "axes fraction"), xytext=(4, 6),
                textcoords="offset points", fontsize=8, color=CRITICAL)

    bounds = pd.concat([
        scoreboard[column] for column in (
            "cv_win_rate", "holdout_win_rate",
            "cv_win_rate_ci_low", "cv_win_rate_ci_high",
            "holdout_win_rate_ci_low", "holdout_win_rate_ci_high",
        ) if column in scoreboard
    ]).dropna()
    span = [float(bounds.min()), float(bounds.max()), 0.50, BREAK_EVEN]
    ax.set_xlim(min(span) - 0.015, max(span) + 0.02)

    ax.set_yticks(positions, scoreboard[name_col], fontsize=9)
    ax.set_ylim(-0.6, len(scoreboard) - 0.4)
    if spec_col in scoreboard:
        # Outside the panel, under the name: identification belongs with the
        # label, not in the plotting area where it would cross the intervals.
        for position, spec in zip(positions, scoreboard[spec_col], strict=True):
            ax.annotate(str(spec), xy=(0, position), xycoords=("axes fraction", "data"),
                        xytext=(-8, -11), textcoords="offset points",
                        ha="right", fontsize=7.5, color=MUTED)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("win rate — every prediction, no bet threshold")
    ax.grid(axis="y", visible=False)
    ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    return ax


def plot_coverage_curve(
    table: pd.DataFrame,
    *,
    title: str = "",
    axes: Any = None,
    ylim: tuple[float, float] = WIN_RATE_YLIM,
) -> Any:
    """Win rate against coverage, with the volume it costs underneath.

    Two stacked panels sharing the x axis rather than one panel with two y
    scales: win rate is a proportion near 0.5 and bet count is in the hundreds,
    and overlaying them on twin axes would let the relative position of the two
    curves be set by the axis limits instead of by the data.

    x runs 100% -> 10% left to right, so reading rightwards is "be more
    selective" -- the direction the decision is made in.

    ``ylim`` focuses the win-rate axis on the range this problem actually
    occupies. It is widened automatically if a win rate falls outside it, so
    the focus never hides a point.
    """
    if axes is None:
        _, axes = plt.subplots(
            2, 1, figsize=(8.4, 6.4), sharex=True,
            gridspec_kw={"height_ratios": [2.4, 1.0]},
            constrained_layout=True,
        )
    top, bottom = axes

    for source, group in table.groupby("source", sort=False):
        group = group.sort_values("target_coverage", ascending=False)
        style = _style(str(source))
        x = group["target_coverage"]
        top.plot(x, group["win_rate"], label=str(source), zorder=3, **style)
        if group["independent"].all():
            top.fill_between(
                x, group["win_rate_ci_low"], group["win_rate_ci_high"],
                color=style["color"], alpha=0.13, linewidth=0, zorder=1,
            )
        bottom.plot(
            x, group["n_bets"], color=style["color"], marker=style["marker"],
            linestyle=style["linestyle"], markersize=5,
        )

    top.axhline(BREAK_EVEN, color=CRITICAL, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    top.annotate(
        f"break-even at -110 ({BREAK_EVEN:.2%})",
        xy=(0.985, BREAK_EVEN), xycoords=("axes fraction", "data"),
        xytext=(0, 4), textcoords="offset points", ha="right",
        fontsize=8, color=CRITICAL,
        # The reference rules run the full width, so their labels have to sit
        # over the series. A surface-coloured plate keeps both readable.
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0},
    )
    top.axhline(0.5, color=LINE_REF, linewidth=1.0, linestyle=":", zorder=2)
    top.annotate(
        "coin flip (50%)",
        xy=(0.985, 0.5), xycoords=("axes fraction", "data"),
        xytext=(0, -11), textcoords="offset points", ha="right",
        fontsize=8, color=MUTED,
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.0},
    )

    top.set_ylim(*win_rate_limits(table["win_rate"], ylim))
    top.set_ylabel("win rate")
    top.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    top.set_title(title, loc="left", color=INK)
    handles = [
        Line2D([], [], label=str(s), **_style(str(s)))
        for s in table["source"].unique()
    ]
    if not table["independent"].all():
        handles.append(
            Line2D([], [], color=MUTED, linestyle="", marker="",
                   label="no band: correlated rows per game")
        )
    top.legend(handles=handles, fontsize=8, loc="best")

    bottom.set_ylabel("bets placed")
    bottom.set_xlabel("share of games kept, ranked by margin (high margin first)")
    bottom.invert_xaxis()
    bottom.set_xticks(list(table["target_coverage"].unique()))
    bottom.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    rotate_xticks(bottom, 0)
    return axes


def plot_margin_buckets(
    buckets: pd.DataFrame, *, title: str = "", ax: Any = None
) -> Any:
    """Win rate per disjoint margin bucket, strongest margin on the left.

    Bars here, not dots: each bucket is an independent sample and the question
    is whether the left of the chart stands above the right, which is a
    magnitude comparison against a meaningful zero (break-even). The whisker is
    a Wilson interval, and its width is the actual finding on most runs.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8.4, 4.0), constrained_layout=True)

    positions = np.arange(len(buckets))
    colours = [
        GOOD if rate > BREAK_EVEN else CRITICAL if pd.notna(rate) else MUTED
        for rate in buckets["win_rate"]
    ]
    bars = ax.bar(positions, buckets["win_rate"], width=0.68, color=colours, alpha=0.85)
    if buckets["independent"].all():
        ax.errorbar(
            positions, buckets["win_rate"],
            yerr=np.vstack([
                buckets["win_rate"] - buckets["win_rate_ci_low"],
                buckets["win_rate_ci_high"] - buckets["win_rate"],
            ]),
            fmt="none", ecolor=INK_2, elinewidth=1.1, capsize=3,
        )
    # Above the whisker rather than the bar top: with a Wilson interval this
    # wide the label would otherwise sit inside the error bar and read as if
    # it were a bound on it.
    tops = buckets["win_rate_ci_high"].fillna(buckets["win_rate"])
    for bar, rate, n, top_y in zip(
        bars, buckets["win_rate"], buckets["n_decided"], tops, strict=True
    ):
        if pd.isna(rate):
            continue
        ax.annotate(
            f"{rate:.0%}\nn={n}",
            (bar.get_x() + bar.get_width() / 2, top_y),
            textcoords="offset points", xytext=(0, 5),
            ha="center", fontsize=7.5, color=INK_2,
        )

    ax.axhline(BREAK_EVEN, color=INK, linewidth=1.3, linestyle=(0, (4, 3)), zorder=4)
    ax.annotate(
        f"break-even {BREAK_EVEN:.1%}", xy=(0.99, BREAK_EVEN),
        xycoords=("axes fraction", "data"), xytext=(0, 4),
        textcoords="offset points", ha="right", fontsize=8, color=INK,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(buckets["pct_from_top"], fontsize=8)
    ax.set_xlabel("margin percentile band (0-10% = the strongest margins)")
    ax.set_ylabel("win rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, max(0.80, float(np.nanmax(tops)) + 0.13))
    ax.set_title(title, loc="left", color=INK)
    ax.spines["bottom"].set_color(AXIS)
    return ax


def plot_coverage_small_multiples(
    table: pd.DataFrame,
    *,
    source: str = "holdout",
    ncols: int = 2,
    ylim: tuple[float, float] = WIN_RATE_YLIM,
) -> Any:
    """One panel per run of the win-rate curve, faceted rather than overlaid.

    Runs of the same strategy share a colour in this project's palette, so
    overlaying six of them would make identity ambiguous exactly where the
    lines cross. Every panel keeps the same y limits, which is what makes the
    panels comparable at a glance.

    Those limits default to ``WIN_RATE_YLIM`` rather than to the data's own
    range, because the confidence bands here span 30 points and the win rates
    span about eight -- scaling to the bands flattens every curve into a
    horizontal line. The limits widen if a win rate would fall outside them.
    """
    view = table[table["source"] == source]
    labels = list(dict.fromkeys(view["label"]))
    if not labels:
        raise ValueError(f"No rows for source={source!r}.")
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.6 * ncols, 3.1 * nrows),
        sharex=True, sharey=True, constrained_layout=True, squeeze=False,
    )
    limits = win_rate_limits(view["win_rate"], ylim)

    for ax, label in zip(axes.flat, labels, strict=False):
        group = view[view["label"] == label].sort_values(
            "target_coverage", ascending=False
        )
        colour = STRATEGY_COLOR.get(str(group["prediction_strategy"].iloc[0]), MUTED)
        ax.plot(group["target_coverage"], group["win_rate"], color=colour, marker="o")
        independent = bool(group["independent"].all())
        if independent:
            ax.fill_between(
                group["target_coverage"],
                group["win_rate_ci_low"], group["win_rate_ci_high"],
                color=colour, alpha=0.13, linewidth=0,
            )
        ax.axhline(BREAK_EVEN, color=CRITICAL, linewidth=1.2, linestyle=(0, (4, 3)))
        # A panel with no band has to SAY it has no band. Silently omitting it
        # reads as a narrow interval, which is the opposite of what it means.
        title = str(label) if independent else f"{label}  (pooled rows: no interval)"
        ax.set_title(textwrap.fill(title, 46), loc="left", fontsize=8.5)
        ax.set_ylim(*limits)
        ax.invert_xaxis()
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    for ax in axes.flat[len(labels):]:
        ax.set_visible(False)

    # sharex hides tick labels on every row but the last, which strands a
    # column whose bottom cell is one of the hidden panels. Give the lowest
    # VISIBLE axis in each column its labels back.
    for column in range(ncols):
        visible = [
            axes[row][column] for row in range(nrows)
            if axes[row][column].get_visible()
        ]
        if visible:
            visible[-1].tick_params(labelbottom=True)
    fig.supxlabel("share of games kept, ranked by margin", fontsize=9, color=INK_2)
    fig.supylabel("win rate", fontsize=9, color=INK_2)
    return fig, axes
