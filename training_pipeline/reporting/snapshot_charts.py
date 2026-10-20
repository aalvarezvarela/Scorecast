"""Reading a pooled-snapshot run against a one-row-per-game reference.

The intermediate-line campaign trains one model on every pre-game snapshot at
once, so a run's headline betting numbers pool several rows per game and cannot
be read as bets. ``training_pipeline.snapshot_scoring`` writes the per-snapshot
breakdown beside each run; this module loads those tables and draws them
against the closing-line runs they are meant to be compared with.

Conventions follow ``theme``: one fixed colour per strategy, the bookmaker's
break-even as a grey reference rule rather than a fourth series, and error bars
ONLY where the quantity really is an interval. The Wilson bounds here qualify --
within a snapshot there is exactly one row per game, so the binomial assumption
holds. The pooled ``ALL`` row does not qualify and is deliberately left out of
every chart.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from training_pipeline.reporting import loaders, theme

#: Written by run_experiment() itself, for any dataset holding several rows
#: per game. Runs in artifacts/experiments/ from before that was automatic
#: carry the same filenames, written by the retired
#: scripts/run_intermediate_snapshot_experiment.py wrapper, so they still load.
SNAPSHOT_FILES = {"cv": "snapshot_cv_metrics.csv", "holdout": "snapshot_holdout_metrics.csv"}

#: The pooled row. Excluded from every chart: its interval and significance are
#: blank by construction because correlated per-game repeats break the binomial
#: assumption the interval rests on.
POOLED_LABEL = "ALL"


def load_snapshot_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """Every per-snapshot table for every run that has one, stacked.

    Runs without the files are skipped and NAMED in the returned frame's
    ``attrs["skipped"]`` -- a loader that drops runs silently is how a
    comparison quietly loses half its cells.
    """
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []

    for _, run in runs.iterrows():
        run_dir = Path(loaders.run_dir_of(run))
        found = False
        for source, filename in SNAPSHOT_FILES.items():
            path = run_dir / filename
            if not path.exists():
                continue
            found = True
            frame = pd.read_csv(path)
            frame["source"] = source
            frame["label"] = run["label"]
            frame["run_name"] = run["run_name"]
            frame["prediction_strategy"] = run["prediction_strategy"]
            frames.append(frame)
        if not found:
            skipped.append(str(run["run_name"]))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.attrs["skipped"] = skipped
    return out


def per_snapshot_only(metrics: pd.DataFrame) -> pd.DataFrame:
    """Drop the pooled row and coerce the snapshot to a sortable number."""
    if metrics.empty:
        return metrics
    out = metrics[metrics["snapshot"].astype(str) != POOLED_LABEL].copy()
    out["snapshot"] = pd.to_numeric(out["snapshot"], errors="coerce")
    return out.dropna(subset=["snapshot"]).sort_values(["label", "snapshot"])


def _reference_lines(
    ax: plt.Axes, references: dict[str, float] | None, *, as_pct: bool
) -> None:
    """Closing-line runs drawn as horizontal rules, not as competing series.

    They are single numbers with no snapshot axis, so plotting them as points
    would invent an x-position they do not have.
    """
    for name, value in (references or {}).items():
        if value is None or pd.isna(value):
            continue
        y = value * 100 if as_pct else value
        colour = theme.STRATEGY_COLOR.get(
            f"{name}_regressor", theme.STRATEGY_COLOR.get(name, theme.LINE_REF)
        )
        ax.axhline(y, color=colour, linestyle="--", linewidth=1.5, alpha=0.75, zorder=1)
        ax.annotate(
            f"closing {name}: {y:.1f}" + ("%" if as_pct else ""),
            xy=(0.995, y), xycoords=("axes fraction", "data"),
            ha="right", va="bottom", fontsize=8, color=colour,
        )


def plot_win_rate_by_snapshot(
    metrics: pd.DataFrame,
    *,
    source: str = "holdout",
    references: dict[str, float] | None = None,
    ax: plt.Axes | None = None,
) -> pd.DataFrame:
    """Win rate against minutes-before-tip, with Wilson intervals.

    The x-axis runs from the longest lead time down to tip-off, so reading left
    to right follows the clock toward the game.
    """
    view = per_snapshot_only(metrics)
    view = view[view["source"] == source]
    if view.empty:
        raise ValueError(f"No per-snapshot rows for source={source!r}.")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.2))

    for label, group in view.groupby("label", sort=False):
        colour = theme.STRATEGY_COLOR.get(
            group["prediction_strategy"].iloc[0], theme.LINE_REF
        )
        rate = group["win_rate"] * 100
        low = (group["win_rate"] - group["win_rate_ci_low"]) * 100
        high = (group["win_rate_ci_high"] - group["win_rate"]) * 100
        ax.errorbar(
            group["snapshot"], rate, yerr=[low, high],
            marker="o", capsize=3, color=colour, label=label,
            linewidth=2, markersize=7, elinewidth=1, alpha=0.95, zorder=3,
        )

    ax.axhline(
        theme.BREAK_EVEN * 100, color=theme.LINE_REF, linewidth=1.5, zorder=1
    )
    ax.annotate(
        f"break-even {theme.BREAK_EVEN * 100:.2f}%",
        xy=(0.005, theme.BREAK_EVEN * 100), xycoords=("axes fraction", "data"),
        va="bottom", fontsize=8, color=theme.MUTED,
    )
    _reference_lines(ax, references, as_pct=True)

    ax.invert_xaxis()
    _hours_axis(ax, sorted(view["snapshot"].unique()))
    ax.set_ylabel("win rate %")
    ax.set_title(f"Win rate by virtual bet time — {source}")
    ax.legend(loc="best")
    return view


def plot_roi_by_snapshot(
    metrics: pd.DataFrame,
    *,
    source: str = "holdout",
    references: dict[str, float] | None = None,
    ax: plt.Axes | None = None,
) -> pd.DataFrame:
    """ROI against minutes-before-tip.

    No error bars: ROI here has no interval attached. The seed spread reported
    elsewhere is an observed min-max across seeds, not a confidence interval,
    and drawing it as a whisker would claim more than it says.
    """
    view = per_snapshot_only(metrics)
    view = view[view["source"] == source]
    if view.empty:
        raise ValueError(f"No per-snapshot rows for source={source!r}.")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.2))

    for label, group in view.groupby("label", sort=False):
        colour = theme.STRATEGY_COLOR.get(
            group["prediction_strategy"].iloc[0], theme.LINE_REF
        )
        ax.plot(
            group["snapshot"], group["roi"] * 100,
            marker="o", color=colour, label=label, linewidth=2, markersize=7, zorder=3,
        )

    ax.axhline(0.0, color=theme.LINE_REF, linewidth=1.5, zorder=1)
    _reference_lines(ax, references, as_pct=True)

    ax.invert_xaxis()
    _hours_axis(ax, sorted(view["snapshot"].unique()))
    ax.set_ylabel("ROI %")
    ax.set_title(f"ROI by virtual bet time — {source}")
    ax.legend(loc="best")
    return view


def cohort_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Holdout window and volume per run — the comparison-legitimacy check.

    Two runs scored on different date ranges are not measuring the same thing,
    however similar their columns look. This is the first table to read.
    """
    columns = [
        "run_name", "prediction_strategy", "train_games",
        "holdout_start", "holdout_end", "holdout_n_games",
        "dataset_checksum", "n_seeds",
    ]
    present = [c for c in columns if c in runs.columns]
    out = runs[present].copy()
    for column in ("holdout_start", "holdout_end"):
        if column in out.columns:
            out[column] = pd.to_datetime(out[column]).dt.date
    return out.sort_values("run_name").reset_index(drop=True)


def headline_table(runs: pd.DataFrame) -> pd.DataFrame:
    """CV and holdout betting metrics, with the seed spread beside them.

    ``seed_roi_range`` is printed next to every ROI on purpose: a difference
    smaller than one run's own spread across seeds is not a result, and the
    two numbers are unreadable apart.
    """
    columns = {
        "run_name": "run", "cv_win_rate": "cv_win%", "cv_roi": "cv_roi%",
        "cv_n_bets": "cv_bets", "win_rate": "ho_win%", "roi": "ho_roi%",
        "n_bets": "ho_bets", "seed_roi_range": "seed_roi_range",
        "cv_minus_holdout_roi": "cv−ho_roi",
    }
    present = {k: v for k, v in columns.items() if k in runs.columns}
    out = runs[list(present)].rename(columns=present)
    for column in out.columns:
        if column.endswith(("%", "roi_range", "roi")):
            out[column] = pd.to_numeric(out[column], errors="coerce") * 100
    return out.round(2).reset_index(drop=True)


def describe_gap(metrics: pd.DataFrame, *, source: str = "holdout") -> dict[str, Any]:
    """Best and worst snapshot per run, and whether any interval clears break-even."""
    view = per_snapshot_only(metrics)
    view = view[view["source"] == source]
    summary: dict[str, Any] = {}
    for label, group in view.groupby("label", sort=False):
        best = group.loc[group["win_rate"].idxmax()]
        summary[label] = {
            "best_snapshot": int(best["snapshot"]),
            "best_win_rate": round(float(best["win_rate"]) * 100, 2),
            "best_ci_low": round(float(best["win_rate_ci_low"]) * 100, 2),
            "clears_break_even": bool(
                (group["win_rate_ci_low"] > theme.BREAK_EVEN).any()
            ),
            "n_snapshots_profitable": int((group["roi"] > 0).sum()),
        }
    return summary


def _hours_axis(ax: plt.Axes, snapshots: list[float]) -> None:
    """Label the snapshot axis in hours, which is how a bet time is thought about.

    Kept as a log scale because the grid is geometric (30 min to 720 min); a
    linear axis would crush the four short horizons into the left margin.
    """
    ax.set_xscale("log")
    ax.set_xticks(snapshots)
    ax.set_xticklabels(
        [f"{t/60:g}h" if t >= 60 else f"{int(t)}m" for t in snapshots]
    )
    ax.set_xlabel("bet placed this long before tip-off  (earlier  →  later)")


def pooled_vs_control(
    runs: pd.DataFrame, metrics: pd.DataFrame, *, snapshot: int = 720
) -> pd.DataFrame:
    """Is pooling the snapshots earning its complexity?

    Compares the pooled model SCORED at ``snapshot`` against the control model
    TRAINED only on that snapshot. This is the cleanest comparison available in
    the campaign: identical holdout window, identical games, identical
    threshold, one row per game on both sides. Unlike the closing-line
    comparison it is properly matched, so a gap here means something.
    """
    per_snapshot = per_snapshot_only(metrics)
    rows: list[dict[str, Any]] = []

    controls = runs[runs["run_name"].str.contains("control", na=False)]
    for _, control in controls.iterrows():
        strategy = control["prediction_strategy"]
        pooled_rows = per_snapshot[
            (per_snapshot["prediction_strategy"] == strategy)
            & (per_snapshot["snapshot"] == snapshot)
        ]
        for source in ("cv", "holdout"):
            pooled = pooled_rows[pooled_rows["source"] == source]
            if pooled.empty:
                continue
            if source == "holdout":
                c_win, c_roi, c_bets = control["win_rate"], control["roi"], control["n_bets"]
            else:
                c_win, c_roi, c_bets = (
                    control["cv_win_rate"], control["cv_roi"], control["cv_n_bets"]
                )
            rows.append({
                "strategy": strategy,
                "source": source,
                "pooled_win_rate": float(pooled["win_rate"].iloc[0]),
                "control_win_rate": float(c_win),
                "pooled_roi": float(pooled["roi"].iloc[0]),
                "control_roi": float(c_roi),
                "roi_gap_pooled_minus_control": float(pooled["roi"].iloc[0]) - float(c_roi),
                "pooled_n_bets": int(pooled["n_bets"].iloc[0]),
                "control_n_bets": int(c_bets),
                "control_seed_roi_range": float(control["seed_roi_range"]),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # A gap smaller than the control's own spread across seeds is not an effect.
    out["gap_exceeds_seed_noise"] = (
        out["roi_gap_pooled_minus_control"].abs() > out["control_seed_roi_range"]
    )
    return out


def plot_pooled_vs_control(comparison: pd.DataFrame, *, ax: plt.Axes | None = None) -> None:
    """Dumbbells: control ROI to pooled ROI, one row per strategy x source.

    Dumbbells rather than paired bars because both ends are meaningful values on
    the same scale and the quantity of interest is the distance between them.
    """
    if comparison.empty:
        raise ValueError("Nothing to plot: no pooled/control pairs were found.")
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.9 * len(comparison) + 1.8))

    labels = [
        f"{row.strategy.replace('_regressor','')} · {row.source}"
        for row in comparison.itertuples()
    ]
    positions = range(len(comparison))
    for y, row in zip(positions, comparison.itertuples(), strict=True):
        colour = theme.STRATEGY_COLOR.get(row.strategy, theme.LINE_REF)
        control, pooled = row.control_roi * 100, row.pooled_roi * 100
        ax.plot([control, pooled], [y, y], color=colour, linewidth=2, alpha=0.45, zorder=2)
        ax.scatter(control, y, s=95, facecolor=theme.SURFACE, edgecolor=colour,
                   linewidth=2, zorder=3, label="control (720-only)" if y == 0 else None)
        ax.scatter(pooled, y, s=95, color=colour, zorder=3,
                   label="pooled (all snapshots)" if y == 0 else None)

    ax.axvline(0.0, color=theme.LINE_REF, linewidth=1.5, zorder=1)
    ax.set_yticks(list(positions))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("ROI %  (filled = pooled, hollow = 720-only control)")
    ax.set_title("Does pooling the snapshots earn its complexity?  scored at 12h")
    ax.legend(loc="best")
    ax.grid(axis="y", visible=False)


# ---------------------------------------------------------------------------
# The simple view: win rate only, everything on one page.
# ---------------------------------------------------------------------------

def _variant(run_name: str) -> str:
    """pooled (trained on every snapshot) vs control (one snapshot only)."""
    return "control" if "control" in str(run_name) else "pooled"


def _experiment_label(run_name: str, strategy: str) -> str:
    """Compact label that still identifies every intermediate-line run."""
    short = strategy.replace("_regressor", "")
    variant = _variant(run_name)
    parts = [short]

    if variant == "pooled":
        parts.append("pooled")
    else:
        found = re.search(r"t(\d+)", str(run_name))
        if found:
            minutes = int(found.group(1))
            horizon = f"{minutes / 60:g}h" if minutes >= 60 else f"{minutes}m"
            parts.append(f"{horizon} control")
        else:
            parts.append("control")

    if "no_time_decay" in str(run_name):
        parts.append("no decay")
    return " · ".join(parts)


def win_rate_overall(runs: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """One CV and one holdout win rate per experiment, across all bet times.

    For a pooled run this is the ``ALL`` row -- every snapshot together. That
    number is a fine point estimate; it is only its *interval* that the repeated
    games invalidate, which is why no interval is carried here.
    """
    rows: list[dict[str, Any]] = []
    for _, run in runs.iterrows():
        variant = _variant(run["run_name"])
        if variant == "pooled":
            pooled = metrics[
                (metrics["run_name"] == run["run_name"])
                & (metrics["snapshot"].astype(str) == POOLED_LABEL)
            ]
            values = {
                source: float(pooled.loc[pooled["source"] == source, "win_rate"].iloc[0])
                for source in ("cv", "holdout")
                if not pooled[pooled["source"] == source].empty
            }
        else:
            values = {"cv": float(run["cv_win_rate"]), "holdout": float(run["win_rate"])}
        for source, value in values.items():
            rows.append({
                "experiment": run["run_name"],
                "label": _experiment_label(
                    str(run["run_name"]), str(run["prediction_strategy"])
                ),
                "strategy": run["prediction_strategy"],
                "variant": variant,
                "source": source,
                "win_rate": value,
            })
    return pd.DataFrame(rows)


def plot_win_rate_overall(overall: pd.DataFrame, *, ax: plt.Axes | None = None) -> None:
    """Dumbbell per experiment: CV win rate to holdout win rate.

    Dumbbells rather than bars because these values live in a narrow band around
    50% on a 0-100 scale -- a bar from zero would show nothing, and a truncated
    bar would exaggerate every difference.
    """
    wide = overall.pivot_table(
        index=["experiment", "label", "strategy", "variant"],
        columns="source",
        values="win_rate",
    ).reset_index()
    wide = wide.sort_values(["strategy", "variant", "experiment"])

    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 0.85 * len(wide) + 1.8))

    for y, row in enumerate(wide.itertuples()):
        colour = theme.STRATEGY_COLOR.get(row.strategy, theme.LINE_REF)
        cv, holdout = row.cv * 100, row.holdout * 100
        ax.plot([cv, holdout], [y, y], color=colour, linewidth=2, alpha=0.45, zorder=2)
        ax.scatter(cv, y, s=95, facecolor=theme.SURFACE, edgecolor=colour, linewidth=2,
                   zorder=3, label="CV" if y == 0 else None)
        ax.scatter(holdout, y, s=95, color=colour, zorder=3,
                   label="holdout" if y == 0 else None)

    ax.axvline(theme.BREAK_EVEN * 100, color=theme.LINE_REF, linewidth=1.5, zorder=1)
    ax.annotate(f"break-even {theme.BREAK_EVEN * 100:.2f}%",
                xy=(theme.BREAK_EVEN * 100, 0.99), xycoords=("data", "axes fraction"),
                ha="center", va="top", fontsize=8, color=theme.MUTED)
    ax.set_yticks(range(len(wide)))
    ax.set_yticklabels(wide["label"])
    ax.invert_yaxis()
    ax.set_xlabel("win rate %   (hollow = CV, filled = holdout)")
    ax.set_title("Win rate per experiment — all bet times together")
    ax.legend(loc="best")
    ax.grid(axis="y", visible=False)


def control_snapshot(run: Any) -> float | None:
    """Which single snapshot a control run was trained on.

    Read from the run name ("..._t720_...") and otherwise from the sliced CSV
    it points at, both of which carry the horizon. Returns None when the run is
    not a single-snapshot control.
    """
    if _variant(run["run_name"]) != "control":
        return None
    for text in (str(run["run_name"]), str(run.get("data_version", ""))):
        found = re.search(r"t(\d+)", text)
        if found:
            return float(found.group(1))
    return None


def with_control_points(metrics: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    """Per-snapshot rows, plus one row per control at its own single horizon.

    A control has no per-snapshot file because it needs none -- it is already one
    row per game -- but on a by-bet-time chart it is a real measurement at one
    time and leaving it out silently drops half the experiments.
    """
    frames = [per_snapshot_only(metrics)]
    for _, run in runs.iterrows():
        snapshot = control_snapshot(run)
        if snapshot is None:
            continue
        for source, win, bets in (
            ("cv", run["cv_win_rate"], run["cv_n_bets"]),
            ("holdout", run["win_rate"], run["n_bets"]),
        ):
            frames.append(pd.DataFrame([{
                "snapshot": snapshot, "source": source,
                "win_rate": float(win), "n_bets": int(bets),
                "label": run["label"], "run_name": run["run_name"],
                "prediction_strategy": run["prediction_strategy"],
            }]))
    return pd.concat(frames, ignore_index=True)


def plot_win_rate_split(metrics: pd.DataFrame, runs: pd.DataFrame | None = None) -> None:
    """Win rate against bet time, CV and holdout as small multiples.

    Hue carries the target, marker fill carries pooled-vs-control, and marker
    shape/line style distinguish the no-decay variants. Every run is grouped
    separately so repeated target/variant combinations cannot be joined into a
    spurious line.
    """
    view = per_snapshot_only(metrics) if runs is None else with_control_points(metrics, runs)
    if view.empty:
        raise ValueError("No per-snapshot rows to plot.")
    view = view.assign(variant=view["run_name"].map(_variant))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, source in zip(axes, ("cv", "holdout"), strict=True):
        panel = view[view["source"] == source]
        for (run_name, strategy, variant), group in panel.groupby(
            ["run_name", "prediction_strategy", "variant"], sort=False
        ):
            colour = theme.STRATEGY_COLOR.get(strategy, theme.LINE_REF)
            no_decay = "no_time_decay" in str(run_name)
            ax.plot(
                group["snapshot"], group["win_rate"] * 100,
                marker="s" if no_decay else "o",
                color=colour,
                linewidth=2,
                markersize=7,
                markerfacecolor=colour if variant == "pooled" else theme.SURFACE,
                markeredgecolor=colour, markeredgewidth=2,
                linestyle=("--" if no_decay else "-") if variant == "pooled" else "none",
                label=_experiment_label(str(run_name), str(strategy)), zorder=3,
            )
        ax.axhline(theme.BREAK_EVEN * 100, color=theme.LINE_REF, linewidth=1.5, zorder=1)
        ax.invert_xaxis()
        _hours_axis(ax, sorted(view["snapshot"].unique()))
        ax.set_title(source)
    axes[0].set_ylabel("win rate %")
    axes[0].annotate(f"break-even {theme.BREAK_EVEN * 100:.2f}%",
                     xy=(0.02, theme.BREAK_EVEN * 100), xycoords=("axes fraction", "data"),
                     va="bottom", fontsize=8, color=theme.MUTED)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.suptitle("Win rate by bet time", y=1.01, fontsize=11, fontweight="semibold")
    fig.tight_layout()
