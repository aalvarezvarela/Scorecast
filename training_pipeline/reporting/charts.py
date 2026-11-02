"""The figures the survey notebook draws.

Each function takes the prepared runs frame, loads whatever artifacts it needs,
draws one figure, and returns the data it computed so the notebook can narrate
from the same numbers it plotted rather than recomputing them.

Form choices worth knowing about, because they are not arbitrary:

- Values that differ by ~0.1 on a base of ~14 (model MAE versus the line) are
  drawn as a dumbbell, never as bars. A bar encodes magnitude by its length, so
  showing that difference as bars would need a truncated axis, and a truncated
  bar misleads. A dot encodes position, where a non-zero axis is honest.
- Per-run curves are faceted by strategy rather than overlaid. Two runs of the
  same strategy share a colour, so a single panel would make identity ambiguous
  and turn six crossing lines into spaghetti.
- The seed-range rule is a plain vertical line, not a capped errorbar: it is an
  observed min-max range, not a symmetric confidence interval, and should not
  be dressed as one.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from training_pipeline.reporting.theme import (
    AXIS,
    BREAK_EVEN,
    CRITICAL,
    GRID,
    INK,
    INK_2,
    MUTED,
    STRATEGY_COLOR,
    SURFACE,
    colour_of,
    strategy_legend,
)

#: Edge ranges in points. Fine at the bottom, where the question "is a 1-point
#: edge better than a 0.5-point one?" actually lives, and coarser in the tail
#: where large edges are rare and half-point bins would hold single digits.
DEFAULT_EDGE_BINS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, np.inf)

#: Below this a bin's win rate is too noisy to read as anything.
MIN_BIN_SIZE = 30


# ------------------------------------------------------- CV vs final test
def plot_cv_vs_test_win_rate(runs: pd.DataFrame) -> pd.DataFrame:
    """Win rate in cross-validation against win rate on the held-out period.

    Both are measured at each run's own bet threshold, so this is the
    like-for-like version of "did it hold up?". CV chose the hyperparameters
    and so flatters the model; the holdout did not but is far smaller. The
    dumbbell makes the direction and size of the move the thing you read.
    """
    view = runs.dropna(subset=["cv_win_rate", "win_rate"]).copy()
    if view.empty:
        return view

    view["win_rate_drop"] = view["cv_win_rate"] - view["win_rate"]
    view = view.sort_values("win_rate", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 0.55 * len(view) + 3))
    y = np.arange(len(view))

    for position, (_, row) in enumerate(view.iterrows()):
        colour = colour_of(row)
        ax.plot([row["cv_win_rate"], row["win_rate"]], [position, position],
                color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
        # Hollow = cross-validation, filled = the holdout it has to survive.
        ax.plot([row["cv_win_rate"]], [position], marker="o", markersize=10,
                markerfacecolor=SURFACE, markeredgecolor=colour,
                markeredgewidth=2.2, zorder=3)
        ax.plot([row["win_rate"]], [position], marker="o", markersize=10,
                color=colour, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
        ax.annotate(
            f"{row['cv_win_rate']:.1%} → {row['win_rate']:.1%}"
            f"   ({int(row['cv_n_bets'])} → {int(row['n_bets'])} bets)",
            (max(row["cv_win_rate"], row["win_rate"]), position),
            textcoords="offset points", xytext=(10, -3), fontsize=8, color=INK_2,
        )

    ax.axvline(BREAK_EVEN, color=CRITICAL, linestyle="--", linewidth=1.5, zorder=2)
    # Top of the axis, not the bottom: the bottom row's own annotation starts
    # just right of the break-even line and the two collide there.
    ax.annotate("break-even 52.38%", xy=(BREAK_EVEN, 1),
                xycoords=("data", "axes fraction"), textcoords="offset points",
                xytext=(6, -12), fontsize=8, color=CRITICAL)
    ax.set_yticks(y, view["label"])
    ax.set_xlabel("Win rate")
    ax.set_title("Cross-validation → held-out test, at each run's own bet threshold")
    ax.set_xlim(0.35, 0.80)
    ax.invert_yaxis()
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markersize=10, markerfacecolor=SURFACE,
               markeredgecolor=MUTED, markeredgewidth=2.2, label="cross-validation"),
        Line2D([], [], marker="o", linestyle="", markersize=10, color=MUTED,
               label="held-out test"),
    ], fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.show()

    return view[[
        "label", "prediction_strategy", "holdout_evaluation",
        "cv_win_rate", "cv_n_bets", "win_rate", "n_bets", "win_rate_drop",
        "cv_roi", "roi",
    ]]


# ------------------------------------------------------- edge ranges
#: Row order for the combined per-run table.
COMBINED_SOURCE_ORDER = ("cross-validation", "holdout", "% of games")


# --------------------------------------------------------------- portfolio
# --------------------------------------------------------------- headline
def plot_roi_with_seed_noise(runs: pd.DataFrame) -> None:
    """Holdout ROI, with the seed range for any run that produced one.

    Single-seed is the default, so usually no run has a range and this degrades
    to plain bars. The title and legend follow suit rather than advertising an
    error bar the chart is not drawing -- a legend entry for a mark that never
    appears reads as "the bars ARE the interval", which is the opposite of true.
    """
    has_seed_range = (
        "seed_roi_range" in runs.columns and runs["seed_roi_range"].notna().any()
    )
    fig, ax = plt.subplots(figsize=(12.5, 5))
    x = np.arange(len(runs))
    ax.bar(x, runs["roi"], color=[colour_of(r) for _, r in runs.iterrows()],
           width=0.6, zorder=2)

    above, below = [], []
    for position, (_, row) in enumerate(runs.iterrows()):
        top = bottom = row["roi"]
        if pd.notna(row.get("seed_roi_range")):
            half = row["seed_roi_range"] / 2
            top, bottom = row["roi"] + half, row["roi"] - half
            ax.plot([position, position], [bottom, top], color=INK, linewidth=2,
                    zorder=4, solid_capstyle="butt")
            ax.plot([position], [row["roi"]], marker="_", markersize=13,
                    color=INK, zorder=5)
        above.append(top)
        below.append(bottom)

    # Label clear of the seed rule -- at the bar top the rule runs straight
    # through the text and the value becomes unreadable.
    for position, (_, row) in enumerate(runs.iterrows()):
        if pd.isna(row["roi"]):
            continue
        positive = row["roi"] >= 0
        anchor = above[position] if positive else below[position]
        ax.annotate(f"{row['roi']:.1%}", (position, anchor),
                    textcoords="offset points", xytext=(0, 6 if positive else -13),
                    ha="center", fontsize=8, color=INK_2, zorder=6)

    margin = (max(above) - min(below)) * 0.12
    ax.set_ylim(min(min(below), 0) - margin, max(above) + margin)
    ax.axhline(0, color=AXIS, linewidth=1.2, zorder=1)
    ax.set_title(
        "Holdout ROI, with the seed-noise range each configuration produces"
        if has_seed_range
        else "Holdout ROI (single seed -- no error bar)"
    )
    ax.set_ylabel("ROI")
    ax.set_xticks(x, runs["label"], rotation=25, ha="right")
    strategy_legend(
        ax,
        runs["prediction_strategy"].unique(),
        extra=(
            [Line2D([], [], color=INK, linewidth=2, label="seed range")]
            if has_seed_range
            else None
        ),
    )
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------- cv vs holdout
# --------------------------------------------------------------- strategy
# --------------------------------------------------------------- regressors
# --------------------------------------------------------------- classifiers
# --------------------------------------------------------------- confidence
# --------------------------------------------------------------- folds
# --------------------------------------------------------------- walk-forward
# --------------------------------------------------------------- lines
# --------------------------------------------------- one factor at a time
def plot_factor_effect(
    contrast_table: pd.DataFrame,
    factor: str,
    *,
    max_cols: int = 3,
) -> None:
    """Small multiples of a controlled comparison: one panel per matched set.

    Within a panel every run is identical except for ``factor``, so the shape
    of the line across the x-axis *is* the effect of that factor -- there is
    nothing else varying that it could be attributed to.

    Both measurements are drawn together and never averaged. Cross-validation
    (hollow) chose the hyperparameters and so reads high; the holdout (filled)
    did not. A factor that moves them in opposite directions has not been shown
    to do anything, and seeing the two lines cross is the fastest way to notice
    that.

    Bet counts are annotated because they are the other half of every ROI here:
    the classifier reaches its win rate on ~150-250 bets where a regressor
    takes ~400, and two win rates on such different volumes are not equally
    believable.
    """
    if contrast_table.empty:
        return

    groups = list(contrast_table.groupby("contrast", sort=False))
    n_cols = min(max_cols, len(groups))
    n_rows = int(np.ceil(len(groups) / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.6 * n_cols, 3.6 * n_rows), squeeze=False
    )
    flat = axes.ravel()

    for ax, (contrast, group) in zip(flat, groups, strict=False):
        group = group.sort_values(factor)
        levels = [str(value) for value in group[factor]]
        x = np.arange(len(group))
        colours = [
            STRATEGY_COLOR.get(strategy, MUTED)
            for strategy in group.get("prediction_strategy", pd.Series(dtype=object))
        ] or [MUTED] * len(group)

        ax.plot(x, group["cv_win_rate"], color=GRID, linewidth=2, zorder=1)
        ax.plot(x, group["win_rate"], color=AXIS, linewidth=2, zorder=1)
        for position, (_, row) in enumerate(group.iterrows()):
            colour = colours[position]
            ax.plot(position, row["cv_win_rate"], marker="o", markersize=9,
                    markerfacecolor=SURFACE, markeredgecolor=colour,
                    markeredgewidth=2.2, zorder=3)
            ax.plot(position, row["win_rate"], marker="o", markersize=9,
                    color=colour, markeredgecolor=SURFACE, markeredgewidth=1.4,
                    zorder=4)
            if pd.notna(row.get("n_bets")):
                ax.annotate(f"{row['n_bets']:.0f} bets",
                            (position, row["win_rate"]), textcoords="offset points",
                            xytext=(0, -15), ha="center", fontsize=7, color=MUTED)

        ax.axhline(BREAK_EVEN, color=CRITICAL, linestyle="--", linewidth=1.3, zorder=2)
        ax.set_xticks(x, levels, fontsize=8)
        ax.set_xlim(-0.45, len(group) - 0.55)
        ax.set_title(contrast, fontsize=9)
        ax.set_xlabel(factor)
        ax.set_ylabel("Win rate")

    for ax in flat[len(groups):]:
        ax.axis("off")

    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markersize=9, markerfacecolor=SURFACE,
               markeredgecolor=MUTED, markeredgewidth=2.2, label="cross-validation"),
        Line2D([], [], marker="o", linestyle="", markersize=9, color=MUTED,
               label="held-out test"),
        Line2D([], [], color=CRITICAL, linestyle="--", label="break-even 52.38%"),
    ], fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Effect of {factor}, everything else held constant", fontsize=11)
    plt.tight_layout(rect=(0, 0.04, 1, 0.97))
    plt.show()


def plot_headline_win_rate_and_roi(
    operating: pd.DataFrame,
    *,
    operating_coverage: float,
    label_col: str = "label",
) -> tuple[plt.Figure, Any]:
    """CV-versus-holdout win rate as paired dumbbells, with CV ROI beside it.

    ``operating`` is one row per (run, source) scored at a single frozen
    threshold -- the survey notebook's operating-point table.

    Two panels rather than one with two x scales: a win rate is a proportion
    near 0.5 and an ROI is a signed fraction near 0, and twin axes would let
    the relative position of the two encodings be set by the axis limits
    instead of by the data.

    The dumbbell is not decoration. Where each run sits matters less than how
    far it moved between the folds that chose its hyperparameters and the
    period that never touched them, and a connector encodes that movement as
    a length the eye reads directly.

    This figure keeps its own report palette rather than the strategy colours:
    here the series is the SOURCE (CV or holdout), consistently across every
    run, and borrowing the strategy hues would imply the opposite mapping.

    ``label_col`` picks which name goes on the y axis. Prefer the short form:
    a tick label long enough to wrap steals width from the encoding it is
    labelling, and the identity it adds is already in the section above.
    """
    if label_col not in operating.columns:
        label_col = "label"
    cv = (
        operating[operating['source'] == 'cross-validation']
        .sort_values(['win_rate', 'label'])
        .reset_index(drop=True)
    )
    holdout = (
        operating[operating['source'] == 'holdout']
        .drop_duplicates('label', keep='last')
        .set_index('label')
    )
    positions = np.arange(len(cv))
    # Restrained report palette: source is encoded consistently across runs,
    # while marker shape remains a redundant CV/holdout cue.
    CV_COLOR = '#244a6b'
    HOLDOUT_COLOR = '#718096'
    CONNECTOR_COLOR = '#aeb7c2'
    ROI_POSITIVE = '#587568'
    ROI_NEGATIVE = '#8a6060'
    colours = [CV_COLOR] * len(cv)
    height = max(6.2, 0.95 * len(cv) + 3.0)
    fig, axes = plt.subplots(
        1, 2, figsize=(20, height), sharey=True,
        gridspec_kw={'width_ratios': [1.45, 1.0]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor('#f4f5f7')

    for ax in axes:
        ax.set_facecolor('#ffffff')
        ax.grid(axis='x', color='#dfe3eb', linewidth=0.9, alpha=0.85)
        ax.grid(axis='y', visible=False)
        ax.tick_params(labelsize=10)
        for position in positions:
            if position % 2 == 0:
                ax.axhspan(position - 0.43, position + 0.43, color='#f3f5f9', zorder=0)

    # A very light neutral band separates chance from flat -110 break-even.
    axes[0].axvspan(0.50, BREAK_EVEN, color='#7a8490', alpha=0.07, zorder=0)
    win_bounds = [0.50, BREAK_EVEN]

    # Paired dumbbells: the connector makes the CV-to-holdout movement visible.
    for position, (_, row), colour in zip(positions, cv.iterrows(), colours, strict=True):
        has_holdout = row['label'] in holdout.index
        cv_position = position - 0.14 if has_holdout else position
        cv_low = row['win_rate'] - row['win_rate_ci_low']
        cv_high = row['win_rate_ci_high'] - row['win_rate']
        win_bounds.extend([row['win_rate_ci_low'], row['win_rate_ci_high']])

        if has_holdout and pd.notna(holdout.loc[row['label'], 'win_rate']):
            test = holdout.loc[row['label']]
            test_position = position + 0.14
            axes[0].plot(
                [row['win_rate'], test['win_rate']], [cv_position, test_position],
                color=CONNECTOR_COLOR, linewidth=2.4, alpha=0.85, zorder=1,
            )
            test_low = test['win_rate'] - test['win_rate_ci_low']
            test_high = test['win_rate_ci_high'] - test['win_rate']
            win_bounds.extend([test['win_rate_ci_low'], test['win_rate_ci_high']])
            axes[0].errorbar(
                test['win_rate'], test_position,
                xerr=np.array([[test_low], [test_high]]),
                fmt='D', color=HOLDOUT_COLOR, markerfacecolor='white',
                markeredgewidth=2.2, elinewidth=2.4, capsize=5,
                capthick=2.0, markersize=9, zorder=4,
            )
            axes[0].annotate(
                f"{test['win_rate']:.1%}", (test['win_rate'], test_position),
                xytext=(9, 9), textcoords='offset points', fontsize=9,
                fontweight='semibold', color=HOLDOUT_COLOR,
            )

        axes[0].errorbar(
            row['win_rate'], cv_position, xerr=np.array([[cv_low], [cv_high]]),
            fmt='o', color=colour, markeredgecolor='white', markeredgewidth=1.3,
            elinewidth=2.4, capsize=5, capthick=2.0, markersize=11, zorder=5,
        )
        axes[0].annotate(
            f"{row['win_rate']:.1%}", (row['win_rate'], cv_position),
            xytext=(9, -16 if has_holdout else 9), textcoords='offset points',
            fontsize=9, fontweight='semibold', color=colour,
        )

    axes[0].plot(
        [], [], 'o', color=CV_COLOR, markeredgecolor='white', markersize=10,
        linestyle='', label='Cross-validation',
    )
    axes[0].plot(
        [], [], marker='D', markerfacecolor='white', markeredgecolor=HOLDOUT_COLOR,
        markeredgewidth=2, markersize=8, linestyle='', label='Holdout',
    )
    axes[0].axvline(
        0.50, color='#6c757d', linewidth=1.6, linestyle=':', label='Chance · 50%',
    )
    axes[0].axvline(
        BREAK_EVEN, color='#e45756', linewidth=1.8, linestyle='--',
        label='Flat -110 break-even · 52.38%',
    )
    win_left = max(0.0, float(np.nanmin(win_bounds)) - 0.025)
    win_right = min(1.0, float(np.nanmax(win_bounds)) + 0.045)
    axes[0].set_xlim(win_left, win_right)
    axes[0].set_title('Correct-side rate', loc='left', fontsize=15, fontweight='bold', pad=14)
    axes[0].set_xlabel('Win rate', fontsize=11, fontweight='semibold')
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[0].legend(
        loc='lower right', ncol=2, fontsize=9, frameon=True,
        facecolor='white', edgecolor='#dfe3eb', framealpha=0.96,
    )

    # ROI remains secondary. Muted sign colours communicate direction without
    # turning the diagnostic panel into the visual focus.
    roi_values = pd.to_numeric(cv['roi'], errors='coerce')
    finite_roi = roi_values[np.isfinite(roi_values)]
    roi_limit = max(0.03, float(finite_roi.abs().max()) * 1.45) if not finite_roi.empty else 0.10
    axes[1].set_xlim(-roi_limit, roi_limit)
    roi_colours = [
        ROI_POSITIVE if value > 0 else ROI_NEGATIVE if value < 0 else '#7a8490'
        for value in roi_values
    ]
    bars = axes[1].barh(
        positions, roi_values, color=roi_colours, height=0.52, alpha=0.92,
        edgecolor='white', linewidth=1.2, zorder=3,
    )
    axes[1].axvline(0.0, color='#596275', linewidth=1.7, zorder=2)
    axes[1].set_title('Flat -110 ROI · CV', loc='left', fontsize=15, fontweight='bold', pad=14)
    axes[1].set_xlabel('Profit per unit staked', fontsize=11, fontweight='semibold')
    axes[1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].text(0.02, 1.01, 'LOSS', transform=axes[1].transAxes, color='#6b7280', fontsize=9, fontweight='bold')
    axes[1].text(
        0.98, 1.01, 'PROFIT', transform=axes[1].transAxes, color='#6b7280',
        fontsize=9, fontweight='bold', ha='right',
    )
    for bar, value in zip(bars, roi_values, strict=True):
        if pd.isna(value):
            continue
        roi_label = '0.0%' if np.isclose(value, 0.0, atol=0.0005) else f'{value:+.1%}'
        offset = 8 if value >= 0 else -8
        axes[1].annotate(
            roi_label, (value, bar.get_y() + bar.get_height() / 2),
            xytext=(offset, 0), textcoords='offset points', va='center',
            ha='left' if value >= 0 else 'right', fontsize=10,
            fontweight='bold', color=INK_2,
        )

    axes[0].set_yticks(positions, cv[label_col], fontsize=10, fontweight='semibold')
    axes[0].set_ylabel('Experiment', fontsize=11, fontweight='semibold')
    axes[0].set_ylim(-0.65, len(cv) - 0.35)
    subtitle = (
        'Every prediction, no bet threshold'
        if operating_coverage >= 1.0
        else f'Dynamic threshold retains at least {operating_coverage:.0%} of CV games'
    )
    fig.suptitle(
        f'Experiment performance · CV versus holdout\n{subtitle}',
        fontsize=19, fontweight='bold', color='#222831',
    )
    return fig, axes
