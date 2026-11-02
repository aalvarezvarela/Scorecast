"""Verdict text generated from the same numbers the charts plot.

Kept out of ``charts`` so the plotting stays pure, and out of the notebook so a
conclusion is stated once, in code that can be corrected, rather than retyped
under each figure where it silently goes stale.

Every function returns Markdown strings for the notebook to display.
"""

from __future__ import annotations

import pandas as pd

from training_pipeline.reporting.theme import BREAK_EVEN, short_name


def coverage(per_strategy: pd.Series) -> list[str]:
    if len(per_strategy) < 2 or per_strategy.max() < 2 * per_strategy.min():
        return []
    return [
        f"⚠️ **Unbalanced coverage**: {per_strategy.max()} run(s) for "
        f"`{short_name(per_strategy.idxmax())}` versus {per_strategy.min()} for "
        f"`{short_name(per_strategy.idxmin())}`. Differences between strategies below "
        "partly reflect how many attempts each got."
    ]


def cohorts(runs: pd.DataFrame) -> list[str]:
    distinct = runs[["holdout_start", "holdout_end", "holdout_n_games"]].drop_duplicates()
    if len(distinct) == 1:
        return ["✅ **Every run was scored on the same holdout window.** ROI is directly comparable."]
    return [
        f"⚠️ **{len(distinct)} different holdout cohorts present.** Runs are only directly "
        "comparable within a cohort; across cohorts the baselines differ too."
    ]


def threshold_units(runs: pd.DataFrame) -> list[str]:
    grouped = runs.groupby("prediction_strategy")["bet_min_edge"].unique()
    lines = [
        f"- `{short_name(strategy)}`: threshold {[float(v) for v in values]} "
        + ("(expected value)" if strategy == "over_under_classifier"
           else "(points vs the line)")
        for strategy, values in grouped.items()
    ]
    return ["**Threshold units by strategy** — the same number is not the same filter:\n\n"
            + "\n".join(lines)]


def significance(runs: pd.DataFrame) -> list[str]:
    proven = runs[runs["win_rate_ci_low"] > BREAK_EVEN]
    if proven.empty:
        return [
            "**No run here has a win-rate interval clearing break-even.** At these volumes that "
            "is the expected outcome rather than a surprise: even a genuine 55% win rate needs "
            "on the order of a thousand bets before its interval separates from 52.38%. Treat "
            "every ROI as a hypothesis."
        ]
    return [
        "**Runs whose interval clears break-even:** "
        + ", ".join(f"`{r.label}` ({r.win_rate:.1%} on {int(r.n_bets)} bets)"
                    for r in proven.itertuples())
        + ". Confirm against the CV column before believing it."
    ]


def leakage(audit: pd.DataFrame) -> list[str]:
    if audit.empty:
        return []
    if audit["leakage_days"].sum() == 0:
        return ["✅ **No run trained on data at or after its own prediction date.**"]
    offenders = audit[audit["leakage_days"] > 0]["label"].tolist()
    return [f"🚨 **Future data leaked into training for {', '.join(offenders)} — those results "
            "are invalid.**"]


def line_comparison(lines: pd.DataFrame) -> list[str]:
    """Per run, what the alternative-line scoring actually licenses you to say.

    The circularity check comes first and short-circuits the rest: if the line
    bets settle into is itself a model feature, a gain against an earlier line
    is not evidence of anything.
    """
    messages: list[str] = []
    for label in lines["label"].unique():
        subset = lines[lines["label"] == label]
        if len(subset) < 2:
            continue
        close, alternatives = subset.iloc[0], subset.iloc[1:]
        for alt in alternatives.itertuples(index=False):
            if pd.isna(alt.roi) or pd.isna(close["roi"]):
                continue

            if close["line_is_a_feature"] and alt.roi > close["roi"]:
                messages.append(
                    f"- 🚨 **{label}**: {close['roi']:+.2%} vs `{close['line_col']}` → "
                    f"**{alt.roi:+.2%}** vs `{alt.line_col}`. **Do not read this as an edge.** "
                    f"`{close['line_col']}` is one of this model's input features, so the model "
                    "was given the market's closing number and then scored against the earlier "
                    "opener. The apparent gain is circular, and the bet is unplaceable: at the "
                    "time the opener is quoted, the closing line does not exist yet. To measure "
                    "this honestly, retrain with the closing line and every feature derived from "
                    "it excluded."
                )
                continue

            if close["roi"] > 0 and alt.roi > 0:
                verdict = ("is present at both lines — it survives at a price you could actually "
                           "have taken.")
            elif close["roi"] > 0 >= alt.roi:
                verdict = "exists against the close only, and does **not** survive at a bettable price."
            elif close["roi"] <= 0 < alt.roi:
                verdict = ("**is absent against the close but positive against the opener** — the "
                           "model is tracking where the line was going to move rather than beating "
                           "the market's final price. That is closing-line value, capturable only "
                           "if you bet early.")
            else:
                verdict = "is negative at both lines."
            messages.append(
                f"- **{label}**: {close['roi']:+.2%} vs `{close['line_col']}` → "
                f"{alt.roi:+.2%} vs `{alt.line_col}` "
                f"(lines differ by {alt.mean_abs_move_vs_first:.2f} points). The edge {verdict}"
            )
    return messages


def config_differences(substantive: pd.DataFrame, n_runs: int) -> list[str]:
    return [
        f"_{len(substantive)} substantive field(s) differ across {n_runs} run(s). Where more "
        "than one differs between any two runs, their difference in outcome cannot be attributed "
        "to a single cause._"
    ]


def tuned_window(window_table: pd.DataFrame, *, tolerance: float = 1.05) -> list[str]:
    """What tuning did with the training window, and what to distrust about it.

    Three separate warnings, because they call for different actions: a window
    that was never tuned needs a config change, one that hit the edge of its
    grid needs a wider grid, and a run with several rows per game needs its
    confidence intervals suppressed rather than believed.
    """
    if window_table.empty:
        return ["_No runs to report._"]

    messages: list[str] = []
    untuned = window_table.loc[~window_table["tuned"], "label"].tolist()
    if untuned:
        messages.append(
            "**The training window was NOT tuned** for: "
            + ", ".join(f"`{name}`" for name in untuned)
            + ". Set `walk_forward.train_games_choices` in the campaign YAML (at "
            "least two distinct values) and re-run, or the reported window is "
            "just the inherited default."
        )
    else:
        messages.append(
            "`train_games` was tuned by Optuna in every run above. `selected` is "
            "the value the reported trial actually used, read from "
            "`metadata.json` -- not `walk_forward.train_games` in `config.json`, "
            "which on a tuned run is only the fallback."
        )

    at_edge = window_table[window_table["at_grid_edge"]]
    if not at_edge.empty:
        messages.append(
            "**Censored by the search grid.** "
            + ", ".join(
                f"`{row['label']}` chose {row['selected']:,} from [{row['choices']}]"
                for _, row in at_edge.iterrows()
            )
            + ". A run that lands on the smallest or largest offered window did "
            "not find an optimum, it ran out of choices -- the better window may "
            "lie outside the grid. Widen `walk_forward.train_games_choices` and "
            "confirm the new values fit with `scripts/preflight_campaign.py`, "
            "which reports the real per-fold training size rather than an "
            "arithmetic guess."
        )

    pooled = window_table[window_table["rows_per_game"] > tolerance]
    if not pooled.empty:
        messages.append(
            "**Several rows per game** in: "
            + ", ".join(
                f"`{row['label']}` ({row['rows_per_game']:.1f} rows/game)"
                for _, row in pooled.iterrows()
            )
            + ". One game appears once per pre-game snapshot and those rows share "
            "a single outcome, so binomial intervals over them would be far too "
            "narrow. Following `training_pipeline.snapshot_scoring`, every "
            "interval and significance verdict for these runs is **suppressed** "
            "below rather than corrected -- an honest interval needs "
            "game-clustered inference. Their per-horizon numbers live in each "
            "run's `snapshot_holdout_metrics.csv`."
        )
    return messages
