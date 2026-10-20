"""Verdict text generated from the same numbers the charts plot.

Kept out of ``charts`` so the plotting stays pure, and out of the notebook so a
conclusion is stated once, in code that can be corrected, rather than retyped
under each figure where it silently goes stale.

Every function returns Markdown strings for the notebook to display.
"""

from __future__ import annotations

import pandas as pd

from training_pipeline.reporting.theme import BREAK_EVEN, short_name


def cv_vs_test_win_rate(table: pd.DataFrame) -> list[str]:
    """How the cross-validated win rate fared on the held-out period."""
    if table.empty:
        return ["_No run reports both a cross-validation and a holdout win rate._"]

    messages: list[str] = []
    held_up = table[table["win_rate"] > BREAK_EVEN]
    both = table[(table["win_rate"] > BREAK_EVEN) & (table["cv_win_rate"] > BREAK_EVEN)]
    median_drop = table["win_rate_drop"].median()

    messages.append(
        f"Across {len(table)} run(s) the median move from cross-validation to the held-out test is "
        f"**{-median_drop:+.2%}** in win rate. "
        + ("A drop is the expected direction: the folds chose the hyperparameters, so they flatter "
           "the model. What matters is whether anything is left after it."
           if median_drop > 0 else
           "The holdout came out *above* cross-validation here, which is more often the small "
           "holdout being lucky than a model that improves out of sample.")
    )

    if both.empty:
        messages.append(
            "⚠️ **No run clears break-even in both measurements.** A win rate above 52.38% in one "
            "and below it in the other is the signature of noise rather than an edge."
        )
    else:
        messages.append(
            "Runs above break-even in **both**: "
            + ", ".join(
                f"`{r.label}` (CV {r.cv_win_rate:.1%} → test {r.win_rate:.1%})"
                for r in both.itertuples()
            )
            + ". These are the only ones worth carrying forward — and still only on "
            f"{int(both['n_bets'].min())}–{int(both['n_bets'].max())} test bets."
        )

    fragile = held_up[~held_up["label"].isin(both["label"])]
    if not fragile.empty:
        messages.append(
            "_Above break-even on the test but not in cross-validation: "
            + ", ".join(f"`{name}`" for name in fragile["label"])
            + ". With the larger CV sample disagreeing, treat the test figure as the fluke._"
        )
    return messages


def edge_ranges(pooled: pd.DataFrame) -> list[str]:
    """Whether a bigger predicted edge actually wins more often."""
    if pooled.empty:
        return ["_No regression runs with usable predictions._"]

    messages: list[str] = []
    for source in pooled["source"].unique():
        subset = pooled[
            (pooled["source"] == source) & (~pooled["too_few_to_read"])
        ].sort_values("edge_low")
        if len(subset) < 3:
            continue

        # Spearman over the readable bins, weighted implicitly by using the bin
        # order: a monotone rise is the claim being tested.
        rho = subset["edge_low"].corr(subset["win_rate"], method="spearman")
        smallest = subset.iloc[0]
        largest = subset.iloc[-1]
        messages.append(
            f"- **{source}**: win rate goes from **{smallest['win_rate']:.1%}** in the "
            f"{smallest['edge_range']} point range ({int(smallest['n_bets'])} bets) to "
            f"**{largest['win_rate']:.1%}** in {largest['edge_range']} "
            f"({int(largest['n_bets'])} bets). Rank correlation across ranges "
            f"**{rho:+.2f}**. "
            + ("A clear upward trend — bigger edges really are better bets."
               if rho > 0.6 else
               "**No usable trend**: a bigger predicted edge does not win more often here."
               if rho < 0.3 else
               "A weak, non-monotone trend — not something to size bets on.")
        )

    thin = pooled[pooled["too_few_to_read"]]
    if not thin.empty:
        ranges = sorted(set(thin["edge_range"]))
        messages.append(
            f"_{len(thin)} range/source cell(s) hold fewer than 30 bets and are shown but not "
            f"readable ({', '.join(ranges)}). Large edges are rare, so their win rates swing wildly "
            "on a handful of games — the widest intervals in the chart._"
        )

    readable = pooled[~pooled["too_few_to_read"]]
    if not readable.empty:
        best = readable.loc[readable["win_rate"].idxmax()]
        messages.append(
            "**What this means for the bet threshold.** If win rate is flat across ranges, raising "
            "the minimum edge does not improve bet quality — it only discards volume, and volume is "
            "what every significance test here is short of. The best readable range is "
            f"`{best['edge_range']}` at {best['win_rate']:.1%} on {int(best['n_bets'])} bets "
            f"({best['source']}); check whether that is meaningfully above the ranges you are "
            "currently discarding before keeping the filter."
        )
    return messages


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


def seed_noise(runs: pd.DataFrame) -> list[str]:
    """Error-bar commentary, for the runs that happen to have one.

    Single-seed is the default, so ``seed_roi_range`` is usually absent from the
    frame ENTIRELY -- not merely NaN. dropna(subset=[...]) raises KeyError on a
    missing column, which is how a helper like this turns "no error bars" into
    "the whole report failed to render".
    """
    messages: list[str] = []
    if "seed_roi_range" not in runs.columns:
        return messages
    seeded = runs.dropna(subset=["seed_roi_range"])
    if len(runs) >= 2 and not seeded.empty:
        between = runs["roi"].max() - runs["roi"].min()
        within = seeded["seed_roi_range"].max()
        best = runs.loc[runs["roi"].idxmax()]
        worst = runs.loc[runs["roi"].idxmin()]
        if between > within:
            tone, verdict = "✅", (
                f"larger than the widest seed range (**{within:.2%}**), so the spread across "
                "runs is not fully explained by seed noise.")
        else:
            tone, verdict = "⚠️", (
                f"**smaller** than the widest seed range (**{within:.2%}**) — this selection "
                "cannot distinguish its runs from one another.")
        n_cohorts = runs[["holdout_start", "holdout_end"]].drop_duplicates().shape[0]
        caveat = "" if n_cohorts == 1 else (
            f" Note these runs span **{n_cohorts} different holdout cohorts**, so part of the "
            "spread is different games rather than different configurations.")
        messages.append(
            f"{tone} Best-to-worst ROI spread is **{between:.2%}** "
            f"({best['label']} {best['roi']:+.2%} vs {worst['label']} {worst['roi']:+.2%}), "
            f"which is {verdict}{caveat}"
        )

    # Only worth saying when the selection is MIXED: with every run single-seed
    # (the default) this would just restate the protocol on every report.
    missing = runs[runs["seed_roi_range"].isna()]
    if not missing.empty and not seeded.empty:
        messages.append(
            f"_{len(missing)} run(s) have no seed data and no error bar: "
            + ", ".join(f"`{name}`" for name in missing["label"])
            + ". Compare them on holdout MAE, which is far less noisy than ROI, "
            "or set `evaluation_seeds: [101, 202]` and rerun to give them one._"
        )
    return messages


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


def cv_agreement(agreement: pd.DataFrame) -> list[str]:
    if agreement.empty:
        return []
    n_agree = int(agreement["same_sign"].sum())
    tail = (
        "A majority disagreeing is itself the finding: at these volumes the two measurements "
        "are largely independent noise, and neither alone should pick a winner."
        if n_agree <= len(agreement) / 2 else
        "Runs agreeing on sign are the ones worth carrying forward."
    )
    return [f"**{n_agree} of {len(agreement)} run(s) agree on the sign** of their edge "
            f"between CV and holdout. {tail}"]


def strategy_spread(summary: pd.DataFrame, runs: pd.DataFrame) -> list[str]:
    # Absent column, not just absent values: single-seed runs never write one.
    if "seed_roi_range" not in runs.columns or summary.empty:
        return []
    if not runs["seed_roi_range"].notna().any():
        return []
    noise = runs["seed_roi_range"].max()
    spread = summary["mean_roi"].max() - summary["mean_roi"].min()
    tail = ("The strategies are separated by more than seed noise."
            if spread > noise else
            "**That is within seed noise** — this selection does not establish that any "
            "strategy beats another.")
    return [f"Mean-ROI spread between strategies is **{spread:.2%}**, against a worst-case seed "
            f"range of **{noise:.2%}**. {tail}"]


def classifier_quality(classifiers: pd.DataFrame) -> list[str]:
    messages = []
    for _, row in classifiers.iterrows():
        improvement = row.get("cv_log_loss_improvement")
        if pd.isna(improvement):
            continue
        tail = ("Positive, so the probabilities carry information beyond the base rate — though "
                "how much is best judged next to the bet volume."
                if improvement > 0 else
                "**Not positive**: the probabilities are no better than always predicting the "
                "historical OVER rate, so any profit is not coming from probability quality.")
        messages.append(f"- **{row['label']}**: CV log-loss improvement **{improvement:+.4f}**. {tail}")
    return messages


def confidence_trend(trend: pd.DataFrame) -> list[str]:
    if trend.empty:
        return []
    strong = trend[(trend["p_value"] < 0.05) & (trend["rho"] > 0)]
    if strong.empty:
        return [
            "⚠️ **No run shows a significant positive relationship between confidence and "
            "winning.** On this evidence a higher minimum-edge threshold is not selecting better "
            "bets — it is only reducing volume, which makes every other number here harder to "
            "prove. Worth testing a lower threshold explicitly before assuming the filter earns "
            "its place."
        ]
    return [
        "✅ Confidence predicts winning for: "
        + ", ".join(f"`{r.label}` (rho {r.rho:+.3f}, p={r.p_value:.3f})"
                    for r in strong.itertuples())
        + ". For these, a selective threshold is doing real work."
    ]


def fold_consistency(consistency: pd.DataFrame) -> list[str]:
    if consistency.empty:
        return []
    return ["`mean_over_sd` below about 1 means the fold-to-fold spread is as large as the "
            "average itself — the signature of an edge carried by a few folds rather than a "
            "repeatable one."]


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
