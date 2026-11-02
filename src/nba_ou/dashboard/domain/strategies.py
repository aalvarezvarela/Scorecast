"""Best single slot, its efficacy over time and mean-of-predictions combinations.

See docs/serving/historical_analysis.md §4. Every result here is measured on the
same rows it was selected from; the UI reports the selection reference with it.
"""

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from nba_ou.dashboard.domain.analysis import (
    FINAL_BUCKET,
    aggregate_cells,
    prepare_analysis,
    wilson_interval,
)
from nba_ou.dashboard.domain.dates import horizon_label

# T−6h belongs to the early window: "≥ 6h" = trained at least six hours out.
EARLY_WINDOW_MINUTES = 360
WINDOWS = ("All horizons", "≥ 6h", "< 6h")
CANDIDATE_KEYS = (
    "candidate",
    "family_key",
    "model_label",
    "dataset_type",
    "model_horizon_minutes",
)
NULL_SIMULATIONS = 20_000


def window_rows(frame: pd.DataFrame, window: str) -> pd.DataFrame:
    if window not in WINDOWS:
        raise ValueError("Unknown horizon window")
    if window == "All horizons":
        return frame
    early = frame["model_horizon_minutes"].ge(EARLY_WINDOW_MINUTES)
    return frame.loc[early if window == "≥ 6h" else ~early]


def candidate_rows(rows: pd.DataFrame, source: str) -> pd.DataFrame:
    """Each intermediate model at its own snapshot, plus each closing Final."""
    if rows.empty:
        return rows.copy()
    work = rows.loc[
        rows["source"].eq(source)
        & (
            rows["on_horizon"]
            | rows["dataset_type"].eq(1) & rows["bucket"].eq(FINAL_BUCKET)
        )
    ].copy()
    work["candidate"] = [
        f"{label} · " + ("Closing Final" if dataset == 1 else horizon_label(horizon))
        for label, dataset, horizon in zip(
            work["model_label"],
            work["dataset_type"],
            work["model_horizon_minutes"],
            strict=True,
        )
    ]
    return work


def rank_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Win % per candidate; only cells with MIN_CELL_BETS wins + losses rank."""
    if candidates.empty:
        return pd.DataFrame()
    cells = aggregate_cells(candidates, by=CANDIDATE_KEYS)
    cells["eligible"] = ~cells["low_sample"] & cells["win_pct"].notna()
    return cells.sort_values(
        ["eligible", "win_pct", "bets"], ascending=False, na_position="last"
    ).reset_index(drop=True)


def selection_chance(bet_counts, best_win_pct: float, *, seed: int = 0) -> float:
    """Share of coin-flip draws whose best of these cells reaches ``best_win_pct``.

    Cells share games, so they are positively correlated and act like fewer
    independent tries; simulating them as independent overstates the chance,
    which makes this a conservative reference rather than an exact p-value.
    """
    counts = np.asarray(bet_counts, dtype=int)
    counts = counts[counts > 0]
    if not len(counts) or not np.isfinite(best_win_pct):
        return float("nan")
    rng = np.random.default_rng(seed)
    wins = rng.binomial(counts, 0.5, size=(NULL_SIMULATIONS, len(counts)))
    best = (100 * wins / counts).max(axis=1)
    return float((best >= best_win_pct - 1e-9).mean())


def cumulative_win_rate(
    frame: pd.DataFrame, game_dates: pd.Series, series: str
) -> pd.DataFrame:
    """Running win % by slate date; ``game_dates`` maps game_id to its date."""
    settled = frame.loc[frame["result"].isin(["WIN", "LOSS"])] if len(frame) else frame
    if settled.empty:
        return pd.DataFrame()
    daily = (
        settled.assign(
            game_date=pd.to_datetime(settled["game_id"].map(game_dates)),
            win=settled["result"].eq("WIN"),
            loss=settled["result"].eq("LOSS"),
        )
        .dropna(subset=["game_date"])
        .groupby("game_date")[["win", "loss"]]
        .sum()
        .cumsum()
    )
    if daily.empty:
        return pd.DataFrame()
    curve = pd.DataFrame(
        {
            "game_date": daily.index,
            "series": series,
            "wins": daily["win"].to_numpy(dtype=int),
            "losses": daily["loss"].to_numpy(dtype=int),
        }
    )
    curve["bets"] = curve["wins"] + curve["losses"]
    curve["win_pct"] = 100 * curve["wins"] / curve["bets"]
    bounds = [
        wilson_interval(wins, losses)
        for wins, losses in zip(curve["wins"], curve["losses"], strict=True)
    ]
    curve["ci_low"], curve["ci_high"] = zip(*bounds, strict=True)
    # Where a coin flip with the same number of bets lands 95% of the time.
    radius = 1.959963984540054 * 50 / np.sqrt(curve["bets"])
    curve["flip_low"], curve["flip_high"] = 50 - radius, 50 + radius
    return curve


def combine_predictions(
    members: pd.DataFrame, *, min_edge: Mapping[int, float] | None = None
) -> pd.DataFrame:
    """Average each game's member edges and bet them at the latest member's line.

    An edge is a prediction minus the line that member was graded at, so
    members from different snapshots, models or totals targets share a scale.
    The bet uses the line, price and minimum edge of the latest member (the
    shortest horizon; closing counts as 0), when the whole mean is known.
    """
    usable = (
        members.loc[np.isfinite(members["comparison_edge"])]
        if len(members)
        else members
    )
    if usable.empty:
        return pd.DataFrame()
    mean = usable.groupby("game_id")["comparison_edge"].agg(["mean", "size"])
    latest = (
        usable.sort_values(["model_horizon_minutes", "target"], kind="stable")
        .groupby("game_id")
        .head(1)
        .set_index("game_id")
    )
    combined = latest.copy()
    combined["prediction_value"] = mean["mean"]
    combined["model_line"] = combined["reference_line"]
    # Line-error convention: the prediction is an offset from model_line.
    combined["target"] = combined["target"].where(combined["target"].eq(3), 1)
    combined = prepare_analysis(combined.reset_index(), min_edge=min_edge)
    combined["members"] = combined["game_id"].map(mean["size"])
    return combined


def strategy_members(
    own: pd.DataFrame, partner: pd.DataFrame | None = None
) -> dict[str, pd.DataFrame]:
    """The fixed, pre-declared combinations; never a search over subsets.

    ``own`` holds one source's own-snapshot intermediate rows; ``partner`` the
    same rows for the other totals target (line error ↔ total points).
    """
    groups: dict[str, pd.DataFrame] = {}
    if own.empty:
        return groups
    families = own.groupby("family_key", sort=True)
    for window in WINDOWS:
        for _, family in families:
            label = family["model_label"].iloc[0]
            groups[f"{label} · mean {window.lower()}"] = window_rows(family, window)
            if partner is not None and not partner.empty:
                other = partner.loc[
                    partner["family_key"].eq(family["family_key"].iloc[0])
                ]
                if not other.empty:
                    groups[
                        f"{label} · line error + total points · {window.lower()}"
                    ] = window_rows(pd.concat([family, other]), window)
        if len(families) > 1:
            groups[f"All models · mean {window.lower()}"] = window_rows(own, window)
    return {label: rows for label, rows in groups.items() if not rows.empty}


def sign_test(only_a: int, only_b: int) -> float:
    """Exact two-sided sign test on the games where exactly one side won."""
    n = only_a + only_b
    if not n:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1))
    return min(1.0, 2 * tail / 2**n)


def compare_with_best(
    strategies: Mapping[str, pd.DataFrame], best: pd.DataFrame
) -> pd.DataFrame:
    """Pooled stats per strategy plus a paired comparison with the best slot."""
    best_settled = best.loc[best["result"].isin(["WIN", "LOSS"]), ["game_id", "result"]]
    rows = []
    for label, frame in strategies.items():
        if frame.empty:
            continue
        stats = aggregate_cells(frame, by=[]).iloc[0]
        settled = frame.loc[
            frame["result"].isin(["WIN", "LOSS"]), ["game_id", "result"]
        ]
        pairs = settled.merge(best_settled, on="game_id", suffixes=("", "_best"))
        only_strategy = int(
            (pairs["result"].eq("WIN") & pairs["result_best"].eq("LOSS")).sum()
        )
        only_best = int(
            (pairs["result"].eq("LOSS") & pairs["result_best"].eq("WIN")).sum()
        )
        rows.append(
            {
                "strategy": label,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "pushes": stats["pushes"],
                "bets": stats["bets"],
                "win_pct": stats["win_pct"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "roi": stats["roi"],
                "mae": stats["mae"],
                "low_sample": stats["low_sample"],
                "matched_games": len(pairs),
                "only_strategy_won": only_strategy,
                "only_best_won": only_best,
                "matched_delta_pp": (
                    100 * (only_strategy - only_best) / len(pairs)
                    if len(pairs)
                    else np.nan
                ),
                "sign_test_p": sign_test(only_strategy, only_best),
            }
        )
    return pd.DataFrame(rows)
