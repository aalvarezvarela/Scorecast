"""Best single slots, their efficacy over time and same-timepoint combinations.

See docs/serving/historical_analysis.md §4. Every result here is measured on the
same rows it was selected from; the UI reports the selection reference with it.
"""

import itertools
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from nba_ou.dashboard.domain.analysis import (
    BREAK_EVEN,
    MIN_CELL_BETS,
    aggregate_cells,
    bucket_label,
    prepare_analysis,
    wilson_interval,
)

# T−6h belongs to the early window: "≥ 6h" = trained at least six hours out.
EARLY_WINDOW_MINUTES = 360
WINDOWS = ("All horizons", "≥ 6h", "< 6h")
CANDIDATE_KEYS = (
    "candidate",
    "family_key",
    "model_label",
    "dataset_type",
    "model_horizon_minutes",
    "bucket",
)
NULL_SIMULATIONS = 20_000
PERMUTATIONS = 2_000
# Skeptical prior for the profit probability: Beta(50, 50), centred on a coin
# flip with the weight of 100 bets (prior SD ≈ 5 pp). The market is close to
# efficient, so a short hot streak should not read as a sure edge.
PRIOR_STRENGTH = 50
TOP_SLOTS = 3
TOP_COMBINATIONS = 10
# Subsets of 2..MAX_COMBINATION_SIZE members of one timepoint; a timepoint with
# more members keeps the MAX_TIMEPOINT_MEMBERS with the most settled bets.
MAX_COMBINATION_SIZE = 4
MAX_TIMEPOINT_MEMBERS = 8
MEAN_EDGE = "Mean edge"
MAJORITY_VOTE = "Majority vote"
TARGET_NAMES = {1: "line error", 2: "total points", 3: "spread"}
_SIDE = {"OVER": 1, "HOME": 1, "UNDER": -1, "AWAY": -1}


def window_rows(frame: pd.DataFrame, window: str) -> pd.DataFrame:
    if window not in WINDOWS:
        raise ValueError("Unknown horizon window")
    if window == "All horizons":
        return frame
    early = frame["model_horizon_minutes"].ge(EARLY_WINDOW_MINUTES)
    return frame.loc[early if window == "≥ 6h" else ~early]


def _phase(dataset_type) -> str:
    return "Closing" if int(dataset_type) == 1 else "Intermediate"


def candidate_rows(rows: pd.DataFrame, source: str) -> pd.DataFrame:
    """Every slot at each timepoint it could bet, in one source.

    An intermediate model at its own snapshot, and a closing model at each
    input timepoint (``bucket``), Final included. Each pair is one candidate.
    """
    if rows.empty:
        return rows.copy()
    work = rows.loc[
        rows["source"].eq(source) & (rows["on_horizon"] | rows["dataset_type"].eq(1))
    ].copy()
    work["candidate"] = [
        f"{label} · {_phase(dataset)} · {bucket_label(int(bucket))}"
        for label, dataset, bucket in zip(
            work["model_label"], work["dataset_type"], work["bucket"], strict=True
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


def timepoint_members(
    candidates: pd.DataFrame, partner: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Candidate rows labelled by ``member``: the model and phase, without time.

    ``partner`` holds the other totals target's candidates (same source); its
    members, and then every member, name their target.
    """
    frames = [candidates]
    if partner is not None and not partner.empty:
        frames.append(partner)
    work = pd.concat(frames, ignore_index=True) if len(frames) > 1 else candidates
    if work.empty:
        return work.copy()
    work = work.copy()
    crossed = len(frames) > 1
    work["member"] = [
        f"{label.replace(' · ', ' ')} {_phase(dataset).lower()}"
        + (f" ({TARGET_NAMES[int(target)]})" if crossed else "")
        for label, dataset, target in zip(
            work["model_label"], work["dataset_type"], work["target"], strict=True
        )
    ]
    return work


def _combined_rows(
    part: pd.DataFrame, subset: tuple[str, ...], timepoint: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One mean-edge and one majority-vote row per game with every member.

    Both bet at the line, price and minimum edge of the member published last,
    when the whole combination is known.
    """
    size = len(subset)
    base = (
        part.sort_values(["available_at", "member"], kind="stable")
        .groupby("game_id")
        .tail(1)
        .set_index("game_id")
    )
    grouped = part.groupby("game_id")
    sides = part["signal"].map(_SIDE).fillna(0)
    up = sides.gt(0).groupby(part["game_id"]).sum()
    down = sides.lt(0).groupby(part["game_id"]).sum()
    # More than half of all members on one side: with two, both must agree.
    side = pd.Series(
        np.select([2 * up > size, 2 * down > size], [1.0, -1.0], 0.0), index=up.index
    )
    base = base.assign(
        model_line=base["reference_line"],
        # Line-error convention: the prediction is an offset from model_line.
        target=base["target"].where(base["target"].eq(3), 1),
        timepoint=timepoint,
        members=size,
        member_names=" + ".join(subset),
    )
    point = bucket_label(int(timepoint))
    names = " + ".join(subset)
    mean = base.assign(
        prediction_value=grouped["comparison_edge"].mean(),
        method=MEAN_EDGE,
        strategy=f"{point} · {MEAN_EDGE.lower()} · {names}",
    )
    vote = base.assign(
        prediction_value=side,
        method=MAJORITY_VOTE,
        strategy=f"{point} · {'both agree' if size == 2 else 'majority'} · {names}",
    )
    return mean.reset_index(), vote.reset_index()


def same_timepoint_combinations(
    members: pd.DataFrame, *, min_edge: Mapping[int, float] | None = None
) -> pd.DataFrame:
    """Graded rows of every combination of members sharing a timepoint.

    Per timepoint, each subset of 2 to ``MAX_COMBINATION_SIZE`` members is
    scored on the games where all of them have a line, as the mean of their
    edges (graded with the betting member's minimum edge) and as a strict
    majority vote of their own signals (no bet without one). An edge is a
    prediction minus the line it was graded at, so members from different
    models, phases and totals targets share a scale.
    """
    if members.empty:
        return pd.DataFrame()
    means, votes = [], []
    usable = members.loc[np.isfinite(members["comparison_edge"])]
    for timepoint, group in usable.groupby("bucket", sort=True):
        settled = group["result"].isin(["WIN", "LOSS"]).groupby(group["member"]).sum()
        names = sorted(
            settled.sort_values(ascending=False, kind="stable")
            .index[:MAX_TIMEPOINT_MEMBERS]
            .tolist()
        )
        for size in range(2, min(MAX_COMBINATION_SIZE, len(names)) + 1):
            for subset in itertools.combinations(names, size):
                part = group.loc[group["member"].isin(subset)]
                complete = part.groupby("game_id")["member"].transform("nunique")
                part = part.loc[complete.eq(size)]
                if part.empty:
                    continue
                mean, vote = _combined_rows(part, subset, int(timepoint))
                means.append(mean)
                votes.append(vote)
    if not means:
        return pd.DataFrame()
    graded_mean = prepare_analysis(
        pd.concat(means, ignore_index=True), min_edge=min_edge
    )
    # A vote is ±1 or 0, so any threshold below 1 places exactly the voted side.
    graded_vote = prepare_analysis(
        pd.concat(votes, ignore_index=True), min_edge={1: 0.5, 3: 0.5}
    )
    # A vote has no point prediction to score.
    graded_vote[["prediction_error", "absolute_error"]] = np.nan
    return pd.concat([graded_mean, graded_vote], ignore_index=True)


def rank_combinations(combined: pd.DataFrame) -> pd.DataFrame:
    """Win % per combination, eligible ones first (``MIN_CELL_BETS`` settled)."""
    if combined.empty:
        return pd.DataFrame()
    counts = pd.crosstab(combined["strategy"], combined["result"])
    for result in ("WIN", "LOSS", "PUSH"):
        if result not in counts:
            counts[result] = 0
    info = combined.groupby("strategy")[
        ["method", "timepoint", "members", "member_names"]
    ].first()
    table = info.join(counts[["WIN", "LOSS", "PUSH"]]).rename(
        columns={"WIN": "wins", "LOSS": "losses", "PUSH": "pushes"}
    )
    decided = table["wins"] + table["losses"]
    table["win_pct"] = (100 * table["wins"] / decided).where(decided.gt(0))
    table["eligible"] = decided.ge(MIN_CELL_BETS) & table["win_pct"].notna()
    return (
        table.reset_index()
        .sort_values(
            ["eligible", "win_pct", "wins", "strategy"],
            ascending=[False, False, False, True],
            na_position="last",
            kind="stable",
        )
        .reset_index(drop=True)
    )


def profit_probability(
    wins: int,
    losses: int,
    *,
    threshold_pct: float = BREAK_EVEN,
    prior: int = PRIOR_STRENGTH,
) -> float:
    """Posterior P(win rate > ``threshold_pct``) under a Beta(prior, prior) prior.

    The posterior is Beta(a, b), a = prior + wins, b = prior + losses. For
    integer a and b, P(X > t) = P(Binomial(a + b − 1, t) ≤ a − 1), so this is
    exact without scipy. It ignores how the slot was selected: for a slot picked
    as the best of many, read it with the selection-aware p-values.
    """
    a, b = prior + int(wins), prior + int(losses)
    if a <= 0 or b <= 0:
        return float("nan")
    n, t = a + b - 1, threshold_pct / 100
    log_t, log_u = math.log(t), math.log1p(-t)
    terms = [
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * log_t
        + (n - k) * log_u
        for k in range(a)
    ]
    peak = max(terms)
    return min(1.0, math.exp(peak) * sum(math.exp(x - peak) for x in terms))


def game_permutation_pvalue(
    frame: pd.DataFrame,
    by: str,
    cells,
    best_win_pct: float,
    *,
    simulations: int = PERMUTATIONS,
    seed: int = 0,
) -> float:
    """Selection-aware p-value of the best of ``cells``, permuting whole games.

    Under the null that picks carry no information about results, each game's
    outcome is flipped at random, the same flip for every cell that bet it, so
    the cells' shared games stay shared (the coin-flip chance treats them as
    independent instead). The p-value is the share of simulations whose best
    cell reaches ``best_win_pct``, with the usual +1 correction. Cells betting
    a game at different lines are treated as tied to one outcome, a mild
    overstatement of their correlation.
    """
    if frame.empty or not np.isfinite(best_win_pct):
        return float("nan")
    settled = frame.loc[
        frame[by].isin(list(cells)) & frame["result"].isin(["WIN", "LOSS"]),
        [by, "game_id", "result"],
    ]
    if settled.empty:
        return float("nan")
    games = pd.Index(settled["game_id"].unique())
    columns = pd.Index(settled[by].unique())
    sign = np.zeros((len(games), len(columns)), dtype=np.float32)
    sign[games.get_indexer(settled["game_id"]), columns.get_indexer(settled[by])] = (
        np.where(settled["result"].eq("WIN"), 1.0, -1.0)
    )
    bets = np.abs(sign).sum(axis=0)
    losses = (sign < 0).sum(axis=0).astype(np.float32)
    rng = np.random.default_rng(seed)
    keep = rng.integers(0, 2, size=(simulations, len(games))).astype(np.float32)
    # A kept game keeps its result; a flipped one swaps win and loss. So a
    # cell's simulated wins are its losses plus the kept games' signs.
    wins = keep @ sign + losses
    best = (100 * wins / bets).max(axis=1)
    return float(((best >= best_win_pct - 1e-9).sum() + 1) / (simulations + 1))


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
