"""Benchmark for guard-edge weights: how well expected rates predict a game's shares.

For each attacker-game, the observed share of his matchup time spent against
each defender he shared the floor with::

    s_ij = matchup_seconds_ij / sum_l matchup_seconds_il

is compared with the share implied by an expected rate and the game's actual
co-floor time (known on a training graph, which is built from the stints)::

    e_ij = rate_ij * cofloor_ij / sum_l rate_il * cofloor_il

The error is the total variation distance ``0.5 * sum_j |s_ij - e_ij|`` (0 =
identical, 1 = disjoint), averaged over attacker-games weighted by the
attacker's matchup seconds. Lower is better. Phase 4B must beat v0 here.

An attacker whose rates do not define a distribution (any rate not finite, or
all zero) falls back to the constant rate, ``e`` proportional to co-floor time,
as a graph falls back to uniform shares. Without this an undefined prediction
would score a perfect 0. :func:`estimator_errors` reports how many attacker-games
fell back.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .expected_guard import shrink

KEYS = ["game_id", "off_player_id", "def_player_id"]
ATTACKER = ["game_id", "off_player_id"]

PRIOR_WEIGHT_BINS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)


def benchmark_frame(pair_game: pd.DataFrame, expected: pd.DataFrame) -> pd.DataFrame:
    """Observed pairs joined to their expected rates, with observed shares ``s``.

    Keeps pairs that shared the floor, have a defined prior, and belong to an
    attacker-game with some matchup time.
    """
    observed = pair_game.loc[
        pair_game["cofloor_seconds"] > 0,
        [*KEYS, "season_year", "cofloor_seconds", "matchup_seconds"],
    ]
    frame = observed.merge(
        expected.drop(columns=["game_date"], errors="ignore"),
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    frame = frame.loc[frame["r_prior"].notna()]
    total = frame.groupby(ATTACKER)["matchup_seconds"].transform("sum")
    frame = frame.loc[total > 0].copy()
    frame["s"] = frame["matchup_seconds"] / total.loc[frame.index]
    return frame.reset_index(drop=True)


def implied_shares(
    frame: pd.DataFrame, rate: pd.Series | np.ndarray
) -> tuple[pd.Series, pd.Series]:
    """``(e, fallback)``: rate-implied shares per pair, and per attacker-game
    whether the constant-rate fallback was used."""
    groups = [frame["game_id"], frame["off_player_id"]]
    cofloor = frame["cofloor_seconds"].to_numpy(float)
    implied = pd.Series(np.asarray(rate, float) * cofloor, index=frame.index)
    finite = np.isfinite(implied).groupby(groups).transform("all")
    total = implied.where(finite, 0.0).groupby(groups).transform("sum")
    usable = finite & (total > 0)
    constant = pd.Series(cofloor, index=frame.index)
    constant = constant / constant.groupby(groups).transform("sum")
    shares = (implied / total).where(usable, constant)
    return shares, (~usable).groupby(groups).first()


def share_tvd(frame: pd.DataFrame, rate: pd.Series | np.ndarray) -> pd.Series:
    """TVD per attacker-game between observed and rate-implied shares."""
    implied, _ = implied_shares(frame, rate)
    groups = [frame["game_id"], frame["off_player_id"]]
    return 0.5 * (frame["s"] - implied).abs().groupby(groups).sum()


def _attacker_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby(ATTACKER).agg(
        weight=("matchup_seconds", "sum"), season_year=("season_year", "first")
    )


def _weighted(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values, weights=weights.loc[values.index]))


def estimator_errors(frame: pd.DataFrame, k_grid: Iterable[float]) -> pd.DataFrame:
    """Weighted TVD per estimator, overall (``all``) and per season, plus the
    share of attacker-games that fell back to the constant rate."""
    attackers = _attacker_table(frame)
    estimators = {
        "constant rate (co-floor only)": np.ones(len(frame)),
        "position prior only (k=inf)": frame["r_prior"],
        "pair history only (k->0)": shrink(
            frame["hist_matchup_seconds_decayed"],
            frame["hist_cofloor_seconds_decayed"],
            frame["r_prior"],
            1e-6,
        )[0],
    }
    for k in k_grid:
        estimators[f"k={k:g}"] = shrink(
            frame["hist_matchup_seconds_decayed"],
            frame["hist_cofloor_seconds_decayed"],
            frame["r_prior"],
            k,
        )[0]
    rows = {}
    for name, rate in estimators.items():
        error = share_tvd(frame, rate)
        _, fallback = implied_shares(frame, rate)
        seasons = attackers["season_year"].loc[error.index]
        row = {
            "fallback": float(fallback.loc[error.index].mean()),
            "all": _weighted(error, attackers["weight"]),
        }
        for season in sorted(seasons.unique()):
            mask = seasons.eq(season)
            row[int(season)] = _weighted(error[mask], attackers["weight"])
        rows[name] = row
    return pd.DataFrame(rows).T


def prior_weight_buckets(frame: pd.DataFrame, k: float) -> pd.DataFrame:
    """Error by how much of an attacker's expected distribution is prior.

    The attacker's prior weight is the mean of his pairs' ``prior_weight``,
    weighted by their implied shares ``e_ij``.
    """
    rate, prior_weight = shrink(
        frame["hist_matchup_seconds_decayed"],
        frame["hist_cofloor_seconds_decayed"],
        frame["r_prior"],
        k,
    )
    groups = [frame["game_id"], frame["off_player_id"]]
    implied, _ = implied_shares(frame, rate)
    attacker_weight = (implied * prior_weight).groupby(groups).sum()
    error = share_tvd(frame, rate)
    attackers = _attacker_table(frame)
    table = pd.DataFrame(
        {
            "tvd": error,
            "weight": attackers["weight"].loc[error.index],
            "bucket": pd.cut(
                attacker_weight.loc[error.index],
                list(PRIOR_WEIGHT_BINS),
                include_lowest=True,
            ),
        }
    )
    return table.groupby("bucket", observed=True).apply(
        lambda group: pd.Series(
            {
                "attacker_games": len(group),
                "tvd": np.average(group["tvd"], weights=group["weight"]),
            }
        ),
        include_groups=False,
    )
