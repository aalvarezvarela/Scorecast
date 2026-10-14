"""The frozen phase 3 ratings as a walk-forward rating book (2_6's cache format).

Phase 3 froze the pipeline (``frozen_decisions_and_future_checks.md``, "Phase 3
closed"): offense and defense shrunk toward the profile prior ``f`` at
lambda 3,000, pace shrunk toward zero at lambda 10,000, 2_6's 180-day
half-life, ``f`` refit monthly on as-of pairs with the alphas of
``alphas.json``, a debut prior for players with a profile but no box scores.
``rapm_sweep.sweep_predictions`` scored that pipeline on stints;
:func:`walk_forward_rating_book` runs the same fit day by day and emits the
ratings instead: one row per ``(as_of_date, player_id)`` with ``o_rating``,
``d_rating``, ``pace_rating``, ``poss_weight``, ``league_ortg``,
``league_pace`` and ``fit_max_game_date``, readable by
``lineups.features.load_rating_book`` / ``RatingBook``.

Each date is fitted on stints strictly before it, with the ``f`` of the latest
monthly checkpoint on or before it and the node profiles as of it. A player
with a prior block keeps his prior while unrated (a player without data gets
exactly ``beta0``); pace, with no prior, is 0 while unrated, as in 2_6. Rows
are emitted for players with exposure or a non-zero offensive / defensive
prior; every other player of the solver has 0 in all three, which is what a
book lookup returns for a missing player.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from nba_ou.data_processing.lineups.player_ratings import _lineup, _NormalEquations

from .profile_prior import HALF_LIFE_DAYS, LAMBDA_OFFDEF, LAMBDA_PACE, PriorProvider
from .rapm_sweep import _add_day, _fit

PIPELINE = "phase3_profile_prior"
#: First monthly checkpoint of ``f`` (phase 3); players first seen on or after
#: it are debutants for the debut prior.
FIRST_CHECKPOINT = pd.Timestamp("2016-12-01")
BOOK_PATH = Path("player_graph") / "rating_book" / f"{PIPELINE}.parquet"


@dataclass(frozen=True)
class BookSpec:
    """The frozen phase 3 configuration (the defaults)."""

    lambda_offdef: float = LAMBDA_OFFDEF
    lambda_pace: float = LAMBDA_PACE
    prior_offdef: bool = True
    prior_pace: bool = False
    half_life_days: float = HALF_LIFE_DAYS


FROZEN = BookSpec()


def debutants(history: pd.DataFrame, first_checkpoint=FIRST_CHECKPOINT) -> set[str]:
    """Players whose first stint is on or after ``first_checkpoint``."""
    first: dict[str, pd.Timestamp] = {}
    for row in history[["game_date", "home_lineup", "away_lineup"]].itertuples(
        index=False
    ):
        for player in (*row.home_lineup, *row.away_lineup):
            first.setdefault(str(int(player)), row.game_date)
    return {p for p, d in first.items() if pd.Timestamp(d) >= first_checkpoint}


def frozen_prior_provider(
    history: pd.DataFrame, *, local_root: Path = Path("data")
) -> PriorProvider:
    """Phase 3's provider as frozen for its 2019-25 evaluation: the evaluation
    pairs (``build_prior_pairs.py --evaluation``), the node profiles of the
    history's seasons and the frozen alphas."""
    root = Path(local_root) / "player_graph"
    seasons = sorted(pd.to_numeric(history["season_year"]).unique())
    profiles = pd.concat(
        [
            pd.read_parquet(root / "node_profiles" / f"season={s}.parquet")
            for s in seasons
        ],
        ignore_index=True,
    )
    pairs = pd.read_parquet(root / "profile_prior" / "pairs_evaluation.parquet")
    frozen = json.loads((root / "profile_prior" / "alphas.json").read_text())
    return PriorProvider(pairs, profiles, debutants(history), frozen["alphas"])


def walk_forward_rating_book(
    history: pd.DataFrame,
    target_dates: Iterable,
    *,
    spec: BookSpec = FROZEN,
    prior: Callable[[pd.Timestamp, list[str]], dict[str, np.ndarray]] | None = None,
) -> pd.DataFrame:
    """Ratings on each target date from ``history`` stints strictly before it."""
    if (spec.prior_offdef or spec.prior_pace) and prior is None:
        raise ValueError("The spec asks for a prior but none was given")
    frame = history.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.normalize()
    frame["home_lineup"] = frame["home_lineup"].map(_lineup)
    frame["away_lineup"] = frame["away_lineup"].map(_lineup)
    players = sorted(
        {
            pid
            for side in ("home", "away")
            for lineup in frame[f"{side}_lineup"]
            for pid in lineup
        },
        key=int,
    )
    index = {pid: i for i, pid in enumerate(players)}
    n = len(players)
    offdef = _NormalEquations(2 * n)
    pace = _NormalEquations(n)
    exposure = np.zeros(n)
    penalty = np.r_[np.full(n, spec.lambda_offdef), np.full(n, spec.lambda_offdef)]
    pace_penalty = np.full(n, spec.lambda_pace)
    previous = pace_previous = None
    requested = {pd.Timestamp(d).normalize() for d in target_dates}
    grouped = dict(tuple(frame.groupby("game_date", sort=False)))
    ids = np.asarray(players)
    out = []
    previous_day, last_fit = None, pd.NaT
    for day in sorted(requested | set(grouped)):
        if previous_day is not None:
            factor = 0.5 ** ((day - previous_day).days / spec.half_life_days)
            offdef.decay(factor)
            pace.decay(factor)
            exposure *= factor
        previous_day = day
        if day in requested and offdef.weight:
            rated = exposure > 0
            zero = np.zeros(n)
            vectors = prior(day, players) if prior is not None else {}
            target = np.r_[
                vectors["o"] if spec.prior_offdef else zero,
                vectors["d"] if spec.prior_offdef else zero,
            ]
            beta, intercept = _fit(offdef, penalty, previous, target)
            previous = beta
            o = beta[:n] if spec.prior_offdef else beta[:n] * rated
            d = beta[n:] if spec.prior_offdef else beta[n:] * rated
            pace_beta, pace_intercept = _fit(
                pace,
                pace_penalty,
                pace_previous,
                vectors["pace"] if spec.prior_pace else zero,
            )
            pace_previous = pace_beta
            p = pace_beta if spec.prior_pace else pace_beta * rated
            keep = rated | (o != 0) | (d != 0) | (p != 0)
            out.append(
                pd.DataFrame(
                    {
                        "as_of_date": day,
                        "player_id": ids[keep],
                        "o_rating": o[keep],
                        "d_rating": d[keep],
                        "pace_rating": p[keep],
                        "poss_weight": exposure[keep],
                        "league_ortg": intercept,
                        "league_pace": pace_intercept,
                        "fit_max_game_date": last_fit,
                    }
                )
            )
        if day in grouped:
            _add_day(grouped[day], index, n, offdef, pace, exposure)
            last_fit = day
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def book_metadata(spec: BookSpec, **extra) -> dict:
    """What a stored book records: the solver version ``load_rating_book``
    checks, the pipeline and its frozen parameters."""
    from nba_ou.data_processing.lineups.player_ratings import RATING_SOLVER_VERSION

    return {
        "solver_version": RATING_SOLVER_VERSION,
        "pipeline": PIPELINE,
        **asdict(spec),
        **extra,
    }
