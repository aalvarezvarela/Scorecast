"""2_6's walk-forward ridge for a grid of penalties, scored on future stints
(plan phase 3, step 2: the zero-prior control).

The same fit as ``lineups.player_ratings.walk_forward_player_ratings`` (design,
weights, decay, free intercept, CG solver and tolerance), so a penalty pair
``(lambda, lambda)`` reproduces 2_6's ratings for that lambda. What changes:

* the normal equations are accumulated once and solved for every penalty on
  each date;
* offense and defense may take separate penalties (2_6 shares one,
  ``lambda_offdef``), as a secondary analysis;
* a penalty of ``inf`` removes that block (its ratings are 0): ``(inf, inf)``
  and a pace penalty of ``inf`` are the lambda -> infinity reference, the
  intercept alone (the decayed historical mean).

Ratings are not stored: each date's fits predict that date's rows (from
``rating_diagnostics.scored_rows``, which carry the players on the floor) and
only the predictions are kept. As in 2_6's output, a player with no offensive
possessions before the date has no rating (0).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg, spsolve

from nba_ou.data_processing.lineups.player_ratings import (
    _lineup,
    _matrix,
    _NormalEquations,
    _poss,
)

from .rating_diagnostics import _clustered_mean


def _solve_penalized(
    matrix: sparse.csr_matrix,
    rhs: np.ndarray,
    penalty: np.ndarray,
    x_sum: np.ndarray,
    weight: float,
    previous: np.ndarray | None,
) -> np.ndarray:
    """``player_ratings._solve`` with a diagonal penalty instead of a scalar."""
    if not np.any(rhs):
        return np.zeros_like(rhs)
    regularized = (matrix + sparse.diags(penalty)).tocsr()
    centered = LinearOperator(
        matrix.shape,
        matvec=lambda beta: regularized @ beta - x_sum * (x_sum @ beta / weight),
        dtype=float,
    )
    result, info = cg(centered, rhs, x0=previous, rtol=1e-7, maxiter=500)
    if info:
        column = sparse.csr_matrix(x_sum[:, None])
        system = sparse.bmat(
            [[regularized, column], [column.T, sparse.csr_matrix([[weight]])]],
            format="csr",
        )
        result = spsolve(system, np.r_[rhs, 0.0])[:-1]
    return np.asarray(result, dtype=float)


def _fit(
    equations: _NormalEquations,
    penalty: np.ndarray,
    previous: np.ndarray | None,
    prior: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Ridge with free intercept toward ``prior`` (0 when None):
    ``(X'X + diag(penalty)) beta = X'y + diag(penalty) prior``, centered.

    A coefficient with an infinite penalty is fixed at its prior; its
    coupling with the free coefficients moves to the right-hand side.
    """
    prior = np.zeros(len(penalty)) if prior is None else prior
    active = np.isfinite(penalty)
    beta = np.where(active, 0.0, prior)
    if not equations.weight:
        return prior.copy(), 0.0
    weight = equations.weight
    mean = equations.weighted_y / weight
    rhs = equations.b_raw - mean * equations.x_sum
    if active.any():
        idx = np.flatnonzero(active)
        fixed = np.flatnonzero(~active)
        rhs_active = rhs[idx] + penalty[idx] * prior[idx]
        if fixed.size and np.any(beta[fixed]):
            rhs_active -= equations.a[idx][:, fixed] @ beta[fixed] - equations.x_sum[
                idx
            ] * (equations.x_sum[fixed] @ beta[fixed] / weight)
        matrix = equations.a if active.all() else equations.a[idx][:, idx]
        beta[idx] = _solve_penalized(
            matrix,
            rhs_active,
            penalty[idx],
            equations.x_sum[idx],
            weight,
            None if previous is None else previous[idx],
        )
    return beta, float(mean - equations.x_sum @ beta / weight)


def sweep_predictions(
    history: pd.DataFrame,
    efficiency_rows: pd.DataFrame,
    pace_rows: pd.DataFrame,
    *,
    offdef_grid: Sequence[tuple],
    pace_grid: Sequence,
    half_life_days: float = 180.0,
    prior: Callable[[pd.Timestamp, list[str]], dict[str, np.ndarray]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predictions for every row and penalty, fitted on ``history`` stints
    dated strictly before each row's date.

    ``offdef_grid`` items are ``(lambda_off, lambda_def)`` (shrink toward 0,
    2_6) or ``(lambda_off, lambda_def, prior_off, prior_def)``; ``pace_grid``
    items are ``lambda_pace`` or ``(lambda_pace, prior)``. A block with a
    prior shrinks toward ``prior(day, players)["o" | "d" | "pace"]`` (one
    value per player, in the solver's player order) instead of 0, and its
    unrated players keep their prior instead of 2_6's 0.

    Returns ``(efficiency, pace)``: frames aligned with the rows, one column per
    grid item.
    """
    offdef_grid = [
        tuple(item) if len(item) == 4 else (*item, False, False) for item in offdef_grid
    ]
    pace_grid = [
        tuple(item) if isinstance(item, tuple) else (item, False) for item in pace_grid
    ]
    if prior is None and (
        any(po or pd_ for *_, po, pd_ in offdef_grid) or any(p for _, p in pace_grid)
    ):
        raise ValueError("A grid item asks for a prior but none was given")
    frame = history.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.normalize()
    frame["home_lineup"] = frame["home_lineup"].map(_lineup)
    frame["away_lineup"] = frame["away_lineup"].map(_lineup)
    players = sorted(
        {
            pid
            for side in ("home", "away")
            for lu in frame[f"{side}_lineup"]
            for pid in lu
        },
        key=int,
    )
    index = {pid: i for i, pid in enumerate(players)}
    n = len(players)

    def indices(column: pd.Series) -> np.ndarray:
        return np.array(
            [[index[pid] for pid in lineup] for lineup in column], dtype=int
        )

    eff_dates = pd.to_datetime(efficiency_rows["game_date"]).dt.normalize().to_numpy()
    pace_dates = pd.to_datetime(pace_rows["game_date"]).dt.normalize().to_numpy()
    off_idx = indices(efficiency_rows["offense_players"])
    def_idx = indices(efficiency_rows["defense_players"])
    pace_idx = indices(pace_rows["players"])
    eff_out = np.full((len(efficiency_rows), len(offdef_grid)), np.nan)
    pace_out = np.full((len(pace_rows), len(pace_grid)), np.nan)
    penalties = [np.r_[np.full(n, lo), np.full(n, ld)] for lo, ld, *_ in offdef_grid]
    pace_penalties = [np.full(n, lp) for lp, _ in pace_grid]
    previous: list[np.ndarray | None] = [None] * len(offdef_grid)
    pace_previous: list[np.ndarray | None] = [None] * len(pace_grid)

    offdef = _NormalEquations(2 * n)
    pace = _NormalEquations(n)
    exposure = np.zeros(n)
    requested = set(pd.DatetimeIndex(eff_dates)) | set(pd.DatetimeIndex(pace_dates))
    grouped = dict(tuple(frame.groupby("game_date", sort=False)))
    previous_day = None
    for day in sorted(requested | set(grouped)):
        if previous_day is not None:
            factor = 0.5 ** ((day - previous_day).days / half_life_days)
            offdef.decay(factor)
            pace.decay(factor)
            exposure *= factor
        previous_day = day
        if day in requested:
            rated = exposure > 0  # 2_6 emits only players with exposure
            zero = np.zeros(n)
            vectors = prior(day, players) if prior is not None else {}
            rows = np.flatnonzero(eff_dates == day)
            for k, penalty in enumerate(penalties):
                _, _, prior_off, prior_def = offdef_grid[k]
                target = np.r_[
                    vectors["o"] if prior_off else zero,
                    vectors["d"] if prior_def else zero,
                ]
                beta, intercept = _fit(offdef, penalty, previous[k], target)
                if np.isfinite(penalty).all():
                    previous[k] = beta
                o = beta[:n] if prior_off else beta[:n] * rated
                d = beta[n:] if prior_def else beta[n:] * rated
                eff_out[rows, k] = (
                    intercept + o[off_idx[rows]].sum(1) - d[def_idx[rows]].sum(1)
                )
            rows = np.flatnonzero(pace_dates == day)
            for k, penalty in enumerate(pace_penalties):
                with_prior = pace_grid[k][1]
                target = vectors["pace"] if with_prior else zero
                beta, intercept = _fit(pace, penalty, pace_previous[k], target)
                if np.isfinite(penalty).all():
                    pace_previous[k] = beta
                values = beta if with_prior else beta * rated
                pace_out[rows, k] = intercept + values[pace_idx[rows]].sum(1)
        if day not in grouped:
            continue
        off_rows, off_y, off_w = [], [], []
        pace_rows_, pace_y, pace_w = [], [], []
        for row in grouped[day].to_dict("records"):
            home, away = row["home_lineup"], row["away_lineup"]
            seconds = (row["end_ds"] - row["start_ds"]) / 10
            home_poss, away_poss = _poss(row, "home"), _poss(row, "away")
            for attacking, defending, poss, points in (
                (home, away, home_poss, row["home_pts"]),
                (away, home, away_poss, row["away_pts"]),
            ):
                if poss > 0:
                    off_rows.append(
                        {
                            **{index[pid]: 1.0 for pid in attacking},
                            **{n + index[pid]: -1.0 for pid in defending},
                        }
                    )
                    off_y.append(100 * float(points) / poss)
                    off_w.append(poss)
                    for pid in attacking:
                        exposure[index[pid]] += poss
            game_poss = (home_poss + away_poss) / 2
            if game_poss > 0 and seconds > 0:
                pace_rows_.append({index[pid]: 1.0 for pid in home + away})
                pace_y.append(game_poss * 2880 / seconds)
                pace_w.append(seconds)
        offdef.add(_matrix(off_rows, 2 * n), np.asarray(off_y), np.asarray(off_w))
        pace.add(_matrix(pace_rows_, n), np.asarray(pace_y), np.asarray(pace_w))

    has_prior = any(po or pd_ for *_, po, pd_ in offdef_grid) or any(
        p for _, p in pace_grid
    )
    if has_prior:
        eff_columns = pd.MultiIndex.from_tuples(
            offdef_grid, names=["lambda_off", "lambda_def", "prior_off", "prior_def"]
        )
        pace_columns = pd.MultiIndex.from_tuples(
            pace_grid, names=["lambda_pace", "prior"]
        )
    else:  # 2_6 sweeps keep their original labels
        eff_columns = pd.MultiIndex.from_tuples(
            [item[:2] for item in offdef_grid], names=["lambda_off", "lambda_def"]
        )
        pace_columns = pd.Index([lp for lp, _ in pace_grid], name="lambda_pace")
    return (
        pd.DataFrame(eff_out, index=efficiency_rows.index, columns=eff_columns),
        pd.DataFrame(pace_out, index=pace_rows.index, columns=pace_columns),
    )


def squared_error_gain(
    rows: pd.DataFrame, better: np.ndarray, worse: np.ndarray
) -> tuple[float, float]:
    """Weighted mean drop in squared error from ``worse`` to ``better``
    predictions on the same rows, and its SE clustered by game."""
    actual = rows["actual"].to_numpy(float)
    return _clustered_mean(
        (actual - worse) ** 2 - (actual - better) ** 2,
        rows["weight"].to_numpy(float),
        rows["game_id"].to_numpy(),
    )
