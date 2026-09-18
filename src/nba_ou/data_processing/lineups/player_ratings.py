"""Walk-forward, regularized player offense/defense and pace ratings.

The defensive coefficient is positive for *points prevented*. Consequently its
design entries are -1 when the player defends. This matches the game projection
formula ``league_ortg + offense - opponent_defense``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import cg, spsolve


def _poss(row, side: str) -> float:
    return (
        float(row[f"{side}_fga"])
        + 0.44 * float(row[f"{side}_fta"])
        - float(row[f"{side}_oreb"])
        + float(row[f"{side}_tov"])
    )


def _solve(matrix: sparse.csr_matrix, rhs: np.ndarray, alpha: float,
           previous: np.ndarray | None = None) -> np.ndarray:
    if not np.any(rhs):
        return np.zeros_like(rhs)
    regularized = matrix + alpha * sparse.eye(matrix.shape[0], format="csr")
    result, info = cg(regularized, rhs, x0=previous, rtol=1e-7, maxiter=500)
    if info:
        result = spsolve(regularized, rhs)
    return np.asarray(result, dtype=float)


class _NormalEquations:
    """Sufficient statistics for an exponentially decayed weighted ridge."""

    def __init__(self, n_features: int) -> None:
        self.a = sparse.csr_matrix((n_features, n_features), dtype=float)
        self.b_raw = np.zeros(n_features)
        self.x_sum = np.zeros(n_features)
        self.weighted_y = 0.0
        self.weight = 0.0
        self.previous: np.ndarray | None = None

    def decay(self, factor: float) -> None:
        self.a *= factor
        self.b_raw *= factor
        self.x_sum *= factor
        self.weighted_y *= factor
        self.weight *= factor

    def add(self, x: sparse.csr_matrix, y: np.ndarray, w: np.ndarray) -> None:
        if not len(y):
            return
        wx = x.multiply(w[:, None])
        self.a = (self.a + x.T @ wx).tocsr()
        self.b_raw += np.asarray(x.T @ (w * y)).ravel()
        self.x_sum += np.asarray(x.T @ w).ravel()
        self.weighted_y += float(w @ y)
        self.weight += float(w.sum())

    def solve(self, alpha: float) -> tuple[np.ndarray, float]:
        mean = self.weighted_y / self.weight if self.weight else 0.0
        rhs = self.b_raw - mean * self.x_sum
        self.previous = _solve(self.a, rhs, alpha, self.previous)
        return self.previous, mean


def _matrix(rows: list[dict[int, float]], n_features: int) -> sparse.csr_matrix:
    return sparse.csr_matrix(
        (
            [value for row in rows for value in row.values()],
            ([i for i, row in enumerate(rows) for _ in row],
             [key for row in rows for key in row]),
        ),
        shape=(len(rows), n_features),
    )


def _lineup(value: object) -> tuple[str, ...]:
    return tuple(str(int(player)) for player in value)


def walk_forward_player_ratings(
    stints: pd.DataFrame,
    target_dates: Iterable,
    *,
    lambda_offdef: float,
    lambda_pace: float,
    half_life_days: float = 180.0,
) -> pd.DataFrame:
    """Emit ratings on each target date using game dates strictly before it.

    Lambda values must be selected by the caller's walk-forward validation;
    there is intentionally no untuned production default. Unseen players have
    implicit zero ratings and do not appear until they have prior stints.
    """
    if lambda_offdef <= 0 or lambda_pace <= 0 or half_life_days <= 0:
        raise ValueError("Ridge lambdas and half-life must be positive")
    required = {"game_date", "home_lineup", "away_lineup", "start_ds", "end_ds",
                "home_pts", "away_pts"}
    required |= {f"{side}_{stat}" for side in ("home", "away")
                 for stat in ("fga", "fta", "oreb", "tov")}
    if missing := required - set(stints):
        raise ValueError(f"Missing stint columns: {sorted(missing)}")
    history = stints.copy()
    history["game_date"] = pd.to_datetime(history.game_date).dt.normalize()
    history["home_lineup"] = history.home_lineup.map(_lineup)
    history["away_lineup"] = history.away_lineup.map(_lineup)
    players = sorted(
        {pid for side in ("home", "away") for lineup in history[f"{side}_lineup"]
         for pid in lineup}, key=int
    )
    index = {pid: i for i, pid in enumerate(players)}
    n = len(players)
    offdef = _NormalEquations(2 * n)
    pace = _NormalEquations(n)
    exposure = np.zeros(n)
    requested = {pd.Timestamp(date).normalize() for date in target_dates}
    all_dates = sorted(requested | set(history.game_date))
    grouped = {date: frame for date, frame in history.groupby("game_date", sort=False)}
    output = []
    previous_date = None
    latest_fit_date = pd.NaT
    for day in all_dates:
        if previous_date is not None:
            factor = 0.5 ** ((day - previous_date).days / half_life_days)
            offdef.decay(factor)
            pace.decay(factor)
            exposure *= factor
        previous_date = day
        if day in requested:
            beta, league_ortg = offdef.solve(lambda_offdef)
            pace_beta, league_pace = pace.solve(lambda_pace)
            for pid, i in index.items():
                if exposure[i] > 0:
                    output.append(dict(
                        as_of_date=day, player_id=pid,
                        o_rating=beta[i], d_rating=beta[n + i],
                        pace_rating=pace_beta[i], poss_weight=exposure[i],
                        league_ortg=league_ortg, league_pace=league_pace,
                        fit_max_game_date=latest_fit_date,
                    ))
        if day not in grouped:
            continue
        off_rows, off_y, off_w = [], [], []
        pace_rows, pace_y, pace_w = [], [], []
        for row in grouped[day].to_dict("records"):
            home, away = row["home_lineup"], row["away_lineup"]
            if len(home) != 5 or len(away) != 5:
                raise ValueError("Every stint requires two five-player lineups")
            seconds = (row["end_ds"] - row["start_ds"]) / 10
            if seconds <= 0:
                raise ValueError("Nonpositive stint duration")
            home_poss, away_poss = _poss(row, "home"), _poss(row, "away")
            for attacking, defending, poss, points in (
                (home, away, home_poss, row["home_pts"]),
                (away, home, away_poss, row["away_pts"]),
            ):
                if poss > 0:
                    off_rows.append({
                        **{index[pid]: 1.0 for pid in attacking},
                        **{n + index[pid]: -1.0 for pid in defending},
                    })
                    off_y.append(100 * float(points) / poss)
                    off_w.append(poss)
                    for pid in attacking:
                        exposure[index[pid]] += poss
            game_poss = (home_poss + away_poss) / 2
            if game_poss > 0:
                pace_rows.append({index[pid]: 1.0 for pid in home + away})
                pace_y.append(game_poss * 2880 / seconds)
                pace_w.append(seconds)
        offdef.add(_matrix(off_rows, 2 * n), np.asarray(off_y), np.asarray(off_w))
        pace.add(_matrix(pace_rows, n), np.asarray(pace_y), np.asarray(pace_w))
        latest_fit_date = day
    return pd.DataFrame(output)
