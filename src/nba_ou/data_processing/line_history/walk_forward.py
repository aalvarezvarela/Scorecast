"""Walk-forward least squares over completed games.

Every estimator here answers the same question for a row at snapshot time T:
fitted on which rows? Only rows of games that **tipped off strictly before T**,
from T's own season and the one before it. A game's label may depend on its
closing line, which exists only at tip-off, so the tip-off -- not the snapshot
time of the labelled row -- is when a game enters the fit.

Each game carries total weight one, however many of its snapshots are labelled,
so a game quoted at ten horizons does not count ten times more than one quoted
at three.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class WalkForwardFit:
    """Per-row output of :func:`walk_forward_least_squares`.

    ``predictions[i]`` is ``x[i] @ coefficients[i]``; both are NaN where the row
    was not predicted (masked out, or too little history).
    """

    predictions: np.ndarray
    coefficients: np.ndarray


def walk_forward_least_squares(
    x: np.ndarray,
    y: np.ndarray,
    *,
    game_ids: pd.Series,
    seasons: np.ndarray,
    tip_ns: np.ndarray,
    snapshot_ns: np.ndarray,
    label_mask: np.ndarray,
    predict_mask: np.ndarray,
    alpha: float,
    unpenalized: tuple[int, ...] = (),
    min_train_games: int = 0,
    min_train_rows: int = 0,
    progress: str | None = None,
) -> WalkForwardFit:
    """Fit ``y ~ x`` (ridge penalty ``alpha``) walk-forward, one fit per row.

    ``label_mask`` selects the rows that may train; ``predict_mask`` the rows
    that receive a prediction. ``unpenalized`` lists design columns exempt from
    the penalty (an intercept). A row is predicted only once the current and
    previous seasons hold at least ``min_train_games`` games and
    ``min_train_rows`` labelled rows. The solve is repeated only when the
    training set has changed since the previous row of the same season.

    ``progress`` labels a tqdm bar over the rows; ``None`` runs silently.
    """
    n_rows, n_features = x.shape
    events = []
    for game, positions in game_ids.groupby(game_ids, sort=False).indices.items():
        idx = np.asarray(positions)
        if len(set(seasons[idx])) != 1 or len(set(tip_ns[idx])) != 1:
            raise ValueError(f"Inconsistent season or tip-off for game {game}")
        labelled = idx[label_mask[idx]]
        if len(labelled) == 0:
            continue
        xx = x[labelled]
        yy = y[labelled]
        weight = 1.0 / len(labelled)
        events.append(
            (
                tip_ns[idx[0]],
                seasons[idx[0]],
                weight * xx.T @ xx,
                weight * xx.T @ yy,
                len(labelled),
            )
        )
    events.sort(key=lambda item: item[0])

    gram: dict = {}
    rhs: dict = {}
    counts: dict = {}
    row_counts: dict = {}
    versions: dict = {}
    fitted: dict = {}
    penalty = np.eye(n_features) * alpha
    for column in unpenalized:
        penalty[column, column] = 0.0
    predictions = np.full(n_rows, np.nan)
    coefficients = np.full((n_rows, n_features), np.nan)
    event_index = 0
    for row_index in tqdm(
        np.argsort(snapshot_ns, kind="stable"),
        desc=progress,
        unit="row",
        mininterval=1.0,
        disable=progress is None,
    ):
        timestamp = snapshot_ns[row_index]
        while event_index < len(events) and events[event_index][0] < timestamp:
            _, season, gg, gy, n_labelled = events[event_index]
            gram.setdefault(season, np.zeros((n_features, n_features)))
            rhs.setdefault(season, np.zeros(n_features))
            gram[season] += gg
            rhs[season] += gy
            counts[season] = counts.get(season, 0) + 1
            row_counts[season] = row_counts.get(season, 0) + n_labelled
            versions[season] = versions.get(season, 0) + 1
            event_index += 1

        if not predict_mask[row_index]:
            continue
        season = seasons[row_index]
        n_games = counts.get(season - 1, 0) + counts.get(season, 0)
        n_labelled_rows = row_counts.get(season - 1, 0) + row_counts.get(season, 0)
        if (
            n_games == 0
            or n_games < min_train_games
            or n_labelled_rows < min_train_rows
        ):
            continue
        version = (versions.get(season - 1, 0), versions.get(season, 0))
        if season not in fitted or fitted[season][0] != version:
            xx = gram.get(season - 1, 0) + gram.get(season, 0)
            xy = rhs.get(season - 1, 0) + rhs.get(season, 0)
            try:
                solution = np.linalg.solve(xx + penalty, xy)
            except np.linalg.LinAlgError:
                solution = None
            fitted[season] = (version, solution)
        solution = fitted[season][1]
        if solution is None:
            continue
        coefficients[row_index] = solution
        predictions[row_index] = x[row_index] @ solution
    return WalkForwardFit(predictions=predictions, coefficients=coefficients)
