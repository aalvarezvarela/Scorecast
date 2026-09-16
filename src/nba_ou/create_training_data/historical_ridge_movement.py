"""Walk-forward Ridge estimate of the anchor total's remaining line movement.

The output is a feature for the game model, not a fitted value on the games
used to train this Ridge. A game's closing snapshot supplies its label only
after tip-off, and all of that game's snapshots enter the fit together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_ou.config.odds_columns import total_line_col

EXPECTED_TOTAL_MOVE_COLUMN = "ODDS_LINE_HIST_RIDGE_EXPECTED_TOTAL_MOVE_TO_CLOSE"
RIDGE_ALPHA = 50.0
MIN_TRAIN_GAMES = 100
MAX_ABS_MOVE = 8.0


def _design_matrix(frame: pd.DataFrame, *, anchor: str) -> np.ndarray:
    """Use fixed, past-independent scales; unavailable windows have zero value."""
    book = f"ODDS_SNAP_TOT_{anchor.upper()}_"
    consensus = "ODDS_SNAP_TOT_CONSENSUS_"
    names_and_scales = (
        (book + "MOVE_FROM_OPEN", 3.0),
        (book + "MOVE_LAST_60", 1.5),
        (book + "MOVE_LAST_120", 2.0),
        (book + "LINE_AGE_MINUTES", 240.0),
        (book + "DEVIATION_FROM_CONSENSUS", 1.0),
        (book + "N_MOVES_SO_FAR", 4.0),
        (consensus + "N_BOOKS_QUOTING", 4.0),
        (book + "HAS_WINDOW_60", 1.0),
        (book + "HAS_WINDOW_120", 1.0),
    )
    required = (names_and_scales[0][0], names_and_scales[3][0])
    missing = [name for name in required if name not in frame]
    if missing:
        raise KeyError(f"Ridge movement needs snapshot columns: {missing}")

    columns = []
    for name, scale in names_and_scales:
        if name in frame:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
            values = np.nan_to_num(values / scale, nan=0.0, posinf=0.0, neginf=0.0)
            columns.append(np.clip(values, -10.0, 10.0))
        else:
            columns.append(np.zeros(len(frame)))
    horizon = pd.to_numeric(frame["TIME_TO_MATCH_MIN"], errors="raise").to_numpy(float)
    if not np.isfinite(horizon).all() or (horizon < 0).any():
        raise ValueError("Ridge movement requires non-negative snapshot horizons")
    h = np.log1p(horizon) / np.log1p(720.0)
    return np.column_stack(
        (np.ones(len(frame)), *columns, h, columns[0] * h, columns[4] * h)
    )


def add_historical_ridge_movement(
    frame: pd.DataFrame,
    closing_lines: pd.DataFrame,
    *,
    anchor: str,
) -> pd.DataFrame:
    """Add E[anchor total at close - anchor total now] to every snapshot.

    ``closing_lines`` contains the *same normalized tick-series* at horizon 0,
    keyed by GAME_ID. The independent closing-odds table is deliberately not
    used: it disagrees with the tick series for some games. For a prediction
    timestamp, only games with an earlier tip-off are eligible. The fit uses
    the current and immediately preceding seasons, with one unit of weight per
    game regardless of how many snapshots its grid contains.

    Until 100 labelled prior games exist, the neutral estimate is zero. A
    horizon-0 snapshot is also exactly zero by definition. All predictions are
    clipped to +/-8 points, as are training labels, to bound bad feed values.
    """
    if EXPECTED_TOTAL_MOVE_COLUMN in frame:
        raise ValueError(f"{EXPECTED_TOTAL_MOVE_COLUMN} already exists")
    required = {
        "GAME_ID",
        "SEASON_YEAR",
        "TIME_TO_MATCH_MIN",
        "TIPOFF_UTC",
        "SNAPSHOT_TS_UTC",
        total_line_col(anchor),
    }
    missing = sorted(required - set(frame))
    if missing:
        raise KeyError(f"Ridge movement needs columns: {missing}")
    if closing_lines["GAME_ID"].duplicated().any():
        raise ValueError("Ridge closing lines must have one row per game")
    if frame.duplicated(["GAME_ID", "TIME_TO_MATCH_MIN"]).any():
        raise ValueError("Ridge movement needs one row per game and snapshot")

    close = closing_lines.set_index("GAME_ID")["CLOSING_LINE"]
    close_values = pd.to_numeric(frame["GAME_ID"].map(close), errors="coerce").to_numpy(
        float
    )
    current = pd.to_numeric(frame[total_line_col(anchor)], errors="coerce").to_numpy(
        float
    )
    horizons = frame["TIME_TO_MATCH_MIN"].to_numpy(int)
    seasons = frame["SEASON_YEAR"].to_numpy(int)
    tips = pd.to_datetime(frame["TIPOFF_UTC"], utc=True, errors="raise")
    snapshots = pd.to_datetime(frame["SNAPSHOT_TS_UTC"], utc=True, errors="raise")
    if tips.isna().any() or snapshots.isna().any():
        raise ValueError("Ridge movement needs tip-off and snapshot timestamps")
    tip_ns = tips.astype("int64").to_numpy()
    snapshot_ns = snapshots.astype("int64").to_numpy()
    if (snapshot_ns > tip_ns).any():
        raise ValueError("A Ridge snapshot cannot be after its game's tip-off")

    x = _design_matrix(frame, anchor=anchor)
    y = np.clip(close_values - current, -MAX_ABS_MOVE, MAX_ABS_MOVE)
    valid_label = np.isfinite(y) & (horizons > 0)
    n_features = x.shape[1]

    # One contribution per completed game; each game has total weight one.
    events = []
    for game, positions in frame.groupby("GAME_ID", sort=False).indices.items():
        idx = np.asarray(positions)
        if len(set(seasons[idx])) != 1 or len(set(tip_ns[idx])) != 1:
            raise ValueError(f"Inconsistent season or tip-off for game {game}")
        labelled = idx[valid_label[idx]]
        if len(labelled) == 0:
            continue
        xx = x[labelled]
        yy = y[labelled]
        weight = 1.0 / len(labelled)
        events.append(
            (tip_ns[idx[0]], seasons[idx[0]], weight * xx.T @ xx, weight * xx.T @ yy)
        )
    events.sort(key=lambda item: item[0])

    gram = {}
    rhs = {}
    counts = {}
    versions = {}
    fitted = {}
    penalty = np.eye(n_features) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    predictions = np.zeros(len(frame), dtype=float)
    event_index = 0
    for row_index in np.argsort(snapshot_ns, kind="stable"):
        timestamp = snapshot_ns[row_index]
        while event_index < len(events) and events[event_index][0] < timestamp:
            _, season, gg, gy = events[event_index]
            gram.setdefault(season, np.zeros((n_features, n_features)))
            rhs.setdefault(season, np.zeros(n_features))
            gram[season] += gg
            rhs[season] += gy
            counts[season] = counts.get(season, 0) + 1
            versions[season] = versions.get(season, 0) + 1
            event_index += 1

        if horizons[row_index] == 0 or not np.isfinite(current[row_index]):
            continue
        season = seasons[row_index]
        n_train = counts.get(season - 1, 0) + counts.get(season, 0)
        if n_train < MIN_TRAIN_GAMES:
            continue
        version = (versions.get(season - 1, 0), versions.get(season, 0))
        if season not in fitted or fitted[season][0] != version:
            xx = gram.get(season - 1, 0) + gram.get(season, 0)
            xy = rhs.get(season - 1, 0) + rhs.get(season, 0)
            fitted[season] = (version, np.linalg.solve(xx + penalty, xy))
        predictions[row_index] = np.clip(
            x[row_index] @ fitted[season][1], -MAX_ABS_MOVE, MAX_ABS_MOVE
        )

    result = frame.copy()
    result[EXPECTED_TOTAL_MOVE_COLUMN] = predictions
    return result
