"""Line-dynamics features summarised over *previous* games.

Distinct from ``movement_features``, which describes the game being predicted.
Here the full open-to-close history of games that have already finished is
summarised and rolled forward. Those games closed before the snapshot, so their
complete movement history -- including their closing line -- was fully known at
T. This is the one place the closing line may legitimately be used.

The signal these carry is not the same as the size of a move. A team whose lines
get re-priced many times is one the market keeps changing its mind about, which
is information about uncertainty rather than about direction.

Naming follows the repo's convention so these sit alongside the existing
features: ``_BEFORE`` marks a leakage-safe pre-game value, ``_TEAM_HOME`` /
``_TEAM_AWAY`` the post-merge side, and ``_DIFF_BEFORE`` the home-minus-away
contrast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_ou.postgre_db.line_history_aiven.fetch import MARKET_TOTALS

from .movement_features import prepare_tick_history

#: Trailing game counts for the team-level rollups.
DEFAULT_WINDOWS: tuple[int, ...] = (5, 10, 20)

#: Metrics summarised per finished game, then rolled forward.
DYNAMICS_METRICS: tuple[str, ...] = (
    "N_MOVES_OPEN_TO_CLOSE",
    "ABS_MOVE_OPEN_TO_CLOSE",
    "MOVE_OPEN_TO_CLOSE",
    "N_PRICE_ONLY_TICKS",
    "LINE_STD_OPEN_TO_CLOSE",
)


def build_game_line_dynamics(
    ticks: pd.DataFrame, *, market: str = MARKET_TOTALS
) -> pd.DataFrame:
    """Summarise each finished game's complete open-to-close line history.

    Aggregated across books with a median so one book's dense ticking does not
    dominate the count.
    """
    if ticks.empty:
        return pd.DataFrame(columns=["game_id", *DYNAMICS_METRICS])

    working = prepare_tick_history(ticks)
    working = working[working["market"].eq(market)]
    if working.empty:
        return pd.DataFrame(columns=["game_id", *DYNAMICS_METRICS])

    per_book = working.groupby(["game_id", "book"], sort=False).agg(
        n_moves=("is_move", "sum"),
        n_price_only=("is_price_only", "sum"),
        abs_move=("line_delta", lambda s: s.abs().sum()),
        line_std=("line", "std"),
        opener=("line", "first"),
        closer=("line", "last"),
    )
    per_book["signed_move"] = per_book["closer"] - per_book["opener"]

    per_game = per_book.groupby("game_id").median(numeric_only=True)
    out = pd.DataFrame(
        {
            "N_MOVES_OPEN_TO_CLOSE": per_game["n_moves"],
            "ABS_MOVE_OPEN_TO_CLOSE": per_game["abs_move"],
            "MOVE_OPEN_TO_CLOSE": per_game["signed_move"],
            "N_PRICE_ONLY_TICKS": per_game["n_price_only"],
            "LINE_STD_OPEN_TO_CLOSE": per_game["line_std"].fillna(0.0),
        }
    )
    return out.reset_index()


def _team_long_frame(games: pd.DataFrame, dynamics: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game), so a team's history is a simple time series."""
    merged = games.merge(dynamics, on="game_id", how="inner")
    sides = []
    for side in ["team_home", "team_away"]:
        frame = merged[["game_id", "game_date", side, *DYNAMICS_METRICS]].copy()
        frame = frame.rename(columns={side: "team"})
        sides.append(frame)
    long_frame = pd.concat(sides, ignore_index=True)
    return long_frame.sort_values(["team", "game_date", "game_id"])


def add_prior_game_line_dynamics(
    games: pd.DataFrame,
    ticks: pd.DataFrame | None,
    *,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    market: str = MARKET_TOTALS,
    dynamics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-game team and league rollups of prior games' line dynamics.

    Returns one row per ``game_id`` with ``_BEFORE`` columns. Every rollup is
    shifted by one game within the team, so a game never contributes to its own
    feature.

    ``dynamics`` is ``build_game_line_dynamics``' frame, given instead of the
    ticks: each game's summary reads its own ticks only, so a tonight-only
    build passes the finished games' summaries from storage, plus tonight's.
    """
    if dynamics is None:
        dynamics = build_game_line_dynamics(ticks, market=market)
    if dynamics.empty or games.empty:
        return pd.DataFrame({"game_id": games.get("game_id", pd.Series(dtype=str))})

    long_frame = _team_long_frame(games, dynamics)
    grouped = long_frame.groupby("team", sort=False)

    for metric in DYNAMICS_METRICS:
        # shift(1) first, so the rolling window sees only prior games.
        prior = grouped[metric].shift(1)
        for window in windows:
            long_frame[f"{metric}_LAST_{window}_BEFORE"] = prior.groupby(
                long_frame["team"], sort=False
            ).transform(lambda s, w=window: s.rolling(w, min_periods=1).mean())
        long_frame[f"{metric}_SEASON_BEFORE"] = prior.groupby(
            long_frame["team"], sort=False
        ).transform(lambda s: s.expanding(min_periods=1).mean())

    feature_columns = [
        column for column in long_frame.columns if column.endswith("_BEFORE")
    ]

    # Back to one row per game, with the two sides side by side.
    home = games[["game_id", "team_home"]].merge(
        long_frame[["game_id", "team", *feature_columns]],
        left_on=["game_id", "team_home"],
        right_on=["game_id", "team"],
        how="left",
    )
    away = games[["game_id", "team_away"]].merge(
        long_frame[["game_id", "team", *feature_columns]],
        left_on=["game_id", "team_away"],
        right_on=["game_id", "team"],
        how="left",
    )

    out = games[["game_id"]].copy()
    for column in feature_columns:
        home_values = home[column].to_numpy()
        away_values = away[column].to_numpy()
        out[f"{column}_TEAM_HOME"] = home_values
        out[f"{column}_TEAM_AWAY"] = away_values
        out[f"{column.replace('_BEFORE', '')}_DIFF_BEFORE"] = home_values - away_values

    league = _league_wide_rollups(games, dynamics, windows=windows)
    return out.merge(league, on="game_id", how="left")


def _league_wide_rollups(
    games: pd.DataFrame, dynamics: pd.DataFrame, *, windows: tuple[int, ...]
) -> pd.DataFrame:
    """League-level movement regime, over strictly earlier game DATES.

    Date-based rather than row-based: at any snapshot before tip, the other
    games on the same slate have not closed either, so a same-date aggregate
    would use lines that did not yet exist.
    """
    merged = games[["game_id", "game_date"]].merge(dynamics, on="game_id", how="inner")
    if merged.empty:
        return pd.DataFrame({"game_id": games["game_id"]})

    daily = merged.groupby("game_date")[list(DYNAMICS_METRICS)].mean().sort_index()

    out = pd.DataFrame({"game_date": daily.index})
    for metric in DYNAMICS_METRICS:
        prior = daily[metric].shift(1)
        for window in windows:
            out[f"LEAGUE_{metric}_LAST_{window}_DAYS_BEFORE"] = (
                prior.rolling(window, min_periods=1).mean().to_numpy()
            )

    return (
        games[["game_id", "game_date"]]
        .merge(out, on="game_date", how="left")
        .drop(columns=["game_date"])
    )


def summarise_dynamics(dynamics: pd.DataFrame) -> pd.Series:
    """Compact description of the movement-count distribution, for sanity checks."""
    if dynamics.empty:
        return pd.Series(dtype="float64")
    return pd.Series(
        {
            "games": float(len(dynamics)),
            "median_moves": float(dynamics["N_MOVES_OPEN_TO_CLOSE"].median()),
            "mean_moves": float(dynamics["N_MOVES_OPEN_TO_CLOSE"].mean()),
            "median_abs_move": float(dynamics["ABS_MOVE_OPEN_TO_CLOSE"].median()),
            "pct_never_moved": float(
                (dynamics["N_MOVES_OPEN_TO_CLOSE"] == 0).mean() * 100.0
            ),
            "mean_signed_move": float(np.nanmean(dynamics["MOVE_OPEN_TO_CLOSE"])),
        }
    )
