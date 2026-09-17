"""Walk-forward Ridge estimates of the anchor book's remaining line movement.

The output is a feature for the game model, not a fitted value on the games
used to train this Ridge. A game's closing snapshot supplies its label only
after tip-off, and all of that game's snapshots enter the fit together.

One estimator per market, in that market's own level units:

* totals -- points of total;
* spread -- points of expected home margin;
* moneyline -- de-vigged home win probability.

The spread design adds cross-market inputs (the moneyline-implied margin gap
and the recent total and moneyline moves). Measured on 2021-2025, a model with
those inputs beat an own-market one for the spread in every season, while
totals and moneyline gained nothing from them
(``docs/intermediate_market_dynamics_plan.md`` section 1.6). The gap also enters
non-linearly -- its absolute value, a clipped copy, and its product with the
spread's size -- which is what brought the linear spread estimate level with a
gradient-boosted one (section 7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_ou.config.odds_columns import total_line_col
from nba_ou.data_processing.line_history.walk_forward import (
    walk_forward_least_squares,
)
from nba_ou.postgre_db.line_history_aiven.fetch import (
    MARKET_MONEYLINE,
    MARKET_SPREAD,
    MARKET_TOTALS,
)

EXPECTED_TOTAL_MOVE_COLUMN = "ODDS_LINE_HIST_RIDGE_EXPECTED_TOTAL_MOVE_TO_CLOSE"
EXPECTED_SPREAD_MOVE_COLUMN = "ODDS_LINE_HIST_RIDGE_EXPECTED_SPREAD_MOVE_TO_CLOSE"
EXPECTED_ML_MOVE_COLUMN = "ODDS_LINE_HIST_RIDGE_EXPECTED_ML_MOVE_TO_CLOSE"
RIDGE_ALPHA = 50.0
MIN_TRAIN_GAMES = 100
MAX_ABS_MOVE = 8.0


@dataclass(frozen=True)
class _MarketDesign:
    short: str
    output_column: str
    max_abs_move: float
    #: Anchor-book snapshot features and fixed scales, in design-matrix order.
    #: The first entry is interacted with the horizon, as is the fifth.
    own_scales: tuple[tuple[str, float], ...]
    #: Full column templates (``{anchor}`` is the upper-case book) and scales.
    cross_scales: tuple[tuple[str, float], ...] = ()
    #: Add the non-linear terms of the moneyline-implied margin gap.
    gap_terms: bool = False


def _own_scales(
    move_from_open: float, move_60: float, move_120: float, deviation: float
) -> tuple[tuple[str, float], ...]:
    return (
        ("MOVE_FROM_OPEN", move_from_open),
        ("MOVE_LAST_60", move_60),
        ("MOVE_LAST_120", move_120),
        ("LINE_AGE_MINUTES", 240.0),
        ("DEVIATION_FROM_CONSENSUS", deviation),
        ("N_MOVES_SO_FAR", 4.0),
        ("CONSENSUS:N_BOOKS_QUOTING", 4.0),
        ("HAS_WINDOW_60", 1.0),
        ("HAS_WINDOW_120", 1.0),
    )


MARKET_DESIGNS: dict[str, _MarketDesign] = {
    MARKET_TOTALS: _MarketDesign(
        short="TOT",
        output_column=EXPECTED_TOTAL_MOVE_COLUMN,
        max_abs_move=MAX_ABS_MOVE,
        own_scales=_own_scales(3.0, 1.5, 2.0, 1.0),
    ),
    MARKET_SPREAD: _MarketDesign(
        short="SPR",
        output_column=EXPECTED_SPREAD_MOVE_COLUMN,
        max_abs_move=5.0,
        own_scales=_own_scales(1.5, 1.0, 1.0, 0.5),
        cross_scales=(
            ("ODDS_SNAP_XMKT_{anchor}_ML_MARGIN_MINUS_SPREAD", 1.0),
            ("ODDS_SNAP_XMKT_{anchor}_ML_MARGIN_MINUS_SPREAD_MOVE_60", 0.5),
            ("ODDS_SNAP_TOT_{anchor}_MOVE_LAST_60", 1.5),
            ("ODDS_SNAP_ML_{anchor}_MOVE_LAST_60", 0.02),
        ),
        gap_terms=True,
    ),
    MARKET_MONEYLINE: _MarketDesign(
        short="ML",
        output_column=EXPECTED_ML_MOVE_COLUMN,
        max_abs_move=0.15,
        own_scales=_own_scales(0.03, 0.015, 0.02, 0.01),
    ),
}

EXPECTED_MOVE_COLUMNS: dict[str, str] = {
    market: design.output_column for market, design in MARKET_DESIGNS.items()
}


def current_line_column(market: str, anchor: str) -> str:
    """The column holding the anchor's line at the snapshot, in label units.

    Totals keep the canonical anchor column (the normalised snapshot line). The
    spread uses the anchor's normalised expected home margin and the moneyline
    its de-vigged home probability -- the same quantities the closing labels are
    read in.
    """
    if market == MARKET_TOTALS:
        return total_line_col(anchor)
    if market == MARKET_SPREAD:
        return f"ODDS_SNAP_SPR_{anchor.upper()}_NORM_LINE"
    if market == MARKET_MONEYLINE:
        return f"ODDS_SNAP_ML_{anchor.upper()}_LEVEL"
    raise ValueError(f"Unknown market {market!r}")


def ridge_input_columns(columns, market: str, anchor: str) -> list[str]:
    """The snapshot columns a market's design can read, from ``columns``."""
    design = MARKET_DESIGNS[market]
    prefixes = (
        f"ODDS_SNAP_{design.short}_{anchor.upper()}_",
        f"ODDS_SNAP_{design.short}_CONSENSUS_",
    )
    cross = {
        template.format(anchor=anchor.upper()) for template, _ in design.cross_scales
    }
    return [c for c in columns if c.startswith(prefixes) or c in cross]


def _scaled(frame: pd.DataFrame, name: str, scale: float) -> np.ndarray:
    if name not in frame:
        return np.zeros(len(frame))
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
    values = np.nan_to_num(values / scale, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(values, -10.0, 10.0)


def _gap_terms(frame: pd.DataFrame, *, anchor: str) -> list[np.ndarray]:
    """|gap|, gap clipped to +/-1, gap x |spread| / 7 and |spread| / 7.

    The gap is the moneyline-implied home margin minus the spread; its pull on
    the spread is not linear and depends on how lopsided the game is.
    """
    gap = _scaled(frame, f"ODDS_SNAP_XMKT_{anchor.upper()}_ML_MARGIN_MINUS_SPREAD", 1.0)
    size = np.abs(_scaled(frame, f"ODDS_SNAP_SPR_{anchor.upper()}_LEVEL", 7.0))
    return [
        np.abs(gap),
        np.clip(gap, -1.0, 1.0),
        np.clip(gap * size, -10.0, 10.0),
        size,
    ]


def _design_matrix(
    frame: pd.DataFrame, *, anchor: str, market: str = MARKET_TOTALS
) -> np.ndarray:
    """Use fixed, past-independent scales; unavailable windows have zero value."""
    design = MARKET_DESIGNS[market]
    book = f"ODDS_SNAP_{design.short}_{anchor.upper()}_"
    consensus = f"ODDS_SNAP_{design.short}_CONSENSUS_"
    names_and_scales = [
        (
            (
                consensus + name.removeprefix("CONSENSUS:")
                if name.startswith("CONSENSUS:")
                else book + name
            ),
            scale,
        )
        for name, scale in design.own_scales
    ]
    required = (names_and_scales[0][0], names_and_scales[3][0])
    missing = [name for name in required if name not in frame]
    if missing:
        raise KeyError(f"Ridge movement needs snapshot columns: {missing}")

    columns = [_scaled(frame, name, scale) for name, scale in names_and_scales]
    columns += [
        _scaled(frame, template.format(anchor=anchor.upper()), scale)
        for template, scale in design.cross_scales
    ]
    if design.gap_terms:
        columns += _gap_terms(frame, anchor=anchor)
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
    market: str = MARKET_TOTALS,
) -> pd.DataFrame:
    """Add E[anchor line at close - anchor line now] to every snapshot.

    ``closing_lines`` contains the *same tick-series* value at horizon 0, keyed
    by GAME_ID, in the units of :func:`current_line_column`. The independent
    closing-odds table is deliberately not used: it disagrees with the tick
    series for some games. For a prediction timestamp, only games with an
    earlier tip-off are eligible. The fit uses the current and immediately
    preceding seasons, with one unit of weight per game regardless of how many
    snapshots its grid contains.

    Until 100 labelled prior games exist, the neutral estimate is zero. A
    horizon-0 snapshot is also exactly zero by definition. Predictions and
    training labels are clipped to the market's limit (8 total points, 5 spread
    points, 0.15 probability) to bound bad feed values.
    """
    if market not in MARKET_DESIGNS:
        raise ValueError(f"Unknown market {market!r}")
    design = MARKET_DESIGNS[market]
    output = design.output_column
    line_column = current_line_column(market, anchor)
    if output in frame:
        raise ValueError(f"{output} already exists")
    required = {
        "GAME_ID",
        "SEASON_YEAR",
        "TIME_TO_MATCH_MIN",
        "TIPOFF_UTC",
        "SNAPSHOT_TS_UTC",
        line_column,
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
    current = pd.to_numeric(frame[line_column], errors="coerce").to_numpy(float)
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

    limit = design.max_abs_move
    x = _design_matrix(frame, anchor=anchor, market=market)
    y = np.clip(close_values - current, -limit, limit)
    fit = walk_forward_least_squares(
        x,
        y,
        game_ids=frame["GAME_ID"].reset_index(drop=True),
        seasons=seasons,
        tip_ns=tip_ns,
        snapshot_ns=snapshot_ns,
        label_mask=np.isfinite(y) & (horizons > 0),
        predict_mask=(horizons != 0) & np.isfinite(current),
        alpha=RIDGE_ALPHA,
        unpenalized=(0,),
        min_train_games=MIN_TRAIN_GAMES,
    )
    predictions = np.clip(np.nan_to_num(fit.predictions, nan=0.0), -limit, limit)

    result = frame.copy()
    result[output] = predictions
    return result
