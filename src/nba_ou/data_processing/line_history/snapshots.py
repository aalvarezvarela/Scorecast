"""Build the as-of view of every market at each pre-game snapshot.

The store records line *changes*, not samples: there is no row for "the line at
14:00", only a row each time a book moved. So a snapshot is a
last-observation-carried-forward read, never an equality match, and the age of
the carried line is itself information -- measured on this data the median
carried line is 30-100 minutes old and the p90 is around five hours. That is why
``line_age_minutes`` is emitted alongside every quote rather than treated as
bookkeeping.

Side orientation, verified against the data rather than assumed:

* ``left`` is OVER for totals and the **away** side for spread and moneyline.
* ``right`` is UNDER / the **home** side.

Two independent measurements pin this. The residual of (home margin - left
spread) has std 13.46 against 19.96 for the opposite orientation, and devigged
moneyline probabilities average 0.551 on the right side versus 0.449 on the
left -- consistent with home-court advantage. Getting this backwards would
silently invert every spread and moneyline feature, so
``tests/test_line_history_snapshots.py`` pins it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from nba_ou.data_processing.odds.normalize_spread_lines import (
    spread_price_extreme_mask,
)
from nba_ou.postgre_db.line_history_aiven.fetch import (
    MARKET_MONEYLINE,
    MARKET_SPREAD,
    MARKET_TOTALS,
)

from .normalization import (
    MARGIN_SIGMA,
    TOTAL_SIGMA,
    center_two_way_line,
    devig_two_way,
)

#: Minutes before tip at which the market is sampled.
#:
#: Roughly geometric rather than evenly spaced, because line-movement
#: information decays in log time. Measured on the store: coverage is ~100% of
#: game-book pairs out to 12h and collapses to ~60% at 24h, which is why the
#: grid stops there. 30 and 60 are both kept even though they often resolve to
#: the same tick -- they bracket the most common betting window.
#:
#: ``0`` is the closing snapshot: bet as late as the market allows. It is not
#: literally tip-off -- ``fetch_pregame_ticks`` already refuses anything inside
#: ``DEFAULT_MIN_MINUTES_BEFORE_TIP``, so T=0 resolves to the last tick at least
#: five minutes out. That makes it this store's closing line, and it is the row
#: that puts the intermediate dataset on the same footing as the closing-line
#: dataset: same bet, same moment, different feature construction. Keep in mind
#: that closing-line value is ~0 by construction on those rows, so any CLV
#: measured over a pooled dataset is diluted by them.
#:
#: The grid is deliberately denser than any single model needs. Snapshots are
#: rows, so an unwanted horizon is removed with a filter on
#: ``TIME_TO_MATCH_MIN`` -- no rebuild required -- whereas adding one back means
#: regenerating the whole dataset. Over-sampling here is the cheap direction.
DEFAULT_SNAPSHOT_GRID: tuple[int, ...] = (
    0,
    30,
    60,
    120,
    180,
    240,
    300,
    360,
    480,
    720,
)

#: Per-market sigma for the -110 centering. Moneyline has no line to center.
MARKET_SIGMAS: dict[str, float] = {
    MARKET_TOTALS: TOTAL_SIGMA,
    MARKET_SPREAD: MARGIN_SIGMA,
}

GROUP_KEYS = ["game_id", "market", "book"]

PANEL_COLUMNS = [
    "game_id",
    "snapshot_minutes",
    "market",
    "book",
    "raw_line",
    "norm_line",
    "norm_minus_raw",
    "level",
    "price_left",
    "price_right",
    "fair_left",
    "fair_right",
    "fair_up",
    "overround",
    "line_age_minutes",
    "tick_minutes_before_tip",
    "n_ticks_so_far",
    "has_quote",
]


def market_level(frame: pd.DataFrame, line_column: str = "raw_line") -> pd.Series:
    """The canonical quantity each market *moves in*.

    Every movement, dispersion and path feature is computed from this one
    series, which is what keeps them comparable within a market and stops them
    mixing scales across one.

    * **Totals / spread** -- the raw line, i.e. the number actually on the
      board. Deliberately the raw rather than the centered line: a move is a
      thing that visibly happened, and mixing a centered "now" against a raw
      "then" produced path features that fell outside their own realised range.
      The pricing correction is not lost -- it is carried separately as
      ``norm_minus_raw`` and by the probability-movement family.
    * **Moneyline** -- the devigged HOME win probability. There is no line, so
      without this the market has no level at all: ``line_delta`` was NaN on
      every row, which silently zeroed every move count, reversal, window flag
      and dispersion figure for the entire market.

    Moneyline movement is therefore in probability units, not points. Nothing
    compares levels across markets, so the mixed units are safe -- but they are
    the reason no cross-market difference feature should be built on ``level``.
    """
    level = pd.to_numeric(frame[line_column], errors="coerce")
    moneyline = frame["market"].eq(MARKET_MONEYLINE)
    if moneyline.any():
        level = level.mask(moneyline, pd.to_numeric(frame["fair_right"]))
    return level


def market_up_probability(frame: pd.DataFrame) -> pd.Series:
    """Devigged probability of the side that wins when ``level`` goes UP.

    Needed because "left" does not mean the same thing across markets, so
    pairing every market's price movement with ``fair_left`` silently flipped
    the sign against ``level`` on two of the three:

    * **Totals** -- level is the total; up means OVER wins. That is ``fair_left``.
    * **Spread** -- level is the expected HOME margin; up means HOME covers.
      That is ``fair_right`` (left is the away side).
    * **Moneyline** -- level *is* the home probability, so up means HOME wins:
      ``fair_right`` again.

    Concretely, on a moneyline moving from +150/-170 to +120/-140 the home
    probability falls 0.0495 while the away probability rises by exactly the
    same amount. Reporting the level move as -0.0495 and the "probability move"
    as +0.0495 describes one event with two opposite signs.
    """
    up = pd.to_numeric(frame["fair_left"], errors="coerce")
    right_is_up = frame["market"].isin([MARKET_SPREAD, MARKET_MONEYLINE])
    if right_is_up.any():
        up = up.mask(right_is_up, pd.to_numeric(frame["fair_right"], errors="coerce"))
    return up


def resolve_line(ticks: pd.DataFrame) -> pd.Series:
    """Collapse the two stored line columns into one signed line per row.

    A valid total quotes the same number on both sides; a valid spread is
    mirrored. Either side alone is therefore sufficient, and taking the second
    as a fallback recovers rows where only one survived the load-time repairs
    (the spread price-bleed fix leaves a valid price with a NULL line).
    """
    left = pd.to_numeric(ticks["left_line"], errors="coerce")
    right = pd.to_numeric(ticks["right_line"], errors="coerce")

    mirrored = ticks["market"].eq(MARKET_SPREAD)
    right_as_left = right.where(~mirrored, -right)

    resolved = left.where(left.notna(), right_as_left)
    # Moneyline carries no line at all; make that explicit rather than relying
    # on both columns happening to be NULL.
    return resolved.mask(ticks["market"].eq(MARKET_MONEYLINE))


def _as_of(ticks_sorted: pd.DataFrame, snapshot_minutes: int) -> pd.DataFrame:
    """Last tick at least ``snapshot_minutes`` before tip, per game/market/book.

    ``>=`` is the leakage filter and it is deliberately inclusive of the
    boundary only: a tick exactly at the horizon was observable, a tick one
    minute later was not.
    """
    eligible = ticks_sorted[ticks_sorted["minutes_before_tip"] >= snapshot_minutes]
    if eligible.empty:
        return eligible

    grouped = eligible.groupby(GROUP_KEYS, sort=False)
    latest = grouped.tail(1).copy()
    latest["n_ticks_so_far"] = (
        grouped.size().reindex(
            pd.MultiIndex.from_frame(latest[GROUP_KEYS])
        ).to_numpy()
    )
    return latest


def build_snapshot_panel(
    ticks: pd.DataFrame,
    *,
    grid: tuple[int, ...] = DEFAULT_SNAPSHOT_GRID,
    normalize: bool | None = None,
    normalize_total_lines: bool = True,
    normalize_spread_lines: bool = True,
    null_extreme_spread_prices: bool = True,
    progress: str | None = None,
) -> pd.DataFrame:
    """Long panel: one row per (game, market, book, snapshot).

    ``progress`` labels a tqdm bar over the snapshot horizons; ``None`` runs
    silently.

    ``normalize_total_lines`` and ``normalize_spread_lines`` center each quote
    onto its -110/-110 equivalent. The raw line is kept regardless -- it is the
    one you could actually have bet, while the centered one is the one that is
    comparable across books and snapshots. ``normalize`` is the backward-
    compatible alias that sets both market-specific flags at once.
    """
    if normalize is not None:
        normalize_total_lines = normalize
        normalize_spread_lines = normalize

    if not grid:
        raise ValueError("grid must not be empty.")
    # 0 is allowed and means "as late as the store allows" -- the closing
    # snapshot. Negative is not: it would place the horizon *after* tip-off and
    # admit in-play ticks, which is a direct look-ahead that every downstream
    # column-name check would pass.
    if any(minutes < 0 for minutes in grid):
        raise ValueError(
            "snapshot horizons must be non-negative minutes before tip; a "
            f"negative horizon reads the market after tip-off. Got {grid}."
        )

    if ticks.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    working = ticks.copy()
    working["resolved_line"] = resolve_line(working)

    # Chronological within each series, so `tail(1)` is "most recent".
    working = working.sort_values(
        [*GROUP_KEYS, "minutes_before_tip"], ascending=[True, True, True, False]
    )

    frames = []
    for snapshot_minutes in tqdm(
        sorted(set(grid)), desc=progress, unit="snapshot", disable=progress is None
    ):
        latest = _as_of(working, snapshot_minutes)
        if latest.empty:
            continue
        latest = latest.assign(snapshot_minutes=int(snapshot_minutes))
        frames.append(latest)

    if not frames:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    panel = pd.concat(frames, ignore_index=True)

    panel = panel.rename(
        columns={
            "resolved_line": "raw_line",
            "left_price": "price_left",
            "right_price": "price_right",
            "minutes_before_tip": "tick_minutes_before_tip",
        }
    )

    # How stale the carried line already is at the snapshot instant.
    panel["line_age_minutes"] = (
        panel["tick_minutes_before_tip"] - panel["snapshot_minutes"]
    )

    if null_extreme_spread_prices:
        spread_rows = panel["market"].eq(MARKET_SPREAD)
        extreme_left = spread_rows & spread_price_extreme_mask(
            panel["price_left"], odds_format="american"
        )
        extreme_right = spread_rows & spread_price_extreme_mask(
            panel["price_right"], odds_format="american"
        )
        panel.loc[extreme_left, "price_left"] = np.nan
        panel.loc[extreme_right, "price_right"] = np.nan

    fair = devig_two_way(panel["price_left"], panel["price_right"])
    panel["fair_left"] = fair["fair_left"]
    panel["fair_right"] = fair["fair_right"]
    panel["overround"] = fair["overround"]

    panel["norm_line"] = np.nan
    normalize_by_market = {
        MARKET_TOTALS: normalize_total_lines,
        MARKET_SPREAD: normalize_spread_lines,
    }
    if any(normalize_by_market.values()):
        for market, sigma in MARKET_SIGMAS.items():
            rows = panel["market"].eq(market)
            if not normalize_by_market[market]:
                panel.loc[rows, "norm_line"] = panel.loc[rows, "raw_line"]
                continue
            if not rows.any():
                continue
            panel.loc[rows, "norm_line"] = center_two_way_line(
                panel.loc[rows, "raw_line"],
                panel.loc[rows, "price_left"],
                panel.loc[rows, "price_right"],
                sigma=sigma,
                # Totals: OVER wins above the line. Spread: the left side is
                # AWAY, which covers when the home margin lands BELOW it.
                left_wins_above=market != MARKET_SPREAD,
            )
        # Where a quote was one-sided the centering is undefined; falling back
        # to the raw line keeps the column populated and loses only the (small)
        # pricing correction, which is better than dropping the row.
        panel["norm_line"] = panel["norm_line"].fillna(panel["raw_line"])
    else:
        panel["norm_line"] = panel["raw_line"]

    # The half-tick a book has priced but not yet taken: it moved the price
    # while leaving the number alone. Invisible in the line itself.
    # Zero rather than NaN on the moneyline, which has no line to correct.
    panel["norm_minus_raw"] = (panel["norm_line"] - panel["raw_line"]).fillna(0.0)
    panel["level"] = market_level(panel)
    panel["fair_up"] = market_up_probability(panel)
    # Explicit availability, so "this book had no quote at T" is a value the
    # model can read rather than a NaN that row-level cleaning may act on.
    panel["has_quote"] = panel["level"].notna().astype(int)

    return panel[PANEL_COLUMNS].reset_index(drop=True)


def snapshot_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """Rows and distinct games per (market, snapshot).

    Used as an acceptance check: coverage must not fall off a cliff at the long
    horizons, because a snapshot that only exists for well-covered games is a
    biased sample rather than a longer lead time.
    """
    if panel.empty:
        return pd.DataFrame(columns=["market", "snapshot_minutes", "rows", "games"])
    return (
        panel.groupby(["market", "snapshot_minutes"])
        .agg(rows=("game_id", "size"), games=("game_id", "nunique"))
        .reset_index()
    )
