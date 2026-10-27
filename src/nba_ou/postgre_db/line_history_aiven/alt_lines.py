"""Drop alternate-line quotes that the SBR/Covers feed files as the main line.

A book offers one main total/spread, priced near -110/-110, and a ladder of
alternate lines priced away from it (238.5 main -> 231.5 at -245/+172). The
line-history feed SBR and Covers share is meant to carry only the main line,
but for spells it also carries the ladder's updates as if they were the main
line's: audited 2026-10-27, 50-84% of the totals and spread ticks captured
2026-10-21 12:00Z .. 2026-10-22 ~20:30Z, every book, and 1-4% in April-June
2026, against <=0.05% before. The raw payload marks them in no way; MoviOdds'
bet365 and OddsLogic's books show the main line unmoved at the same minutes.
Moneyline has no ladder and is clean.

Such a tick says nothing about the main line, so it is dropped rather than
corrected: the book's previous tick stays its line as of that moment. Centering
it to -110 recovers only ~75% of the offset (the ladders are priced flatter
than the centering's sigma), and the flips would still read as moves.

A totals or spread tick is an alternate line when its prices are valid and

* **unbalanced**: the devigged probability of either side is outside
  ``BALANCED_PROBABILITY`` -- no main-line tick of 2025-26 was;
* **impossible**: the overround is outside ``OVERROUND_RANGE`` (both sides
  plus-money, or the prices of two different lines paired); or
* **a near alternate**: ``MIN_OFFSET_POINTS`` or more from the median of the
  other books' lines at that moment and priced for that offset (the side the
  offset favours at ``MIN_PRICE_LEAN`` or more). A book leading a real move is
  off the others but priced near -110, so it stays. The median reads only
  ticks not dropped yet, and the rule is repeated until it drops nothing more:
  several books' alternates quoted at once would otherwise hold the median up
  between them.

The rule needs each book's whole history: a reader that keeps only the last
ticks must drop alternates before cutting (``fetch_closing_ticks``), or a tail
of alternates would hide the main line before it.

Only seasons from ``FIRST_FILTERED_SEASON`` on are filtered: earlier seasons
feed the frozen 2_5 base, whose columns must not change, and carry ~0.1% of
such ticks (2025-26 playoffs 3.8%). ``filtered_fingerprints`` marks the
filtered seasons' fingerprints, so memos keyed on them are rebuilt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_ou.data_processing.line_history.normalization import devig_two_way

#: First season whose ticks are filtered.
FIRST_FILTERED_SEASON = 2026

#: Changes whenever the rule below changes: it is part of the filtered
#: seasons' fingerprints, so memos built from the earlier rule are not reused.
FILTER_VERSION = "alt-lines-v1"

#: Devigged probability range of a main line. 2025-26: 0.0% of the ticks
#: within 2 points of the other books fell outside it.
BALANCED_PROBABILITY = (0.40, 0.60)

#: Overround range of a real two-way quote. 2025-26 main-line ticks: 0.1st
#: percentile 0.036, 99.9th 0.089.
OVERROUND_RANGE = (0.02, 0.10)

#: Offset from the other books' median that a near alternate needs.
MIN_OFFSET_POINTS = 3.0

#: How far from 0.5 a near alternate's price must lean towards the side its
#: offset favours.
MIN_PRICE_LEAN = 0.03

#: Other books needed for a median.
MIN_OTHER_BOOKS = 2

#: Passes of the near-alternate rule. Each pass takes the median over the
#: ticks no pass has dropped yet, so alternates of several books quoted at
#: once stop propping each other up; the passes stop once one drops nothing.
MAX_NEAR_ALTERNATE_PASSES = 10

_TOTALS = "totals"
_SPREAD = "point_spread"


def _level(ticks: pd.DataFrame) -> pd.Series:
    """The line in the market's own terms: the total, or the away handicap
    (= the implied home margin) for spreads, from either stored side."""
    left = pd.to_numeric(ticks["left_line"], errors="coerce")
    right = pd.to_numeric(ticks["right_line"], errors="coerce")
    mirrored = ticks["market"].eq(_SPREAD)
    return left.where(left.notna(), right.where(~mirrored, -right))


def _other_books_median(
    ticks: pd.DataFrame, level: pd.Series, usable: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Per tick, the median of the other books' latest ``usable`` lines at its
    time, and how many books that median read."""
    median = pd.Series(np.nan, index=ticks.index)
    count = pd.Series(0, index=ticks.index)
    frame = pd.DataFrame(
        {
            "game_id": ticks["game_id"],
            "market": ticks["market"],
            "book": ticks["book"],
            "ts": ticks["line_ts"],
            "level": level.where(usable),
        }
    )
    for _, group in frame.groupby(["game_id", "market"], sort=False):
        wide = (
            group.dropna(subset=["level"])
            .pivot_table(index="ts", columns="book", values="level", aggfunc="last")
            .reindex(sorted(group["ts"].unique()))
            .ffill()
        )
        if wide.shape[1] <= MIN_OTHER_BOOKS - 1:
            continue
        books = {book: i for i, book in enumerate(wide.columns)}
        rows = wide.to_numpy(dtype="float64")[wide.index.get_indexer(group["ts"])]
        own = group["book"].map(books)
        has_own = own.notna().to_numpy()
        rows[np.flatnonzero(has_own), own[has_own].astype(int).to_numpy()] = np.nan
        others = (~np.isnan(rows)).sum(axis=1)
        with np.errstate(all="ignore"):
            values = np.where(
                others >= MIN_OTHER_BOOKS,
                np.nanmedian(np.where(others[:, None] > 0, rows, 0.0), axis=1),
                np.nan,
            )
        median.loc[group.index] = values
        count.loc[group.index] = others
    return median, count


def alternate_line_mask(ticks: pd.DataFrame) -> pd.Series:
    """True for every tick of a filtered season that is an alternate line."""
    mask = pd.Series(False, index=ticks.index)
    if ticks.empty:
        return mask
    season = pd.to_numeric(ticks["season_year"], errors="coerce")
    scope = season.ge(FIRST_FILTERED_SEASON) & ticks["market"].isin([_TOTALS, _SPREAD])
    if not scope.any():
        return mask
    rows = ticks[scope]
    left_price = pd.to_numeric(rows["left_price"], errors="coerce")
    right_price = pd.to_numeric(rows["right_price"], errors="coerce")
    # American prices are at least 100 in size; anything else is not a price
    # (the old spread price-bleed), so the tick cannot be judged by its prices.
    priced = left_price.abs().ge(100) & right_price.abs().ge(100)
    fair = devig_two_way(left_price.where(priced), right_price.where(priced))
    fair_left = fair["fair_left"]
    low, high = BALANCED_PROBABILITY
    unbalanced = priced & (fair_left.lt(low) | fair_left.gt(high))
    impossible = priced & (
        fair["overround"].lt(OVERROUND_RANGE[0])
        | fair["overround"].gt(OVERROUND_RANGE[1])
    )

    level = _level(rows)
    # Totals: OVER is left, so a line above the others makes OVER the side
    # less likely. Spreads: the level is the away handicap, so a line above
    # the others makes the AWAY side (left) more likely.
    lean = (fair_left - 0.5).where(rows["market"].eq(_SPREAD), 0.5 - fair_left)
    leaning = priced & lean.abs().ge(MIN_PRICE_LEAN)
    dropped = unbalanced | impossible
    for _ in range(MAX_NEAR_ALTERNATE_PASSES):
        median, _ = _other_books_median(rows, level, usable=level.notna() & ~dropped)
        offset = level - median
        near_alternate = (
            leaning
            & offset.abs().ge(MIN_OFFSET_POINTS)
            & (np.sign(lean) == np.sign(offset))
        )
        if not (near_alternate & ~dropped).any():
            break
        dropped = dropped | near_alternate
    mask.loc[rows.index] = dropped
    return mask


def drop_alternate_lines(ticks: pd.DataFrame) -> pd.DataFrame:
    """``ticks`` without its alternate-line quotes (see the module docstring).

    Keeps the frame's attributes and row order; prints how many it dropped.
    """
    mask = alternate_line_mask(ticks)
    if not mask.any():
        return ticks
    attrs = dict(ticks.attrs)
    out = ticks[~mask].reset_index(drop=True)
    out.attrs = attrs
    scope = pd.to_numeric(ticks["season_year"], errors="coerce").ge(
        FIRST_FILTERED_SEASON
    ) & ticks["market"].isin([_TOTALS, _SPREAD])
    print(
        f"  alternate lines: dropped {int(mask.sum()):,} of "
        f"{int(scope.sum()):,} totals/spread ticks "
        f"(seasons {FIRST_FILTERED_SEASON}+)"
    )
    return out


def filtered_fingerprints(fingerprints: dict[int, list]) -> dict[int, list]:
    """``fingerprints`` with the filter's version on each filtered season."""
    return {
        season: (
            [*fingerprint, FILTER_VERSION]
            if int(season) >= FIRST_FILTERED_SEASON
            else fingerprint
        )
        for season, fingerprint in fingerprints.items()
    }
