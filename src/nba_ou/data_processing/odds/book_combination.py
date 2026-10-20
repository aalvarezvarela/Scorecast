"""Reconcile the discontinued Caesars book with fanatics_sportsbook.

The sportsbook-review scrape no longer serves Caesars (see
``nba_ou.postgre_db.line_history_aiven.update.DISCONTINUED_BOOKS``), and
``fanatics_sportsbook`` only exists from the 2025 season. Left alone, that
makes fanatics_sportsbook a de facto season indicator: a column present
exactly when ``season_year == 2025``. Folding Caesars into fanatics_sportsbook
(fanatics values kept where present, NaNs backfilled from Caesars) gives one
continuously-covered book slot instead of two disjoint, season-correlated
ones.
"""

from __future__ import annotations

import pandas as pd

CAESARS_BOOK = "caesars"
FANATICS_BOOK = "fanatics_sportsbook"

#: (market prefix, suffixes) pairs describing every wide odds column family
#: that carries a per-book name, e.g. ``total_<book>_line_over``.
BOOK_COLUMN_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("total", ("line_over", "line_under", "price_over", "price_under")),
    ("spread", ("line_home", "line_away", "price_home", "price_away")),
    ("ml", ("price_home", "price_away")),
)


def resolve_combine_books(
    *,
    combine: bool | None,
    exclude_caesars: bool = False,
    exclude_fanatics: bool = False,
) -> bool:
    """Resolve the ``combine_fanatics_and_caesars`` tri-state into a plain bool.

    Combining is the default because it is the only option that fixes the
    season-leak without throwing data away. But "default on" must not turn a
    caller's explicit ``exclude_*`` into a hard error, so the parameter is
    tri-state:

    * ``None`` (default) -- combine, unless an exclusion was explicitly asked
      for, in which case the caller clearly wants the standalone book gone.
    * ``True`` -- combine, and reject a simultaneous exclusion as the genuine
      contradiction it is: there is no standalone book left to exclude.
    * ``False`` -- leave the books as they are.
    """
    if combine is None:
        return not (exclude_caesars or exclude_fanatics)
    if combine and (exclude_caesars or exclude_fanatics):
        raise ValueError(
            "combine_fanatics_and_caesars=True cannot be combined with "
            "exclude_caesars or exclude_fanatics: once Caesars is merged into "
            "fanatics_sportsbook there is no standalone book left to exclude. "
            "Leave combine_fanatics_and_caesars unset to let an explicit "
            "exclusion win."
        )
    return combine


def combine_caesars_and_fanatics(
    df_odds: pd.DataFrame,
    *,
    exclude_caesars: bool = False,
    combine_with_fanatics: bool = False,
) -> pd.DataFrame:
    """Reconcile Caesars and fanatics_sportsbook columns in a wide odds frame.

    ``combine_with_fanatics=True`` coalesces every
    ``<market>_fanatics_sportsbook_<suffix>`` column with its
    ``<market>_caesars_<suffix>`` counterpart (fanatics values kept where
    present, Caesars used to fill NaNs) and drops the standalone Caesars
    columns.

    ``exclude_caesars=True`` (with combining off) drops every Caesars-named
    column outright.

    The two are mutually exclusive: once Caesars has been merged into
    fanatics_sportsbook there is no standalone Caesars column left to
    exclude.
    """
    if exclude_caesars and combine_with_fanatics:
        raise ValueError(
            "exclude_caesars and combine_with_fanatics cannot both be True: "
            "once Caesars is merged into fanatics_sportsbook there is no "
            "standalone Caesars column left to exclude."
        )
    if not exclude_caesars and not combine_with_fanatics:
        return df_odds

    df_odds = df_odds.copy()

    if combine_with_fanatics:
        for market, suffixes in BOOK_COLUMN_FAMILIES:
            for suffix in suffixes:
                fanatics_col = f"{market}_{FANATICS_BOOK}_{suffix}"
                caesars_col = f"{market}_{CAESARS_BOOK}_{suffix}"
                if caesars_col not in df_odds.columns:
                    continue
                if fanatics_col in df_odds.columns:
                    df_odds[fanatics_col] = df_odds[fanatics_col].fillna(
                        df_odds[caesars_col]
                    )
                else:
                    df_odds[fanatics_col] = df_odds[caesars_col]
                df_odds = df_odds.drop(columns=[caesars_col])
        return df_odds

    caesars_cols = [
        f"{market}_{CAESARS_BOOK}_{suffix}"
        for market, suffixes in BOOK_COLUMN_FAMILIES
        for suffix in suffixes
    ]
    return df_odds.drop(columns=[col for col in caesars_cols if col in df_odds.columns])
