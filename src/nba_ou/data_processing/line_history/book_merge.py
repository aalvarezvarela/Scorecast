"""Fold discontinued Caesars ticks into fanatics_sportsbook.

Mirrors ``nba_ou.data_processing.odds.book_combination`` for the tidy Aiven
line-history tick store, where ``book`` is a row value rather than a
column-name suffix.
"""

from __future__ import annotations

import pandas as pd

CAESARS_BOOK = "caesars"
FANATICS_BOOK = "fanatics_sportsbook"


def merge_caesars_into_fanatics_ticks(ticks: pd.DataFrame) -> pd.DataFrame:
    """Relabel Caesars ticks as fanatics_sportsbook where fanatics has no native tick.

    For each ``(game_id, market)``, Caesars ticks are kept -- relabelled to
    fanatics_sportsbook -- only when no native fanatics_sportsbook tick exists
    for that (game_id, market); any remaining standalone Caesars ticks (where
    fanatics_sportsbook already has native coverage) are dropped. That makes
    fanatics_sportsbook one continuously-covered book instead of two disjoint,
    season-correlated ones.

    Every downstream consumer (``build_snapshot_panel``, ``add_movement_features``,
    ``aggregate_across_books``, ``add_book_deviation``, ``add_prior_game_line_dynamics``)
    treats ``book`` as an opaque groupby key, so this single row-level relabel
    upstream is sufficient -- none of those need to change.
    """
    if ticks.empty or "book" not in ticks.columns:
        return ticks

    has_native_fanatics = (
        ticks.loc[ticks["book"] == FANATICS_BOOK, ["game_id", "market"]]
        .drop_duplicates()
        .assign(_has_fanatics=True)
    )

    is_caesars = ticks["book"] == CAESARS_BOOK
    caesars = ticks.loc[is_caesars].merge(
        has_native_fanatics, on=["game_id", "market"], how="left"
    )
    relabelled = caesars.loc[caesars["_has_fanatics"].isna()].drop(
        columns=["_has_fanatics"]
    )
    relabelled = relabelled.assign(book=FANATICS_BOOK)

    return pd.concat([ticks.loc[~is_caesars], relabelled], ignore_index=True)
