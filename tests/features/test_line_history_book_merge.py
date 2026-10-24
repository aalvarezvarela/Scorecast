"""Tests for the tidy Aiven-tick-store Caesars/fanatics_sportsbook merge.

Mirrors test_book_combination.py for the wide pipeline. ``book`` is a row
value here, not a column-name suffix, so the merge is a relabel rather than a
column coalesce -- see the module docstring in book_merge.py for why a
row-level relabel upstream of build_snapshot_panel is sufficient and no
downstream consumer needs to change.
"""

from __future__ import annotations

import pandas as pd
from nba_ou.data_processing.line_history.book_merge import (
    merge_caesars_into_fanatics_ticks,
)


def test_caesars_relabelled_to_fanatics_when_fanatics_has_no_native_tick():
    ticks = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "market": ["totals", "totals"],
            "book": ["caesars", "betmgm"],
            "raw_line": [220.0, 221.0],
        }
    )
    out = merge_caesars_into_fanatics_ticks(ticks)
    assert set(out["book"]) == {"fanatics_sportsbook", "betmgm"}
    assert len(out) == len(ticks)


def test_caesars_dropped_when_fanatics_already_has_a_native_tick():
    ticks = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "market": ["totals", "totals"],
            "book": ["caesars", "fanatics_sportsbook"],
            "raw_line": [219.5, 220.5],
        }
    )
    out = merge_caesars_into_fanatics_ticks(ticks)
    assert out["book"].tolist() == ["fanatics_sportsbook"]
    assert out["raw_line"].tolist() == [220.5]


def test_merge_is_scoped_per_game_and_market_independently():
    """Same game, different markets: each market decides independently."""
    ticks = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g1"],
            "market": ["totals", "totals", "spread"],
            "book": ["caesars", "fanatics_sportsbook", "caesars"],
            "raw_line": [219.5, 220.5, -3.0],
        }
    )
    out = merge_caesars_into_fanatics_ticks(ticks)
    totals_books = set(out.loc[out["market"] == "totals", "book"])
    spread_books = set(out.loc[out["market"] == "spread", "book"])
    assert totals_books == {"fanatics_sportsbook"}
    assert spread_books == {"fanatics_sportsbook"}
    assert len(out) == 2


def test_other_books_are_untouched():
    ticks = pd.DataFrame(
        {
            "game_id": ["g1", "g1"],
            "market": ["totals", "totals"],
            "book": ["betmgm", "draftkings"],
            "raw_line": [220.0, 220.5],
        }
    )
    out = merge_caesars_into_fanatics_ticks(ticks)
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), ticks.reset_index(drop=True)
    )


def test_empty_frame_is_returned_unchanged():
    empty = pd.DataFrame(columns=["game_id", "market", "book", "raw_line"])
    out = merge_caesars_into_fanatics_ticks(empty)
    assert out.empty
