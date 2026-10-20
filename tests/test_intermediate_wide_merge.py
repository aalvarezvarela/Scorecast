"""The wide snapshot blocks must join without duplicating a column.

``create_intermediate_line_df`` pivots the per-book and cross-book consensus
features into wide blocks and joins them on ``(game_id, snapshot_minutes)``. A
plain chain of ``merge`` calls is silent about overlap -- it renames a collision
to ``<name>_x`` / ``<name>_y`` and keeps both copies.

That is not hypothetical: the consensus block was pivoted and merged twice, so
``intermediate_line_data_20260412_7snap.csv`` carries 63 pairs of bit-identical
consensus columns (``SNAP_TOT_CONSENSUS_LINE_x`` equals ``..._y`` exactly). No
column-name check could see it -- both names are well-formed -- while the twins
inflate the feature count, split permutation importance between them and count
twice against ``cleaning.max_na_per_row``.
"""

from __future__ import annotations

import pandas as pd
import pytest
from nba_ou.create_training_data.create_intermediate_line_df import _merge_wide_parts


def _block(*columns: str) -> pd.DataFrame:
    frame = pd.DataFrame({"game_id": ["G1", "G1"], "snapshot_minutes": [30, 60]})
    for index, column in enumerate(columns):
        frame[column] = [float(index), float(index) + 0.5]
    return frame


def test_disjoint_blocks_join_into_one_row_per_key():
    merged = _merge_wide_parts([_block("A_LINE"), _block("B_LINE"), _block("C_LINE")])

    assert list(merged.columns) == [
        "game_id",
        "snapshot_minutes",
        "A_LINE",
        "B_LINE",
        "C_LINE",
    ]
    assert len(merged) == 2


def test_a_repeated_block_raises_instead_of_being_suffixed():
    consensus = _block("SNAP_TOT_CONSENSUS_LINE")

    with pytest.raises(ValueError, match="would be duplicated"):
        _merge_wide_parts([_block("A_LINE"), consensus, consensus])


def test_the_error_names_the_offending_columns():
    consensus = _block("SNAP_TOT_CONSENSUS_LINE", "SNAP_SPR_CONSENSUS_LINE")

    with pytest.raises(ValueError) as raised:
        _merge_wide_parts([consensus, consensus])

    message = str(raised.value)
    assert "SNAP_TOT_CONSENSUS_LINE" in message
    assert "SNAP_SPR_CONSENSUS_LINE" in message


def test_no_x_y_suffix_can_survive_the_join():
    """The observable symptom the guard exists to make impossible."""
    consensus = _block("SNAP_TOT_CONSENSUS_LINE")

    try:
        merged = _merge_wide_parts([_block("A_LINE"), consensus, consensus])
    except ValueError:
        return
    assert not [c for c in merged.columns if c.endswith(("_x", "_y"))]


def test_empty_blocks_are_skipped_not_merged():
    """A book with no quotes yields an empty frame; it must not break the join
    or contribute key-only columns."""
    merged = _merge_wide_parts([_block("A_LINE"), pd.DataFrame(), _block("B_LINE")])

    assert list(merged.columns) == ["game_id", "snapshot_minutes", "A_LINE", "B_LINE"]


def test_all_blocks_empty_yields_the_keys_only():
    merged = _merge_wide_parts([pd.DataFrame(), pd.DataFrame()])

    assert list(merged.columns) == ["game_id", "snapshot_minutes"]
    assert merged.empty


def test_keys_themselves_are_not_treated_as_a_collision():
    """Every block carries the join keys by construction -- that overlap is the
    point of the join, not an error."""
    merged = _merge_wide_parts([_block("A_LINE"), _block("B_LINE")])

    assert merged["game_id"].tolist() == ["G1", "G1"]
