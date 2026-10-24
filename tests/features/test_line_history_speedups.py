"""The line-history speed-ups must not change a single value.

Each optimised function is checked, bit for bit, against a verbatim copy of the
implementation it replaced.
"""

from __future__ import annotations

import pandas as pd
import pytest
from nba_ou.data_processing.line_history import movement_features
from nba_ou.data_processing.line_history.movement_features import (
    _first_nonzero_sign,
    _history_aggregates,
    _reversal_counts,
    add_movement_features,
    prepare_tick_history,
)
from nba_ou.data_processing.line_history.snapshots import (
    GROUP_KEYS,
    build_snapshot_panel,
)

from .frame_identity import assert_identical, random_ticks

GRID = (0, 30, 60, 120, 360, 720)


# ---- movement_features._history_aggregates ---------------------------------


def _history_aggregates_before(working: pd.DataFrame, snapshot_minutes: int):
    """The implementation with per-group Python callables inside ``agg``."""
    eligible = working[working["minutes_before_tip"] >= snapshot_minutes]
    if eligible.empty:
        return pd.DataFrame()

    grouped = eligible.groupby(GROUP_KEYS, sort=False)

    aggregates = grouped.agg(
        n_ticks_total=("line_ts", "size"),
        n_moves_so_far=("is_move", "sum"),
        n_price_only_ticks=("is_price_only", "sum"),
        n_distinct_levels=("line", "nunique"),
        line_max_so_far=("line", "max"),
        line_min_so_far=("line", "min"),
        line_std_so_far=("line", "std"),
        first_minutes_before_tip=("minutes_before_tip", "max"),
        first_move_direction=("move_sign", _first_nonzero_sign),
        abs_move_total=("line_delta", lambda s: s.abs().sum()),
    )

    labelled_opener = (
        eligible["line"]
        .where(eligible["is_opener"])
        .groupby([eligible[key] for key in GROUP_KEYS], sort=False)
        .first()
    )
    aggregates["opener_line"] = labelled_opener.reindex(aggregates.index).fillna(
        grouped["line"].first()
    )
    aggregates["opener_fair_left"] = grouped["fair_up"].first()

    reversals = _reversal_counts(eligible)
    aggregates["n_reversals"] = (
        reversals.reindex(aggregates.index).fillna(0.0) if not reversals.empty else 0.0
    )

    aggregates["snapshot_minutes"] = int(snapshot_minutes)
    return aggregates.reset_index()


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("snapshot", [0, 30, 360, 720, 1590, 5000])
def test_history_aggregates_match_the_callable_implementation(seed, snapshot):
    working = prepare_tick_history(random_ticks(seed=seed))
    expected = _history_aggregates_before(working, snapshot)
    actual = _history_aggregates(working, snapshot)
    assert_identical(actual, expected)


def test_abs_sums_fall_back_when_series_are_not_contiguous(monkeypatch):
    """Unsorted input takes the original lambda instead of slicing."""
    working = prepare_tick_history(random_ticks(seed=4))
    shuffled = working.sample(frac=1.0, random_state=1)
    grouped = shuffled.groupby(GROUP_KEYS, sort=False)
    assert movement_features._abs_sums(shuffled, grouped, "line_delta") is None
    assert_identical(
        _history_aggregates(shuffled, 60), _history_aggregates_before(shuffled, 60)
    )


def test_movement_features_match_end_to_end(monkeypatch):
    ticks = random_ticks(n_games=20, seed=5)
    panel = build_snapshot_panel(ticks, grid=GRID)
    actual = add_movement_features(panel, ticks, grid=GRID)
    monkeypatch.setattr(
        movement_features, "_history_aggregates", _history_aggregates_before
    )
    expected = add_movement_features(panel, ticks, grid=GRID)
    assert_identical(actual, expected)


# ---- cross_book.aggregate_across_books ----------------------------------------


def _aggregate_across_books_before(panel: pd.DataFrame) -> pd.DataFrame:
    """The implementation with per-group Python callables."""
    from nba_ou.data_processing.line_history.cross_book import (
        CONSENSUS_KEYS,
        _weighted_agreement,
        steam_move_column,
    )

    if panel.empty:
        return pd.DataFrame(columns=CONSENSUS_KEYS)

    move_column = steam_move_column(panel)
    grouped = panel.groupby(CONSENSUS_KEYS, sort=False)
    consensus = grouped.agg(
        consensus_line=("level", "median"),
        consensus_line_mean=("level", "mean"),
        crossbook_std=("level", lambda s: s.std(ddof=0)),
        crossbook_range=("level", lambda s: s.max() - s.min()),
        consensus_norm_line=("norm_line", "median"),
        consensus_raw_line=("raw_line", "median"),
        consensus_fair_left=("fair_left", "median"),
        consensus_overround=("overround", "median"),
        n_books_quoting=("level", "count"),
        median_line_age=("line_age_minutes", "median"),
        max_line_age=("line_age_minutes", "max"),
        consensus_move_from_open=("move_from_open", "median"),
        consensus_move_recent=(move_column, "median"),
        consensus_n_moves=("n_moves_so_far", "median"),
        consensus_opener_line=("opener_line", "median"),
        consensus_has_quote=("has_quote", "sum"),
    )
    steam = grouped.apply(
        _weighted_agreement, move_column=move_column, include_groups=False
    )
    consensus = consensus.join(steam)
    consensus["crossbook_std"] = consensus["crossbook_std"].fillna(0.0)
    consensus["crossbook_range"] = consensus["crossbook_range"].fillna(0.0)
    return consensus.reset_index()


@pytest.mark.parametrize("seed", [0, 3])
@pytest.mark.parametrize("windows", [(15, 60, 180), (120, 360)])
def test_cross_book_aggregation_matches_the_callable_implementation(seed, windows):
    from nba_ou.data_processing.line_history.cross_book import aggregate_across_books

    ticks = random_ticks(n_games=25, seed=seed)
    panel = build_snapshot_panel(ticks, grid=GRID)
    panel = add_movement_features(panel, ticks, grid=GRID, windows=windows)
    # Groups where every book is missing a level, for the NaN paths.
    panel.loc[panel.sample(frac=0.05, random_state=seed).index, "level"] = float("nan")
    assert_identical(
        aggregate_across_books(panel), _aggregate_across_books_before(panel)
    )


# ---- anchor_total_path ---------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 6])
@pytest.mark.parametrize("windows", [(15, 60, 180), (120, 360)])
def test_anchor_path_matches_the_previous_implementation(seed, windows):
    from nba_ou.data_processing.line_history.anchor_total_path import (
        add_anchor_total_path_features,
    )

    from .legacy_anchor_total_path import add_anchor_total_path_features_before

    ticks = random_ticks(n_games=25, seed=seed)
    panel = build_snapshot_panel(ticks, grid=GRID)
    panel = add_movement_features(panel, ticks, grid=GRID, windows=windows)
    assert_identical(
        add_anchor_total_path_features(panel, ticks, anchor="bet365"),
        add_anchor_total_path_features_before(panel, ticks, anchor="bet365"),
    )
