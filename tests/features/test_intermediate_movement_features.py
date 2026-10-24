"""Tests for snapshot movement features."""

from __future__ import annotations

import pytest
from nba_ou.data_processing.line_history.movement_features import (
    MISSING_SENTINEL,
    add_movement_features,
    extended_grid,
    prepare_tick_history,
)
from nba_ou.data_processing.line_history.snapshots import build_snapshot_panel

from .test_line_history_snapshots import make_ticks


def build(rows, grid, windows=(60, 180)):
    ticks = make_ticks(rows)
    panel = build_snapshot_panel(ticks, grid=grid)
    return add_movement_features(panel, ticks, grid=grid, windows=windows)


def test_extended_grid_covers_every_window_lookback():
    assert extended_grid((30, 60), (60, 180)) == (30, 60, 90, 120, 210, 240)


def test_move_from_open_is_current_minus_opener():
    out = build(
        [
            {"left_line": 220.0, "minutes_before_tip": 900.0, "is_opener": True},
            {"left_line": 223.5, "minutes_before_tip": 300.0},
        ],
        grid=(120,),
    )
    assert out["opener_line"].iloc[0] == 220.0
    assert out["move_from_open"].iloc[0] == pytest.approx(3.5)
    assert out["abs_move_from_open"].iloc[0] == pytest.approx(3.5)
    assert out["move_direction"].iloc[0] == 1.0


def test_only_level_changes_count_as_moves():
    """A re-price at the same number is pressure, not a move."""
    ticks = make_ticks(
        [
            {"left_line": 220.0, "minutes_before_tip": 900.0},
            {
                "left_line": 220.0,
                "minutes_before_tip": 600.0,
                "left_price": -120.0,
                "right_price": 100.0,
            },
            {"left_line": 221.0, "minutes_before_tip": 300.0},
        ]
    )
    history = prepare_tick_history(ticks)
    assert history["is_move"].sum() == 1
    assert history["is_price_only"].sum() == 1


def test_windowed_move_uses_the_line_one_window_earlier():
    out = build(
        [
            {"left_line": 220.0, "minutes_before_tip": 400.0},
            {"left_line": 226.0, "minutes_before_tip": 100.0},
        ],
        grid=(60,),
        windows=(60,),
    )
    # At T=60 the line is 226.0; at T+60=120 it was still 220.0.
    assert out["has_window_60"].iloc[0] == 1
    assert out["move_last_60"].iloc[0] == pytest.approx(6.0)
    assert out["velocity_last_60"].iloc[0] == pytest.approx(6.0)


def test_absent_window_is_flagged_and_sentinel_filled_not_nan():
    """The ``max_na_per_row`` trap.

    Long horizons are systematically the ones whose look-back does not reach.
    Bare NaNs there would let row-level NaN cleaning delete exactly the
    long-lead snapshots this dataset exists to compare.
    """
    out = build(
        [{"left_line": 220.0, "minutes_before_tip": 200.0}],
        grid=(120,),
        windows=(180,),
    )
    assert out["has_window_180"].iloc[0] == 0
    assert out["move_last_180"].iloc[0] == MISSING_SENTINEL
    assert not out["move_last_180"].isna().any()


def test_move_counts_grow_toward_tip():
    rows = [
        {"left_line": 220.0, "minutes_before_tip": 900.0},
        {"left_line": 221.0, "minutes_before_tip": 500.0},
        {"left_line": 222.0, "minutes_before_tip": 200.0},
        {"left_line": 223.0, "minutes_before_tip": 45.0},
    ]
    out = build(rows, grid=(30, 240, 720)).sort_values("snapshot_minutes")
    counts = out.set_index("snapshot_minutes")["n_moves_so_far"]
    assert counts.loc[720] == 0
    assert counts.loc[240] == 1
    assert counts.loc[30] == 3


def test_position_in_range_defaults_to_midpoint_when_line_never_moved():
    out = build([{"left_line": 220.0, "minutes_before_tip": 900.0}], grid=(120,))
    assert out["line_range_so_far"].iloc[0] == 0.0
    assert out["position_in_range"].iloc[0] == 0.5


def test_reversal_is_counted_when_direction_flips():
    out = build(
        [
            {"left_line": 220.0, "minutes_before_tip": 900.0},
            {"left_line": 224.0, "minutes_before_tip": 600.0},
            {"left_line": 221.0, "minutes_before_tip": 300.0},
        ],
        grid=(120,),
    )
    assert out["n_reversals"].iloc[0] == 1


def test_single_observation_series_reports_zero_dispersion_not_nan():
    out = build([{"left_line": 220.0, "minutes_before_tip": 900.0}], grid=(120,))
    assert out["line_std_so_far"].iloc[0] == 0.0


def make_moneyline_ticks(rows):
    """Moneyline ticks: prices only, no line at all."""
    return make_ticks(
        [
            {"market": "money_line", "left_line": None, "right_line": None, **row}
            for row in rows
        ]
    )


def test_moneyline_movement_is_measured_in_devigged_probability():
    """Regression: the moneyline used to have no level, so every count was zero.

    ``line_delta`` was NaN on every moneyline row, which silently zeroed
    ``is_move``, ``is_price_only``, the reversal count, the window flags and the
    cross-book dispersion for the whole market.
    """
    ticks = make_moneyline_ticks(
        [
            {"minutes_before_tip": 900.0, "left_price": 150.0, "right_price": -170.0},
            {"minutes_before_tip": 300.0, "left_price": 120.0, "right_price": -140.0},
        ]
    )
    history = prepare_tick_history(ticks)
    assert history["line"].notna().all()
    assert history["is_move"].sum() == 1

    out = add_movement_features(
        build_snapshot_panel(ticks, grid=(120,)), ticks, grid=(120,), windows=(60,)
    )
    assert out["n_moves_so_far"].iloc[0] == 1
    assert out["move_from_open"].iloc[0] != 0.0
    # Home price shortened from -170 to -140... no: it lengthened, so the home
    # win probability FELL between the opener and now.
    assert out["move_from_open"].iloc[0] < 0.0


def test_moneyline_price_only_ticks_are_structurally_zero():
    """On the moneyline the level IS the price, so the categories collapse."""
    ticks = make_moneyline_ticks(
        [
            {"minutes_before_tip": 900.0, "left_price": 150.0, "right_price": -170.0},
            {"minutes_before_tip": 300.0, "left_price": 120.0, "right_price": -140.0},
        ]
    )
    history = prepare_tick_history(ticks)
    assert history["is_price_only"].sum() == 0


def test_position_in_range_stays_within_bounds():
    """Regression: mixing a centered "now" with a raw path put 946 spread rows
    outside their own realised range."""
    ticks = make_ticks(
        [
            {
                "market": "point_spread",
                "left_line": 4.5,
                "right_line": -4.5,
                "minutes_before_tip": 900.0,
                "left_price": -130.0,
                "right_price": 110.0,
            },
            {
                "market": "point_spread",
                "left_line": 6.0,
                "right_line": -6.0,
                "minutes_before_tip": 300.0,
                "left_price": -125.0,
                "right_price": 105.0,
            },
        ]
    )
    out = add_movement_features(
        build_snapshot_panel(ticks, grid=(120,)), ticks, grid=(120,), windows=(60,)
    )
    position = out["position_in_range"].iloc[0]
    assert 0.0 <= position <= 1.0


def test_non_positive_windows_are_rejected():
    """A negative window reads a LATER moment than the snapshot: look-ahead."""
    ticks = make_ticks([{"left_line": 220.0, "minutes_before_tip": 900.0}])
    panel = build_snapshot_panel(ticks, grid=(120,))
    for bad in [(-60,), (0,), (60, -30)]:
        with pytest.raises(ValueError, match="positive minutes"):
            add_movement_features(panel, ticks, grid=(120,), windows=bad)


def test_probability_move_has_the_same_sign_as_the_level_move():
    """Regression: level and prob_move used opposite sides on 2 of 3 markets.

    ``level`` is the home probability on the moneyline and the expected home
    margin on the spread, while the probability family used ``fair_left`` --
    the AWAY side on both. One event was reported with two opposite signs.
    """
    for market, rows in {
        "money_line": [
            {
                "left_line": None,
                "right_line": None,
                "left_price": 150.0,
                "right_price": -170.0,
                "minutes_before_tip": 900.0,
            },
            {
                "left_line": None,
                "right_line": None,
                "left_price": 120.0,
                "right_price": -140.0,
                "minutes_before_tip": 300.0,
            },
        ],
        "point_spread": [
            {
                "left_line": 4.5,
                "right_line": -4.5,
                "left_price": -110.0,
                "right_price": -110.0,
                "minutes_before_tip": 900.0,
            },
            {
                "left_line": 4.5,
                "right_line": -4.5,
                "left_price": 110.0,
                "right_price": -130.0,
                "minutes_before_tip": 300.0,
            },
        ],
    }.items():
        ticks = make_ticks([{"market": market, **row} for row in rows])
        out = add_movement_features(
            build_snapshot_panel(ticks, grid=(120,)),
            ticks,
            grid=(120,),
            windows=(60,),
        )
        level_move = out["move_from_open"].iloc[0]
        prob_move = out["prob_move_from_open"].iloc[0]
        if level_move != 0 and prob_move != 0:
            assert (level_move > 0) == (prob_move > 0), market


def test_opposes_opening_direction_uses_the_first_move_not_the_net_move():
    """A line that rose then rose further is NOT opposing its opening direction.

    The old form compared the last hour against the NET opener-to-now
    direction, so a line that fell then recovered past its open scored the same
    as one that simply kept rising.
    """
    ticks = make_ticks(
        [
            {"left_line": 220.0, "minutes_before_tip": 900.0},
            {"left_line": 224.0, "minutes_before_tip": 600.0},  # first move: UP
            {"left_line": 222.0, "minutes_before_tip": 200.0},  # now moving DOWN
        ]
    )
    out = add_movement_features(
        build_snapshot_panel(ticks, grid=(120,)), ticks, grid=(120,), windows=(60,)
    )
    assert out["first_move_direction"].iloc[0] == 1.0
    # Net move is still up (222 > 220), so the net-based form would say "no".
    assert out["move_direction"].iloc[0] == 1.0
    assert out["net_opposes_opening_direction"].iloc[0] == 0


def test_windows_without_60_minutes_do_not_break_cross_book():
    """Steam derived the shortest window instead of assuming 60 exists."""
    from nba_ou.data_processing.line_history.cross_book import (
        aggregate_across_books,
        steam_move_column,
    )

    ticks = make_ticks(
        [
            {"left_line": 220.0, "minutes_before_tip": 900.0},
            {"left_line": 223.0, "minutes_before_tip": 400.0},
        ]
    )
    out = add_movement_features(
        build_snapshot_panel(ticks, grid=(120,)),
        ticks,
        grid=(120,),
        windows=(120, 360),
    )
    assert steam_move_column(out) == "move_last_120"
    consensus = aggregate_across_books(out)
    assert "steam_fraction" in consensus.columns
    assert "consensus_move_recent" in consensus.columns
