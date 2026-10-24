"""Tests for the as-of snapshot builder.

The leakage-critical property is negative: a tick that had not yet happened at
the snapshot horizon must never reach the row.
"""

from __future__ import annotations

import pandas as pd
import pytest
from nba_ou.data_processing.line_history.snapshots import (
    build_snapshot_panel,
    resolve_line,
    snapshot_coverage,
)


def make_ticks(rows: list[dict]) -> pd.DataFrame:
    """Build a tick frame shaped like ``fetch_pregame_ticks`` output."""
    base = {
        "game_id": "0022300001",
        "season_year": 2023,
        "market": "totals",
        "book": "bet365",
        "is_opener": False,
        "left_price": -110.0,
        "right_price": -110.0,
        "right_line": None,
    }
    records = []
    for index, row in enumerate(rows):
        record = {**base, **row}
        record.setdefault("right_line", record["left_line"])
        record["line_ts"] = pd.Timestamp("2023-11-01", tz="UTC") + pd.Timedelta(
            minutes=index
        )
        records.append(record)
    return pd.DataFrame(records)


def test_snapshot_carries_the_last_tick_at_or_before_the_horizon():
    ticks = make_ticks(
        [
            {"left_line": 220.0, "minutes_before_tip": 600.0},
            {"left_line": 222.0, "minutes_before_tip": 200.0},
            {"left_line": 225.0, "minutes_before_tip": 100.0},
        ]
    )
    panel = build_snapshot_panel(ticks, grid=(120,))
    assert panel["raw_line"].iloc[0] == 222.0
    # The 100-minute tick is later than the 120-minute horizon and must not leak.
    assert 225.0 not in panel["raw_line"].tolist()


def test_a_tick_exactly_on_the_horizon_is_observable():
    ticks = make_ticks(
        [
            {"left_line": 220.0, "minutes_before_tip": 600.0},
            {"left_line": 223.0, "minutes_before_tip": 120.0},
        ]
    )
    panel = build_snapshot_panel(ticks, grid=(120,))
    assert panel["raw_line"].iloc[0] == 223.0


def test_line_age_measures_staleness_at_the_snapshot_instant():
    ticks = make_ticks(
        [
            {"left_line": 220.0, "minutes_before_tip": 500.0},
        ]
    )
    panel = build_snapshot_panel(ticks, grid=(120,))
    assert panel["line_age_minutes"].iloc[0] == 380.0


def test_series_with_no_tick_yet_produces_no_row():
    """No quote existed at that horizon, so there is nothing to carry forward."""
    ticks = make_ticks([{"left_line": 220.0, "minutes_before_tip": 100.0}])
    panel = build_snapshot_panel(ticks, grid=(720,))
    assert panel.empty


def test_spread_line_is_mirrored_when_only_the_right_side_survives():
    """The price-bleed repair can leave one side NULL; the mirror recovers it."""
    ticks = make_ticks(
        [
            {
                "market": "point_spread",
                "left_line": None,
                "right_line": 4.5,
                "minutes_before_tip": 300.0,
            }
        ]
    )
    assert resolve_line(ticks).iloc[0] == -4.5


def test_moneyline_carries_no_line():
    ticks = make_ticks(
        [{"market": "money_line", "left_line": None, "minutes_before_tip": 300.0}]
    )
    assert resolve_line(ticks).isna().all()


def test_left_is_away_and_right_is_home_for_spread():
    """Orientation pinned deliberately.

    Verified against the store: the residual of (home margin - left spread) has
    std 13.46 versus 19.96 under the opposite reading, and devigged moneyline
    probabilities average 0.551 on the right side against 0.449 on the left,
    consistent with home-court advantage. Inverting this would silently flip
    every spread and moneyline feature, so it is asserted rather than assumed.
    """
    ticks = make_ticks(
        [
            {
                "market": "point_spread",
                "left_line": 4.5,
                "right_line": -4.5,
                "minutes_before_tip": 300.0,
            }
        ]
    )
    # left_line is the AWAY side: away +4.5 means home is favoured by 4.5, so a
    # positive left line implies a positive expected home margin.
    assert resolve_line(ticks).iloc[0] == 4.5


def test_each_book_is_carried_independently():
    ticks = pd.concat(
        [
            make_ticks([{"left_line": 220.0, "minutes_before_tip": 300.0}]),
            make_ticks(
                [{"book": "fanduel", "left_line": 224.0, "minutes_before_tip": 300.0}]
            ),
        ]
    )
    panel = build_snapshot_panel(ticks, grid=(120,))
    assert set(panel["book"]) == {"bet365", "fanduel"}
    assert sorted(panel["raw_line"]) == [220.0, 224.0]


def test_a_negative_horizon_is_refused():
    """It would place the snapshot AFTER tip-off and admit in-play ticks -- a
    look-ahead that every downstream column-name check would pass."""
    ticks = make_ticks([{"left_line": 220.0, "minutes_before_tip": 300.0}])
    with pytest.raises(ValueError, match="non-negative minutes"):
        build_snapshot_panel(ticks, grid=(-30,))


def test_zero_is_the_closing_snapshot_and_is_allowed():
    """T=0 means "bet as late as the market allows". It is not tip-off: the
    fetch layer already refuses ticks inside its safety margin, so the latest
    tick this can resolve to is still comfortably pre-game."""
    ticks = make_ticks(
        [
            {"left_line": 220.0, "minutes_before_tip": 900.0},
            {"left_line": 223.0, "minutes_before_tip": 45.0},
            {"left_line": 224.0, "minutes_before_tip": 6.0},
        ]
    )

    panel = build_snapshot_panel(ticks, grid=(0,))

    assert panel["raw_line"].tolist() == [224.0]
    assert panel["tick_minutes_before_tip"].tolist() == [6.0]
    # Age is measured from the snapshot, which is 0 -- so it is the tick's own
    # distance from tip.
    assert panel["line_age_minutes"].tolist() == [6.0]


def test_the_closing_snapshot_sees_every_tick():
    ticks = make_ticks(
        [
            {"left_line": 220.0, "minutes_before_tip": 900.0},
            {"left_line": 223.0, "minutes_before_tip": 45.0},
            {"left_line": 224.0, "minutes_before_tip": 6.0},
        ]
    )

    panel = build_snapshot_panel(ticks, grid=(0, 60))
    at_close = panel[panel.snapshot_minutes == 0]
    at_60 = panel[panel.snapshot_minutes == 60]

    assert at_close["n_ticks_so_far"].iloc[0] == 3
    assert at_60["n_ticks_so_far"].iloc[0] == 1


def test_coverage_reports_rows_and_games_per_snapshot():
    ticks = make_ticks(
        [
            {"left_line": 220.0, "minutes_before_tip": 900.0},
            {"left_line": 222.0, "minutes_before_tip": 100.0},
        ]
    )
    coverage = snapshot_coverage(build_snapshot_panel(ticks, grid=(60, 720)))
    assert coverage["games"].tolist() == [1, 1]
