"""Compact tick-path and peer-state features for the anchor total."""

from __future__ import annotations

import pandas as pd
from nba_ou.data_processing.line_history.anchor_total_path import (
    add_anchor_total_path_features,
)
from nba_ou.data_processing.line_history.cross_book import (
    add_book_deviation,
)
from nba_ou.data_processing.line_history.movement_features import add_movement_features
from nba_ou.data_processing.line_history.snapshots import build_snapshot_panel

from .test_line_history_snapshots import make_ticks


def _panel(ticks: pd.DataFrame, windows=(60,)) -> pd.DataFrame:
    panel = build_snapshot_panel(ticks, grid=(50,))
    return add_movement_features(panel, ticks, grid=(50,), windows=windows)


def test_peer_gap_excludes_own_quote_and_caps_bad_feed_values():
    panel = pd.DataFrame(
        {
            "game_id": ["g"] * 4,
            "market": ["totals"] * 4,
            "snapshot_minutes": [50] * 4,
            "book": ["bet365", "a", "b", "bad"],
            "level": [220.0, 221.0, 222.0, 300.0],
        }
    )
    got = add_book_deviation(panel, pd.DataFrame()).set_index("book")
    # Peer median for bet365 is 221.5 after removing the corrupt 300 quote.
    assert got.loc["bet365", "deviation_from_consensus"] == -1.5
    assert got.loc["bad", "deviation_from_consensus"] == 10.0
    assert got.loc["bad", "abs_deviation_from_consensus"] == 10.0


def test_last_level_change_ignores_price_only_tick_and_future_move():
    ticks = make_ticks(
        [
            {"left_line": 220.0, "minutes_before_tip": 900.0},
            {"left_line": 221.0, "minutes_before_tip": 100.0},
            {"left_line": 220.0, "minutes_before_tip": 80.0},
            {"left_line": 220.0, "minutes_before_tip": 55.0, "left_price": -120.0},
            {"left_line": 225.0, "minutes_before_tip": 40.0},
        ]
    )
    panel = _panel(ticks)
    got = add_anchor_total_path_features(panel, ticks, anchor="bet365").iloc[0]
    prefix = "ODDS_SNAP_TOT_BET365_"
    assert panel.iloc[0]["line_age_minutes"] == 5.0
    assert got[prefix + "MINUTES_SINCE_LAST_LEVEL_MOVE"] == 30.0
    assert got[prefix + "ABS_LEVEL_PATH_60"] == 2.0
    assert got[prefix + "SIGNED_MOVE_STREAK_60"] == -1
    assert got[prefix + "LAST_TWO_LEVEL_MOVES_GAP_MIN"] == 20.0


def test_peers_can_move_while_anchor_does_not():
    ticks = pd.concat(
        [
            make_ticks([{"left_line": 220.0, "minutes_before_tip": 900.0}]),
            make_ticks(
                [
                    {"book": "a", "left_line": 220.0, "minutes_before_tip": 900.0},
                    {"book": "a", "left_line": 221.0, "minutes_before_tip": 80.0},
                ]
            ),
            make_ticks(
                [
                    {"book": "b", "left_line": 220.5, "minutes_before_tip": 900.0},
                    {"book": "b", "left_line": 221.5, "minutes_before_tip": 70.0},
                ]
            ),
        ],
        ignore_index=True,
    )
    panel = _panel(ticks)
    for available_windows in ((60,), (120,)):
        tested = panel if available_windows == (60,) else _panel(ticks, (120,))
        got = add_anchor_total_path_features(tested, ticks, anchor="bet365")
        assert got["ODDS_SNAP_TOT_BET365_PEERS_MOVED_ANCHOR_STILL_60"].iloc[0] == 1


def test_no_moves_uses_neutral_path_and_capped_gap():
    ticks = make_ticks([{"left_line": 220.0, "minutes_before_tip": 900.0}])
    got = add_anchor_total_path_features(_panel(ticks), ticks, anchor="bet365").iloc[0]
    prefix = "ODDS_SNAP_TOT_BET365_"
    assert got[prefix + "ABS_LEVEL_PATH_60"] == 0.0
    assert got[prefix + "SIGNED_MOVE_STREAK_60"] == 0
    assert got[prefix + "LAST_TWO_LEVEL_MOVES_GAP_MIN"] == 720.0
    assert got[prefix + "PEERS_MOVED_ANCHOR_STILL_60"] == 0


def test_peer_state_is_signed_by_peer_direction():
    ticks = pd.concat(
        [
            make_ticks([{"left_line": 220.0, "minutes_before_tip": 900.0}]),
            make_ticks(
                [
                    {"book": "a", "left_line": 221.0, "minutes_before_tip": 900.0},
                    {"book": "a", "left_line": 220.0, "minutes_before_tip": 80.0},
                ]
            ),
            make_ticks(
                [
                    {"book": "b", "left_line": 221.5, "minutes_before_tip": 900.0},
                    {"book": "b", "left_line": 220.5, "minutes_before_tip": 70.0},
                ]
            ),
        ],
        ignore_index=True,
    )
    got = add_anchor_total_path_features(_panel(ticks), ticks, anchor="bet365")
    assert got["ODDS_SNAP_TOT_BET365_PEERS_MOVED_ANCHOR_STILL_60"].iloc[0] == -1
