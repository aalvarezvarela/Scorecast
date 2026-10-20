"""The configured snapshot grid and look-back windows, and what depends on them.

Snapshots are ROWS (``TIME_TO_MATCH_MIN``), not columns, so the grid decides how
many rows each game contributes; the windows decide how many columns each book
contributes. Both are defaults a caller can override, but the defaults are what
the shipped datasets are built with, so they are pinned here rather than left to
drift silently.

The pair also sets the build cost: every ``(snapshot, window)`` combination needs
its own as-of read at ``T + w``, so work scales with the *product*, not with
either alone.
"""

from __future__ import annotations

import pandas as pd
import pytest
from nba_ou.data_processing.line_history.cross_book import (
    STEAM_WINDOW_MINUTES,
    steam_move_column,
)
from nba_ou.data_processing.line_history.movement_features import (
    DEFAULT_WINDOWS,
    add_movement_features,
    extended_grid,
)
from nba_ou.data_processing.line_history.snapshots import (
    DEFAULT_SNAPSHOT_GRID,
    build_snapshot_panel,
)

TIP = pd.Timestamp("2025-01-15T00:10:00Z")


def _ticks(*, book: str = "bet365", moves: list[tuple[float, float]] | None = None):
    moves = moves or [(1500.0, 224.5), (700.0, 225.0), (300.0, 225.5), (58.0, 226.0)]
    return pd.DataFrame(
        [
            {
                "game_id": "0022400500",
                "season_year": 2025,
                "market": "totals",
                "book": book,
                "line_ts": TIP - pd.Timedelta(minutes=minutes),
                "minutes_before_tip": minutes,
                "is_opener": index == 0,
                "left_line": line,
                "right_line": line,
                "left_price": -110,
                "right_price": -110,
            }
            for index, (minutes, line) in enumerate(moves)
        ]
    )


# --- the configured defaults ------------------------------------------------


def test_the_grid_includes_the_closing_snapshot():
    assert 0 in DEFAULT_SNAPSHOT_GRID


def test_the_grid_is_sorted_and_free_of_duplicates():
    assert list(DEFAULT_SNAPSHOT_GRID) == sorted(set(DEFAULT_SNAPSHOT_GRID))


#: The horizons present in the schema-2.5 intermediate dataset, and therefore
#: the set a model can be trained at and promoted to a slot. Kept as a literal
#: rather than read from the 4.4 GB CSV so the test stays cheap and states the
#: contract out loud.
TRAINED_HORIZONS = (
    0,
    30,
    60,
    120,
    180,
    240,
    300,
    360,
    420,
    480,
    540,
    600,
    660,
    720,
    840,
    960,
    1080,
)


def test_the_grid_covers_the_hours_a_bettor_actually_uses():
    """3h, 5h and 6h were added so the afternoon window is sampled rather than
    jumped over -- the old grid went straight from 4h to 8h."""
    for horizon in (0, 30, 60, 120, 180, 240, 300, 360, 480, 720):
        assert horizon in DEFAULT_SNAPSHOT_GRID


def test_the_grid_samples_every_horizon_a_model_can_be_promoted_at():
    """The promotable-horizon contract.

    A slot is one ``(target, horizon)``, and the daily build's grid bounds which
    slots can be refit -- promote a horizon the build does not sample and the
    slot silently never updates, which is the failure
    ``registry.check_horizon_is_buildable`` exists to catch. So the grid must be
    a SUPERSET of every horizon the training data carries, not merely overlap
    it. Before this was enforced the default stopped at 720 while campaigns
    trained at 420, 540, 600, 660, 840, 960 and 1080.
    """
    missing = [t for t in TRAINED_HORIZONS if t not in DEFAULT_SNAPSHOT_GRID]
    assert not missing, (
        f"horizons {missing} are in the training data but are not sampled by "
        "the daily build, so a model promoted at them could never refit"
    )


def test_the_grid_stops_at_eighteen_hours():
    """Coverage falls off with lead time but does not collapse until ~24h.

    Measured on ``intermediate_line_data_2_5_20260613.csv`` as games carrying a
    row at each horizon, against 8,902 at T-0: 720 -> 95.8%, 840 -> 93.1%,
    960 -> 87.6%, 1080 -> 79.5%. At 24h it reaches ~60%, where a longer lead
    buys a biased sample of well-covered games rather than more information.
    """
    assert max(DEFAULT_SNAPSHOT_GRID) == 1080


def test_windows_include_the_short_end():
    assert 15 in DEFAULT_WINDOWS
    assert 30 in DEFAULT_WINDOWS


def test_windows_are_sorted_and_free_of_duplicates():
    assert list(DEFAULT_WINDOWS) == sorted(set(DEFAULT_WINDOWS))


def test_every_window_is_positive():
    """A zero or negative window would read the panel at or after the snapshot
    itself -- a look-ahead wearing a look-back's name."""
    assert all(window > 0 for window in DEFAULT_WINDOWS)


def test_acceleration_stays_computable():
    """``move_acceleration`` is only emitted when both 60 and 180 are configured."""
    assert 60 in DEFAULT_WINDOWS
    assert 180 in DEFAULT_WINDOWS


# --- what the pair costs ----------------------------------------------------


def test_the_extended_grid_is_the_union_of_snapshots_and_lookbacks():
    extended = extended_grid(DEFAULT_SNAPSHOT_GRID, DEFAULT_WINDOWS)

    assert set(DEFAULT_SNAPSHOT_GRID).issubset(extended)
    for snapshot in DEFAULT_SNAPSHOT_GRID:
        for window in DEFAULT_WINDOWS:
            assert snapshot + window in extended


def test_a_short_window_at_the_closing_snapshot_is_reachable():
    """T=0 with a 15-minute window needs an as-of read at 15 minutes, which only
    exists because 0 is a legal horizon."""
    assert 15 in extended_grid((0,), (15,))


# --- the columns each window produces ---------------------------------------


def test_each_window_emits_its_five_columns():
    ticks = _ticks()
    panel = build_snapshot_panel(ticks, grid=(60, 240))
    out = add_movement_features(panel, ticks, grid=(60, 240), windows=(15, 60))

    for window in (15, 60):
        for prefix in (
            "has_window",
            "move_last",
            "abs_move_last",
            "velocity_last",
            "prob_move_last",
        ):
            assert f"{prefix}_{window}" in out.columns


def test_a_quiet_short_window_reads_zero_and_is_flagged_as_present():
    """The 15-minute window is mostly zero on real data. That must be readable
    as "the market did not move", distinct from "no history reaches back"."""
    ticks = _ticks(moves=[(1500.0, 224.5), (700.0, 225.0)])
    panel = build_snapshot_panel(ticks, grid=(60,))
    out = add_movement_features(panel, ticks, grid=(60,), windows=(15,))

    assert out["has_window_15"].iloc[0] == 1
    assert out["move_last_15"].iloc[0] == 0.0


# --- steam ------------------------------------------------------------------


def test_steam_is_pinned_to_its_own_window_not_the_shortest():
    """Adding a shorter window must not silently redefine steam. Cross-book
    agreement needs books to have moved; at 15 minutes almost none have."""
    panel = pd.DataFrame(
        {
            "move_last_15": [0.0],
            "move_last_30": [0.0],
            "move_last_60": [0.5],
            "move_last_180": [1.0],
        }
    )

    assert steam_move_column(panel) == f"move_last_{STEAM_WINDOW_MINUTES}"
    assert steam_move_column(panel) == "move_last_60"


def test_steam_falls_back_to_the_shortest_window_when_its_own_is_absent():
    """The fallback is why this is derived rather than hardcoded: a caller
    configuring windows=(120, 360) used to hit a bare KeyError."""
    panel = pd.DataFrame({"move_last_120": [0.5], "move_last_360": [1.0]})

    assert steam_move_column(panel) == "move_last_120"


def test_steam_raises_a_named_error_when_movement_features_were_not_run():
    with pytest.raises(ValueError, match="add_movement_features"):
        steam_move_column(pd.DataFrame({"level": [225.0]}))


def test_the_default_windows_give_steam_its_pinned_window():
    assert STEAM_WINDOW_MINUTES in DEFAULT_WINDOWS
