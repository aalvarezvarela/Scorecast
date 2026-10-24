"""The cross-book total spread is measured over books' raw closes only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.config.odds_columns import (
    extract_total_line_books,
    resolve_main_total_line_col,
)
from nba_ou.data_processing.merged_home_away_data.global_market_features import (
    add_global_market_features,
)

DERIVED = "ODDS_TOTAL_LINE_{}_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME"


def _frame() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "GAME_ID": ["g1", "g2", "g3", "g4"],
            "GAME_DATE": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "TOTAL_POINTS": [220.0, 230.0, 210.0, 225.0],
            "ODDS_TOTAL_LINE_bet365": [221.0, 228.5, 212.0, 224.0],
            "ODDS_TOTAL_LINE_fanduel": [222.0, 229.5, np.nan, 225.0],
            # An opener far from the closes: included, it would dominate the std.
            "ODDS_TOTAL_LINE_consensus_opener": [210.0, 240.0, 200.0, 230.0],
        }
    )
    # Derived columns sharing the prefix, on a completely different scale.
    df[DERIVED.format("bet365")] = [100.0, 110.0, 120.0, 130.0]
    df[DERIVED.format("fanduel")] = [5.0, 6.0, 7.0, 8.0]
    df["ODDS_TOTAL_LINE_bet365_TREND_SLOPE_LAST_5_BEFORE_TEAM_AWAY"] = 0.3
    return df


def test_extract_total_line_books_ignores_derived_columns():
    assert extract_total_line_books(_frame()) == [
        "bet365",
        "consensus_opener",
        "fanduel",
    ]


def test_main_line_fallback_never_picks_a_derived_column():
    df = _frame().drop(columns=["ODDS_TOTAL_LINE_fanduel"])
    df = df.drop(columns=["ODDS_TOTAL_LINE_consensus_opener"])
    # Only bet365's derived columns and the raw bet365 close remain; the
    # configured book is absent, so the fallback must land on the raw close.
    assert resolve_main_total_line_col(df, "draftkings") == "ODDS_TOTAL_LINE_bet365"


def test_crossbook_std_and_range_use_books_only():
    out = add_global_market_features(_frame()).set_index("GAME_ID")

    std = out["THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE"]
    rng = out["THIS_GAME_CROSSBOOK_TOTAL_RANGE_BEFORE"]
    # Two books one point apart: ddof=0 std 0.5, range 1.0.
    assert std["g1"] == pytest.approx(0.5)
    assert std["g2"] == pytest.approx(0.5)
    assert rng["g4"] == pytest.approx(1.0)
    # One book only: no disagreement to measure, not "the books agree".
    assert np.isnan(std["g3"])
    assert np.isnan(rng["g3"])


def test_crossbook_rollup_skips_single_book_games():
    out = add_global_market_features(_frame()).set_index("GAME_ID")
    rollup = out["GLOBAL_CROSSBOOK_TOTAL_STD_AVG_15G_BEFORE"]
    assert np.isnan(rollup["g1"])  # no earlier games
    assert rollup["g4"] == pytest.approx(0.5)  # g1, g2; g3 has no value


def test_single_book_frame_emits_no_crossbook_columns():
    df = _frame().drop(columns=["ODDS_TOTAL_LINE_fanduel"])
    out = add_global_market_features(df)
    assert "THIS_GAME_CROSSBOOK_TOTAL_STD_BEFORE" not in out.columns
