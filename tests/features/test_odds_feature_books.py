"""``engineer_odds_features`` aggregates over books, never over derived columns."""

from __future__ import annotations

import pandas as pd
import pytest
from nba_ou.config.odds_columns import extract_spread_line_home_books
from nba_ou.data_processing.merged_home_away_data.odds_feature_engeneer import (
    engineer_odds_features,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ODDS_TOTAL_LINE_bet365": [220.0],
            "ODDS_TOTAL_LINE_fanduel": [222.0],
            # A derived column sharing the prefix: a trend of value 0. Read as a
            # book, it made three "books" with mean 147.33 and std 104.18.
            "ODDS_TOTAL_LINE_bet365_TREND_SLOPE_LAST_5_BEFORE_TEAM_HOME": [0.0],
            "ODDS_TOTAL_LINE_fanduel_LAST_ALL_5_MATCHES_BEFORE_TEAM_AWAY": [218.0],
            "total_bet365_price_over": [1.91],
            "total_bet365_price_under": [1.91],
            "total_fanduel_price_over": [1.87],
            "total_fanduel_price_under": [1.95],
        }
    )


def test_book_aggregates_use_raw_book_lines_only():
    out = engineer_odds_features(_frame()).iloc[0]

    assert out["ODDS_n_books_total_line_present"] == 2
    assert out["ODDS_n_books_total_line_present_ratio"] == 1.0
    assert out["ODDS_n_books_total_price_present"] == 2
    assert out["ODDS_total_line_books_mean"] == pytest.approx(221.0)
    assert out["ODDS_total_line_books_median"] == pytest.approx(221.0)
    assert out["ODDS_total_line_books_std"] == pytest.approx(1.0)
    assert out["ODDS_total_line_books_range"] == pytest.approx(2.0)


def test_no_per_book_columns_for_derived_columns():
    out = engineer_odds_features(_frame())
    per_book = sorted(c for c in out if c.startswith("ODDS_book_total_line_mid_"))
    assert per_book == [
        "ODDS_book_total_line_mid_bet365",
        "ODDS_book_total_line_mid_fanduel",
    ]


def test_spread_books_ignore_derived_columns():
    df = pd.DataFrame(
        columns=[
            "ODDS_SPREAD_LINE_HOME_bet365",
            "ODDS_SPREAD_LINE_HOME_consensus_opener",
            "ODDS_SPREAD_LINE_HOME_bet365_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME",
        ]
    )
    assert extract_spread_line_home_books(df) == ["bet365", "consensus_opener"]
