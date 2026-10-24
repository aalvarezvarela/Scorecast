from __future__ import annotations

import pandas as pd
import pytest
from nba_ou.data_processing.odds.normalize_spread_lines import (
    normalize_spread_lines_inplace,
    spread_price_extreme_mask,
)
from nba_ou.data_processing.odds.normalize_total_lines import odds_to_decimal
from nba_ou.postgre_db.odds.merge_odds_data import merge_yahoo_sportsbook_odds


def _spread_quote(
    *,
    line_home: float = -4.5,
    line_away: float = 4.5,
    price_home: float = -130.0,
    price_away: float = 110.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["1"],
            "game_date": ["2026-01-01"],
            "season_year": [2025],
            "spread_bet365_line_home": [line_home],
            "spread_bet365_line_away": [line_away],
            "spread_bet365_price_home": [price_home],
            "spread_bet365_price_away": [price_away],
        }
    )


def test_spread_price_extreme_mask_identifies_wrong_market_prices():
    prices = pd.Series([1.91, 1.49, 2.51, None])
    assert spread_price_extreme_mask(prices).tolist() == [
        False,
        True,
        True,
        False,
    ]


def test_merge_normalizes_spread_quotes_after_decimal_conversion():
    normalized = merge_yahoo_sportsbook_odds(pd.DataFrame(), _spread_quote())
    original = merge_yahoo_sportsbook_odds(
        pd.DataFrame(),
        _spread_quote(),
        normalize_spread_lines=False,
    )

    assert normalized.at[0, "spread_bet365_line_home"] == pytest.approx(-6.0)
    assert normalized.at[0, "spread_bet365_line_away"] == pytest.approx(6.0)
    assert normalized.at[0, "spread_bet365_price_home"] == pytest.approx(1.91)
    assert normalized.at[0, "spread_bet365_price_away"] == pytest.approx(1.91)

    assert original.at[0, "spread_bet365_line_home"] == pytest.approx(-4.5)
    assert original.at[0, "spread_bet365_line_away"] == pytest.approx(4.5)
    assert original.at[0, "spread_bet365_price_home"] == pytest.approx(
        round(odds_to_decimal(-130.0, "american"), 2)
    )
    assert original.at[0, "spread_bet365_price_away"] == pytest.approx(
        round(odds_to_decimal(110.0, "american"), 2)
    )


def test_extreme_spread_prices_are_nulled_not_centered():
    normalized = merge_yahoo_sportsbook_odds(
        pd.DataFrame(),
        _spread_quote(line_home=19.5, line_away=-19.5, price_home=-649, price_away=375),
    )

    assert normalized.at[0, "spread_bet365_line_home"] == pytest.approx(19.5)
    assert normalized.at[0, "spread_bet365_line_away"] == pytest.approx(-19.5)
    assert pd.isna(normalized.at[0, "spread_bet365_price_home"])
    assert pd.isna(normalized.at[0, "spread_bet365_price_away"])


def test_invalid_spread_line_pairs_are_nulled():
    quote = _spread_quote(line_home=-4.5, line_away=5.0)
    normalize_spread_lines_inplace(quote, odds_format="american")

    assert pd.isna(quote.at[0, "spread_bet365_line_home"])
    assert pd.isna(quote.at[0, "spread_bet365_line_away"])
    assert pd.isna(quote.at[0, "spread_bet365_price_home"])
    assert pd.isna(quote.at[0, "spread_bet365_price_away"])
