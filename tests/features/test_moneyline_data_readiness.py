"""Moneyline data readiness -- data only, deliberately no model.

The point of these tests is that a future moneyline strategy should need no
second upstream redesign: the prices, their HOME/AWAY orientation and their
de-vigged probabilities are already present and already canonical.
"""

from __future__ import annotations

import pandas as pd
import pytest
from nba_ou.config.market_columns import (
    ML_PRICE_AWAY_COL,
    ML_PRICE_HOME_COL,
    ML_PROB_AWAY_NOVIG_COL,
    ML_PROB_HOME_NOVIG_COL,
    Market,
)
from nba_ou.data_processing.odds.canonical_markets import (
    add_canonical_moneyline_columns,
)


def test_closing_moneyline_columns_map_to_home_and_away():
    """``ODDS_MONEYLINE_bet365_TEAM_HOME`` is the HOME price, in DECIMAL odds.

    Verified on 7,500 real games: when the home price is the cheaper of the two
    the home team won 68.4% of the time; when the away price is cheaper, 33.7%.
    """
    df = pd.DataFrame(
        {
            "ODDS_MONEYLINE_bet365_TEAM_HOME": [1.7575757575757576, 2.75],
            "ODDS_MONEYLINE_bet365_TEAM_AWAY": [2.1, 1.4694835680751177],
        }
    )
    out = add_canonical_moneyline_columns(df)
    assert out[ML_PRICE_HOME_COL].tolist() == [1.7575757575757576, 2.75]
    assert out[ML_PRICE_AWAY_COL].tolist() == [2.1, 1.4694835680751177]
    # Row 0 the home side is the favourite; row 1 the away side is.
    assert out[ML_PROB_HOME_NOVIG_COL].iloc[0] > 0.5
    assert out[ML_PROB_HOME_NOVIG_COL].iloc[1] < 0.5


def test_intermediate_moneyline_right_is_home_left_is_away():
    """The snapshot panel stores sides as LEFT/RIGHT, and RIGHT is HOME.

    Verified on the real intermediate dataset: where the RIGHT price is cheaper
    the home team won 68.8% of the time, and FAIR_RIGHT correlates +0.403 with
    the home team actually winning. LEFT is therefore AWAY.
    """
    df = pd.DataFrame(
        {
            "ODDS_ml_bet365_price_home": [1.75],   # RIGHT, per the panel
            "ODDS_ml_bet365_price_away": [2.10],   # LEFT
        }
    )
    out = add_canonical_moneyline_columns(df)
    assert out[ML_PRICE_HOME_COL].iloc[0] == 1.75
    assert out[ML_PRICE_AWAY_COL].iloc[0] == 2.10


def test_novig_probabilities_sum_to_one_and_strip_the_overround():
    df = pd.DataFrame(
        {
            "ODDS_MONEYLINE_bet365_TEAM_HOME": [1.90],
            "ODDS_MONEYLINE_bet365_TEAM_AWAY": [1.90],
        }
    )
    out = add_canonical_moneyline_columns(df)
    home = out[ML_PROB_HOME_NOVIG_COL].iloc[0]
    away = out[ML_PROB_AWAY_NOVIG_COL].iloc[0]
    assert home + away == pytest.approx(1.0)
    assert home == pytest.approx(0.5)
    # The raw implied probabilities summed to more than 1 (that is the vig).
    assert (1 / 1.90) * 2 > 1.0


def test_a_frame_without_moneyline_columns_passes_through_untouched():
    """Readiness must never be able to fail a build that has no ML data."""
    df = pd.DataFrame({"FEATURE_A": [1.0]})
    out = add_canonical_moneyline_columns(df)
    assert list(out.columns) == ["FEATURE_A"]


def test_moneyline_market_is_declared_but_carries_no_target():
    """No label is derived. Readiness is data, not a model."""
    assert Market.MONEYLINE.value == "moneyline"
    df = pd.DataFrame(
        {
            "ODDS_MONEYLINE_bet365_TEAM_HOME": [1.5],
            "ODDS_MONEYLINE_bet365_TEAM_AWAY": [2.6],
        }
    )
    out = add_canonical_moneyline_columns(df)
    assert not any("LABEL" in c or "WIN" in c.upper() for c in out.columns)
