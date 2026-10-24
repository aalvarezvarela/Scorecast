"""The spread target's algebra, sign conventions and push semantics.

Every number asserted here about a REAL column's orientation was measured
against realised game outcomes before the code was written (see the
implementation report). They are pinned as tests because a sign error in this
target is invisible: it produces no NaNs, no exception and a plausible-looking
MAE, and it would be wrong by twice the spread on every row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.config.market_columns import (
    HOME_MARGIN_COL,
    SPREAD_ERROR_COL,
    home_margin,
    spread_error,
    spread_line_home_from_handicap,
    spread_line_home_from_implied_margin,
)
from nba_ou.data_processing.odds.canonical_markets import (
    add_canonical_market_columns,
    devig_two_way_prices,
)


def test_home_margin_is_home_minus_away():
    df = pd.DataFrame({"PTS_TEAM_HOME": [126, 99], "PTS_TEAM_AWAY": [121, 121]})
    assert home_margin(df).tolist() == [5, -22]


def test_spread_error_is_margin_minus_line():
    margin = pd.Series([5.0, -22.0, 1.0])
    line = pd.Series([3.5, -5.0, 1.0])
    assert spread_error(margin, line).tolist() == [1.5, -17.0, 0.0]


@pytest.mark.parametrize(
    ("pts_home", "pts_away", "handicap", "line_home", "error", "side"),
    [
        # Real games, taken from the verification table built against
        # data/train_data/training_data_2_0_20260819.csv joined to the season
        # box scores. HOME favourite, half-point line, home covers.
        (126, 121, -3.5, 3.5, 1.5, "HOME"),
        # HOME favourite, away covers.
        (113, 118, -7.5, 7.5, -12.5, "AWAY"),
        # AWAY favourite (positive handicap), home covers.
        (112, 102, 3.5, -3.5, 13.5, "HOME"),
        # AWAY favourite, away covers.
        (99, 121, 5.0, -5.0, -17.0, "AWAY"),
        # Integer line landing exactly on the number: a PUSH.
        (117, 119, 2.0, -2.0, 0.0, "PUSH"),
        (116, 109, -7.0, 7.0, 0.0, "PUSH"),
    ],
)
def test_closing_handicap_convention_on_real_games(
    pts_home, pts_away, handicap, line_home, error, side
):
    """ODDS_SPREAD_bet365_TEAM_HOME is a HANDICAP: negative means home favoured.

    Measured across 7,607 real games: as-is it correlates -0.457 with the
    realised home margin; negated, +0.457 with a mean error of -0.069 points.
    The negated orientation is the market-implied home margin.
    """
    df = pd.DataFrame(
        {
            "PTS_TEAM_HOME": [pts_home],
            "PTS_TEAM_AWAY": [pts_away],
            "handicap": [handicap],
        }
    )
    resolved = spread_line_home_from_handicap(df["handicap"])
    assert resolved.iloc[0] == line_home

    margin = home_margin(df)
    err = spread_error(margin, resolved)
    assert err.iloc[0] == error

    observed = "HOME" if err.iloc[0] > 0 else "AWAY" if err.iloc[0] < 0 else "PUSH"
    assert observed == side


def test_snapshot_line_is_already_implied_margin_not_negated():
    """ODDS_SNAP_SPR_*_RAW_LINE is the OPPOSITE convention to the closing column.

    Measured on the real intermediate dataset: as-is it correlates +0.460 with
    the realised home margin (mean error -0.015). It must therefore be passed
    through, not negated. This test exists because the two datasets disagreeing
    about a sign is the single most damaging thing that could happen to this
    target, and nothing in either column's NAME reveals it.
    """
    implied = pd.Series([3.5, -5.0])
    assert spread_line_home_from_implied_margin(implied).tolist() == [3.5, -5.0]
    # The two helpers must genuinely disagree; if someone "simplified" them into
    # one, this is what fails.
    assert spread_line_home_from_handicap(implied).tolist() == [-3.5, 5.0]


def test_push_is_a_valid_regression_target_not_a_dropped_row():
    """A push is SPREAD_ERROR == 0.0, kept, and never NaN.

    The market was exactly right on those games. Dropping them would bias the
    error distribution toward the market being wrong.
    """
    margin = pd.Series([7.0])
    line = pd.Series([7.0])
    err = spread_error(margin, line)
    assert err.iloc[0] == 0.0
    assert err.notna().all()


def test_canonical_spread_columns_normalise_every_book():
    df = pd.DataFrame(
        {
            "ODDS_SPREAD_bet365_TEAM_HOME": [-3.5, 4.0],
            "ODDS_spread_home_line_fanduel": [-3.0, 4.5],
            "ODDS_spread_home_line_draftkings": [-4.0, 5.0],
        }
    )
    out = add_canonical_market_columns(df)
    assert out["ODDS_SPREAD_LINE_HOME_bet365"].tolist() == [3.5, -4.0]
    assert out["ODDS_SPREAD_LINE_HOME_fanduel"].tolist() == [3.0, -4.5]
    assert out["ODDS_SPREAD_BOOK_COUNT"].tolist() == [3, 3]
    assert out["ODDS_SPREAD_CONSENSUS_MEDIAN"].tolist() == [3.5, -4.5]


def test_consensus_median_ignores_corrupt_book_lines():
    """A price that bled into a line field must not drag the consensus.

    Real closing data contains ODDS_spread_home_line_draftkings == -120 and
    ..._betmgm == +77.5. A mean would follow them; the median plus the
    plausibility filter must not, and the book count must say so.
    """
    df = pd.DataFrame(
        {
            "ODDS_SPREAD_bet365_TEAM_HOME": [-3.5],
            "ODDS_spread_home_line_fanduel": [-3.0],
            "ODDS_spread_home_line_draftkings": [120.0],  # corrupt
        }
    )
    out = add_canonical_market_columns(df)
    assert out["ODDS_SPREAD_BOOK_COUNT"].iloc[0] == 2
    assert out["ODDS_SPREAD_CONSENSUS_MEDIAN"].iloc[0] == pytest.approx(3.25)


def test_moneyline_home_away_mapping_and_devig():
    """ODDS_MONEYLINE_bet365_TEAM_HOME is the HOME price, in DECIMAL odds.

    Verified on real games: when it is the cheaper side the home team won 68.4%
    of the time; when the away price is cheaper, 33.7%.
    """
    df = pd.DataFrame(
        {
            "ODDS_MONEYLINE_bet365_TEAM_HOME": [1.5],
            "ODDS_MONEYLINE_bet365_TEAM_AWAY": [2.6],
        }
    )
    out = add_canonical_market_columns(df)
    assert out["ODDS_ML_PRICE_HOME"].iloc[0] == 1.5
    assert out["ODDS_ML_PRICE_AWAY"].iloc[0] == 2.6
    p_home = out["ODDS_ML_PROB_HOME_NOVIG"].iloc[0]
    p_away = out["ODDS_ML_PROB_AWAY_NOVIG"].iloc[0]
    assert p_home + p_away == pytest.approx(1.0)
    assert p_home > p_away  # the cheaper side is the favourite


def test_devig_returns_nan_for_invalid_prices():
    p_home, p_away = devig_two_way_prices(
        pd.Series([1.0, np.nan, 2.0]), pd.Series([2.0, 2.0, np.nan])
    )
    assert p_home.isna().tolist() == [True, True, True]
    assert p_away.isna().tolist() == [True, True, True]


def test_target_and_outcome_columns_are_named_as_expected():
    """Guards the constants the upstream gates and training pipeline agree on."""
    assert HOME_MARGIN_COL == "HOME_MARGIN"
    assert SPREAD_ERROR_COL == "SPREAD_ERROR"
