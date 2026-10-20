"""Canonical, market-neutral column names and the sign conventions behind them.

Why this module exists: the same market is spelled two different ways in the two
training datasets, **with opposite signs**, and nothing in either name says so.

Measured on real games (see ``tests/test_spread_sign_convention.py``, which pins
every number below):

* Closing dataset -- ``ODDS_SPREAD_bet365_TEAM_HOME`` is a **handicap**: it is
  what the home team gives away, so it is NEGATIVE when the home team is
  favoured. Its correlation with the realised home margin is **-0.457**.
* Intermediate dataset -- ``ODDS_SNAP_SPR_BET365_RAW_LINE`` is **already** the
  market-implied home margin, POSITIVE when the home team is favoured.
  Correlation **+0.460**.

Feeding one into the formula written for the other produces a target that is
wrong by twice the spread on every row, with no error and no NaNs -- the run
completes and reports ordinary-looking numbers. That is the failure this module
exists to make impossible: every caller converts to ``SPREAD_LINE_HOME`` through
the helpers here, and the conversion is named after the source convention rather
than after the column.

One convention, stated once, for everything downstream:

    SPREAD_LINE_HOME = the market-implied FINAL HOME MARGIN
        > 0  home expected to win by that many points
        < 0  home expected to lose by that many points

    HOME_MARGIN  = PTS_TEAM_HOME - PTS_TEAM_AWAY
    SPREAD_ERROR = HOME_MARGIN - SPREAD_LINE_HOME
        > 0  home beat the Bet365 spread
        < 0  away beat the Bet365 spread
        = 0  push (a valid regression target -- see training_pipeline.data)

Moneyline is carried in canonical form too, so a future moneyline model needs no
second upstream redesign. No moneyline TARGET is derived here: readiness is
data, not a label.
"""

from __future__ import annotations

from enum import StrEnum

import pandas as pd


class Market(StrEnum):
    """The betting markets this repo carries data for.

    MONEYLINE is present so the data layer can be market-complete. It has no
    prediction strategy yet, and adding one is deliberately out of scope.
    """

    TOTALS = "totals"
    SPREAD = "spread"
    MONEYLINE = "moneyline"


#: The book that defines every target. Never substituted: swapping in another
#: book for the rows Bet365 is missing would silently change what the target
#: MEANS between observations, which is worse than losing those rows.
TARGET_ANCHOR_BOOK = "bet365"

# --- canonical outcome / target columns ---------------------------------------

#: PTS_TEAM_HOME - PTS_TEAM_AWAY. The outcome the spread market settles against.
HOME_MARGIN_COL = "HOME_MARGIN"
#: Market-implied final home margin from the anchor book (see module docstring).
SPREAD_LINE_HOME_COL = "SPREAD_LINE_HOME"
#: HOME_MARGIN - SPREAD_LINE_HOME.
SPREAD_ERROR_COL = "SPREAD_ERROR"

#: Per-team final scores. Kept through data generation ONLY so HOME_MARGIN can be
#: derived and bets settled; they are outcome facts and never features.
PTS_HOME_COL = "PTS_TEAM_HOME"
PTS_AWAY_COL = "PTS_TEAM_AWAY"

# --- canonical moneyline columns ----------------------------------------------

ML_PRICE_HOME_COL = "ODDS_ML_PRICE_HOME"
ML_PRICE_AWAY_COL = "ODDS_ML_PRICE_AWAY"
ML_PROB_HOME_NOVIG_COL = "ODDS_ML_PROB_HOME_NOVIG"
ML_PROB_AWAY_NOVIG_COL = "ODDS_ML_PROB_AWAY_NOVIG"

# --- canonical spread price columns -------------------------------------------

SPREAD_PRICE_HOME_COL = "ODDS_SPREAD_PRICE_HOME"
SPREAD_PRICE_AWAY_COL = "ODDS_SPREAD_PRICE_AWAY"

#: Cross-book spread features. Median rather than mean for the headline
#: consensus: measured on the closing data, per-book spread columns contain
#: out-of-range values (``ODDS_spread_home_line_draftkings`` reaches -120,
#: ``..._betmgm`` reaches +77.5) where a price bled into the line field. A mean
#: is dragged by those; a median is not.
SPREAD_CONSENSUS_MEDIAN_COL = "ODDS_SPREAD_CONSENSUS_MEDIAN"
SPREAD_BET365_MINUS_CONSENSUS_COL = "ODDS_SPREAD_BET365_MINUS_CONSENSUS"
SPREAD_CROSS_BOOK_STD_COL = "ODDS_SPREAD_CROSS_BOOK_STD"
SPREAD_CROSS_BOOK_RANGE_COL = "ODDS_SPREAD_CROSS_BOOK_RANGE"
SPREAD_BOOK_COUNT_COL = "ODDS_SPREAD_BOOK_COUNT"

#: Lines outside this range are transport damage, not quotes. The widest real
#: NBA spread in the closing data is 28.5; anything beyond 60 is a price that
#: landed in a line field. Applied only when building CONSENSUS features, never
#: to the anchor target column -- a corrupt anchor must drop the row loudly
#: rather than be quietly repaired into a plausible-looking number.
PLAUSIBLE_SPREAD_ABS_MAX = 60.0


def home_margin(
    df: pd.DataFrame,
    *,
    pts_home_col: str = PTS_HOME_COL,
    pts_away_col: str = PTS_AWAY_COL,
) -> pd.Series:
    """``PTS_TEAM_HOME - PTS_TEAM_AWAY`` as a numeric series."""
    return pd.to_numeric(df[pts_home_col], errors="coerce") - pd.to_numeric(
        df[pts_away_col], errors="coerce"
    )


def spread_line_home_from_handicap(handicap: pd.Series) -> pd.Series:
    """Convert a HOME-HANDICAP column into ``SPREAD_LINE_HOME``.

    Use for the closing dataset's ``ODDS_SPREAD_bet365_TEAM_HOME`` and every
    per-book ``ODDS_spread_<book>_line_home``: those are negative when the home
    team is favoured, so the implied home margin is their negation.
    """
    return -pd.to_numeric(handicap, errors="coerce")


def spread_line_home_from_implied_margin(implied_margin: pd.Series) -> pd.Series:
    """Pass through a column that is ALREADY the implied home margin.

    Use for the intermediate dataset's ``ODDS_SNAP_SPR_*_RAW_LINE`` /
    ``_NORM_LINE``. It exists as a named function rather than an assignment so
    the call site records WHICH convention the source column was in -- the whole
    point of this module.
    """
    return pd.to_numeric(implied_margin, errors="coerce")


def spread_error(home_margin_values: pd.Series, spread_line_home: pd.Series) -> pd.Series:
    """``HOME_MARGIN - SPREAD_LINE_HOME``.

    Zero is a push and is returned as ``0.0``, not dropped: for the regression
    target a push is a real, correctly-measured observation ("the market was
    exactly right"), and removing those rows would bias the error distribution
    toward the market being wrong.
    """
    return pd.to_numeric(home_margin_values, errors="coerce") - pd.to_numeric(
        spread_line_home, errors="coerce"
    )
