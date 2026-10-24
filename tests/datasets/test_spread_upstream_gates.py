"""The upstream selection gates must let the spread target through -- and only it.

These gates are allow-lists, so a new outcome column is dropped by DEFAULT. That
is the safe direction, but it means "the spread column is in the CSV" is a
property that has to be tested rather than assumed.
"""

from __future__ import annotations

import pandas as pd
from nba_ou.config.market_columns import HOME_MARGIN_COL, SPREAD_ERROR_COL
from nba_ou.config.odds_columns import (
    TARGET_NAMED_COLUMNS,
    find_unprefixed_odds_columns,
    resolve_main_spread_line_col,
    spread_line_home_col,
)
from nba_ou.create_training_data.select_intermediate_columns import (
    TARGET_COLUMNS as INTERMEDIATE_TARGETS,
)
from nba_ou.create_training_data.select_intermediate_columns import is_kept_column
from nba_ou.data_processing.merged_home_away_data.select_train_columns import (
    TARGET_COLUMNS as CLOSING_TARGETS,
)

# --- closing gate -----------------------------------------------------------


def test_closing_gate_carries_the_outcome_columns():
    for column in ("TOTAL_POINTS", "HOME_MARGIN", "PTS_TEAM_HOME", "PTS_TEAM_AWAY"):
        assert column in CLOSING_TARGETS


def test_totals_target_is_still_first_and_unchanged():
    """Existing totals behaviour must be untouched."""
    assert CLOSING_TARGETS[0] == "TOTAL_POINTS"


# --- intermediate gate ------------------------------------------------------


def test_intermediate_gate_keeps_the_spread_target_and_outcome():
    for column in (SPREAD_ERROR_COL, HOME_MARGIN_COL, "PTS_TEAM_HOME", "PTS_TEAM_AWAY"):
        assert column in INTERMEDIATE_TARGETS
        assert is_kept_column(column)


def test_intermediate_gate_keeps_the_snapshot_spread_anchor():
    """It is the target's reference line and the price bets settle into."""
    assert is_kept_column(spread_line_home_col("bet365"))


def test_intermediate_gate_still_blocks_closing_market_columns():
    """The safeguard that must not be weakened: closing values stay out."""
    for column in (
        "ODDS_CLOSING_SPREAD_LINE_HOME_bet365",
        "ODDS_CLOSING_TOTAL_LINE_bet365",
        "ODDS_CLOSING_ML_PRICE_HOME",
    ):
        assert not is_kept_column(column)


def test_intermediate_gate_still_keeps_totals_targets():
    assert is_kept_column("TOTAL_POINTS")
    assert is_kept_column("LINE_ERROR")


def test_a_bare_market_column_is_still_rejected():
    """Random non-_BEFORE market columns must not slip through the gate."""
    assert not is_kept_column("ODDS_SPREAD_bet365")
    assert not is_kept_column("ODDS_spread_fanduel_line_home")


# --- the ODDS_ prefix invariant ---------------------------------------------


def test_spread_error_is_exempt_from_the_odds_prefix_invariant():
    """It is a TARGET named after a relationship, not a market feature.

    Without the exemption ``apply_odds_prefix`` would rename it to
    ODDS_SPREAD_ERROR and every consumer selecting on is_odds_column() would
    treat the training target as a market feature.
    """
    assert SPREAD_ERROR_COL in TARGET_NAMED_COLUMNS
    assert find_unprefixed_odds_columns([SPREAD_ERROR_COL]) == []


def test_the_canonical_spread_column_carries_the_odds_marker():
    """It IS a market quote, so it must be selectable as one."""
    assert spread_line_home_col("bet365") == "ODDS_SPREAD_LINE_HOME_bet365"
    assert find_unprefixed_odds_columns([spread_line_home_col("bet365")]) == []


def test_a_genuinely_unprefixed_market_column_is_still_caught():
    """Mutation guard: the exemption must not have disabled the invariant."""
    assert find_unprefixed_odds_columns(["spread_bet365_line_home"]) == [
        "spread_bet365_line_home"
    ]


# --- anchor resolution ------------------------------------------------------


def test_spread_anchor_never_falls_back_to_another_book():
    """Substituting a book would change what the target MEANS between rows."""
    df = pd.DataFrame({"ODDS_SPREAD_LINE_HOME_fanduel": [3.5]})
    assert resolve_main_spread_line_col(df, "bet365") is None


def test_spread_anchor_resolves_when_present():
    df = pd.DataFrame({"ODDS_SPREAD_LINE_HOME_bet365": [3.5]})
    assert resolve_main_spread_line_col(df, "bet365") == "ODDS_SPREAD_LINE_HOME_bet365"
