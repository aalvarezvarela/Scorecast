"""The ODDS_ marker must be an invariant, not a convention.

The point of the prefix is that ``[c for c in df.columns if is_odds_column(c)]``
selects *every* odds-derived feature. That only pays off if nothing can slip
through: an odds column arriving without the marker is invisible to every
consumer selecting on it, and no existing check would surface the omission.

These tests pin the guard itself and the two places it is wired in, so a new
odds feature that reaches an output frame unprefixed fails the build instead of
being silently excluded from every odds-based selection downstream.
"""

from __future__ import annotations

import pandas as pd
import pytest
from nba_ou.config.odds_columns import (
    apply_odds_prefix,
    assert_odds_columns_prefixed,
    find_unprefixed_odds_columns,
    is_odds_column,
    is_odds_shaped_column,
    strip_odds_prefix,
)
from nba_ou.create_training_data.select_intermediate_columns import (
    select_intermediate_training_columns,
)
from nba_ou.data_processing.merged_home_away_data.odds_feature_engeneer import (
    engineer_odds_features,
)
from nba_ou.data_processing.merged_home_away_data.select_train_columns import (
    select_training_columns,
)

#: One example per name shape the pipeline actually produces, taken from a real
#: training CSV header rather than invented: the canonical post-merge uppercase
#: names, the raw lowercase per-book market columns, and the rolling _BEFORE
#: variants built on top of them (which the exact-membership rename used to miss).
ODDS_SHAPED_EXAMPLES = [
    "TOTAL_LINE_bet365",
    "SPREAD_bet365_TEAM_HOME",
    "MONEYLINE_bet365_TEAM_AWAY",
    "total_bet365_price_over",
    "total_consensus_opener_line_under",
    "spread_consensus_opener_line_home",
    "ml_bet365_price_home",
    "moneyline_pct_bets_home",
    "ml_bet365_price_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME",
    "moneyline_pct_money_SEASON_BEFORE_AVG_DIFF_BEFORE",
    "total_bet365_price_over_LAST_HOME_AWAY_5_MATCHES_DIFF_BEFORE",
]

#: Columns that must never be mistaken for odds. TOTAL_POINTS is the target and
#: shares a prefix with TOTAL_LINE_ up to the underscore; the rest are the
#: ordinary team/metadata vocabulary surrounding it.
NON_ODDS_EXAMPLES = [
    "TOTAL_POINTS",
    "TOTAL_POINTS_LAST_5_GAMES_BEFORE_TEAM_HOME",
    "GAME_ID",
    "GAME_DATE",
    "SEASON_YEAR",
    "TEAM_ABBREVIATION_TEAM_HOME",
    "IS_OVERTIME",
    "PTS_LAST_5_GAMES_BEFORE_TEAM_AWAY",
    "TOTAL_KM_IN_LAST_7_DAYS_TEAM_HOME",
    "SPREADSHEET_ID",
]


@pytest.mark.parametrize("column", ODDS_SHAPED_EXAMPLES)
def test_every_odds_name_shape_is_recognised(column):
    assert is_odds_shaped_column(column)
    assert find_unprefixed_odds_columns([column]) == [column]


@pytest.mark.parametrize("column", NON_ODDS_EXAMPLES)
def test_non_odds_columns_are_not_false_positives(column):
    assert not is_odds_shaped_column(column)
    assert find_unprefixed_odds_columns([column]) == []


@pytest.mark.parametrize("column", ODDS_SHAPED_EXAMPLES)
def test_prefixed_columns_are_not_reported_as_offenders(column):
    assert find_unprefixed_odds_columns([f"ODDS_{column}"]) == []


def test_apply_odds_prefix_marks_every_odds_shape_and_leaves_the_rest():
    df = pd.DataFrame(
        {column: [1.0] for column in ODDS_SHAPED_EXAMPLES + NON_ODDS_EXAMPLES}
    )

    out = apply_odds_prefix(df)

    assert list(out.columns) == [f"ODDS_{c}" for c in ODDS_SHAPED_EXAMPLES] + (
        NON_ODDS_EXAMPLES
    )
    assert out.shape == df.shape


def test_apply_odds_prefix_is_idempotent():
    df = pd.DataFrame({column: [1.0] for column in ODDS_SHAPED_EXAMPLES})

    once = apply_odds_prefix(df)
    twice = apply_odds_prefix(once)

    assert list(twice.columns) == list(once.columns)
    assert not any(c.startswith("ODDS_ODDS_") for c in twice.columns)


def test_assert_passes_once_every_odds_column_carries_the_marker():
    df = apply_odds_prefix(
        pd.DataFrame({c: [1.0] for c in ODDS_SHAPED_EXAMPLES + NON_ODDS_EXAMPLES})
    )

    assert_odds_columns_prefixed(df.columns, context="test")

    assert all(is_odds_column(c) for c in df.columns if is_odds_shaped_column(c))


def test_assert_names_the_offenders_and_the_context():
    columns = ["GAME_ID", "ODDS_TOTAL_LINE_bet365", "total_bet365_price_over"]

    with pytest.raises(ValueError) as excinfo:
        assert_odds_columns_prefixed(columns, context="my_pipeline")

    message = str(excinfo.value)
    assert "total_bet365_price_over" in message
    assert "my_pipeline" in message
    # The already-correct columns must not be reported as problems.
    assert "ODDS_TOTAL_LINE_bet365" not in message
    assert "GAME_ID" not in message


def test_assert_truncates_but_still_counts_a_large_offender_list():
    columns = [f"total_book{i}_price_over" for i in range(30)]

    with pytest.raises(ValueError, match=r"30 odds-derived column\(s\)"):
        assert_odds_columns_prefixed(columns, context="test")


def test_strip_odds_prefix_round_trips_and_leaves_bare_names_alone():
    assert strip_odds_prefix("ODDS_TOTAL_LINE_bet365") == "TOTAL_LINE_bet365"
    assert strip_odds_prefix("TOTAL_LINE_bet365") == "TOTAL_LINE_bet365"
    assert strip_odds_prefix("GAME_ID") == "GAME_ID"


def test_intermediate_gate_rejects_an_unprefixed_odds_feature():
    """The gate keeps anything containing _BEFORE, so a rolling odds feature
    added without the marker would sail through it. The invariant is what stops
    that."""
    df = pd.DataFrame(
        {
            "GAME_ID": ["0022400001"],
            "TOTAL_POINTS": [221.0],
            "ODDS_SNAP_TOT_BET365_NORM_LINE": [220.5],
            "total_bet365_price_over_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME": [1.91],
        }
    )

    with pytest.raises(ValueError, match="unified 'ODDS_' prefix"):
        select_intermediate_training_columns(df)


def test_intermediate_gate_accepts_the_same_feature_once_prefixed():
    df = pd.DataFrame(
        {
            "GAME_ID": ["0022400001"],
            "TOTAL_POINTS": [221.0],
            "ODDS_SNAP_TOT_BET365_NORM_LINE": [220.5],
            "ODDS_total_bet365_price_over_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME": [1.91],
        }
    )

    out = select_intermediate_training_columns(df)

    assert (
        "ODDS_total_bet365_price_over_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME"
        in out.columns
    )


# --- where the marker may be applied --------------------------------------
#
# The invariant says every odds column ends up prefixed. It does NOT say the
# prefix may go on at any point: engineer_odds_features resolves its inputs by
# their raw market names, so prefixing before it runs hides them and silently
# drops every vig / no-vig / price-dispersion feature. These two tests pin the
# ordering constraint that keeps the invariant from costing features.


def test_selection_leaves_raw_market_names_for_the_odds_engineer():
    selected = select_training_columns(
        pd.DataFrame(
            {
                "GAME_ID": [1],
                "TOTAL_POINTS": [220.0],
                "ODDS_TOTAL_LINE_bet365": [220.5],
                "total_bet365_price_over": [1.91],
                "total_bet365_price_under": [1.91],
            }
        ),
        original_columns=[],
    )

    assert "total_bet365_price_over" in selected.columns
    assert "total_bet365_price_under" in selected.columns


def test_odds_engineer_still_produces_its_price_derived_features():
    selected = select_training_columns(
        pd.DataFrame(
            {
                "GAME_ID": [1, 2],
                "TOTAL_POINTS": [220.0, 224.0],
                "ODDS_TOTAL_LINE_bet365": [220.5, 224.0],
                "total_bet365_price_over": [1.91, 1.95],
                "total_bet365_price_under": [1.91, 1.87],
            }
        ),
        original_columns=[],
    )

    engineered = engineer_odds_features(selected)

    # Each of these is derived from the raw over/under prices and is exactly
    # what disappears if the prefix is applied too early.
    assert "ODDS_book_total_vig_bet365" in engineered.columns
    assert "ODDS_book_total_prob_diff_novig_bet365" in engineered.columns
    assert "ODDS_n_books_total_price_present" in engineered.columns

    # ...and once the marker does go on, the invariant holds over the result.
    final = apply_odds_prefix(engineered)
    assert_odds_columns_prefixed(final.columns, context="test")
