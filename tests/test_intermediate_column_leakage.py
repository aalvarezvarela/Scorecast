"""Tests for the intermediate dataset's leakage gate.

Each test plants the leak it claims to catch, so a guard that stopped working
would fail here rather than pass quietly. That is the mutation check: the
"before" state is constructed in the test itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.create_training_data.select_intermediate_columns import (
    LEAKY_BEFORE_COLUMNS,
    assert_no_bare_closing_odds,
    audit_closing_line_reconstruction,
    feature_columns,
    is_kept_column,
    select_intermediate_training_columns,
)


def test_before_columns_named_after_closing_prices_are_dropped():
    """``_BEFORE`` does not mean safe in this dataset.

    These four are computed from the current game's CLOSING prices despite the
    suffix. They are legitimate in the closing-line dataset and leakage here.
    """
    for column in LEAKY_BEFORE_COLUMNS:
        assert not is_kept_column(column), column


def test_ordinary_before_columns_are_kept():
    assert is_kept_column("PTS_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME")
    assert is_kept_column("GLOBAL_CROSSBOOK_TOTAL_STD_AVG_15G_BEFORE")


def test_current_game_closing_odds_are_dropped():
    for column in [
        "ODDS_TOTAL_LINE_betmgm",
        "total_bet365_price_over",
        "spread_fanduel_line_home",
        "ml_caesars_price_away",
        "total_pct_bets_over",
    ]:
        assert not is_kept_column(column), column


def test_snapshot_and_schedule_columns_are_kept():
    assert is_kept_column("ODDS_SNAP_TOT_BET365_NORM_LINE")
    assert is_kept_column("TOTAL_KM_IN_LAST_7_DAYS_HOME_TEAM")
    assert is_kept_column("JETLAG_HOURS_FROM_LAST_GAME_AWAY_TEAM")


def test_gate_raises_if_a_leaky_column_is_forced_through():
    """Mutation check: re-inject the leak and the gate must refuse it."""
    frame = pd.DataFrame({"GAME_ID": ["1"], "PTS_SEASON_BEFORE_AVG_TEAM_HOME": [110.0]})
    assert not select_intermediate_training_columns(frame).empty

    import nba_ou.create_training_data.select_intermediate_columns as module

    original = module.is_kept_column
    module.is_kept_column = lambda column: True  # the bug being guarded against
    try:
        leaked = frame.assign(IMPLIED_PTS_HOME_BEFORE=[113.0])
        with pytest.raises(ValueError, match="survived the intermediate gate"):
            module.select_intermediate_training_columns(leaked)
    finally:
        module.is_kept_column = original


def test_feature_columns_is_a_reporting_helper_not_a_safety_boundary():
    """It excludes metadata and targets, but nothing downstream consults it.

    Superseded ``test_scoring_only_closing_columns_are_carried_but_not_features``,
    which asserted that closing lines were *kept* in the training frame and
    filtered by this helper -- the assumption that produced the leak.
    """
    frame = pd.DataFrame(
        {
            "GAME_ID": ["1"],
            "TOTAL_POINTS": [221.0],
            "PTS_SEASON_BEFORE_AVG_TEAM_HOME": [110.0],
        }
    )
    assert feature_columns(frame) == ["PTS_SEASON_BEFORE_AVG_TEAM_HOME"]


def test_audit_catches_a_pair_that_reconstructs_the_closing_line():
    """The IMPLIED_PTS case, reproduced.

    Neither column alone looks alarming -- each correlates about 0.8 with the
    line -- but they sum to it exactly. A single-column correlation screen
    misses this, which is why the audit also searches pairs.
    """
    rng = np.random.default_rng(0)
    closing = pd.Series(rng.normal(224.0, 10.0, 400))
    spread = pd.Series(rng.normal(0.0, 12.0, 400))
    frame = pd.DataFrame(
        {
            "IMPLIED_PTS_HOME_LOOKALIKE_BEFORE": closing / 2 - spread / 2,
            "IMPLIED_PTS_AWAY_LOOKALIKE_BEFORE": closing / 2 + spread / 2,
        }
    )
    findings = audit_closing_line_reconstruction(frame, closing)
    assert not findings.empty
    assert (findings["kind"] == "pair_sum").any()


def test_audit_is_quiet_on_genuinely_safe_features():
    rng = np.random.default_rng(1)
    closing = pd.Series(rng.normal(224.0, 10.0, 400))
    frame = pd.DataFrame(
        {
            # Correlated with the line, as a team's rolling history genuinely is,
            # but carrying no exact reconstruction of it.
            "PTS_LAST_ALL_5_MATCHES_BEFORE_TEAM_HOME": closing * 0.4
            + rng.normal(0, 8, 400),
            "REST_DAYS_BEFORE_MATCH_TEAM_HOME": rng.integers(0, 4, 400).astype(float),
        }
    )
    assert audit_closing_line_reconstruction(frame, closing).empty


def test_bare_closing_odds_check_allows_only_the_designated_anchor():
    frame = pd.DataFrame(
        {"ODDS_TOTAL_LINE_bet365": [224.5], "ODDS_TOTAL_LINE_fanduel": [225.0]}
    )
    with pytest.raises(ValueError, match="ODDS_TOTAL_LINE_fanduel"):
        assert_no_bare_closing_odds(frame, allowed=("ODDS_TOTAL_LINE_bet365",))


def test_anchor_column_alone_passes_the_bare_odds_check():
    """In this dataset the anchor column holds the SNAPSHOT line, by design."""
    frame = pd.DataFrame({"ODDS_TOTAL_LINE_bet365": [224.5]})
    assert_no_bare_closing_odds(frame, allowed=("ODDS_TOTAL_LINE_bet365",))


def test_closing_columns_never_survive_the_gate():
    """Regression, and the most serious defect found in review.

    These were previously KEPT behind a ``CLOSING_`` prefix, on the assumption
    that ``feature_columns()`` would filter them downstream. It does not:
    ``training_pipeline.data.build_feature_matrix`` drops only the *configured*
    exclusions, so every closing line entered X. Being absent from the frame is
    the only guarantee that does not depend on a caller remembering something.
    """
    frame = pd.DataFrame(
        {
            "GAME_ID": ["1"],
            "PTS_SEASON_BEFORE_AVG_TEAM_HOME": [110.0],
            "ODDS_CLOSING_TOTAL_LINE_bet365": [224.5],
            "ODDS_CLOSING_TOTAL_LINE_caesars": [225.0],
        }
    )
    gated = select_intermediate_training_columns(frame)
    assert not [c for c in gated.columns if c.startswith("ODDS_CLOSING_")]
    assert not is_kept_column("ODDS_CLOSING_TOTAL_LINE_draftkings")


def test_bare_closing_odds_check_also_flags_closing_prefixed_columns():
    frame = pd.DataFrame({"ODDS_CLOSING_TOTAL_LINE_bet365": [224.5]})
    with pytest.raises(ValueError, match="ODDS_CLOSING_TOTAL_LINE_bet365"):
        assert_no_bare_closing_odds(frame, allowed=("ODDS_TOTAL_LINE_bet365",))


def test_consensus_opener_survives_under_its_own_name():
    """It is the configured `comparison_line_cols` baseline and is leakage-safe.

    Openers land a median ~25h before tip, outside the whole snapshot grid.
    Sweeping it into ``CLOSING_*`` silently disabled that baseline.
    """
    assert is_kept_column("ODDS_TOTAL_LINE_consensus_opener")
    frame = pd.DataFrame({"ODDS_TOTAL_LINE_consensus_opener": [223.0]})
    gated = select_intermediate_training_columns(frame)
    assert "ODDS_TOTAL_LINE_consensus_opener" in gated.columns
    assert_no_bare_closing_odds(frame, allowed=("ODDS_TOTAL_LINE_bet365",))


def test_raw_timestamps_are_not_features():
    """A raw tipoff timestamp lets a model pin individual games."""
    assert not is_kept_column("TIPOFF_UTC")
    assert not is_kept_column("SNAPSHOT_TS_UTC")
