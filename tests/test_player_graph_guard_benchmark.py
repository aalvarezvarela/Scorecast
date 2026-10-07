"""The guard-edge benchmark: TVD between observed and rate-implied shares."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.player_graph.guard_benchmark import (
    benchmark_frame,
    estimator_errors,
    share_tvd,
)


def _frame():
    """One attacker, two defenders with equal co-floor time."""
    pair_game = pd.DataFrame(
        {
            "game_id": "g",
            "off_player_id": "a",
            "def_player_id": ["d1", "d2", "d3"],
            "season_year": 2018,
            "cofloor_seconds": [600.0, 600.0, 0.0],
            "matchup_seconds": [90.0, 30.0, 5.0],
        }
    )
    expected = pd.DataFrame(
        {
            "game_id": "g",
            "off_player_id": "a",
            "def_player_id": ["d1", "d2", "d3"],
            "r_prior": 0.1,
            "hist_matchup_seconds_decayed": [300.0, 100.0, 0.0],
            "hist_cofloor_seconds_decayed": [2000.0, 2000.0, 0.0],
        }
    )
    return benchmark_frame(pair_game, expected)


def test_pairs_without_shared_floor_are_left_out_and_shares_sum_to_one():
    frame = _frame()
    assert frame["def_player_id"].tolist() == ["d1", "d2"]
    assert frame["s"].tolist() == pytest.approx([0.75, 0.25])


def test_tvd_is_zero_for_a_perfect_rate_and_one_for_a_disjoint_one():
    frame = _frame()
    assert share_tvd(frame, [0.15, 0.05]).iloc[0] == pytest.approx(0.0)
    assert share_tvd(frame, [0.0, 1.0]).iloc[0] == pytest.approx(0.75)
    # Constant rate: shares follow co-floor time, 0.5 / 0.5.
    assert share_tvd(frame, np.ones(2)).iloc[0] == pytest.approx(0.25)


def test_pair_history_beats_the_flat_prior_here():
    errors = estimator_errors(_frame(), [300.0])
    assert errors.loc["pair history only (k->0)", "all"] == pytest.approx(0.0, abs=1e-8)
    assert errors.loc["position prior only (k=inf)", "all"] == pytest.approx(0.25)
    assert 0 < errors.loc["k=300", "all"] < 0.25
    assert list(errors.columns) == ["all", 2018]
