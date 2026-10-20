"""Spread bets settle through the existing betting layer, unchanged.

``evaluate_betting`` was written for totals but its arithmetic is
market-agnostic: an edge picks a side, an outcome is compared to a line, and a
tie is a push. These tests pin that a spread bet handed to it settles correctly,
including that a push returns the stake rather than counting as a loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training_pipeline.betting import (
    DECIMAL_ODDS_MINUS_110,
    OUTCOME_COLUMN,
    evaluate_betting,
    outcome_from_predictions,
)
from training_pipeline.config import (
    BettingConfig,
    DataConfig,
    ExperimentConfig,
    Market,
    PredictionStrategy,
)
from training_pipeline.decisions import (
    collect_prices,
    decisions_from_pooled_predictions,
)

PRICE = DECIMAL_ODDS_MINUS_110


def _spread_config(**overrides):
    kwargs = {
        "experiment_name": "s",
        "prediction_strategy": PredictionStrategy.SPREAD_ERROR_REGRESSOR,
        "data": DataConfig(csv_path="x.csv"),
        "save_experiment_artifacts": False,
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def test_home_cover_away_cover_and_push_settle_correctly():
    """Three games, one of each outcome, all bet at the same price.

    Row 0: model says HOME by 6, home margin 8 vs line 3   -> HOME covers, WIN
    Row 1: model says AWAY by 6, home margin -8 vs line 3  -> AWAY covers, WIN
    Row 2: model says HOME by 6, home margin 3 vs line 3   -> PUSH
    """
    metrics = evaluate_betting(
        predicted_edge=np.array([6.0, -6.0, 6.0]),
        actual_total=np.array([8.0, -8.0, 3.0]),   # HOME_MARGIN
        line=np.array([3.0, 3.0, 3.0]),            # SPREAD_LINE_HOME
        min_edge=0.0,
        flat_decimal_odds=PRICE,
    )
    assert metrics.n_bets == 3
    assert metrics.n_wins == 2
    assert metrics.n_losses == 0
    assert metrics.n_pushes == 1
    # A push returns the stake: profit is two wins, nothing lost or gained on it.
    assert metrics.profit_units == pytest.approx(2 * (PRICE - 1.0))
    # ...and it is excluded from the win rate rather than counted as a win.
    assert metrics.win_rate == pytest.approx(1.0)


def test_a_push_is_never_scored_as_a_loss():
    """Isolated, because 'push counted as loss' is a plausible off-by-one and
    would quietly depress every spread ROI by the integer-line push rate."""
    metrics = evaluate_betting(
        predicted_edge=np.array([4.0]),
        actual_total=np.array([7.0]),
        line=np.array([7.0]),
        min_edge=0.0,
        flat_decimal_odds=PRICE,
    )
    assert metrics.n_pushes == 1
    assert metrics.n_losses == 0
    assert metrics.n_wins == 0
    assert metrics.profit_units == 0.0


def test_losing_sides_are_scored_as_losses():
    metrics = evaluate_betting(
        predicted_edge=np.array([5.0, -5.0]),
        actual_total=np.array([-2.0, 10.0]),
        line=np.array([3.0, 3.0]),
        min_edge=0.0,
        flat_decimal_odds=PRICE,
    )
    assert metrics.n_wins == 0
    assert metrics.n_losses == 2
    assert metrics.profit_units == pytest.approx(-2.0)


def test_edge_sign_selects_home_then_away():
    """positive edge => HOME. The whole betting layer depends on this mapping."""
    home_only = evaluate_betting(
        predicted_edge=np.array([5.0]),
        actual_total=np.array([10.0]),
        line=np.array([3.0]),
        min_edge=0.0,
        flat_decimal_odds=PRICE,
    )
    assert home_only.n_wins == 1  # home covered and we backed home

    away_only = evaluate_betting(
        predicted_edge=np.array([-5.0]),
        actual_total=np.array([10.0]),
        line=np.array([3.0]),
        min_edge=0.0,
        flat_decimal_odds=PRICE,
    )
    assert away_only.n_losses == 1  # home covered but we backed away


def test_asymmetric_home_away_prices_are_applied_to_the_right_side():
    """Backing HOME must pay the HOME price."""
    metrics = evaluate_betting(
        predicted_edge=np.array([5.0]),
        actual_total=np.array([10.0]),
        line=np.array([3.0]),
        min_edge=0.0,
        flat_decimal_odds=PRICE,
        decimal_odds_over=np.array([3.0]),   # HOME side
        decimal_odds_under=np.array([1.2]),  # AWAY side
    )
    assert metrics.n_wins == 1
    assert metrics.profit_units == pytest.approx(2.0)  # 3.0 - 1


def test_collect_prices_maps_home_to_the_first_slot():
    """HOME occupies the OVER slot because both are selected by a positive edge.

    RIGHT is HOME and LEFT is AWAY; this test uses decimal prices because raw
    snapshot PRICE_RIGHT/PRICE_LEFT values are American odds in the real data.
    """
    config = _spread_config(
        betting=BettingConfig(
            home_price_col="ODDS_SNAP_SPR_BET365_PRICE_RIGHT",
            away_price_col="ODDS_SNAP_SPR_BET365_PRICE_LEFT",
        )
    )
    assert config.market is Market.SPREAD
    df = pd.DataFrame(
        {
            "ODDS_SNAP_SPR_BET365_PRICE_RIGHT": [1.80],
            "ODDS_SNAP_SPR_BET365_PRICE_LEFT": [2.05],
        }
    )
    home_side, away_side = collect_prices(df, config)
    assert home_side.tolist() == [1.80]
    assert away_side.tolist() == [2.05]


@pytest.mark.parametrize("american_price", [-110.0, 100.0])
def test_collect_prices_rejects_american_odds(american_price):
    config = _spread_config(
        betting=BettingConfig(
            home_price_col="HOME_PRICE",
            away_price_col="AWAY_PRICE",
        )
    )
    df = pd.DataFrame(
        {
            "HOME_PRICE": [american_price],
            "AWAY_PRICE": [1.91],
        }
    )

    with pytest.raises(ValueError, match="must contain decimal odds"):
        collect_prices(df, config)


def test_totals_price_columns_still_route_to_over_under():
    config = _spread_config(
        prediction_strategy=PredictionStrategy.LINE_ERROR_REGRESSOR,
        betting=BettingConfig(
            over_price_col="ODDS_total_bet365_price_over",
            under_price_col="ODDS_total_bet365_price_under",
        ),
    )
    df = pd.DataFrame(
        {
            "ODDS_total_bet365_price_over": [1.91],
            "ODDS_total_bet365_price_under": [1.95],
        }
    )
    over, under = collect_prices(df, config)
    assert over.tolist() == [1.91]
    assert under.tolist() == [1.95]


def test_spread_prediction_is_already_the_edge():
    """No line is subtracted -- the residual model predicts the edge directly."""
    config = _spread_config()
    decisions = decisions_from_pooled_predictions(
        np.array([2.5, -1.5]),
        target_line=np.array([3.0, 3.0]),
        config=config,
    )
    assert decisions.predicted_edge.tolist() == [2.5, -1.5]
    # Adding the line back recovers the implied HOME MARGIN.
    assert decisions.predicted_total.tolist() == [5.5, 1.5]
    assert decisions.selection_score.tolist() == [2.5, 1.5]


def test_mismatched_prices_are_rejected_by_config():
    with pytest.raises(ValueError, match="home_price_col"):
        BettingConfig(home_price_col="a")


def test_outcome_column_falls_back_for_archived_runs():
    """Archived runs wrote TOTAL_POINTS; new ones write actual_outcome."""
    new = pd.DataFrame({OUTCOME_COLUMN: [8.0]})
    old = pd.DataFrame({"TOTAL_POINTS": [221.0]})
    assert outcome_from_predictions(new).tolist() == [8.0]
    assert outcome_from_predictions(old).tolist() == [221.0]
    with pytest.raises(KeyError, match="no realised outcome"):
        outcome_from_predictions(pd.DataFrame({"y_pred": [1.0]}))
