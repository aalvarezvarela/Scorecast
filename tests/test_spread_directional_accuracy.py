"""Directional (``ou_acc``) semantics for a spread run.

The metric NAME is deliberately not changed -- registry and archived-artifact
compatibility outrank naming tidiness -- so what it MEANS for a spread run has to
be pinned by a test instead of by its name. For SPREAD_ERROR it is HOME/AWAY
line-side accuracy with pushes excluded.
"""

from __future__ import annotations

import numpy as np
from nba_ou.modeling.scorers import over_under_betting_accuracy_error_line

from training_pipeline.config import DataConfig, ExperimentConfig, PredictionStrategy
from training_pipeline.tuning import LineErrorStrategy, get_strategy


def test_spread_and_line_error_share_one_residual_strategy():
    """Which is what makes the push convention identical for both, for free."""

    def config(strategy):
        return ExperimentConfig(
            experiment_name="s",
            prediction_strategy=strategy,
            data=DataConfig(csv_path="x.csv"),
            save_experiment_artifacts=False,
        )

    spread = get_strategy(config(PredictionStrategy.SPREAD_ERROR_REGRESSOR))
    totals = get_strategy(config(PredictionStrategy.LINE_ERROR_REGRESSOR))
    assert isinstance(spread, LineErrorStrategy)
    assert isinstance(totals, LineErrorStrategy)


def test_push_rows_are_excluded_from_directional_accuracy():
    """A true error of 0 must not be scored as HOME or AWAY.

    Two correct calls, one push. Accuracy is 2/2, not 2/3.
    """
    y_true = np.array([5.0, -5.0, 0.0])
    y_pred = np.array([2.0, -2.0, 3.0])
    assert over_under_betting_accuracy_error_line(y_true, y_pred) == 1.0


def test_a_push_is_not_silently_counted_as_a_loss_either():
    """The complementary error: 1 correct, 1 wrong, 1 push -> 0.5, not 1/3."""
    y_true = np.array([5.0, 5.0, 0.0])
    y_pred = np.array([2.0, -2.0, 3.0])
    assert over_under_betting_accuracy_error_line(y_true, y_pred) == 0.5


def test_an_exactly_zero_prediction_is_excluded_too():
    """The existing convention: a market-neutral prediction picks no side.

    Documented rather than changed -- this is what line_error_regressor already
    does, and spread inherits it by using the same scorer.
    """
    y_true = np.array([5.0, -5.0])
    y_pred = np.array([0.0, -2.0])
    # Only the second row scores, and it is correct.
    assert over_under_betting_accuracy_error_line(y_true, y_pred) == 1.0


def test_all_pushes_returns_zero_not_a_crash():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, -1.0])
    assert over_under_betting_accuracy_error_line(y_true, y_pred) == 0.0


def test_home_side_is_positive_and_away_side_is_negative():
    """Sign mapping, stated once so it cannot drift from the betting layer."""
    y_true = np.array([4.0, -4.0])
    # Predicting HOME on the home-cover game and AWAY on the away-cover game.
    assert over_under_betting_accuracy_error_line(y_true, np.array([1.0, -1.0])) == 1.0
    # Both backwards.
    assert over_under_betting_accuracy_error_line(y_true, np.array([-1.0, 1.0])) == 0.0
