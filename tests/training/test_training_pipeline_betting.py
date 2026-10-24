import numpy as np
import pandas as pd
import pytest

from training_pipeline.betting import (
    DECIMAL_ODDS_MINUS_110,
    betting_threshold_sweep,
    break_even_win_rate,
    decimal_odds_from_american,
    evaluate_betting,
    wilson_interval,
)

BREAK_EVEN_MINUS_110 = 110.0 / 210.0  # 0.523809...


def test_minus_110_decimal_odds_and_break_even_rate():
    assert DECIMAL_ODDS_MINUS_110 == pytest.approx(1.9090909, abs=1e-6)
    assert decimal_odds_from_american(-110) == pytest.approx(DECIMAL_ODDS_MINUS_110)
    assert decimal_odds_from_american(150) == pytest.approx(2.5)
    assert break_even_win_rate(DECIMAL_ODDS_MINUS_110) == pytest.approx(
        BREAK_EVEN_MINUS_110
    )


def test_break_even_win_rate_rejects_invalid_odds():
    with pytest.raises(ValueError):
        break_even_win_rate(1.0)


def test_all_winning_bets_return_the_full_price():
    metrics = evaluate_betting(
        predicted_edge=[5.0] * 10, actual_total=[220.0] * 10, line=[210.0] * 10
    )
    assert metrics.n_bets == 10
    assert metrics.n_wins == 10
    assert metrics.win_rate == pytest.approx(1.0)
    assert metrics.roi == pytest.approx(DECIMAL_ODDS_MINUS_110 - 1.0)
    assert metrics.beats_break_even is True


def test_all_losing_bets_lose_the_full_stake():
    metrics = evaluate_betting(
        predicted_edge=[5.0] * 10, actual_total=[200.0] * 10, line=[210.0] * 10
    )
    assert metrics.n_losses == 10
    assert metrics.roi == pytest.approx(-1.0)
    assert metrics.beats_break_even is False


def test_exactly_break_even_win_rate_yields_zero_roi():
    """11 wins / 10 losses at -110 is the definition of break-even: the win
    rate equals 110/210 and profit is exactly zero.
    """
    metrics = evaluate_betting(
        predicted_edge=[5.0] * 21,
        actual_total=[220.0] * 11 + [200.0] * 10,
        line=[210.0] * 21,
    )
    assert metrics.win_rate == pytest.approx(BREAK_EVEN_MINUS_110, abs=1e-6)
    assert metrics.break_even_rate == pytest.approx(BREAK_EVEN_MINUS_110)
    assert metrics.roi == pytest.approx(0.0, abs=1e-9)


def test_under_bets_win_when_total_lands_below_the_line():
    metrics = evaluate_betting(
        predicted_edge=[-5.0] * 4, actual_total=[200.0] * 4, line=[210.0] * 4
    )
    assert metrics.n_wins == 4
    assert metrics.roi > 0


def test_pushes_return_the_stake_and_are_excluded_from_the_win_rate():
    metrics = evaluate_betting(
        predicted_edge=[5.0] * 4,
        actual_total=[210.0, 210.0, 220.0, 200.0],
        line=[210.0] * 4,
    )
    assert metrics.n_pushes == 2
    assert metrics.n_wins == 1
    assert metrics.n_losses == 1
    assert metrics.win_rate == pytest.approx(0.5)  # 1 of 2 decided bets


def test_min_edge_threshold_filters_bets():
    metrics = evaluate_betting(
        predicted_edge=[0.5, 3.0],
        actual_total=[220.0, 220.0],
        line=[210.0, 210.0],
        min_edge=2.0,
    )
    assert metrics.n_candidates == 2
    assert metrics.n_bets == 1
    assert metrics.bet_rate == pytest.approx(0.5)


def test_zero_edge_baseline_places_no_bets():
    """The pure "trust the line" baseline has exactly zero edge on every row,
    so it never bets -- which is precisely why its OU accuracy is undefined
    rather than 0%.
    """
    metrics = evaluate_betting(
        predicted_edge=[0.0] * 10, actual_total=[220.0] * 10, line=[210.0] * 10
    )
    assert metrics.n_bets == 0
    assert metrics.roi is None
    assert metrics.win_rate is None
    assert metrics.is_significant is False


def test_non_finite_rows_are_excluded_from_candidates():
    metrics = evaluate_betting(
        predicted_edge=[5.0, np.nan, 5.0],
        actual_total=[220.0, 220.0, np.nan],
        line=[210.0, 210.0, 210.0],
    )
    assert metrics.n_candidates == 1
    assert metrics.n_bets == 1


def test_real_prices_override_the_flat_price():
    """A longer price on the winning side must produce a bigger payout."""
    flat = evaluate_betting(
        predicted_edge=[5.0] * 4, actual_total=[220.0] * 4, line=[210.0] * 4
    )
    priced = evaluate_betting(
        predicted_edge=[5.0] * 4,
        actual_total=[220.0] * 4,
        line=[210.0] * 4,
        decimal_odds_over=[2.5] * 4,
        decimal_odds_under=[1.5] * 4,
    )
    assert priced.roi == pytest.approx(1.5)  # 2.5 decimal -> +1.5 units per win
    assert priced.roi > flat.roi
    assert priced.break_even_rate == pytest.approx(1 / 2.5)


def test_invalid_prices_fall_back_to_the_flat_price():
    metrics = evaluate_betting(
        predicted_edge=[5.0] * 3,
        actual_total=[220.0] * 3,
        line=[210.0] * 3,
        decimal_odds_over=[np.nan, 0.5, 2.5],
        decimal_odds_under=[1.91] * 3,
    )
    expected = np.mean([DECIMAL_ODDS_MINUS_110 - 1, DECIMAL_ODDS_MINUS_110 - 1, 1.5])
    assert metrics.roi == pytest.approx(expected)


def test_significance_requires_the_ci_lower_bound_to_clear_break_even():
    """55% on 40 bets looks like an edge but is not distinguishable from
    break-even; the same rate on a large sample is.
    """
    small = evaluate_betting(
        predicted_edge=[5.0] * 40,
        actual_total=[220.0] * 22 + [200.0] * 18,
        line=[210.0] * 40,
    )
    assert small.win_rate == pytest.approx(0.55)
    assert small.beats_break_even is True
    assert small.is_significant is False

    large = evaluate_betting(
        predicted_edge=[5.0] * 4000,
        actual_total=[220.0] * 2200 + [200.0] * 1800,
        line=[210.0] * 4000,
    )
    assert large.win_rate == pytest.approx(0.55)
    assert large.is_significant is True


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(55, 100)
    assert low < 0.55 < high
    assert 0.0 < low and high < 1.0


def test_wilson_interval_is_nan_without_trials():
    low, high = wilson_interval(0, 0)
    assert np.isnan(low) and np.isnan(high)


def test_threshold_sweep_returns_one_row_per_threshold_with_shrinking_volume():
    edges = pd.Series([0.5, 1.5, 2.5, 3.5, 4.5])
    sweep = betting_threshold_sweep(
        predicted_edge=edges,
        actual_total=[220.0] * 5,
        line=[210.0] * 5,
        thresholds=(0.0, 2.0, 4.0),
    )
    assert list(sweep["min_edge"]) == [0.0, 2.0, 4.0]
    assert list(sweep["n_bets"]) == [5, 3, 1]
    # Volume must be non-increasing as the threshold tightens.
    assert sweep["n_bets"].is_monotonic_decreasing
