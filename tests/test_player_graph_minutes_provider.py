"""Reconciliation of the v1 minutes to 0 <= m <= 48, sum = 240."""

import numpy as np
import pytest
from nba_ou.data_processing.player_graph.minutes_provider import reconcile


def test_without_the_cap_it_is_proportional_rescaling():
    raw = np.array([30.0, 28.0, 25.0, 20.0, 18.0, 15.0, 12.0, 8.0])
    out = reconcile(raw)
    assert out.feasible and out.capped == 0
    assert out.minutes.sum() == pytest.approx(240.0)
    assert out.minutes == pytest.approx(raw * 240.0 / raw.sum())


def test_the_cap_binds_and_the_excess_goes_to_the_others():
    raw = np.array([60.0, 30.0, 20.0, 15.0, 10.0, 5.0])
    out = reconcile(raw)
    assert out.feasible and out.capped >= 1
    assert out.minutes.max() == pytest.approx(48.0)
    assert out.minutes.sum() == pytest.approx(240.0)
    assert (out.minutes >= 0).all()
    uncapped = out.minutes < 48 - 1e-9
    ratio = out.minutes[uncapped] / raw[uncapped]
    assert ratio == pytest.approx(np.full(uncapped.sum(), ratio[0]))


def test_zeros_stay_zero_and_too_few_players_are_infeasible():
    out = reconcile(np.array([30.0, 0.0, 25.0, 20.0, 18.0, 15.0]))
    assert out.minutes[1] == 0.0 and out.minutes.sum() == pytest.approx(240.0)
    short = reconcile(np.array([30.0, 30.0, 30.0, 30.0]))
    assert not short.feasible and short.minutes.sum() < 240.0
