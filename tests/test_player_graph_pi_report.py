"""The pre-registered statistics of the phase 5 report."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = (
    Path(__file__).parents[1] / "scripts" / "player_graph" / "report_pi_walk_forward.py"
)
_spec = importlib.util.spec_from_file_location("report_pi_walk_forward", _PATH)
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def test_clustered_mean_matches_a_hand_computation():
    values = np.array([1.0, 3.0, 2.0, 6.0, 4.0])
    clusters = np.array(["d1", "d1", "d2", "d3", "d3"])
    mean, se = report.clustered_mean(values, clusters)
    assert mean == pytest.approx(3.2)
    sums = np.array([(1 - 3.2) + (3 - 3.2), 2 - 3.2, (6 - 3.2) + (4 - 3.2)])
    expected = np.sqrt((sums**2).sum() * 3 / 2) / 5
    assert se == pytest.approx(expected)


def test_decision_rule_needs_the_interval_and_the_seeds():
    agree = np.array([0.01] * 8 + [-0.01] * 2)
    assert report.decide(0.001, 0.02, agree) == "improves"
    assert report.decide(-0.001, 0.02, agree) == "null"  # interval spans 0
    split = np.array([0.01] * 7 + [-0.01] * 3)
    assert report.decide(0.001, 0.02, split) == "null"  # 7 of 10 seeds
    assert report.decide(-0.02, -0.001, -agree) == "hurts"


def test_wilson_interval_covers_the_rate():
    low, high = report.wilson(55, 100)
    assert low < 0.55 < high
    assert low == pytest.approx(0.452, abs=0.002)
    assert np.isnan(report.wilson(0, 0)[0])
