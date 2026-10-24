"""A shrinkage fit continued from stored sums equals the fit over everything."""

from __future__ import annotations

import numpy as np
import pytest
from nba_ou.data_processing.past_injuries.injury_effects import (
    _expanding_empirical_bayes_weights,
)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_prior_sums_continue_the_fit_exactly(seed):
    rng = np.random.default_rng(seed)
    n = 4000
    dates = np.sort(rng.integers(737000, 737400, n))
    raw = rng.normal(0, 3, n)
    se = np.abs(rng.normal(1, 0.3, n))
    inverse_n = rng.uniform(0.05, 1.0, n)
    raw[rng.random(n) < 0.2] = np.nan
    cutoff = int(dates[n // 2])
    full, _ = _expanding_empirical_bayes_weights(dates, raw, se, inverse_n)
    _, diagnostics = _expanding_empirical_bayes_weights(
        dates, raw, se, inverse_n, totals_before=cutoff
    )
    later = dates >= cutoff
    weights, _ = _expanding_empirical_bayes_weights(
        dates[later],
        raw[later],
        se[later],
        inverse_n[later],
        prior=diagnostics["totals_before"],
    )
    np.testing.assert_array_equal(weights, full[later])
    assert np.isfinite(weights).any()
