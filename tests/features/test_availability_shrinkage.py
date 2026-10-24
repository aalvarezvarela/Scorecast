"""The availability-effect shrinkage weight is fitted, not assumed.

`shrinkage_k` used to be a hand-picked 10.0 applied to every metric. These
tests pin the empirical-Bayes replacement: the weight must follow the measured
ratio of real player-to-player variation to sampling noise, and it must be
estimated only from games that precede the row it is applied to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.past_injuries.injury_effects import (
    MIN_SHRINKAGE_FIT_SAMPLES,
    _expanding_empirical_bayes_weights,
    add_top3_availability_effect_features_for_columns,
)


def _population(
    *,
    tau: float,
    standard_error: float,
    n_entries: int = 4000,
    sigma2: float = 10.0,
    seed: int = 0,
) -> tuple[np.ndarray, ...]:
    """Effects drawn as true player value + sampling noise, one game per date."""
    rng = np.random.default_rng(seed)
    true_effect = rng.normal(0.0, tau, n_entries) if tau > 0 else np.zeros(n_entries)
    raw = true_effect + rng.normal(0.0, standard_error, n_entries)
    dates = np.arange(n_entries, dtype="int64")
    standard_errors = np.full(n_entries, standard_error, dtype="float64")
    # se2 = sigma2 * inverse_n, so the two stay mutually consistent.
    inverse_n = np.full(n_entries, standard_error**2 / sigma2, dtype="float64")
    return dates, raw, standard_errors, inverse_n


def test_weight_matches_the_signal_to_noise_ratio_it_measures():
    """tau=3, se=1 implies a weight of 9/(9+1); nothing here is told that."""
    weights, diagnostics = _expanding_empirical_bayes_weights(
        *_population(tau=3.0, standard_error=1.0)
    )

    assert diagnostics["tau2"] == pytest.approx(9.0, rel=0.15)
    assert weights[-1] == pytest.approx(0.9, rel=0.05)
    assert diagnostics["implied_k"] == pytest.approx(10.0 / 9.0, rel=0.2)


def test_pure_noise_shrinks_effects_to_nothing():
    """No real player-to-player variation means no estimate worth keeping.

    tau2 floors at zero when the observed spread is fully explained by sampling
    noise, and the honest effect is then zero rather than a small random number.
    """
    weights, diagnostics = _expanding_empirical_bayes_weights(
        *_population(tau=0.0, standard_error=1.0)
    )

    assert diagnostics["tau2"] == pytest.approx(0.0, abs=0.1)
    assert np.nanmax(weights) < 0.15


def test_a_strong_signal_measured_precisely_is_barely_shrunk():
    weights, diagnostics = _expanding_empirical_bayes_weights(
        *_population(tau=10.0, standard_error=0.5)
    )

    assert diagnostics["implied_k"] < 1.0
    assert weights[-1] > 0.99


def test_nothing_is_shrunk_using_games_that_come_later():
    """The window is expanding, so early rows have no fit and stay NaN."""
    dates, raw, standard_errors, inverse_n = _population(tau=3.0, standard_error=1.0)
    weights, _ = _expanding_empirical_bayes_weights(
        dates, raw, standard_errors, inverse_n
    )

    assert np.isnan(weights[:MIN_SHRINKAGE_FIT_SAMPLES]).all()
    assert np.isfinite(weights[MIN_SHRINKAGE_FIT_SAMPLES:]).all()


def test_a_noisier_estimate_on_the_same_date_is_shrunk_harder():
    """The weight uses each player's own precision, not just their game count."""
    dates, raw, standard_errors, inverse_n = _population(tau=3.0, standard_error=1.0)
    sigma2 = 10.0
    for index, precise_se in ((-2, 0.25), (-1, 4.0)):
        standard_errors[index] = precise_se
        inverse_n[index] = precise_se**2 / sigma2
        raw[index] = 5.0

    weights, _ = _expanding_empirical_bayes_weights(
        dates, raw, standard_errors, inverse_n
    )

    assert weights[-2] > weights[-1]
    assert weights[-2] > 0.9
    assert weights[-1] < 0.5


def test_a_missing_standard_error_is_imputed_rather_than_dropped():
    """One game on a side gives no spread, but the effect still gets a weight."""
    dates, raw, standard_errors, inverse_n = _population(tau=3.0, standard_error=1.0)
    standard_errors[-1] = np.nan
    inverse_n[-1] = 1.0 / 1.0 + 1.0 / 8.0

    weights, _ = _expanding_empirical_bayes_weights(
        dates, raw, standard_errors, inverse_n
    )

    assert np.isfinite(weights[-1])
    # sigma2 ~= 10 and inverse_n ~= 1.125 imply se2 ~= 11, well above tau2 ~= 9.
    assert weights[-1] < 0.5


def test_too_little_history_reports_no_fit():
    dates, raw, standard_errors, inverse_n = _population(
        tau=3.0, standard_error=1.0, n_entries=MIN_SHRINKAGE_FIT_SAMPLES - 1
    )
    weights, diagnostics = _expanding_empirical_bayes_weights(
        dates, raw, standard_errors, inverse_n
    )

    assert np.isnan(weights).all()
    assert diagnostics["n_fit"] == 0.0


def _planted_effect_frame(n_games: int = 600, n_teams: int = 8) -> tuple:
    """Eight home stars worth different amounts of spread; none worth any totals.

    Between-player variation is the quantity `tau2` estimates, so the population
    has to contain several distinct players. A single player repeated across
    rows has none by construction, and is shrunk to zero -- see
    `test_a_single_player_population_has_no_between_player_variance`.
    """
    rng = np.random.default_rng(7)
    # Signed effects would average to zero across teams and hide themselves in
    # the aggregate; a star being out costing the team points is also the real
    # direction, so the planted effects are all positive.
    true_effect = rng.uniform(4.0, 10.0, n_teams)

    rows, availability = [], {}
    for i in range(n_games):
        slot = i % n_teams
        home_team, away_team = 10 + slot, 30 + slot
        home_star, away_star = 100 + slot, 300 + slot
        star_out = (i // n_teams) % 3 == 0
        game_id = i + 1
        rows.append(
            {
                "GAME_ID": game_id,
                "GAME_DATE": pd.Timestamp("2024-10-01") + pd.Timedelta(days=i),
                "SEASON_YEAR": 2024,
                "TEAM_ID_TEAM_HOME": home_team,
                "TEAM_ID_TEAM_AWAY": away_team,
                "TOTAL_POINTS": 220 + rng.normal(0, 12),
                "HOME_MARGIN": (3.0 - true_effect[slot] * star_out + rng.normal(0, 11)),
                "ODDS_TOTAL_LINE_bet365": 220.0,
                "ODDS_SPREAD_LINE_HOME_bet365": 3.0,
                "HOME_PLAYER": home_star,
                "AWAY_PLAYER": away_star,
            }
        )
        availability[str(game_id)] = {
            str(home_team): {
                "available": [] if star_out else [str(home_star)],
                "injured": [str(home_star)] if star_out else [],
            },
            str(away_team): {"available": [str(away_star)], "injured": []},
        }
    return pd.DataFrame(rows), availability


def _build(frame, availability, **overrides):
    return add_top3_availability_effect_features_for_columns(
        frame,
        injured_dict={},
        availability_dict=availability,
        total_line_book="bet365",
        spread_line_book="bet365",
        home_player_cols=("HOME_PLAYER",),
        away_player_cols=("AWAY_PLAYER",),
        out_prefix="AV",
        **overrides,
    )


def test_the_fitted_weight_is_wired_in_and_leaves_no_gaps():
    """End-to-end wiring: the fit runs, and every aggregate stays finite.

    Effect magnitude is deliberately not asserted. Eight distinct players is
    far too small a population to separate a real effect from sampling noise at
    NBA scale, and the fit correctly reports `tau2 = 0` for some metrics on
    this frame. The estimator's statistics are pinned by the direct tests
    above, which use thousands of independent observations; this test only
    proves it is connected and does not leave NaN behind.
    """
    frame, availability = _planted_effect_frame()
    late = _build(frame, availability).iloc[-100:]

    for metric in ("TOTAL_POINTS", "DIFF_FROM_LINE", "SPREAD_ERROR"):
        assert late[f"AV_HOME_MEAN_{metric}"].notna().all(), metric
        assert late[f"AV_AWAY_MEAN_{metric}"].notna().all(), metric
    assert late["AV_HOME_MEAN_SE_SPREAD_ERROR"].notna().any()
    assert late["AV_HOME_MEAN_TOTAL_POINTS"].abs().max() > 0.0


def test_fitting_changes_the_result_versus_the_fixed_prior():
    """Whatever the fitted weight is, it must not be the hard-coded k."""
    frame, availability = _planted_effect_frame()
    fitted = _build(frame, availability).iloc[-100:]
    fallback = _build(frame, availability, fit_shrinkage=False).iloc[-100:]

    assert not np.allclose(
        fitted["AV_HOME_MEAN_SPREAD_ERROR"].to_numpy(),
        fallback["AV_HOME_MEAN_SPREAD_ERROR"].to_numpy(),
    )


def test_a_single_player_population_has_no_between_player_variance():
    """One player repeated across rows gives `tau2 = 0`, so effects go to zero.

    This is correct, not a bug: with a single player there is no player-to-
    player variation for the estimator to find, and every observed difference
    is sampling noise. It is recorded because it is the one case where the
    fitted weight collapses a feature that a fixed `k` would have kept.
    """
    frame, availability = _planted_effect_frame(n_games=600, n_teams=1)
    late = _build(frame, availability).iloc[-100:]

    assert late["AV_HOME_MEAN_SPREAD_ERROR"].abs().max() == pytest.approx(0.0)
