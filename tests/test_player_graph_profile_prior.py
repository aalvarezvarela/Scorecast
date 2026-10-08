"""Profile prior f: pairs, pseudo-targets and the weighted ridge."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.player_graph.node_profiles import PROFILE_COLUMNS
from nba_ou.data_processing.player_graph.profile_prior import (
    FEATURES,
    S_MIN,
    debut_prior,
    fit_prior,
    monthly_checkpoints,
    prior_pairs,
    select_alpha,
)


def _profiles(n=400, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        rng.normal(size=(n, len(PROFILE_COLUMNS))), columns=list(PROFILE_COLUMNS)
    )
    frame["games_in_data"] = rng.integers(1, 200, n)
    frame["has_box_history"] = True
    return frame.assign(
        as_of_date=pd.Timestamp("2018-01-01"), player_id=[str(k) for k in range(n)]
    )


def _ratings(profiles, true, exposure):
    lam = 3_000.0
    s = exposure / (exposure + lam)
    return pd.DataFrame(
        {
            "as_of_date": profiles["as_of_date"],
            "player_id": profiles["player_id"],
            "o_rating": s * true,  # what a diagonal ridge would report
            "d_rating": s * true,
            "pace_rating": 0.0,
            "poss_weight": exposure,
            "def_poss_weight": exposure,
            "seconds_weight": exposure * 30,
        }
    )


def test_pseudo_targets_undo_the_diagonal_shrinkage():
    profiles = _profiles()
    true = 2.0 * profiles["pts_per36"].to_numpy()
    exposure = np.linspace(50, 8000, len(profiles))
    pairs = prior_pairs(_ratings(profiles, true, exposure), profiles)
    assert np.allclose(pairs["t_o"], true)
    assert pairs["s_o"].between(0, 1).all()


def test_f_recovers_a_linear_profile_effect_on_the_right_scale():
    profiles = _profiles()
    true = 2.0 * profiles["pts_per36"].to_numpy()
    exposure = np.linspace(50, 8000, len(profiles))
    pairs = prior_pairs(_ratings(profiles, true, exposure), profiles)
    model = fit_prior(pairs, "o", alpha=0.01)
    assert np.allclose(model.predict(pairs), true, atol=1e-2)
    # Pairs below S_MIN never reach the fit: corrupt their targets, same model.
    low = pairs["s_o"] < S_MIN
    assert low.any()
    noisy = pairs.assign(t_o=np.where(low, 1e6, pairs["t_o"]))
    again = fit_prior(noisy, "o", alpha=0.01)
    assert np.allclose(again.coef, model.coef)


def test_alpha_is_chosen_on_the_pairs_it_is_given():
    profiles = _profiles(seed=1)
    true = profiles["ast_per36"].to_numpy()
    rng = np.random.default_rng(2)
    exposure = rng.uniform(500, 6000, len(profiles))
    pairs = prior_pairs(_ratings(profiles, true, exposure), profiles)
    alpha, table = select_alpha(pairs, "o", [0.01, 1e6])
    assert alpha == 0.01 and table[0.01] < table[1e6]


def test_debut_prior_reads_only_debutants_in_their_first_games():
    pairs = pd.DataFrame(
        {
            "player_id": ["rookie", "rookie_later", "veteran"],
            "games_in_data": [10, 60, 10],
            "t_o": [-3.0, 5.0, 9.0],
            "s_o": [0.1, 0.5, 0.5],
        }
    )
    assert debut_prior(pairs, "o", {"rookie", "rookie_later"}) == pytest.approx(-3.0)


def test_checkpoints_are_the_first_game_date_of_each_month():
    dates = pd.to_datetime(["2018-10-16", "2018-10-20", "2018-11-01", "2018-11-02"])
    assert monthly_checkpoints(dates) == list(
        pd.to_datetime(["2018-10-16", "2018-11-01"])
    )


def test_features_exist_in_the_node_profiles():
    derived = {"log_games_in_data"}
    assert set(FEATURES) - derived <= set(PROFILE_COLUMNS)
