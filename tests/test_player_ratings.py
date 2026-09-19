"""Adjusted ratings must fit only earlier games and recover planted signal."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.lineups.player_ratings import (
    _matrix,
    _NormalEquations,
    walk_forward_player_ratings,
)
from nba_ou.data_processing.lineups.rating_cache import build_player_rating_cache
from nba_ou.data_processing.lineups.rating_cv import (
    score_stint_predictions,
    tune_rating_lambdas,
)


def _stint(date, game_id, *, special, home_pts):
    return dict(
        game_date=date,
        game_id=game_id,
        start_ds=0,
        end_ds=7200,
        home_lineup=(special, 2, 3, 4, 5),
        away_lineup=(11, 12, 13, 14, 15),
        home_pts=home_pts,
        away_pts=20,
        home_fga=20,
        home_fta=0,
        home_oreb=0,
        home_tov=0,
        away_fga=20,
        away_fta=0,
        away_oreb=0,
        away_tov=0,
    )


def test_planted_offense_signal_recovers_positive_difference():
    history = pd.DataFrame(
        [
            _stint(
                f"2025-01-{day:02d}",
                f"g{day}",
                special=(1 if day % 2 else 6),
                home_pts=(22 if day % 2 else 20),
            )
            for day in range(1, 21)
        ]
    )
    ratings = walk_forward_player_ratings(
        history, ["2025-01-21"], lambda_offdef=1, lambda_pace=10
    ).set_index("player_id")
    assert ratings.loc["1", "o_rating"] > ratings.loc["6", "o_rating"]
    assert ratings.loc["1", "o_rating"] - ratings.loc["6", "o_rating"] == pytest.approx(
        10, abs=2
    )


def test_incremental_normal_equations_match_batch():
    rows = [{0: 1.0, 2: -1.0}, {1: 1.0, 2: -1.0}, {0: 1.0, 3: -1.0}]
    x = _matrix(rows, 4)
    y = np.array([110.0, 100.0, 90.0])
    w = np.array([10.0, 20.0, 15.0])
    batch = _NormalEquations(4)
    batch.add(x, y, w)
    incremental = _NormalEquations(4)
    incremental.add(x[:1], y[:1], w[:1])
    incremental.add(x[1:], y[1:], w[1:])
    np.testing.assert_allclose(batch.solve(5)[0], incremental.solve(5)[0], atol=1e-8)
    assert batch.solve(5)[1] == pytest.approx(incremental.solve(5)[1])

    decayed = _NormalEquations(4)
    decayed.add(x[:2], y[:2], w[:2])
    decayed.decay(0.5)
    decayed.add(x[2:], y[2:], w[2:])
    batch_decayed = _NormalEquations(4)
    batch_decayed.add(x, y, w * np.array([0.5, 0.5, 1.0]))
    np.testing.assert_allclose(
        decayed.solve(5)[0], batch_decayed.solve(5)[0], atol=1e-8
    )


def test_target_and_same_day_changes_cannot_change_prior_ratings():
    history = pd.DataFrame(
        [
            _stint("2025-01-01", "g1", special=1, home_pts=20),
            _stint("2025-01-02", "g2", special=1, home_pts=22),
            _stint("2025-01-02", "g3", special=6, home_pts=18),
        ]
    )
    target = ["2025-01-02"]
    before = walk_forward_player_ratings(
        history, target, lambda_offdef=1, lambda_pace=10
    )
    changed = history.copy()
    changed.loc[changed.game_date.eq("2025-01-02"), "home_pts"] = 100
    changed.at[2, "home_lineup"] = (99, 98, 97, 96, 95)
    after = walk_forward_player_ratings(
        changed, target, lambda_offdef=1, lambda_pace=10
    )
    pd.testing.assert_frame_equal(before, after)
    assert (before.fit_max_game_date < before.as_of_date).all()


def test_rating_cv_emits_each_candidate_and_selects_finite_scores():
    history = pd.DataFrame(
        [
            _stint(
                f"2025-01-{day:02d}",
                f"g{day}",
                special=(1 if day % 2 else 6),
                home_pts=(22 if day % 2 else 20),
            )
            for day in range(1, 8)
        ]
    )
    results, best = tune_rating_lambdas(
        history,
        ["2025-01-05", "2025-01-06", "2025-01-07"],
        offdef_candidates=[1, 10],
        pace_candidates=[10, 100],
    )
    assert len(results) == 4
    assert set(best) == {"lambda_offdef", "lambda_pace"}
    assert best["lambda_offdef"] in {1, 10}
    assert best["lambda_pace"] in {10, 100}
    assert np.isfinite(results.offdef_mae.dropna()).all()
    assert np.isfinite(results.pace_mae.dropna()).all()


def test_stint_scoring_does_not_read_future_ratings():
    history = pd.DataFrame(
        [
            _stint("2025-01-01", "g1", special=1, home_pts=20),
            _stint("2025-01-02", "g2", special=1, home_pts=22),
        ]
    )
    ratings = walk_forward_player_ratings(
        history, ["2025-01-02"], lambda_offdef=1, lambda_pace=10
    )
    baseline = score_stint_predictions(history, ratings)
    future = ratings.copy()
    future["as_of_date"] = pd.Timestamp("2025-01-03")
    future["o_rating"] = 1e9
    combined = pd.concat([ratings, future], ignore_index=True)
    assert score_stint_predictions(history, combined) == baseline


def test_rating_cache_includes_serving_date_and_uses_only_prior_games():
    history = pd.DataFrame(
        [
            _stint("2025-01-01", "g1", special=1, home_pts=20),
            _stint("2025-01-03", "g2", special=1, home_pts=22),
        ]
    )
    cache = build_player_rating_cache(
        history,
        as_of_from="2025-01-02",
        as_of_to="2025-01-04",
        lambda_offdef=1,
        lambda_pace=10,
    )
    assert set(cache.as_of_date) == {
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-04"),
    }
    assert cache.fit_max_game_date.lt(cache.as_of_date).all()
