import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.player_graph.matchup_defense import (
    as_of_sums,
    attacker_norms,
    doubled_exposure_games,
    row_halves,
    shrunk_as_of,
    shrunk_on_dates,
    split_half,
)


def _frame(rows):
    frame = pd.DataFrame(rows, columns=["key", "game_id", "game_date", "y", "w"])
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    return frame


def test_as_of_sums_use_only_earlier_dates_with_decay():
    frame = _frame(
        [
            ("a", "g1", "2018-01-01", 2.0, 1.0),
            ("a", "g2", "2018-01-02", 4.0, 1.0),
            ("a", "g2", "2018-01-02", 1.0, 1.0),  # same date: never sees itself
            ("b", "g2", "2018-01-02", 9.0, 1.0),
            ("a", "g3", "2018-01-03", 0.0, 1.0),
        ]
    )
    sums = as_of_sums(frame, ["key"], ["y", "w"], half_life_days=1.0)
    assert sums.loc[0, "y"] == 0.0
    assert sums.loc[1, "y"] == pytest.approx(1.0)  # 2 * 0.5 ** 1
    assert sums.loc[2, "y"] == pytest.approx(1.0)
    assert sums.loc[3, "y"] == 0.0  # other key
    assert sums.loc[4, "y"] == pytest.approx(2 * 0.25 + 5 * 0.5)
    league = as_of_sums(frame, [], ["y"], half_life_days=1.0)
    assert league.loc[3, "y"] == pytest.approx(1.0)
    assert league.loc[4, "y"] == pytest.approx(0.5 + 14 * 0.5)


def test_attacker_norms_shrink_toward_position_rate():
    pairs = pd.DataFrame(
        {
            "off_player_id": ["x", "y", "x"],
            "position": ["G", "G", "G"],
            "game_date": pd.to_datetime(["2018-01-01", "2018-01-01", "2018-01-02"]),
            "y": [10.0, 2.0, 0.0],
            "w": [10.0, 10.0, 5.0],
        }
    )
    norms = attacker_norms(pairs, "y", "w", prior_weight=10.0, half_life_days=1e9)
    assert np.isnan(norms.iloc[0])  # no history at all
    # x on day 2: own 10/10, position 12/20, prior weight 10.
    assert norms.iloc[2] == pytest.approx((10 + 10 * 0.6) / (10 + 10))


def test_row_halves_alternate_games_by_group():
    frame = _frame(
        [
            ("a", "g1", "2018-01-01", 0, 1),
            ("a", "g1", "2018-01-01", 0, 1),
            ("a", "g2", "2018-01-02", 0, 1),
            ("a", "g3", "2018-01-03", 0, 1),
            ("b", "g3", "2018-01-03", 0, 1),
        ]
    )
    assert row_halves(frame, ["key"]).tolist() == [0, 0, 1, 0, 0]


def _synthetic(effect_sd, n_keys=400, games=40, w=50.0, seed=0):
    rng = np.random.default_rng(seed)
    effects = rng.normal(0.0, effect_sd, n_keys)
    rows = []
    for k in range(n_keys):
        for g in range(games):
            r = rng.normal(effects[k] * w, np.sqrt(w))  # noise variance 1 per unit
            rows.append(
                (
                    k,
                    f"g{k}_{g}",
                    pd.Timestamp("2018-01-01") + pd.Timedelta(days=g),
                    r,
                    w,
                )
            )
    frame = pd.DataFrame(rows, columns=["key", "game_id", "game_date", "r", "w"])
    frame["half"] = row_halves(frame, ["key"])
    return frame


def test_split_half_recovers_signal_variance_and_shrinkage():
    frame = _synthetic(effect_sd=0.05)
    result = split_half(frame, ["key"], min_exposure=100.0)
    assert result.keys == 400
    assert result.signal_var == pytest.approx(0.05**2, rel=0.25)
    assert result.noise_per_unit == pytest.approx(1.0, rel=0.1)
    assert result.shrinkage == pytest.approx(1.0 / 0.05**2, rel=0.3)
    assert 0.0 < result.reliability(2000.0) < 1.0


def test_split_half_without_signal_is_near_zero():
    frame = _synthetic(effect_sd=0.0, seed=1)
    result = split_half(frame, ["key"], min_exposure=100.0)
    assert abs(result.signal_var) < 3 * result.signal_se
    noisy = shrunk_as_of(frame, ["key"], shrinkage=np.inf)
    assert (noisy == 0.0).all()


def test_doubled_exposure_games_flags_games_at_twice_the_guard_rate():
    pairs = pd.DataFrame(
        {
            "game_id": ["ok", "ok", "doubled", "doubled", "empty"],
            "matchup_seconds": [40.0, 40.0, 80.0, 80.0, 0.0],
            "cofloor_seconds": [500.0, 500.0, 500.0, 500.0, 0.0],
        }
    )
    assert doubled_exposure_games(pairs) == {"doubled"}


def test_shrunk_on_dates_matches_shrunk_as_of_on_the_rows_dates():
    frame = _synthetic(effect_sd=0.05, n_keys=5, games=10)
    expected = shrunk_as_of(frame, ["key"], shrinkage=20.0)
    queries = frame[["game_date", "key"]]
    got = shrunk_on_dates(frame, "key", queries, shrinkage=20.0)
    assert np.allclose(got, expected.to_numpy())
