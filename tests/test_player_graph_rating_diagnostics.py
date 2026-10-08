"""RAPM exposure diagnostic: it must see exactly what 2_6's ratings saw."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.lineups.player_ratings import walk_forward_player_ratings
from nba_ou.data_processing.lineups.rating_cv import score_stint_predictions
from nba_ou.data_processing.player_graph.rating_diagnostics import (
    bucket_report,
    decayed_exposure,
    decayed_league_means,
    error_summary,
    exposure_buckets,
    scored_rows,
)

PLAYERS = [str(100 + k) for k in range(16)]


def _stints(n_days=12, per_day=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for day in range(n_days):
        date = pd.Timestamp("2018-10-16") + pd.Timedelta(days=2 * day)
        for k in range(per_day):
            # The last players only appear late, so some are thin or unrated.
            pool = PLAYERS[: 10 + min(6, day // 2)]
            ten = rng.choice(pool, 10, replace=False)
            start = 600 * k
            row = {
                "game_id": f"00218{day:03d}{k // 3:02d}",
                "seg_idx": k,
                "game_date": date,
                "start_ds": start,
                "end_ds": start + int(rng.integers(5, 3000)),
                "home_lineup": np.array(sorted(ten[:5])),
                "away_lineup": np.array(sorted(ten[5:])),
            }
            for side in ("home", "away"):
                row |= {
                    f"{side}_pts": float(rng.integers(0, 15)),
                    f"{side}_fga": float(rng.integers(0, 9)),
                    f"{side}_fta": float(rng.integers(0, 4)),
                    f"{side}_oreb": float(rng.integers(0, 3)),
                    f"{side}_tov": float(rng.integers(0, 3)),
                }
            rows.append(row)
    return pd.DataFrame(rows)


def _ratings(stints, lambda_offdef=50.0, lambda_pace=500.0):
    dates = sorted(stints["game_date"].unique())
    return walk_forward_player_ratings(
        stints,
        dates,
        lambda_offdef=lambda_offdef,
        lambda_pace=lambda_pace,
        half_life_days=30.0,
    )


def test_exposure_reproduces_2_6_poss_weight():
    stints = _stints()
    ratings = _ratings(stints)
    exposure = decayed_exposure(
        stints, ratings["as_of_date"].unique(), half_life_days=30.0
    )
    joined = ratings.merge(exposure, on=["as_of_date", "player_id"], how="left")
    assert joined["poss_weight_y"].notna().all()
    assert np.allclose(joined["poss_weight_x"], joined["poss_weight_y"])


def test_league_means_are_the_infinite_lambda_intercepts():
    stints = _stints()
    huge = _ratings(stints, lambda_offdef=1e12, lambda_pace=1e12)
    means = decayed_league_means(
        stints, huge["as_of_date"].unique(), half_life_days=30.0
    )
    intercepts = huge.groupby("as_of_date")[["league_ortg", "league_pace"]].first()
    joined = means.set_index("as_of_date").join(intercepts, how="inner")
    assert len(joined) > 5
    assert np.allclose(joined["ortg_inf"], joined["league_ortg"], rtol=1e-6)
    assert np.allclose(joined["pace_inf"], joined["league_pace"], rtol=1e-6)


def test_rows_reproduce_2_6_stint_scoring():
    stints = _stints()
    ratings = _ratings(stints)
    dates = ratings["as_of_date"].unique()
    efficiency, pace = scored_rows(
        stints,
        ratings,
        decayed_exposure(stints, dates, half_life_days=30.0),
        lambda_offdef=50.0,
        lambda_pace=500.0,
        league_means=decayed_league_means(stints, dates, half_life_days=30.0),
    )
    reference = score_stint_predictions(stints, ratings)

    def mae(rows):
        error = (rows["actual"] - rows["predicted"]).abs()
        return float((error * rows["weight"]).sum() / rows["weight"].sum())

    assert mae(efficiency) == pytest.approx(reference["offdef_mae"])
    assert mae(pace) == pytest.approx(reference["pace_mae"])
    # Exposure is read as of the stint's date, never from the stint itself: a
    # player's first stints see him with exposure 0.
    debuts = {}
    for row in stints.itertuples():
        for player in (*row.home_lineup, *row.away_lineup):
            debuts.setdefault(player, row.game_date)
    scored = set(efficiency["game_date"])
    late = {p: d for p, d in debuts.items() if d in scored}
    assert late
    keyed = stints.set_index(["game_id", "seg_idx"])
    for row in efficiency.itertuples():
        stint = keyed.loc[(row.game_id, row.seg_idx)]
        ten = {*stint["home_lineup"], *stint["away_lineup"]}
        if any(late.get(p) == row.game_date for p in ten):
            assert row.min_exposure_10 == 0
    assert efficiency["baseline_inf"].notna().all()


def test_summary_and_buckets_add_up():
    rows = pd.DataFrame(
        {
            "game_id": ["a", "a", "b", "c"],
            "weight": [1.0, 3.0, 2.0, 2.0],
            "actual": [100.0, 110.0, 90.0, 105.0],
            "predicted": [102.0, 108.0, 95.0, 104.0],
            "baseline_inf": [105.0, 105.0, 105.0, 105.0],
            "min_exposure_off": [0.0, 50.0, 2000.0, 6000.0],
        }
    )
    summary = error_summary(rows)
    error = rows["actual"] - rows["predicted"]
    assert summary["bias"] == pytest.approx(np.average(error, weights=rows["weight"]))
    zero = rows["actual"] - rows["baseline_inf"]
    assert summary["mse_gain"] == pytest.approx(
        np.average(zero**2 - error**2, weights=rows["weight"])
    )
    table = bucket_report(rows, exposure_buckets(rows["min_exposure_off"]))
    assert list(table.index) == ["0 (unrated)", "(0, 100]", "(1k, 3k]", "> 5k", "all"]
    assert table.loc["all", "weight_share"] == 1
    assert table.drop(index="all")["weight_share"].sum() == pytest.approx(1)


def test_sweep_reproduces_2_6_and_the_infinite_lambda_reference():
    from nba_ou.data_processing.player_graph.rapm_sweep import sweep_predictions

    stints = _stints()
    ratings = _ratings(stints, lambda_offdef=50.0, lambda_pace=500.0)
    dates = ratings["as_of_date"].unique()
    efficiency, pace = scored_rows(
        stints,
        ratings,
        decayed_exposure(stints, dates, half_life_days=30.0),
        lambda_offdef=50.0,
        lambda_pace=500.0,
        league_means=decayed_league_means(stints, dates, half_life_days=30.0),
    )
    inf = float("inf")
    eff, pac = sweep_predictions(
        stints,
        efficiency,
        pace,
        offdef_grid=[(50.0, 50.0), (inf, inf), (50.0, inf), (inf, 50.0)],
        pace_grid=[500.0, inf],
        half_life_days=30.0,
    )
    assert np.allclose(eff[(50.0, 50.0)], efficiency["predicted"], atol=1e-5)
    assert np.allclose(pac[500.0], pace["predicted"], atol=1e-5)
    assert np.allclose(eff[(inf, inf)], efficiency["baseline_inf"])
    assert np.allclose(pac[inf], pace["baseline_inf"])
    # One block removed: a different model from both the full fit and the mean.
    for config in ((50.0, inf), (inf, 50.0)):
        assert not np.allclose(eff[config], eff[(50.0, 50.0)])
        assert not np.allclose(eff[config], eff[(inf, inf)])
