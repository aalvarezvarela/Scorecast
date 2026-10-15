"""Phase 6 out-of-fold folds, baselines, prediction store and evaluation."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.player_graph.pretraining_eval import (
    ladder,
    paired_gain,
    target_errors,
)
from nba_ou.data_processing.player_graph.pretraining_oof import (
    PACE,
    PTS,
    BookSpec,
    Fold,
    PredictionStore,
    check_fold,
    constant_predictions,
    decayed_means,
    monthly_folds,
    rapm_predictions,
)
from nba_ou.data_processing.player_graph.stint_graph import LABELS


def test_monthly_folds_skip_months_without_games():
    dates = pd.to_datetime(["2016-12-03", "2017-01-15", "2017-06-01", "2017-10-20"])
    folds = monthly_folds(dates, first="2016-12-01")
    assert [f.label for f in folds] == ["2016-12", "2017-01", "2017-06", "2017-10"]
    assert folds[-1].until == pd.Timestamp("2017-11-01")


def test_fold_masks_train_strictly_before_the_checkpoint():
    fold = Fold(pd.Timestamp("2017-01-01"), pd.Timestamp("2017-02-01"))
    dates = pd.to_datetime(["2016-12-31", "2017-01-01", "2017-01-31", "2017-02-01"])
    train, predict = fold.masks(dates.to_numpy())
    assert train.tolist() == [True, False, False, False]
    assert predict.tolist() == [False, True, True, False]
    check_fold(fold, dates[train], dates[predict])
    with pytest.raises(ValueError, match="trains on"):
        check_fold(fold, dates[:2], dates[1:3])
    with pytest.raises(ValueError, match="outside"):
        check_fold(fold, dates[:1], dates[1:])


def _labels(n, value):
    labels = np.full((n, 2, len(LABELS)), value, float)
    return labels, np.ones_like(labels)


def test_decayed_means_halve_the_weight_each_half_life():
    labels, weights = _labels(2, 0.0)
    labels[1] = 3.0
    cutoff = pd.Timestamp("2017-01-01")
    dates = np.array([cutoff - pd.Timedelta(days=180), cutoff - pd.Timedelta(days=1)])
    dates = pd.to_datetime(dates).to_numpy()
    means = decayed_means(labels, weights, dates, cutoff, half_life_days=180.0)
    w_old, w_new = 0.5, 0.5 ** (1 / 180)
    assert means[0] == pytest.approx(3.0 * w_new / (w_old + w_new))
    # Rows on or after the cutoff never count.
    later = np.append(dates, cutoff.to_datetime64())
    more_labels = np.concatenate([labels, np.full((1, 2, len(LABELS)), 99.0)])
    assert decayed_means(
        more_labels, np.ones_like(more_labels), later, cutoff, 180.0
    ) == pytest.approx(means)


def test_constant_baseline_only_reads_earlier_folds():
    labels, weights = _labels(3, 1.0)
    labels[2] = 50.0  # the month being predicted
    dates = pd.to_datetime(["2016-12-05", "2016-12-20", "2017-01-10"]).to_numpy()
    folds = monthly_folds(dates, first="2017-01-01")
    out = constant_predictions(labels, weights, dates, folds)
    assert np.isnan(out[:2]).all()
    assert out[2] == pytest.approx(1.0)


def _stint(day, home, away, home_pts, away_pts):
    return {
        "game_id": f"00216{day:05d}",
        "season_year": 2016,
        "game_date": pd.Timestamp("2016-11-01") + pd.Timedelta(days=day),
        "start_ds": 0,
        "end_ds": 3000,
        "home_lineup": np.array(home),
        "away_lineup": np.array(away),
        "home_pts": home_pts,
        "away_pts": away_pts,
        "home_fga": 10,
        "away_fga": 10,
        "home_fta": 0,
        "away_fta": 0,
        "home_oreb": 0,
        "away_oreb": 0,
        "home_tov": 0,
        "away_tov": 0,
    }


def test_rapm_baseline_uses_the_ratings_as_of_the_checkpoint():
    good = [str(k) for k in range(1, 6)]
    bad = [str(k) for k in range(6, 11)]
    history = pd.DataFrame(
        [_stint(day, good, bad, 14, 8) for day in range(40)]  # Nov - Dec
        + [_stint(70, good, bad, 0, 30)]  # in the predicted month: never read
    )
    players = np.array([good + bad])
    dates = pd.to_datetime([history["game_date"].iloc[-1]]).to_numpy()
    folds = monthly_folds(dates, first="2017-01-01")
    spec = BookSpec(lambda_offdef=10.0, lambda_pace=10.0, prior_offdef=False)
    out, book = rapm_predictions(players, dates, folds, history, prior=None, spec=spec)
    assert pd.to_datetime(book["fit_max_game_date"]).max() < folds[0].checkpoint
    assert out[0, 0, PTS] > out[0, 1, PTS]  # the good five still scores more
    assert out[0, 0, PACE] == out[0, 1, PACE]
    others = [k for k in range(len(LABELS)) if k not in (PTS, PACE)]
    assert np.isnan(out[0][:, others]).all()


def test_prediction_store_round_trip_and_refuses_overlapping_folds(tmp_path):
    store = PredictionStore(tmp_path)
    fold = Fold(pd.Timestamp("2017-01-01"), pd.Timestamp("2017-02-01"))
    preds = np.arange(2 * 2 * len(LABELS), dtype=float).reshape(2, 2, len(LABELS))
    store.write("fm", fold, np.array(["g1", "g2"]), np.array([0, 3]), preds, {"a": 1})
    assert store.has("fm", fold)
    back = store.read("fm", np.array(["g2", "g1", "g3"]), np.array([3, 0, 0]))
    assert np.array_equal(back[0], preds[1]) and np.array_equal(back[1], preds[0])
    assert np.isnan(back[2]).all()
    other = Fold(pd.Timestamp("2017-02-01"), pd.Timestamp("2017-03-01"))
    store.write("fm", other, np.array(["g1"]), np.array([0]), preds[:1])
    with pytest.raises(ValueError, match="two folds"):
        store.read("fm", np.array(["g1"]), np.array([0]))


def test_paired_gain_is_zero_for_identical_and_positive_for_better():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.normal(size=(n, 2, 1))
    w = np.ones_like(y)
    dates = np.repeat(np.arange(40), 10)
    worse = np.zeros_like(y)
    better = 0.8 * y
    same = paired_gain(y, w, worse, worse, dates, ["t"])
    assert same.loc["t", "gain"] == 0.0
    gain = paired_gain(y, w, better, worse, dates, ["t"])
    assert gain.loc["t", "gain"] > 5 * gain.loc["t", "se"]
    errors = target_errors(y, w, better, ["t"])
    assert errors.loc["t", "mse"] == pytest.approx(np.mean((0.2 * y) ** 2))


def test_paired_comparison_needs_the_same_rows():
    y = np.ones((4, 2, 1))
    w = np.ones_like(y)
    better = y.copy()
    better[0] = np.nan
    with pytest.raises(ValueError, match="different rows"):
        paired_gain(y, w, better, np.zeros_like(y), np.arange(4), ["t"])
    # The ladder keeps the rows every model predicts.
    table = ladder(
        y, w, {"ref": np.zeros_like(y), "m": better}, np.arange(4), ["t"], "ref"
    )
    assert table.loc[0, "rows"] == 6 and table.loc[0, "gain"] == 1.0


def test_daily_rapm_never_reads_the_stints_own_day():
    from nba_ou.data_processing.player_graph.pretraining_oof import (
        daily_rapm_predictions,
    )

    good = [str(k) for k in range(1, 6)]
    bad = [str(k) for k in range(6, 11)]
    stints = [_stint(day, good, bad, 14, 8) for day in range(40)]
    # Day 50: the good five collapse. Their own day must not see it.
    collapse = _stint(50, good, bad, 0, 40)
    history = pd.DataFrame([*stints, collapse])
    later = pd.DataFrame([*stints, collapse, _stint(51, good, bad, 14, 8)])
    players = np.array([good + bad] * 2)
    dates = pd.to_datetime(
        [collapse["game_date"], collapse["game_date"] + pd.Timedelta(days=1)]
    ).to_numpy()
    spec = BookSpec(lambda_offdef=10.0, lambda_pace=10.0, prior_offdef=False)
    out, _ = daily_rapm_predictions(
        players, dates, history, prior=None, first="2016-11-01", spec=spec
    )
    out_more, _ = daily_rapm_predictions(
        players, dates, later, prior=None, first="2016-11-01", spec=spec
    )
    # The collapse day is predicted as before it; the next day has seen it.
    assert out[0, 0, PTS] > out[0, 1, PTS]
    assert out[1, 0, PTS] < out[0, 0, PTS]
    assert out[0, 0, PTS] == out_more[0, 0, PTS]


def test_residual_stints_subtract_the_base_where_it_exists():
    from nba_ou.data_processing.player_graph.pretraining_oof import residual_stints

    @dataclass(frozen=True)
    class Stints:
        labels: np.ndarray
        label_weight: np.ndarray

    labels, weights = _labels(2, 1.0)
    base = np.full(labels.shape, np.nan)
    base[0, :, PTS] = 0.25
    out = residual_stints(
        Stints(labels, weights), base, ["pts_per_poss", "poss_per_48"]
    )
    assert out.labels[0, 0, PTS] == 0.75 and out.label_weight[0, 0, PTS] == 1.0
    assert np.isnan(out.labels[1, 0, PTS]) and out.label_weight[1, 0, PTS] == 0.0
    assert np.isnan(out.labels[0, 0, PACE]) and out.label_weight[0, 0, PACE] == 0.0
    other = LABELS.index("fg3a_per_fga")
    assert (out.labels[:, :, other] == 1.0).all()
