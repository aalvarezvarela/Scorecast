"""Rolling-origin CV, tuned n_estimators, tuned training window, phase reporting.

These four changes exist to make the Optuna stage resemble the walk-forward test
the model is actually judged by. Each one replaces something that used to happen
implicitly, so most of what is tested here is the ABSENCE of the old behaviour --
no eval_set, no derived round count, no 50-tree floor. Absences are exactly what
a silent no-op looks like, so every test below was mutation-checked: the fix was
reverted, the test observed to fail, and the fix restored.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
from optuna.distributions import IntDistribution
from sklearn.metrics import mean_absolute_error

from training_pipeline import pipeline as pipeline_module
from training_pipeline import tuning as tuning_module
from training_pipeline.config import (
    N_ESTIMATORS_RANGES,
    CleaningConfig,
    DataConfig,
    ExperimentConfig,
    HoldoutConfig,
    ObjectiveAggregation,
    OptunaConfig,
    PredictionStrategy,
    SearchSpaceConfig,
    WalkForwardConfig,
)
from training_pipeline.data import PreparedDataset
from training_pipeline.pipeline import resolve_selected_train_games, run_experiment
from training_pipeline.reporting import factors
from training_pipeline.season_phase import month_to_phase, phases_present
from training_pipeline.splits import (
    TRAIN_GAMES_PARAM,
    build_rolling_origin_plan,
    build_split_provider,
)
from training_pipeline.tuning import n_estimators_from_trial

optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- fixtures ---------------------------------------------------------------


def _schedule(
    seasons: tuple[tuple[str, str, int], ...],
    *,
    games_per_day: int = 4,
    dark_every: int | None = 7,
) -> pd.DataFrame:
    """A believable NBA calendar: several seasons, a summer gap, and dark days.

    ``dark_every`` blanks one date in every N so the "next four game-days" logic
    is exercised against real gaps rather than a dense calendar where game-days
    and calendar days coincide.
    """
    rows: list[dict] = []
    for start, end, season in seasons:
        for offset, day in enumerate(pd.date_range(start, end, freq="D")):
            if day.month in (5, 6):
                continue
            if dark_every and offset % dark_every == 3:
                continue
            for _ in range(games_per_day):
                rows.append({"GAME_DATE": day, "SEASON_YEAR": season})

    df = pd.DataFrame(rows)
    rng = np.random.default_rng(11)
    line = rng.uniform(205, 240, len(df)).round(1)
    df["ODDS_TOTAL_LINE_bet365"] = line
    df["TOTAL_POINTS"] = (line + rng.normal(0, 12, len(df))).round(1)
    df["LINE_ERROR"] = df["TOTAL_POINTS"] - df["ODDS_TOTAL_LINE_bet365"]
    df["FEATURE_A"] = rng.normal(size=len(df))
    df["FEATURE_B"] = rng.normal(size=len(df))
    return df


THREE_SEASONS = (
    ("2023-10-20", "2024-04-14", 2023),
    ("2024-10-20", "2025-04-14", 2024),
    ("2025-10-20", "2026-03-10", 2025),
)


@pytest.fixture
def dev_frame() -> pd.DataFrame:
    return _schedule(THREE_SEASONS)


def _rolling_config(tmp_path, **overrides) -> ExperimentConfig:
    kwargs: dict = {
        "experiment_name": "rolling",
        "prediction_strategy": PredictionStrategy.LINE_ERROR_REGRESSOR,
        "data": DataConfig(csv_path="x.csv"),
        "cleaning": CleaningConfig(verbose=0),
        "holdout": HoldoutConfig(test_size=None, test_days=40),
        "walk_forward": WalkForwardConfig(
            strategy="rolling_origin",
            retrain_every_days=4,
            eval_span_games=200,
            min_train_games=100,
            max_folds=None,
            train_games=400,
        ),
        "optuna": OptunaConfig(
            n_trials=2,
            tune_n_estimators=True,
            objective_aggregation=ObjectiveAggregation.POOLED,
            search_space=SearchSpaceConfig(n_estimators_range=None),
        ),
        "save_experiment_artifacts": False,
        "experiment_root_dir": tmp_path / "artifacts",
        "model_output_root": tmp_path / "models",
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


# --- fold layout ------------------------------------------------------------


def test_training_never_contains_a_game_dated_on_or_after_the_origin(
    dev_frame, tmp_path
):
    """The property the whole design rests on. Checked per fold and per window,
    because a leak that only appears at one train_games value would read as an
    unexplained result rather than an error."""
    config = _rolling_config(
        tmp_path,
        walk_forward=WalkForwardConfig(
            strategy="rolling_origin",
            retrain_every_days=4,
            eval_span_games=200,
            min_train_games=100,
            max_folds=None,
            train_games_choices=(200, 400, 800),
        ),
    )
    provider = build_split_provider(dev_frame, config)
    dates = dev_frame["GAME_DATE"].to_numpy()

    for window in (200, 400, 800):
        for train_idx, valid_idx in provider.splits_for(window):
            assert dates[train_idx].max() < dates[valid_idx].min()
            assert len(train_idx) == window


def test_validation_windows_are_four_game_days_and_never_overlap(dev_frame, tmp_path):
    """Four GAME-days, not four dates: the fixture blanks one date in seven, so a
    date-counting implementation would produce short folds here."""
    plan = build_rolling_origin_plan(dev_frame, _rolling_config(tmp_path))

    seen: set[int] = set()
    for fold in plan.folds:
        # Four unless the fold was closed early at a season boundary.
        assert 1 <= len(fold.valid_dates) <= 4
        assert len(set(fold.valid_dates)) == len(fold.valid_dates)
        # Consecutive game-days in the frame, with no game-day skipped between.
        assert fold.valid_dates[0] == fold.origin_date
        overlap = seen & set(fold.valid_idx.tolist())
        assert not overlap, f"fold {fold.fold} re-scores games {sorted(overlap)[:3]}"
        seen |= set(fold.valid_idx.tolist())

    assert len(seen) == plan.n_validation_games
    assert sum(len(f.valid_dates) for f in plan.folds) == plan.n_validation_days
    # The overwhelming majority are full four-day windows; only season edges cut.
    full = sum(1 for f in plan.folds if len(f.valid_dates) == 4)
    assert full >= plan.n_folds - len(THREE_SEASONS)


def test_calendar_days_without_games_are_skipped_not_counted(tmp_path):
    """A week-long break must not consume a validation window."""
    # A contiguous block, a 13-day blackout, then another block. The blackout
    # has no games at all, so a date-counting implementation would burn three
    # windows on it.
    before = pd.date_range("2025-10-01", "2025-11-05", freq="D")
    after_break = pd.date_range("2025-11-19", "2025-11-30", freq="D")
    rows = [
        {"GAME_DATE": day, "SEASON_YEAR": 2025}
        for day in list(before) + list(after_break)
        for _ in range(3)
    ]
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(3)
    df["ODDS_TOTAL_LINE_bet365"] = 220.0
    df["TOTAL_POINTS"] = (220.0 + rng.normal(0, 10, len(df))).round(1)
    df["LINE_ERROR"] = df["TOTAL_POINTS"] - 220.0

    config = _rolling_config(
        tmp_path,
        walk_forward=WalkForwardConfig(
            strategy="rolling_origin",
            retrain_every_days=4,
            eval_span_games=63,
            min_train_games=30,
            max_folds=None,
            train_games=60,
        ),
    )
    plan = build_rolling_origin_plan(df, config)

    blackout = set(pd.date_range("2025-11-06", "2025-11-18", freq="D"))
    validation_days = [day for fold in plan.folds for day in fold.valid_dates]

    # Nothing validates inside the blackout, because nothing happened there.
    assert not blackout & set(validation_days)
    # Every day with games is consumed exactly once.
    assert len(validation_days) == len(set(validation_days)) == 21

    # One window spans the break: its four game-days straddle 13 empty dates, so
    # it covers 16 calendar days. A date-counting implementation would instead
    # have produced three empty folds here.
    spans = [(fold.valid_end - fold.valid_start).days for fold in plan.folds]
    assert max(spans) > 4, f"no window spanned the break (spans={spans})"
    straddling = next(
        fold for fold in plan.folds if (fold.valid_end - fold.valid_start).days > 4
    )
    assert len(straddling.valid_dates) == 4


def test_excluded_months_never_validate_but_stay_available_to_train_on(
    dev_frame, tmp_path
):
    """exclude_test_months is about what a fold is SCORED on. Removing those games
    from training as well would be a different decision, and not this one."""
    config = _rolling_config(
        tmp_path,
        walk_forward=WalkForwardConfig(
            strategy="rolling_origin",
            retrain_every_days=4,
            eval_span_games=200,
            min_train_games=100,
            max_folds=None,
            train_games=400,
            exclude_test_months=(5, 6, 11),
        ),
    )
    plan = build_rolling_origin_plan(dev_frame, config)
    months = pd.to_datetime(dev_frame["GAME_DATE"])

    for fold in plan.folds:
        assert not months.iloc[fold.valid_idx].dt.month.isin({5, 6, 11}).any()

    # November games are still trainable: the last fold's history spans them.
    last = plan.folds[-1]
    assert months.iloc[last.history_idx].dt.month.eq(11).any()


def test_a_window_larger_than_the_earliest_fold_raises_instead_of_shrinking(
    dev_frame, tmp_path
):
    """The silent-shrink failure. tail(n) returns what it has, so without this
    guard the earliest folds train on less than asked and the run looks healthy."""
    config = _rolling_config(tmp_path)
    plan = build_rolling_origin_plan(dev_frame, config)

    plan.assert_window_fits(plan.min_history_games)
    with pytest.raises(ValueError, match="exceeds the .* GAMES the earliest"):
        plan.assert_window_fits(plan.min_history_games + 1)


def test_max_folds_cannot_silently_trim_a_requested_eval_span(dev_frame, tmp_path):
    """_base.yaml sets max_folds: 12 globally; inheriting it would cut a 25-fold
    rolling CV to 12 and quietly deliver half the volume configured."""
    config = _rolling_config(
        tmp_path,
        walk_forward=WalkForwardConfig(
            strategy="rolling_origin",
            retrain_every_days=4,
            eval_span_games=200,
            min_train_games=100,
            max_folds=3,
            train_games=400,
        ),
    )
    with pytest.raises(ValueError, match="would trim the"):
        build_rolling_origin_plan(dev_frame, config)


def test_folds_are_built_from_dev_only_so_the_holdout_is_untouched(
    dev_frame, tmp_path
):
    """CV must not see a game the holdout will be scored on."""
    config = _rolling_config(tmp_path)
    df_dev, df_test = pipeline_module.build_holdout_split(dev_frame, config)
    provider = build_split_provider(df_dev, config)

    latest_cv_date = max(
        pd.Timestamp(df_dev["GAME_DATE"].to_numpy()[fold.valid_idx].max())
        for fold in provider.plan.folds
    )
    assert latest_cv_date < pd.Timestamp(df_test["GAME_DATE"].min())


# --- n_estimators -----------------------------------------------------------


def _frozen(params: dict, user_attrs: dict) -> optuna.trial.FrozenTrial:
    return optuna.trial.create_trial(
        params=params,
        distributions={
            key: IntDistribution(1, 5000) for key in params if key == "n_estimators"
        },
        value=1.0,
        user_attrs=user_attrs,
    )


def test_the_fifty_tree_floor_is_gone(dev_frame):
    """max(50, ...) raised a selected 10-round model to 50 in 16 of 38 runs."""
    legacy = _frozen({}, {"median_best_iteration": 10})
    assert n_estimators_from_trial(legacy) == 10

    tuned = _frozen({"n_estimators": 23}, {})
    assert n_estimators_from_trial(tuned) == 23


def test_a_tuned_round_count_wins_over_a_legacy_best_iteration():
    """A trial carrying both must be replayed at the value it was SCORED at."""
    both = _frozen({"n_estimators": 180}, {"median_best_iteration": 41})
    assert n_estimators_from_trial(both) == 180


def test_every_fold_trains_the_same_tuned_rounds_with_no_eval_set(
    dev_frame, tmp_path, monkeypatch
):
    """The core of change #1: one round count per trial, held across folds, and
    the fold's own answers never handed to fit()."""
    observed: list[dict] = []
    real_fit = tuning_module.XGBRegressor.fit

    def spy_fit(self, X, y, **kwargs):
        observed.append(
            {
                "n_estimators": self.get_params()["n_estimators"],
                "early_stopping_rounds": self.get_params().get(
                    "early_stopping_rounds"
                ),
                "eval_set": kwargs.get("eval_set"),
                "n_train": len(X),
            }
        )
        return real_fit(self, X, y, **kwargs)

    monkeypatch.setattr(tuning_module.XGBRegressor, "fit", spy_fit)

    config = _rolling_config(tmp_path, optuna=OptunaConfig(
        n_trials=1, tune_n_estimators=True,
        objective_aggregation=ObjectiveAggregation.POOLED,
    ))
    df_dev, _ = pipeline_module.build_holdout_split(dev_frame, config)
    provider = build_split_provider(df_dev, config)
    X = df_dev[["ODDS_TOTAL_LINE_bet365", "FEATURE_A", "FEATURE_B"]]
    y = df_dev["LINE_ERROR"]

    study = tuning_module.get_strategy(config).tune(
        X=X, y=y, config=config, split_provider=provider, dates=df_dev["GAME_DATE"]
    )

    assert observed, "no fits were recorded"
    assert all(record["eval_set"] is None for record in observed)
    assert all(record["early_stopping_rounds"] is None for record in observed)
    # One value for the whole trial, and it is the value Optuna sampled.
    assert len({record["n_estimators"] for record in observed}) == 1
    sampled = study.trials[0].params["n_estimators"]
    assert observed[0]["n_estimators"] == sampled
    low, high = N_ESTIMATORS_RANGES[PredictionStrategy.LINE_ERROR_REGRESSOR].low, \
        N_ESTIMATORS_RANGES[PredictionStrategy.LINE_ERROR_REGRESSOR].high
    assert low <= sampled <= high


def test_legacy_mode_still_early_stops_on_the_fold(dev_frame, tmp_path, monkeypatch):
    """Backward compatibility: nothing changes unless the new mode is selected."""
    observed: list = []
    real_fit = tuning_module.XGBRegressor.fit

    def spy_fit(self, X, y, **kwargs):
        observed.append(kwargs.get("eval_set"))
        return real_fit(self, X, y, **kwargs)

    monkeypatch.setattr(tuning_module.XGBRegressor, "fit", spy_fit)

    config = _rolling_config(
        tmp_path,
        optuna=OptunaConfig(
            n_trials=1,
            tune_n_estimators=False,
            search_space=SearchSpaceConfig(n_estimators=8, early_stopping_rounds=4),
        ),
    )
    df_dev, _ = pipeline_module.build_holdout_split(dev_frame, config)
    provider = build_split_provider(df_dev, config)
    tuning_module.get_strategy(config).tune(
        X=df_dev[["ODDS_TOTAL_LINE_bet365", "FEATURE_A", "FEATURE_B"]],
        y=df_dev["LINE_ERROR"],
        config=config,
        split_provider=provider,
        dates=df_dev["GAME_DATE"],
    )
    assert observed and all(entry is not None for entry in observed)


# --- pooled objective -------------------------------------------------------


def test_pooled_mae_weights_games_not_folds(dev_frame, tmp_path):
    """Pooled MAE must equal the SIZE-WEIGHTED mean of the fold MAEs, which is
    what distinguishes it from the unweighted mean the objective used before."""
    config = _rolling_config(tmp_path, optuna=OptunaConfig(
        n_trials=1, tune_n_estimators=True,
        objective_aggregation=ObjectiveAggregation.POOLED,
    ))
    df_dev, _ = pipeline_module.build_holdout_split(dev_frame, config)
    provider = build_split_provider(df_dev, config)
    study = tuning_module.get_strategy(config).tune(
        X=df_dev[["ODDS_TOTAL_LINE_bet365", "FEATURE_A", "FEATURE_B"]],
        y=df_dev["LINE_ERROR"],
        config=config,
        split_provider=provider,
        dates=df_dev["GAME_DATE"],
    )
    trial = study.trials[0]
    folds = trial.user_attrs["fold_metrics"]
    sizes = np.array([fold["n_valid"] for fold in folds], dtype=float)
    maes = np.array([fold["mae"] for fold in folds], dtype=float)

    weighted = float((sizes * maes).sum() / sizes.sum())
    assert trial.user_attrs["pooled_mae"] == pytest.approx(weighted, rel=1e-9)
    assert trial.value == pytest.approx(trial.user_attrs["pooled_mae"], rel=1e-12)
    assert trial.user_attrs["pooled_n_games"] == provider.n_validation_games

    # Fold sizes vary here, so the unweighted mean is a different number -- if it
    # were not, this test could pass with the pooling removed.
    assert sizes.std() > 0
    assert trial.user_attrs["mean_mae"] != pytest.approx(weighted, rel=1e-9)


def test_mean_aggregation_still_returns_the_unweighted_fold_mean(dev_frame, tmp_path):
    config = _rolling_config(tmp_path, optuna=OptunaConfig(
        n_trials=1, tune_n_estimators=True,
        objective_aggregation=ObjectiveAggregation.MEAN,
    ))
    df_dev, _ = pipeline_module.build_holdout_split(dev_frame, config)
    provider = build_split_provider(df_dev, config)
    study = tuning_module.get_strategy(config).tune(
        X=df_dev[["ODDS_TOTAL_LINE_bet365", "FEATURE_A", "FEATURE_B"]],
        y=df_dev["LINE_ERROR"],
        config=config,
        split_provider=provider,
        dates=df_dev["GAME_DATE"],
    )
    trial = study.trials[0]
    maes = [fold["mae"] for fold in trial.user_attrs["fold_metrics"]]
    assert trial.value == pytest.approx(float(np.mean(maes)))
    assert "pooled_mae" not in trial.user_attrs


def test_pooled_metrics_reuse_the_shared_scorers():
    """Pooled and per-fold numbers must come from one definition."""
    pooled = tuning_module.PooledPredictions(
        y_true=np.array([3.0, -2.0, 5.0, -1.0]),
        y_pred=np.array([1.0, -0.5, -2.0, -4.0]),
    )
    metrics = tuning_module.pooled_line_error_metrics(pooled)
    assert metrics["mae"] == pytest.approx(
        mean_absolute_error(pooled.y_true, pooled.y_pred)
    )
    # Three of four sides called correctly (game 3 predicted UNDER, went OVER).
    assert metrics["ou_acc"] == pytest.approx(0.75)
    assert metrics["n_games"] == 4


# --- pruning ----------------------------------------------------------------


def test_pruner_warmup_scales_with_folds_but_never_below_the_historical_five(
    tmp_path,
):
    config = _rolling_config(tmp_path)
    assert config.resolve_pruner_warmup_steps(12) == 5  # legacy layout unchanged
    assert config.resolve_pruner_warmup_steps(28) == 7
    assert config.resolve_pruner_warmup_steps(60) == 15
    assert config.resolve_pruner_warmup_steps(None) == 5

    explicit = _rolling_config(
        tmp_path,
        optuna=OptunaConfig(n_trials=1, pruner_warmup_steps=9),
    )
    assert explicit.resolve_pruner_warmup_steps(28) == 9


def test_a_pruned_trial_records_how_much_it_had_actually_seen(dev_frame, tmp_path):
    """"Pruned" is not auditable; "pruned after 7 folds / 210 games" is."""
    config = _rolling_config(tmp_path, optuna=OptunaConfig(
        n_trials=1, tune_n_estimators=True,
        objective_aggregation=ObjectiveAggregation.POOLED,
    ))
    df_dev, _ = pipeline_module.build_holdout_split(dev_frame, config)
    provider = build_split_provider(df_dev, config)
    X = df_dev[["ODDS_TOTAL_LINE_bet365", "FEATURE_A", "FEATURE_B"]]

    def objective(trial):
        # Force a prune on the second fold by reporting a rising metric.
        return tuning_module.run_objective(
            trial,
            X=X,
            y=df_dev["LINE_ERROR"],
            config=config,
            evaluate_fold=lambda model, Xv, yv, *, fold, n_train: (
                tuning_module._error_line.evaluate_fold_error_line(
                    model, Xv, yv, fold=fold, n_train=n_train
                )
            ),
            split_provider=provider,
            pooled_metrics=tuning_module.pooled_line_error_metrics,
            dates=df_dev["GAME_DATE"],
        )

    # A pruner that always prunes after the warmup makes this deterministic.
    class _AlwaysPrune(optuna.pruners.BasePruner):
        def prune(self, study, trial) -> bool:
            return trial.last_step is not None and trial.last_step >= 2

    study = optuna.create_study(direction="minimize", pruner=_AlwaysPrune())
    study.optimize(objective, n_trials=1)

    pruned = study.trials[0]
    assert pruned.state == optuna.trial.TrialState.PRUNED
    assert pruned.user_attrs["pruned_after_folds"] == 2
    assert pruned.user_attrs["pruned_after_games"] > 0


# --- tuned training window --------------------------------------------------


def test_train_games_is_sampled_once_and_held_across_every_fold(
    dev_frame, tmp_path, monkeypatch
):
    sizes: list[int] = []
    real_fit = tuning_module.XGBRegressor.fit

    def spy_fit(self, X, y, **kwargs):
        sizes.append(len(X))
        return real_fit(self, X, y, **kwargs)

    monkeypatch.setattr(tuning_module.XGBRegressor, "fit", spy_fit)

    config = _rolling_config(
        tmp_path,
        walk_forward=WalkForwardConfig(
            strategy="rolling_origin",
            retrain_every_days=4,
            eval_span_games=200,
            min_train_games=100,
            max_folds=None,
            train_games_choices=(200, 400, 800),
        ),
        optuna=OptunaConfig(
            n_trials=1, tune_n_estimators=True,
            objective_aggregation=ObjectiveAggregation.POOLED,
        ),
    )
    df_dev, _ = pipeline_module.build_holdout_split(dev_frame, config)
    provider = build_split_provider(df_dev, config)
    study = tuning_module.get_strategy(config).tune(
        X=df_dev[["ODDS_TOTAL_LINE_bet365", "FEATURE_A", "FEATURE_B"]],
        y=df_dev["LINE_ERROR"],
        config=config,
        split_provider=provider,
        dates=df_dev["GAME_DATE"],
    )
    selected = study.trials[0].params[TRAIN_GAMES_PARAM]
    assert selected in (200, 400, 800)
    assert set(sizes) == {selected}, "a fold trained on a different window"
    assert study.trials[0].user_attrs["train_games"] == selected


def test_every_train_games_choice_is_scored_on_identical_validation_games(
    dev_frame, tmp_path
):
    """Otherwise a trial preferring 800 games would be compared against one
    preferring 200 on a different cohort, and the result would measure the cohort."""
    config = _rolling_config(
        tmp_path,
        walk_forward=WalkForwardConfig(
            strategy="rolling_origin",
            retrain_every_days=4,
            eval_span_games=200,
            min_train_games=100,
            max_folds=None,
            train_games_choices=(200, 400, 800),
        ),
    )
    provider = build_split_provider(dev_frame, config)
    reference = [valid.tolist() for _, valid in provider.splits_for(200)]
    for window in (400, 800):
        assert [valid.tolist() for _, valid in provider.splits_for(window)] == reference


def test_the_selected_window_is_what_the_run_reports_and_refits_on(tmp_path):
    config = _rolling_config(
        tmp_path,
        walk_forward=WalkForwardConfig(
            strategy="rolling_origin",
            train_games=400,
            train_games_choices=(200, 400, 800),
            eval_span_games=200,
            min_train_games=100,
            max_folds=None,
        ),
    )
    trial = optuna.trial.create_trial(
        params={TRAIN_GAMES_PARAM: 800},
        distributions={
            TRAIN_GAMES_PARAM: optuna.distributions.CategoricalDistribution(
                [200, 400, 800]
            )
        },
        value=1.0,
    )
    assert resolve_selected_train_games(trial, config) == 800
    # No trial (skip_tuning) or an untuned trial falls back to the config.
    assert resolve_selected_train_games(None, config) == 400


def test_test_anchored_window_choices_keep_validation_games_fixed(dev_frame, tmp_path):
    config = _rolling_config(
        tmp_path,
        walk_forward=WalkForwardConfig(
            strategy="test_anchored",
            test_games=80,
            step_games_between_tests=1,
            min_train_games=100,
            max_folds=8,
            train_games=400,
            train_games_choices=(200, 400, 800),
        ),
    )
    provider = build_split_provider(dev_frame, config)
    small = provider.splits_for(200)
    large = provider.splits_for(800)
    assert len(small) == len(large) == 8
    for (short_train, short_valid), (long_train, long_valid) in zip(
        small, large, strict=True
    ):
        np.testing.assert_array_equal(short_valid, long_valid)
        np.testing.assert_array_equal(short_train, long_train[-200:])
        assert len(long_train) == 800
        assert len(short_valid) >= 80
        assert dev_frame.GAME_DATE.iloc[long_train].max() < dev_frame.GAME_DATE.iloc[long_valid].min()


def test_test_anchored_tuned_window_rejects_short_history(dev_frame, tmp_path):
    config = _rolling_config(
        tmp_path,
        walk_forward=WalkForwardConfig(
            strategy="test_anchored",
            test_games=80,
            step_games_between_tests=1,
            min_train_games=100,
            max_folds=8,
            train_games_choices=(200, 100000),
        ),
    )
    with pytest.raises(ValueError, match="exceeds the .* GAMES"):
        build_split_provider(dev_frame, config)


def test_tuning_the_window_is_rejected_under_last_n_seasons_folds(tmp_path):
    with pytest.raises(ValueError, match="requires walk_forward.strategy"):
        _rolling_config(
            tmp_path,
            walk_forward=WalkForwardConfig(
                strategy="last_n_seasons", train_seasons=2,
                train_games_choices=(200, 400)
            ),
        )


# --- end to end -------------------------------------------------------------


def _prepared(df: pd.DataFrame) -> PreparedDataset:
    features = ["ODDS_TOTAL_LINE_bet365", "FEATURE_A", "FEATURE_B"]
    return PreparedDataset(
        df_full=df,
        X=df[features],
        y=df["LINE_ERROR"],
        baseline_line_col="ODDS_TOTAL_LINE_bet365",
        target_line_col="ODDS_TOTAL_LINE_bet365",
        feature_names=features,
        dataset_checksum="sha256:test",
    )


def test_run_experiment_end_to_end_with_rolling_origin_and_tuned_window(
    dev_frame, tmp_path, monkeypatch
):
    config = _rolling_config(
        tmp_path,
        walk_forward=WalkForwardConfig(
            strategy="rolling_origin",
            retrain_every_days=4,
            eval_span_games=160,
            min_train_games=100,
            max_folds=None,
            train_games=200,
            train_games_choices=(200, 400),
        ),
        optuna=OptunaConfig(
            n_trials=2, tune_n_estimators=True,
            objective_aggregation=ObjectiveAggregation.POOLED,
            search_space=SearchSpaceConfig(
                n_estimators_range=None, n_estimators=8, early_stopping_rounds=4
            ),
        ),
    )
    monkeypatch.setattr(
        pipeline_module, "prepare_dataset", lambda cfg: _prepared(dev_frame)
    )
    result = run_experiment(config)

    assert result.cv_betting is not None
    cv = result.cv_betting
    assert cv.n_folds > 8
    assert cv.n_games >= 140

    # The selected window reached the walk-forward evaluation.
    selected = resolve_selected_train_games(result.selected_trial, config)
    assert selected in (200, 400)
    assert set(result.walk_forward_result.daily_results["train_n_games"]) == {selected}


def test_cv_fold_and_prediction_tables_carry_month_season_and_phase(
    dev_frame, tmp_path, monkeypatch
):
    config = _rolling_config(
        tmp_path,
        walk_forward=WalkForwardConfig(
            strategy="rolling_origin",
            retrain_every_days=4,
            eval_span_games=160,
            min_train_games=100,
            max_folds=None,
            train_games=200,
        ),
        optuna=OptunaConfig(
            n_trials=1, tune_n_estimators=True,
            objective_aggregation=ObjectiveAggregation.POOLED,
        ),
    )
    monkeypatch.setattr(
        pipeline_module, "prepare_dataset", lambda cfg: _prepared(dev_frame)
    )
    result = run_experiment(config)
    cv = result.cv_betting

    for column in ("game_month", "season_phase", "season"):
        assert column in cv.fold_metrics.columns
    for column in ("game_month", "season_phase", "SEASON_YEAR"):
        assert column in cv.predictions.columns

    # Phase-matched metrics exist and are computed on a real subset of the pool.
    assert cv.holdout_phases
    assert 0 < cv.n_games_phase_matched <= cv.n_games
    assert cv.betting_phase_matched is not None
    summary = cv.summary()
    assert summary["holdout_phases"] == cv.holdout_phases
    assert summary["n_games_phase_matched"] == cv.n_games_phase_matched

    # Auditable: the subset can be rebuilt from the predictions table alone.
    rebuilt = cv.predictions["season_phase"].isin(cv.holdout_phases.split("+")).sum()
    assert int(rebuilt) == cv.n_games_phase_matched


def test_phase_matching_uses_one_definition_for_cv_and_the_holdout(dev_frame):
    """CV and holdout phases must not be derived by two different rules."""
    assert month_to_phase(10) == "early"
    assert month_to_phase(1) == "mid"
    assert month_to_phase(3) == "late"
    assert month_to_phase(5) == "playoffs"

    march_only = dev_frame[pd.to_datetime(dev_frame["GAME_DATE"]).dt.month == 3]
    assert phases_present(march_only["GAME_DATE"]) == frozenset({"late"})


# --- reporting --------------------------------------------------------------


def test_every_new_knob_is_registered_as_a_reporting_factor():
    """An unregistered knob makes the summary notebook treat old and new runs as
    the same experiment -- the silent no-op this list exists to prevent."""
    for source in (
        "walk_forward.retrain_every_days",
        "walk_forward.eval_span_games",
        "optuna.objective_aggregation",
        "optuna.tune_n_estimators",
        "optuna.pruner_warmup_fraction",
    ):
        assert source in factors.FACTOR_SOURCES.values(), source

    for derived in ("train_games_tuned", "n_estimators_range"):
        assert derived in factors.DERIVED_FACTORS

    # And each one can be rendered in a label, or describe_labels would drop it.
    for factor in (
        "retrain_every_days",
        "eval_span_games",
        "objective_aggregation",
        "tune_n_estimators",
        "train_games_tuned",
        "n_estimators_range",
    ):
        assert factor in factors._DEVIATION_TAGS


def test_a_tuned_window_run_never_matches_a_fixed_window_run(tmp_path):
    """train_games reads 400 in both, but means different things."""
    fixed = {"walk_forward": {"train_games": 400, "train_games_choices": None}}
    tuned = {"walk_forward": {"train_games": 400, "train_games_choices": [200, 400]}}
    flat_fixed = {"walk_forward.train_games_choices": None}
    flat_tuned = {"walk_forward.train_games_choices": [200, 400]}

    assert factors._derived_factors(flat_fixed, {})["train_games_tuned"] is False
    assert factors._derived_factors(flat_tuned, {})["train_games_tuned"] is True
    assert fixed != tuned


# ---------------------------------------------------------------------------
# min_validation_games: a floor on fold size that keeps folds date-aligned
# ---------------------------------------------------------------------------


def _lumpy_schedule(
    day_game_counts: list[int],
    *,
    start: str = "2025-11-01",
    season: int = 2025,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """A calendar whose days hold DIFFERENT numbers of games.

    The real NBA schedule swings from 2 to 15 games a night, which is the whole
    reason this knob exists -- and an evenly-spaced fixture would make every
    fold identical and hide exactly the behaviour under test.
    """
    rows: list[dict] = []
    day = pd.Timestamp(start)
    for position, count in enumerate(day_game_counts):
        year = season if seasons is None else seasons[position]
        for _ in range(count):
            rows.append({"GAME_DATE": day, "SEASON_YEAR": year})
        day += pd.Timedelta(days=1)

    df = pd.DataFrame(rows)
    rng = np.random.default_rng(3)
    line = rng.uniform(205, 240, len(df)).round(1)
    df["ODDS_TOTAL_LINE_bet365"] = line
    df["TOTAL_POINTS"] = (line + rng.normal(0, 12, len(df))).round(1)
    df["LINE_ERROR"] = df["TOTAL_POINTS"] - df["ODDS_TOTAL_LINE_bet365"]
    df["FEATURE_A"] = rng.normal(size=len(df))
    df["FEATURE_B"] = rng.normal(size=len(df))
    return df


def _plan_for(df, tmp_path, **wf):
    kwargs = {
        "strategy": "rolling_origin",
        "retrain_every_days": 4,
        "eval_span_games": None,
        "min_train_games": 10,
        "max_folds": None,
        "train_games": 20,
    }
    kwargs.update(wf)
    config = _rolling_config(tmp_path, walk_forward=WalkForwardConfig(**kwargs))
    return build_rolling_origin_plan(df, config)


def test_a_thin_stretch_of_days_is_extended_until_the_game_floor_is_met():
    """4 days of 2 games each is 8, under a floor of 10, so the fold absorbs a
    fifth whole day. This is the pathological case: on cell A's real layout the
    smallest fold held 2 games against a median of 30."""
    df = _lumpy_schedule([2] * 24)
    plan = _plan_for(df, Path("/tmp"), min_validation_games=10, min_train_games=10)

    first = plan.folds[0]
    assert len(first.valid_dates) == 5
    assert len(first.valid_idx) == 10
    assert all(len(f.valid_idx) >= 10 for f in plan.folds)


def test_a_dense_night_keeps_the_plain_four_day_fold():
    """12 games a night already clears the floor at day 4, so nothing is
    absorbed -- the knob is a floor, not a target."""
    df = _lumpy_schedule([12] * 12)
    plan = _plan_for(df, Path("/tmp"), min_validation_games=25)

    first = plan.folds[0]
    assert len(first.valid_dates) == 4
    assert len(first.valid_idx) == 48


def test_without_the_floor_the_layout_is_exactly_what_it_always_was():
    """Backward compatibility is the default. None must reproduce the old
    fold layout game for game, or every existing experiment silently changes."""
    df = _lumpy_schedule([2] * 24)

    before = _plan_for(df, Path("/tmp"), min_train_games=10)
    after = _plan_for(df, Path("/tmp"), min_validation_games=None, min_train_games=10)

    # Folds of four 2-game days, with the remainder standing alone: the old
    # layout, floor or no floor.
    assert [len(f.valid_idx) for f in before.folds] == [8, 8, 8, 8, 6]
    assert [f.valid_dates for f in before.folds] == [
        f.valid_dates for f in after.folds
    ]


def test_a_fold_never_takes_part_of_a_calendar_date():
    """Extending must add WHOLE days. Half a date on each side of the split is
    the leak the daily walk-forward exists to prevent, and a floor expressed in
    games is precisely the knob that would tempt a partial day."""
    df = _lumpy_schedule([3, 3, 3, 3, 9] * 4)
    plan = _plan_for(df, Path("/tmp"), min_validation_games=14, min_train_games=10)

    dates = pd.to_datetime(df["GAME_DATE"]).dt.normalize().to_numpy()
    for fold in plan.folds:
        validated = set(dates[fold.valid_idx])
        for day in validated:
            # Every game on a validated date is in the fold; none leaked out.
            assert (dates == day).sum() == (dates[fold.valid_idx] == day).sum()


def test_the_floor_never_pulls_a_fold_across_a_season_boundary():
    """require_same_season_test still wins. Gluing late April onto late October
    to reach a game count would give the fold an origin five months before the
    games it predicts, which is worse than a short fold."""
    df = _lumpy_schedule([3] * 24, seasons=[2024] * 12 + [2025] * 12)
    plan = _plan_for(df, Path("/tmp"), min_validation_games=20, min_train_games=6)

    seasons = df["SEASON_YEAR"].to_numpy()
    for fold in plan.folds:
        assert len(set(seasons[fold.valid_idx])) == 1

    # The season's own remainder is absorbed BACKWARDS into a predecessor from
    # the same season, never forwards across the summer -- so the floor is met
    # without any fold acquiring an origin months before the games it predicts.
    assert [len(f.valid_idx) for f in plan.folds] == [30, 36]
    assert plan.n_folds_below_min == 0


def test_a_short_fold_that_opens_a_season_is_left_alone():
    """The one case merging must NOT fix. Gluing a thin October onto the
    previous April would give that fold an origin five months before the games
    it predicts, which is worse than a short fold. It stays short, and is
    counted so nobody has to notice it by eye."""
    df = _lumpy_schedule([6] * 12 + [2] * 3, seasons=[2024] * 12 + [2025] * 3)
    plan = _plan_for(df, Path("/tmp"), min_validation_games=20, min_train_games=6)

    seasons = df["SEASON_YEAR"].to_numpy()
    opener = plan.folds[-1]
    assert set(seasons[opener.valid_idx]) == {2025}
    assert len(opener.valid_idx) == 6
    assert plan.n_folds_below_min == 1


def test_a_short_trailing_fold_is_merged_rather_than_dropped():
    """13 days at 3 games with a floor of 10 leaves a 1-day remainder. Emitting
    it as a 3-game fold recreates the problem; DROPPING it would silently shrink
    the OOF cohort and make this cell incomparable with every other. Merge."""
    df = _lumpy_schedule([2] * 24)

    without = _plan_for(df, Path("/tmp"), min_train_games=10)
    with_floor = _plan_for(
        df, Path("/tmp"), min_validation_games=10, min_train_games=10
    )

    # Plain: four 8-game folds. Floored: 10, 10, and an 18-game tail -- the
    # 2-day remainder absorbed into its predecessor instead of standing alone.
    assert [len(f.valid_idx) for f in without.folds] == [8, 8, 8, 8, 6]
    assert [len(f.valid_idx) for f in with_floor.folds] == [10, 10, 18]
    assert min(len(f.valid_idx) for f in with_floor.folds) >= 10
    # And nothing was thrown away to achieve it.
    assert with_floor.n_validation_games == without.n_validation_games


def test_the_floor_changes_grouping_only_never_the_oof_cohort():
    """The strongest guarantee available: the same games, each exactly once,
    regrouped. If this failed, two cells differing only in this knob would be
    scored on different populations."""
    lumpy = [2, 7, 3, 11, 4, 2, 9, 3, 6, 2, 8, 4, 3, 5, 12, 2, 7, 4, 9, 3, 6, 11, 2, 8]
    df = _lumpy_schedule(lumpy)

    without = _plan_for(df, Path("/tmp"), min_train_games=10)
    with_floor = _plan_for(
        df, Path("/tmp"), min_validation_games=25, min_train_games=10
    )

    plain = np.concatenate([f.valid_idx for f in without.folds])
    floored = np.concatenate([f.valid_idx for f in with_floor.folds])

    assert sorted(plain.tolist()) == sorted(floored.tolist())
    assert len(set(floored.tolist())) == len(floored)   # nothing duplicated
    assert with_floor.n_folds < without.n_folds         # fewer, larger folds


def test_an_unreachable_floor_is_an_error_not_a_layout():
    """Every fold short means the knob did nothing and every fold is 'small'.
    Fail loudly rather than run a layout nobody asked for."""
    # Folds ARE creatable here -- the season boundary just caps every one of
    # them below the floor, so the knob would silently do nothing.
    df = _lumpy_schedule([3] * 24, seasons=[2024] * 12 + [2025] * 12)

    with pytest.raises(ValueError, match="met by none of the"):
        _plan_for(df, Path("/tmp"), min_validation_games=50, min_train_games=6)


def test_the_plan_reports_fold_sizes_in_both_games_and_days():
    lumpy = [2, 7, 3, 11, 4, 2, 9, 3, 6, 2, 8, 4, 3, 5, 12, 2, 7, 4, 9, 3, 6, 11, 2, 8]
    df = _lumpy_schedule(lumpy)
    plan = _plan_for(df, Path("/tmp"), min_validation_games=20, min_train_games=10)

    assert plan.fold_game_counts == [
        int(n) for n in plan.fold_info["test_n_games"]
    ]
    assert list(plan.fold_info["n_valid_days"]) == [
        len(f.valid_dates) for f in plan.folds
    ]
    assert plan.min_validation_games == 20


def test_the_floor_is_rejected_outside_rolling_origin():
    """Under test_anchored the fold is already sized in games, so the knob
    would be a silent no-op -- the exact failure class this repo keeps hitting."""
    with pytest.raises(ValueError, match="requires .*rolling_origin"):
        WalkForwardConfig(strategy="test_anchored", min_validation_games=25)


def test_a_floor_larger_than_the_evaluation_region_is_rejected():
    with pytest.raises(ValueError, match="exceeds walk_forward.eval_span_games"):
        WalkForwardConfig(
            strategy="rolling_origin", eval_span_games=100, min_validation_games=200
        )


# ---------------------------------------------------------------------------
# the shipped bundle must record the window that was SELECTED
# ---------------------------------------------------------------------------


def test_the_saved_bundle_records_the_selected_window_not_the_fallback(
    dev_frame, tmp_path, monkeypatch
):
    """The bug this guards, end to end:

    walk_forward.train_games is only the fallback for when tuning is skipped.
    When the window is tuned, the selected trial may choose something else --
    and the run correctly used the selected value for the CV folds, the daily
    walk-forward AND this very refit. But the bundle metadata was written from
    the config fallback.

    That field is not decoration. retraining_utils.build_retraining_settings_
    from_production_metadata reads it off the shipped bundle and makes it THE
    window for every daily retrain. So: Optuna selects 200, everything measured
    uses 200, the bundle says 400, and from the next morning production trains
    on 400 forever -- while every recorded metric still describes the 200-game
    model. Nothing errors and nothing looks wrong.

    The fallback is pinned to a value NOT in train_games_choices, so a
    regression cannot pass by coincidence.
    """
    config = _rolling_config(
        tmp_path,
        walk_forward=WalkForwardConfig(
            strategy="rolling_origin",
            retrain_every_days=4,
            eval_span_games=160,
            min_train_games=100,
            max_folds=None,
            # Deliberately unreachable by any trial.
            train_games=333,
            train_games_choices=(200, 400),
        ),
        optuna=OptunaConfig(
            n_trials=2,
            tune_n_estimators=True,
            objective_aggregation=ObjectiveAggregation.POOLED,
            search_space=SearchSpaceConfig(
                n_estimators_range=None, n_estimators=8, early_stopping_rounds=4
            ),
        ),
        refit={"train_production_model": True},
    )
    monkeypatch.setattr(
        pipeline_module, "prepare_dataset", lambda cfg: _prepared(dev_frame)
    )
    # save_model=True on purpose: the assertion below reads the file that
    # actually ships to the registry, not an in-memory copy of it.
    result = run_experiment(config, save_model=True)

    selected = resolve_selected_train_games(result.selected_trial, config)
    assert selected in (200, 400)
    assert selected != config.walk_forward.train_games

    shipped = json.loads(Path(result.meta_path).read_text())
    recorded = shipped["training_metrics"]["train_games"]
    assert recorded == selected
    assert recorded != config.walk_forward.train_games

    # And the thing the field is read for actually agrees with it: the games the
    # production model was fitted on.
    assert set(result.walk_forward_result.daily_results["train_n_games"]) == {selected}
