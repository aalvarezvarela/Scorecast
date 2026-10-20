"""The planted-signal diagnostic: a feature carrying KNOWN target information.

This is the one place in the pipeline where deriving a feature from the target
is deliberate, so most of these tests are about containment rather than about
the feature working: it must be absent unless asked for, it must not disturb
anything else, and no path may turn a run carrying it into a shipped model.

Mutation-checked: every assertion below was confirmed to fail with its guard
reverted, then the guard restored. A containment test that passes either way is
worse than none.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from training_pipeline import data as data_module
from training_pipeline import pipeline as pipeline_module
from training_pipeline.config import (
    CleaningConfig,
    DataConfig,
    DiagnosticsConfig,
    ExperimentConfig,
    HoldoutConfig,
    ObjectiveAggregation,
    OptunaConfig,
    PlantedSignalConfig,
    PredictionStrategy,
    RefitConfig,
    WalkForwardConfig,
)
from training_pipeline.diagnostics import (
    DIAGNOSTIC_NAME_PREFIX,
    build_planted_signal,
    measure_planted_signal,
    planted_feature_importance,
)

PLANTED = "PLANTED_SIGNAL"


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def training_csv(tmp_path) -> tuple[str, int]:
    """A believable CSV: real column names, a schedule, and a few features."""
    rng = np.random.default_rng(5)
    days = [
        day
        for day in pd.date_range("2024-10-20", "2026-01-31", freq="D")
        if day.month not in (5, 6, 7, 8, 9)
    ]
    rows = [
        {
            "GAME_ID": "0022400001",
            "GAME_DATE": day,
            "SEASON_TYPE": "Regular Season",
            "SEASON_YEAR": 2024 if day < pd.Timestamp("2025-08-01") else 2025,
        }
        for day in days
        for _ in range(4)
    ]
    frame = pd.DataFrame(rows)
    n = len(frame)
    frame["ODDS_TOTAL_LINE_bet365"] = rng.uniform(205, 240, n).round(1)
    frame["TOTAL_POINTS"] = (
        frame["ODDS_TOTAL_LINE_bet365"] + rng.normal(0, 12, n)
    ).round(1)
    for index in range(10):
        frame[f"FEAT_{index}_BEFORE"] = rng.normal(size=n)

    path = tmp_path / "planted.csv"
    frame.to_csv(path, index=False)
    return str(path), n


def _config(csv_path: str, tmp_path, *, variance: float | None, **overrides):
    """A line_error config, diagnostic when ``variance`` is not None."""
    diagnostics = (
        DiagnosticsConfig()
        if variance is None
        else DiagnosticsConfig(
            planted_signal=PlantedSignalConfig(
                enabled=True, variance_explained=variance, seed=12345
            )
        )
    )
    kwargs: dict = {
        "experiment_name": (
            "normal_run" if variance is None else "diag_planted_test"
        ),
        "prediction_strategy": PredictionStrategy.LINE_ERROR_REGRESSOR,
        "data": DataConfig(csv_path=csv_path, season_year_floor=2024),
        "cleaning": CleaningConfig(verbose=0),
        "holdout": HoldoutConfig(test_size=None, test_days=40),
        "walk_forward": WalkForwardConfig(
            strategy="rolling_origin",
            retrain_every_days=4,
            eval_span_games=120,
            min_train_games=80,
            max_folds=None,
            train_games=200,
        ),
        "optuna": OptunaConfig(
            n_trials=1,
            tune_n_estimators=True,
            objective_aggregation=ObjectiveAggregation.POOLED,
        ),
        "diagnostics": diagnostics,
        "save_experiment_artifacts": False,
        "experiment_root_dir": tmp_path / "artifacts",
        "model_output_root": tmp_path / "models",
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


# --- the construction -------------------------------------------------------


def test_zero_variance_gives_a_feature_with_no_target_information():
    """The 0% cell must be a real control: the column exists, and carries
    nothing. Otherwise it measures a different experiment, not a baseline."""
    target = pd.Series(np.random.default_rng(1).normal(0, 19.5, 8000))
    # Seed deliberately unequal to the target's: see
    # test_measured_correlation_is_the_number_to_trust for why that matters.
    config = PlantedSignalConfig(enabled=True, variance_explained=0.0, seed=4242)
    planted = build_planted_signal(target, config=config)

    result = measure_planted_signal(
        pd.DataFrame({"LINE_ERROR": target, PLANTED: planted}),
        target_col="LINE_ERROR",
        config=config,
    )
    # Sampling error on r at n=8000 is ~1/sqrt(n) = 0.011.
    assert abs(result.measured_correlation) < 0.04
    assert result.measured_variance_explained < 0.002


def test_measured_correlation_is_the_number_to_trust_not_the_requested_one():
    """Why every run records the EMPIRICAL correlation, not just the config.

    numpy's ``default_rng(k).normal(0, s, n)`` and
    ``default_rng(k).standard_normal(n)`` are the same underlying draws, one
    scaled by ``s``. So a synthetic target and a planted feature seeded with the
    same number are perfectly correlated -- at a REQUESTED variance of zero.
    That is a fixture hazard rather than a pipeline one (a real target comes
    from played games, not from a seeded generator), but it is exactly the shape
    of accident that produces a spectacular fake result, and the only thing that
    catches it is measuring what the column actually carries.

    This test asserts the measurement notices. If it ever stops noticing, the
    diagnostic's headline number becomes unfalsifiable.
    """
    collided_seed = 1
    target = pd.Series(np.random.default_rng(collided_seed).normal(0, 19.5, 2000))
    config = PlantedSignalConfig(
        enabled=True, variance_explained=0.0, seed=collided_seed
    )
    planted = build_planted_signal(target, config=config)

    result = measure_planted_signal(
        pd.DataFrame({"LINE_ERROR": target, PLANTED: planted}),
        target_col="LINE_ERROR",
        config=config,
    )
    # Requested 0.0, actually carries everything -- and the artifact says so.
    assert result.requested_variance_explained == 0.0
    assert result.measured_variance_explained > 0.99


@pytest.mark.parametrize("variance", [0.005, 0.01, 0.02, 0.05])
def test_the_feature_explains_the_variance_it_was_asked_to(variance):
    """The construction is rho*z + sqrt(1-rho^2)*e, so R2 should land on the
    requested fraction rather than merely 'increase with it'."""
    target = pd.Series(np.random.default_rng(2).normal(0, 19.5, 20000))
    config = PlantedSignalConfig(
        enabled=True, variance_explained=variance, seed=7
    )
    planted = build_planted_signal(target, config=config)
    result = measure_planted_signal(
        pd.DataFrame({"LINE_ERROR": target, PLANTED: planted}),
        target_col="LINE_ERROR",
        config=config,
    )
    assert result.measured_variance_explained == pytest.approx(variance, abs=0.004)
    # Unit variance by construction; a drifting scale would silently change what
    # min_child_weight and gamma mean for this feature relative to the others.
    assert planted.std() == pytest.approx(1.0, abs=0.05)


def test_stronger_requests_produce_a_stronger_relationship():
    target = pd.Series(np.random.default_rng(3).normal(0, 19.5, 20000))
    measured = []
    for variance in (0.0, 0.005, 0.01, 0.02):
        config = PlantedSignalConfig(
            enabled=True, variance_explained=variance, seed=11
        )
        planted = build_planted_signal(target, config=config)
        measured.append(
            measure_planted_signal(
                pd.DataFrame({"LINE_ERROR": target, PLANTED: planted}),
                target_col="LINE_ERROR",
                config=config,
            ).measured_variance_explained
        )
    assert measured == sorted(measured)
    assert measured[-1] > measured[0] + 0.015


def test_the_same_seed_reproduces_the_same_column():
    target = pd.Series(np.random.default_rng(4).normal(0, 19.5, 500))
    same = [
        build_planted_signal(
            target,
            config=PlantedSignalConfig(
                enabled=True, variance_explained=0.01, seed=999
            ),
        )
        for _ in range(2)
    ]
    other = build_planted_signal(
        target,
        config=PlantedSignalConfig(
            enabled=True, variance_explained=0.01, seed=1000
        ),
    )
    assert np.array_equal(same[0], same[1])
    assert not np.array_equal(same[0], other)


def test_the_feature_is_not_a_copy_of_the_target():
    """Trivial leakage would answer a different, useless question."""
    target = pd.Series(np.random.default_rng(6).normal(0, 19.5, 4000))
    planted = build_planted_signal(
        target,
        config=PlantedSignalConfig(
            enabled=True, variance_explained=0.02, seed=3
        ),
    )
    assert abs(float(pd.Series(planted).corr(target))) < 0.25


# --- containment: absent unless asked for -----------------------------------


def test_the_feature_is_absent_when_the_diagnostic_is_off(training_csv, tmp_path):
    csv_path, _ = training_csv
    prepared = data_module.prepare_dataset(
        _config(csv_path, tmp_path, variance=None)
    )
    assert PLANTED not in prepared.X.columns
    assert PLANTED not in prepared.df_full.columns
    assert prepared.planted_signal is None


def test_the_feature_reaches_the_feature_matrix_when_enabled(
    training_csv, tmp_path
):
    """It must survive cleaning AND feature selection. It is deliberately not
    force-kept, so this is a real test of the path a normal feature takes."""
    csv_path, _ = training_csv
    prepared = data_module.prepare_dataset(
        _config(csv_path, tmp_path, variance=0.02)
    )
    assert PLANTED in prepared.df_full.columns
    assert PLANTED in prepared.X.columns
    assert prepared.planted_signal is not None
    assert prepared.planted_signal.requested_variance_explained == 0.02


def test_enabling_it_changes_nothing_else(training_csv, tmp_path):
    """One extra column, same rows, same target, same real features -- byte for
    byte. Anything else would confound the comparison the campaign rests on."""
    csv_path, _ = training_csv
    off = data_module.prepare_dataset(_config(csv_path, tmp_path, variance=None))
    on = data_module.prepare_dataset(_config(csv_path, tmp_path, variance=0.02))

    assert set(on.X.columns) - set(off.X.columns) == {PLANTED}
    assert on.X[list(off.X.columns)].equals(off.X)
    assert on.y.equals(off.y)
    assert len(on.df_full) == len(off.df_full)
    for column in ("GAME_DATE", "TOTAL_POINTS", "ODDS_TOTAL_LINE_bet365", "LINE_ERROR"):
        pd.testing.assert_series_equal(on.df_full[column], off.df_full[column])


def test_the_measurement_describes_the_final_frame(training_csv, tmp_path):
    """Rows are dropped after the feature is generated, so the recorded number
    must be measured on what the model actually gets."""
    csv_path, _ = training_csv
    prepared = data_module.prepare_dataset(
        _config(csv_path, tmp_path, variance=0.02)
    )
    result = prepared.planted_signal
    assert result is not None
    assert result.n_rows == len(prepared.df_full)
    recomputed = prepared.df_full[PLANTED].corr(prepared.df_full["LINE_ERROR"])
    assert result.measured_correlation == pytest.approx(float(recomputed))


def test_a_planted_feature_dropped_by_cleaning_raises_instead_of_no_opping(
    training_csv, tmp_path
):
    """The failure this whole diagnostic is most exposed to.

    The planted column is deliberately NOT force-kept through cleaning, so that
    it travels the same path a real feature does. The cost of that choice is
    that cleaning COULD drop it -- and a diagnostic that silently measures
    nothing while reporting a clean run is worse than no diagnostic. Simulated
    here by excluding it by name, which is exactly what a stray entry in
    cleaning.exclude_cols_containing would do.
    """
    csv_path, _ = training_csv
    config = _config(
        csv_path,
        tmp_path,
        variance=0.02,
        cleaning=CleaningConfig(verbose=0, exclude_cols_containing=("PLANTED",)),
    )
    with pytest.raises(ValueError, match="did not survive cleaning"):
        data_module.prepare_dataset(config)


def test_a_planted_feature_excluded_from_the_matrix_raises(training_csv, tmp_path):
    """Second half of the same hazard: it survives cleaning but never reaches X,
    so no model would ever see it."""
    csv_path, _ = training_csv
    config = _config(
        csv_path,
        tmp_path,
        variance=0.02,
        exclude_cols=["TOTAL_POINTS", "SEASON_YEAR", "GAME_DATE", PLANTED],
    )
    with pytest.raises(ValueError, match="never reached the feature matrix"):
        data_module.prepare_dataset(config)


# --- containment: cannot masquerade as production ---------------------------


def test_a_diagnostic_run_must_be_named_as_one(training_csv, tmp_path):
    csv_path, _ = training_csv
    with pytest.raises(ValueError, match=DIAGNOSTIC_NAME_PREFIX):
        _config(csv_path, tmp_path, variance=0.01, experiment_name="line_error_x")


def test_a_diagnostic_run_cannot_be_configured_to_train_a_production_model(
    training_csv, tmp_path
):
    csv_path, _ = training_csv
    with pytest.raises(ValueError, match="train_production_model must be false"):
        _config(
            csv_path,
            tmp_path,
            variance=0.01,
            refit=RefitConfig(train_production_model=True),
        )


def test_the_save_model_override_cannot_bypass_the_config_guard(
    training_csv, tmp_path
):
    """The other door into the same room: save_model=True at the call site,
    which the config never sees."""
    csv_path, _ = training_csv
    config = _config(csv_path, tmp_path, variance=0.01)
    with pytest.raises(ValueError, match="Refusing to train a production model"):
        pipeline_module.run_experiment(config, save_model=True)


def test_promote_refuses_a_diagnostic_run(training_csv, tmp_path):
    """The last door, and the one most likely to be walked through months later
    by someone reading a leaderboard rather than a run's metadata."""
    from training_pipeline import promote as promote_module

    csv_path, _ = training_csv
    config = _config(csv_path, tmp_path, variance=0.01)
    run_dir = tmp_path / "diag_planted_test_20260101_000000"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(config.model_dump_json())
    (run_dir / "optuna_selected_trial.json").write_text(
        '{"selected_trial": {"number": 0, "params": {"n_estimators": 50}, '
        '"user_attrs": {}}}'
    )

    with pytest.raises(ValueError, match="Refusing to promote"):
        promote_module.train_production_model_from_run(run_dir, save=False)


def test_a_normal_run_is_unaffected_by_any_of_these_guards(training_csv, tmp_path):
    """The guards must be inert on every real config, or they are a liability
    rather than a safeguard."""
    csv_path, _ = training_csv
    config = _config(csv_path, tmp_path, variance=None)
    assert config.is_diagnostic is False
    assert config.diagnostics.planted_signal.enabled is False
    # Naming and production training are both unconstrained here.
    assert ExperimentConfig(
        **{
            **config.model_dump(),
            "experiment_name": "anything_at_all",
            "refit": {"train_production_model": True},
        }
    ).experiment_name == "anything_at_all"


# --- artifacts --------------------------------------------------------------


def test_the_run_records_the_diagnostic_in_its_artifacts(training_csv, tmp_path):
    csv_path, _ = training_csv
    config = _config(
        csv_path,
        tmp_path,
        variance=0.02,
        save_experiment_artifacts=True,
        experiment_root_dir=tmp_path / "artifacts",
    )
    result = pipeline_module.run_experiment(config)

    assert result.run_dir is not None
    import json

    metadata = json.loads((result.run_dir / "metadata.json").read_text())
    assert metadata["is_diagnostic"] is True
    assert metadata["planted_requested_variance_explained"] == 0.02
    assert metadata["planted_column"] == PLANTED

    planted = json.loads((result.run_dir / "planted_signal.json").read_text())
    assert planted["is_diagnostic"] is True
    assert planted["planted_seed"] == 12345
    # The importance diagnostics come off the CV fold models, which are fitted
    # anyway -- so their absence would mean the capture silently no-opped.
    assert "fold_use_rate" in planted
    assert 0.0 <= planted["fold_use_rate"] <= 1.0

    # And the run directory names itself as diagnostic.
    assert result.run_dir.name.startswith(DIAGNOSTIC_NAME_PREFIX)


def test_a_normal_run_writes_no_planted_artifact(training_csv, tmp_path):
    csv_path, _ = training_csv
    config = _config(
        csv_path, tmp_path, variance=None, save_experiment_artifacts=True
    )
    result = pipeline_module.run_experiment(config)
    assert result.run_dir is not None
    assert not (result.run_dir / "planted_signal.json").exists()

    import json

    metadata = json.loads((result.run_dir / "metadata.json").read_text())
    assert metadata["is_diagnostic"] is False
    assert "planted_column" not in metadata


def test_fold_importance_columns_appear_only_on_a_diagnostic_run(
    training_csv, tmp_path
):
    csv_path, _ = training_csv
    on = pipeline_module.run_experiment(_config(csv_path, tmp_path, variance=0.02))
    off = pipeline_module.run_experiment(_config(csv_path, tmp_path, variance=None))

    assert on.cv_betting is not None and off.cv_betting is not None
    assert "planted_gain" in on.cv_betting.fold_metrics.columns
    assert not [
        c for c in off.cv_betting.fold_metrics.columns if c.startswith("planted_")
    ]


# --- importance reporting ---------------------------------------------------


def test_importance_reports_an_unused_feature_as_zero_not_missing():
    """A feature XGBoost never split on is absent from get_score entirely.
    Reporting that as NaN would make "the model ignored it" and "the artifact
    lost it" indistinguishable -- and the first is the headline result."""
    scores = {"gain": {"FEAT_A": 10.0, "FEAT_B": 4.0}}
    out = planted_feature_importance(scores, column=PLANTED, n_features=1458)
    assert out["planted_gain"] == 0.0
    assert out["planted_gain_used"] == 0.0
    assert out["planted_gain_rank"] == 1458


def test_importance_ranks_the_planted_feature_among_the_others():
    scores = {"gain": {"FEAT_A": 10.0, PLANTED: 7.0, "FEAT_B": 4.0}}
    out = planted_feature_importance(scores, column=PLANTED, n_features=3)
    assert out["planted_gain"] == 7.0
    assert out["planted_gain_rank"] == 2
    assert out["planted_gain_used"] == 1.0


# --- the campaign as checked in ---------------------------------------------


def test_the_four_cells_differ_only_in_planted_strength():
    """The entire design rests on this. If any other knob varies, a difference
    between cells cannot be attributed to the planted signal."""
    from pathlib import Path

    from training_pipeline.cli import load_config

    campaign = Path("experiments/diagnostics_planted_signal_2026_08")
    configs = [load_config(path) for path in sorted(campaign.glob("*.yaml"))]
    assert len(configs) == 4

    variances = sorted(
        c.diagnostics.planted_signal.variance_explained for c in configs
    )
    assert variances == [0.0, 0.005, 0.01, 0.02]

    def invariant(config):
        return (
            config.walk_forward.model_dump_json(),
            config.optuna.model_dump_json(),
            config.cleaning.model_dump_json(),
            config.data.model_dump_json(),
            config.random_state,
            config.diagnostics.planted_signal.seed,
            config.strategy,
            config.evaluation_seeds,
        )

    assert len({invariant(c) for c in configs}) == 1
    # And each is a real diagnostic, named as one, that cannot ship a model.
    for config in configs:
        assert config.is_diagnostic
        assert config.experiment_name.startswith(DIAGNOSTIC_NAME_PREFIX)
        assert config.refit.train_production_model is False
    # Four distinct studies: a planted signal changes what every trial means.
    assert len({c.fingerprint() for c in configs}) == 4
