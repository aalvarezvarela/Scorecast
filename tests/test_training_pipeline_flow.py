"""End-to-end checks that run_experiment follows the intended flow:

    Optuna on dev CV  ->  daily walk-forward across the test period
                      ->  [optional] production refit on the full-data tail
"""

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from training_pipeline import pipeline as pipeline_module
from training_pipeline.config import (
    CleaningConfig,
    DataConfig,
    ExperimentConfig,
    HoldoutConfig,
    HoldoutEvaluation,
    OptunaConfig,
    RefitConfig,
    SearchSpaceConfig,
    TargetFamily,
    WalkForwardConfig,
)
from training_pipeline.data import PreparedDataset
from training_pipeline.pipeline import run_experiment

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _prepared(n_games: int = 260, games_per_day: int = 4) -> PreparedDataset:
    rng = np.random.default_rng(0)
    n_days = n_games // games_per_day
    dates = np.repeat(
        pd.date_range("2025-11-01", periods=n_days, freq="D").to_numpy(), games_per_day
    )
    line = rng.uniform(200, 240, size=n_games).round(1)
    df = pd.DataFrame(
        {
            "GAME_DATE": dates,
            "SEASON_YEAR": 2025,
            "TOTAL_POINTS": (line + rng.normal(0, 12, n_games)).round(1),
            "ODDS_TOTAL_LINE_bet365": line,
            "FEATURE_A": rng.normal(size=n_games),
            "FEATURE_B": rng.normal(size=n_games),
        }
    )
    df["LINE_ERROR"] = df["TOTAL_POINTS"] - df["ODDS_TOTAL_LINE_bet365"]
    features = ["ODDS_TOTAL_LINE_bet365", "FEATURE_A", "FEATURE_B"]
    return PreparedDataset(
        df_full=df,
        X=df[features],
        y=df["TOTAL_POINTS"],
        baseline_line_col="ODDS_TOTAL_LINE_bet365",
        target_line_col="ODDS_TOTAL_LINE_bet365",
        feature_names=features,
        dataset_checksum="sha256:test",
    )


def _config(tmp_path, **overrides) -> ExperimentConfig:
    kwargs = {
        "experiment_name": "flow",
        "target_family": TargetFamily.TOTAL_POINTS,
        "line_col": "ODDS_TOTAL_LINE_bet365",
        "data": DataConfig(csv_path="x.csv"),
        "cleaning": CleaningConfig(verbose=0),
        "holdout": HoldoutConfig(test_size=None, test_games=20),
        "walk_forward": WalkForwardConfig(
            test_games=20, step_games_between_tests=40, train_games=120,
            min_train_games=40, max_folds=2,
        ),
        "optuna": OptunaConfig(
            n_trials=1,
            search_space=SearchSpaceConfig(n_estimators=8, early_stopping_rounds=4),
        ),
        "backtest": None,
        "save_experiment_artifacts": False,
        "experiment_root_dir": tmp_path / "artifacts",
        "model_output_root": tmp_path / "models",
    }
    kwargs.pop("backtest")
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


@pytest.fixture
def patched(monkeypatch):
    prepared = _prepared()
    monkeypatch.setattr(pipeline_module, "prepare_dataset", lambda cfg: prepared)
    return prepared


def test_daily_walk_forward_is_the_default_test_evaluation(patched, tmp_path):
    config = _config(tmp_path)
    assert config.holdout_evaluation == HoldoutEvaluation.DAILY_WALK_FORWARD

    result = run_experiment(config)

    assert result.walk_forward_result is not None
    assert result.holdout_result is None
    # One retrain per test game-day, pooled into a single set of predictions.
    assert result.walk_forward_result.n_days == 5      # 20 games / 4 per day
    assert result.walk_forward_result.n_games == 20


def test_walk_forward_never_trains_on_a_future_day(patched, tmp_path):
    result = run_experiment(_config(tmp_path))
    daily = result.walk_forward_result.daily_results
    for _, row in daily.iterrows():
        assert row["train_end_date"] < row["date"]


def test_rolling_window_slides_forward_across_test_days(patched, tmp_path):
    """With a fixed train_games the window SLIDES rather than grows: each day
    trains on the same number of games, but on more recent ones, because
    completed test days have entered history.
    """
    config = _config(tmp_path)
    daily = run_experiment(config).walk_forward_result.daily_results

    assert (daily["train_n_games"] == config.walk_forward.train_games).all()
    # The window's leading edge advances every day.
    ends = pd.to_datetime(daily["train_end_date"]).tolist()
    assert ends == sorted(ends)
    assert ends[-1] > ends[0]


def test_expanding_window_grows_as_test_days_are_played(patched, tmp_path):
    """With train_games=None every completed test day is added to history."""
    config = _config(
        tmp_path,
        walk_forward=WalkForwardConfig(
            test_games=20, step_games_between_tests=40, train_games=None,
            min_train_games=40, max_folds=2,
        ),
    )
    sizes = run_experiment(config).walk_forward_result.daily_results[
        "train_n_games"
    ].tolist()

    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]
    assert sizes[1] - sizes[0] == 4  # one game-day of history added


def test_no_production_model_unless_the_flag_is_on(patched, tmp_path):
    result = run_experiment(_config(tmp_path))
    assert result.model is None
    assert result.model_path is None


def test_production_model_is_fitted_on_the_full_dataset_tail(patched, tmp_path, monkeypatch):
    """The shipped model must see dev AND test -- production holds nothing back."""
    calls: list[dict] = []
    real = pipeline_module.fit_final_model

    def spy(*, X_dev, y_dev, dates_dev=None, **kwargs):
        calls.append(
            {
                "n_rows": len(X_dev),
                "max_date": pd.Timestamp(dates_dev.max()) if dates_dev is not None else None,
            }
        )
        return real(X_dev=X_dev, y_dev=y_dev, dates_dev=dates_dev, **kwargs)

    monkeypatch.setattr(pipeline_module, "fit_final_model", spy)

    config = _config(tmp_path, refit=RefitConfig(train_production_model=True))
    result = run_experiment(config)

    assert result.model is not None
    # The production fit is the last one, on the rolling window over the FULL
    # dataset -- and it reaches the very last game, i.e. into the test period.
    production = calls[-1]
    assert production["n_rows"] == config.walk_forward.train_games
    assert production["max_date"] == patched.df_full["GAME_DATE"].max()


def test_production_refit_uses_the_configured_window(patched, tmp_path):
    """The shipped model is fitted on the last train_games rows it is given."""
    from training_pipeline.evaluation import train_production_model
    from training_pipeline.tuning import get_strategy

    config = _config(tmp_path)
    strategy = get_strategy(config)
    df = patched.df_full
    X = df[patched.feature_names]
    y = df["TOTAL_POINTS"]

    fitted: dict = {}

    class _Spy:
        def __getattr__(self, name):
            return getattr(strategy, name)

        def fit_best(self, *, X_dev, y_dev, study, trial, config, dates_dev=None):
            fitted["n_rows"] = len(X_dev)
            fitted["max_date"] = pd.Timestamp(dates_dev.max())
            return "model"

    study = optuna.create_study()
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)

    train_production_model(
        _Spy(), X_dev=X, y_dev=y, dates_dev=df["GAME_DATE"],
        study=study, config=config,
    )

    assert fitted["n_rows"] == config.walk_forward.train_games
    assert fitted["max_date"] == df["GAME_DATE"].max()


def test_single_shot_mode_skips_the_daily_loop(patched, tmp_path):
    config = _config(tmp_path, holdout_evaluation=HoldoutEvaluation.SINGLE_SHOT)
    result = run_experiment(config)

    assert result.holdout_result is not None
    assert result.walk_forward_result is None


def test_save_model_kwarg_overrides_the_config_flag(patched, tmp_path):
    config = _config(tmp_path, refit=RefitConfig(train_production_model=True))
    assert run_experiment(config, save_model=False).model is None


# ---------------------------------------------------------------------------
# the drift null on the walk-forward path
# ---------------------------------------------------------------------------


def _betting_metrics(run_dir):
    import json
    from pathlib import Path

    return json.loads((Path(run_dir) / "betting_metrics.json").read_text())


def test_walk_forward_runs_record_a_real_drift_null(patched, tmp_path):
    """DailyBacktestResult carried no bias-corrected null, so run_experiment
    fell back to the model's OWN metrics when holdout_result was None -- which
    is every daily_walk_forward run, i.e. the default. betting_metrics.json
    then held a byte-copy of the model under the null's name, and the
    leaderboard's roi_vs_bias_baseline (roi - bias_baseline_roi) was 0.00 by
    construction. Nothing errored and nothing looked wrong.
    """
    config = _config(tmp_path, save_experiment_artifacts=True)
    result = run_experiment(config)
    assert result.holdout_result is None  # the branch that used to fall back

    metrics = _betting_metrics(result.run_dir)
    model = metrics["primary"]
    null = metrics["baseline_bias_corrected_betting"]

    assert model != null
    # The null bets every candidate game; the model selects among them.
    assert null["n_bets"] == null["n_candidates"]
    assert metrics["baseline_bias_corrected"]["mae"] != pytest.approx(
        result.walk_forward_result.mae
    )


def test_both_evaluation_modes_score_against_the_same_dev_drift(patched, tmp_path):
    """The bias must come from dev either way, or the two modes are measured
    against different nulls and their roi_vs_bias_baseline is not comparable."""
    # Distinct names: the run directory is <name>_<timestamp> to the second,
    # so two same-named runs in one test collide.
    walk_forward = run_experiment(
        _config(tmp_path, experiment_name="wf", save_experiment_artifacts=True)
    )
    single_shot = run_experiment(
        _config(
            tmp_path,
            experiment_name="ss",
            save_experiment_artifacts=True,
            holdout_evaluation=HoldoutEvaluation.SINGLE_SHOT,
        )
    )

    assert _betting_metrics(walk_forward.run_dir)["dev_line_error_bias"] == (
        pytest.approx(_betting_metrics(single_shot.run_dir)["dev_line_error_bias"])
    )


# ---------------------------------------------------------------------------
# baseline aggregation must follow the objective aggregation
# ---------------------------------------------------------------------------


def _cv_numbers(run_dir):
    import json
    from pathlib import Path

    metrics = json.loads((Path(run_dir) / "final_test_metrics.json").read_text())
    baseline = json.loads((Path(run_dir) / "baseline_metrics.json").read_text())
    folds = pd.read_csv(Path(run_dir) / "baseline_fold_metrics.csv")
    return metrics["cv"]["mae"], baseline["cv"]["mae"], folds


def test_pooled_runs_compare_the_model_against_a_pooled_baseline(patched, tmp_path):
    """The bug: cv_mae read pooled_mae while baseline_cv_mae stayed the mean of
    fold MAEs, so every model-vs-line CV number under
    objective_aggregation: pooled subtracted two different kinds of average.
    On a real run it turned a +0.02 edge into a reported -0.19 deficit."""
    from training_pipeline.config import ObjectiveAggregation, OptunaConfig

    config = _config(
        tmp_path,
        experiment_name="pooled_base",
        save_experiment_artifacts=True,
        optuna=OptunaConfig(
            n_trials=1,
            objective_aggregation=ObjectiveAggregation.POOLED,
            search_space=SearchSpaceConfig(n_estimators=8, early_stopping_rounds=4),
        ),
    )
    _, baseline_cv_mae, folds = _cv_numbers(run_experiment(config).run_dir)

    pooled = float((folds["mae"] * folds["n_games"]).sum() / folds["n_games"].sum())
    fold_mean = float(folds["mae"].mean())

    assert baseline_cv_mae == pytest.approx(pooled)
    if pooled != pytest.approx(fold_mean):
        assert baseline_cv_mae != pytest.approx(fold_mean)


def test_mean_runs_keep_the_fold_mean_baseline(patched, tmp_path):
    """The other half: runs that aggregate by mean must be unchanged."""
    from training_pipeline.config import ObjectiveAggregation, OptunaConfig

    config = _config(
        tmp_path,
        experiment_name="mean_base",
        save_experiment_artifacts=True,
        optuna=OptunaConfig(
            n_trials=1,
            objective_aggregation=ObjectiveAggregation.MEAN,
            search_space=SearchSpaceConfig(n_estimators=8, early_stopping_rounds=4),
        ),
    )
    _, baseline_cv_mae, folds = _cv_numbers(run_experiment(config).run_dir)

    assert baseline_cv_mae == pytest.approx(float(folds["mae"].mean()))


# ---------------------------------------------------------------------------
# the tie band reaches the run, and is recorded
# ---------------------------------------------------------------------------


def _metadata(run_dir):
    import json
    from pathlib import Path

    return json.loads((Path(run_dir) / "metadata.json").read_text())


def test_the_tie_band_is_recorded_in_the_run_metadata(patched, tmp_path):
    """Which metric actually chose the model is a property of the run. Before
    this it was unrecoverable from the artifacts: the tolerance was a constant
    in the config and the candidate count was never written down."""
    from training_pipeline.config import OptunaConfig

    config = _config(
        tmp_path,
        experiment_name="tie",
        save_experiment_artifacts=True,
        optuna=OptunaConfig(
            n_trials=4,
            search_space=SearchSpaceConfig(n_estimators=8, early_stopping_rounds=4),
        ),
    )
    metadata = _metadata(run_experiment(config).run_dir)

    assert metadata["tie_policy"] == "quantile"
    assert metadata["tie_n_completed"] >= 1
    assert 0 < metadata["tie_n_candidates"] <= metadata["tie_n_completed"]
    assert metadata["tie_tolerance"] <= 0.10   # the configured hard cap
    assert "tie_candidate_fraction" in metadata


def test_the_saved_candidate_table_is_the_set_the_selector_ranked(
    patched, tmp_path
):
    """Two independently-derived cutoffs would let the audit trail disagree
    with the decision it is supposed to explain."""
    from training_pipeline.config import OptunaConfig

    config = _config(
        tmp_path,
        experiment_name="tie_candidates",
        save_experiment_artifacts=True,
        optuna=OptunaConfig(
            n_trials=6,
            search_space=SearchSpaceConfig(n_estimators=8, early_stopping_rounds=4),
        ),
    )
    result = run_experiment(config)
    metadata = _metadata(result.run_dir)
    candidates = pd.read_csv(Path(result.run_dir) / "optuna_lexicographic_candidates.csv")

    assert len(candidates) == metadata["tie_n_candidates"]


def test_fold_sizes_are_recorded_so_a_two_game_fold_cannot_hide(patched, tmp_path):
    """"30 folds, 855 games" hides the difference between thirty 28-game folds
    and a layout containing a 2-game one."""
    config = _config(tmp_path, experiment_name="folds", save_experiment_artifacts=True)
    metadata = _metadata(run_experiment(config).run_dir)

    counts = metadata["cv_fold_game_counts"]
    assert sum(counts) == metadata["cv_n_validation_games"]
    assert len(counts) == metadata["cv_n_folds"]
