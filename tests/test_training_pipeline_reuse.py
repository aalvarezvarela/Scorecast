import json
from pathlib import Path

import pytest

from training_pipeline.config import (
    DataConfig,
    ExperimentConfig,
    OptunaConfig,
    SampleWeightConfig,
    TargetFamily,
)
from training_pipeline.reuse import load_run_hyperparameters


def _write_trials(
    tmp_path: Path,
    *,
    params: dict,
    user_attrs: dict,
    number: int = 7,
    which: str = "selected",
) -> Path:
    run_dir = tmp_path / "run_20260101_120000"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"number": number, "value": 13.5, "params": params, "user_attrs": user_attrs}
    filename = (
        "optuna_selected_trial.json" if which == "selected" else "optuna_best_trial.json"
    )
    key = "selected_trial" if which == "selected" else "best_trial"
    (run_dir / filename).write_text(json.dumps({key: payload}))
    return run_dir


def test_recovers_params_and_boosting_rounds_from_a_saved_run(tmp_path):
    run_dir = _write_trials(
        tmp_path,
        params={"max_depth": 3, "learning_rate": 0.02},
        user_attrs={"median_best_iteration": 143, "mean_mae": 13.42, "mean_ou_acc": 0.55},
    )

    recovered = load_run_hyperparameters(run_dir)

    assert recovered.params == {"max_depth": 3, "learning_rate": 0.02}
    assert recovered.n_estimators == 143
    assert recovered.trial_number == 7
    assert recovered.source == "selected"
    assert recovered.cv_mae == pytest.approx(13.42)


def test_early_stopping_cap_does_not_override_the_median_best_iteration(tmp_path):
    """Regression: an early-stopping trial records user_attrs n_estimators=1000,
    which is the cap, not the rounds. The holdout backtest fitted the median
    (51); promotion read the cap and would have shipped a 1000-round model.
    """
    run_dir = _write_trials(
        tmp_path,
        params={"max_depth": 3},
        user_attrs={
            "n_estimators": 1000,
            "median_best_iteration": 51,
            "mean_best_iteration": 61,
        },
    )

    assert load_run_hyperparameters(run_dir).n_estimators == 51


def test_tuned_rounds_still_win_over_recorded_attrs(tmp_path):
    run_dir = _write_trials(
        tmp_path,
        params={"max_depth": 3, "n_estimators": 240},
        user_attrs={"n_estimators": 240},
    )

    assert load_run_hyperparameters(run_dir).n_estimators == 240


def test_sample_weight_lambda_is_split_out_of_the_xgb_params(tmp_path):
    """sample_weight_lambda is a training-protocol parameter; feeding it to
    XGBRegressor would be silently ignored.
    """
    run_dir = _write_trials(
        tmp_path,
        params={"max_depth": 3, "sample_weight_lambda": 0.004},
        user_attrs={"median_best_iteration": 120},
    )

    recovered = load_run_hyperparameters(run_dir)

    assert "sample_weight_lambda" not in recovered.params
    assert recovered.sample_weight_lambda == pytest.approx(0.004)


def test_falls_back_to_the_best_trial_when_no_selected_trial_exists(tmp_path):
    run_dir = _write_trials(
        tmp_path,
        params={"max_depth": 2},
        user_attrs={"mean_best_iteration": 90},
        which="best",
    )
    recovered = load_run_hyperparameters(run_dir)
    assert recovered.source == "best"
    assert recovered.n_estimators == 90


def test_raises_a_helpful_error_when_nothing_is_recoverable(tmp_path):
    empty = tmp_path / "empty_run"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No Optuna trial artifacts"):
        load_run_hyperparameters(empty)


def test_yaml_block_is_valid_and_round_trips_into_a_config(tmp_path):
    """The printed block must actually paste into an experiment file and
    produce a config that skips tuning.
    """
    import yaml

    run_dir = _write_trials(
        tmp_path,
        params={"max_depth": 3, "learning_rate": 0.02, "sample_weight_lambda": 0.004},
        user_attrs={"median_best_iteration": 143, "mean_mae": 13.42},
    )
    block = load_run_hyperparameters(run_dir).to_yaml_block()

    parsed = yaml.safe_load(block)
    config = ExperimentConfig(
        experiment_name="reused",
        target_family=TargetFamily.LINE_ERROR,
        data=DataConfig(csv_path="x.csv"),
        optuna=OptunaConfig(**parsed["optuna"]),
    )

    assert config.optuna.skip_tuning is True
    assert config.optuna.fixed_params == {"max_depth": 3, "learning_rate": 0.02}
    assert config.optuna.fixed_n_estimators == 143
    assert config.optuna.fixed_sample_weight_lambda == pytest.approx(0.004)


def test_total_points_now_accepts_recency_weighting():
    """Regression: sample weighting used to be rejected for TOTAL_POINTS
    because upstream's tuner had no such parameter. training_pipeline supplies
    its own objective and final fit, so both families support it.
    """
    config = ExperimentConfig(
        experiment_name="weighted_total_points",
        target_family=TargetFamily.TOTAL_POINTS,
        line_col="ODDS_TOTAL_LINE_bet365",
        data=DataConfig(csv_path="x.csv"),
        sample_weight=SampleWeightConfig(enabled=True, lambda_=0.005),
    )
    assert config.sample_weight.enabled is True
    assert config.sample_weight.lambda_ == pytest.approx(0.005)


def test_trial_that_declined_weighting_is_not_re_enabled(tmp_path):
    """Regression: a trial choosing use_sample_weight=False records no lambda,
    and the config fallback must not quietly reinstate weighting for the final
    fit.
    """
    from training_pipeline.tuning import USE_SAMPLE_WEIGHT_PARAM, resolve_final_params

    run_dir = _write_trials(
        tmp_path,
        params={"max_depth": 3, USE_SAMPLE_WEIGHT_PARAM: False},
        user_attrs={"median_best_iteration": 100},
    )
    recovered = load_run_hyperparameters(run_dir)
    assert recovered.sample_weight_lambda is None
    assert USE_SAMPLE_WEIGHT_PARAM not in recovered.params

    class _Trial:
        params = {"max_depth": 3, USE_SAMPLE_WEIGHT_PARAM: False}
        user_attrs = {"median_best_iteration": 100}

    config = ExperimentConfig(
        experiment_name="t",
        target_family=TargetFamily.LINE_ERROR,
        data=DataConfig(csv_path="x.csv"),
        # Config supplies a lambda, but the trial said no.
        sample_weight=SampleWeightConfig(enabled=True, lambda_=0.005, tune_lambda=True),
    )
    params, n_estimators, lambda_ = resolve_final_params(_Trial(), config)
    assert lambda_ is None
    assert USE_SAMPLE_WEIGHT_PARAM not in params
