"""Recency weighting must be *evaluated* the same way it is later *applied*.

Two ways that can silently break, both regression-tested here:

1. Optuna samples `use_sample_weight` / `sample_weight_lambda`, but the CV fits
   that score those choices run unweighted -- so the search selects a training
   option it never actually tested, and the shipped model is fitted under a
   regime no trial measured.
2. A trial that explicitly chooses "no weighting" is represented as
   ``lambda=None``, which a downstream `None`-means-unset fallback then turns
   back into the configured lambda -- reinstating exactly what the trial
   rejected.
"""

import numpy as np
import optuna
import pandas as pd
import pytest

from training_pipeline import backtest as backtest_module
from training_pipeline import tuning as tuning_module
from training_pipeline.config import (
    CleaningConfig,
    DataConfig,
    ExperimentConfig,
    HoldoutConfig,
    OptunaConfig,
    SampleWeightConfig,
    SearchSpaceConfig,
    TargetFamily,
    WalkForwardConfig,
)
from training_pipeline.data import PreparedDataset
from training_pipeline.tuning import get_strategy

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _frame(n_games: int = 240, games_per_day: int = 4) -> pd.DataFrame:
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
    return df


def _prepared(df: pd.DataFrame) -> PreparedDataset:
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


def _config(tmp_path, *, target_family, sample_weight, **overrides):
    kwargs = {
        "experiment_name": "sw",
        "target_family": target_family,
        "line_col": (
            "ODDS_TOTAL_LINE_bet365"
            if target_family == TargetFamily.TOTAL_POINTS
            else None
        ),
        "data": DataConfig(csv_path="x.csv"),
        "cleaning": CleaningConfig(verbose=0),
        "holdout": HoldoutConfig(test_size=None, test_games=20),
        "walk_forward": WalkForwardConfig(
            test_games=20, step_games_between_tests=40, train_games=120,
            min_train_games=40, max_folds=2,
        ),
        "optuna": OptunaConfig(
            n_trials=2,
            search_space=SearchSpaceConfig(n_estimators=6, early_stopping_rounds=3),
        ),
        "sample_weight": sample_weight,
        "save_experiment_artifacts": False,
        "experiment_root_dir": tmp_path / "artifacts",
        "model_output_root": tmp_path / "models",
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


@pytest.fixture
def weight_spy(monkeypatch):
    """Record every lambda that actually reached a weight computation."""
    calls: list[float] = []
    real = tuning_module.build_recency_sample_weights

    def spy(dates, *, lambda_):
        calls.append(float(lambda_))
        return real(dates, lambda_=lambda_)

    monkeypatch.setattr(tuning_module, "build_recency_sample_weights", spy)
    monkeypatch.setattr(backtest_module, "build_recency_sample_weights", spy)
    return calls


# --- bug 1: the tuned option must actually be exercised during tuning -------


@pytest.mark.parametrize(
    "target_family", [TargetFamily.TOTAL_POINTS, TargetFamily.LINE_ERROR]
)
def test_cv_fits_are_weighted_for_both_target_families(
    tmp_path, weight_spy, target_family
):
    """Optuna must not select a weighting it never applied.

    TOTAL_POINTS regressed here once: its strategy called the shared objective
    without forwarding `dates`, and the objective silently skips weighting when
    dates are absent. The sampler still drew sample_weight_lambda, so the trial
    recorded a decay rate that no CV fit had ever used.
    """
    df = _frame()
    config = _config(
        tmp_path,
        target_family=target_family,
        sample_weight=SampleWeightConfig(
            enabled=True, tune_lambda=True, allow_unweighted=False
        ),
    )
    target = (
        "TOTAL_POINTS" if target_family == TargetFamily.TOTAL_POINTS else "LINE_ERROR"
    )
    X = df[["ODDS_TOTAL_LINE_bet365", "FEATURE_A", "FEATURE_B"]]

    get_strategy(config).tune(
        X=X,
        y=df[target],
        splits=[(np.arange(0, 150), np.arange(150, 200))],
        config=config,
        dates=df["GAME_DATE"],
    )

    assert weight_spy, (
        f"{target_family.value}: no sample weights were computed during tuning, "
        "so Optuna scored every trial on an unweighted fit while still recording "
        "a sample_weight_lambda."
    )


def test_tuned_lambda_is_the_one_the_folds_were_scored_with(tmp_path, weight_spy):
    """The lambda recorded on the trial must be the lambda the folds used."""
    df = _frame()
    config = _config(
        tmp_path,
        target_family=TargetFamily.TOTAL_POINTS,
        sample_weight=SampleWeightConfig(
            enabled=True, tune_lambda=False, lambda_=0.002
        ),
    )

    get_strategy(config).tune(
        X=df[["ODDS_TOTAL_LINE_bet365", "FEATURE_A", "FEATURE_B"]],
        y=df["TOTAL_POINTS"],
        splits=[(np.arange(0, 150), np.arange(150, 200))],
        config=config,
        dates=df["GAME_DATE"],
    )

    assert weight_spy == pytest.approx([0.002] * len(weight_spy))


# --- bug 2: "unweighted" must survive to the model that gets fitted --------


def test_a_trial_choosing_unweighted_is_not_re_weighted_by_the_config(
    tmp_path, weight_spy
):
    """`lambda=None` means "this trial chose not to weight", but a
    None-means-unset fallback downstream read it as "unspecified" and restored
    the configured lambda -- reinstating precisely what the trial rejected.
    """
    df = _frame()
    config = _config(
        tmp_path,
        target_family=TargetFamily.TOTAL_POINTS,
        # The dangerous combination: a concrete fallback lambda IS configured,
        # and Optuna is allowed to decline weighting.
        sample_weight=SampleWeightConfig(
            enabled=True, tune_lambda=True, lambda_=0.004, allow_unweighted=True
        ),
    )
    prepared = _prepared(df)

    backtest_module.run_walk_forward_evaluation(
        config,
        prepared=prepared,
        df_history=df.iloc[:200].reset_index(drop=True),
        df_evaluation=df.iloc[200:].reset_index(drop=True),
        train_games=120,
        xgb_params={"max_depth": 2},
        n_estimators=5,
        sample_weight_lambda=None,  # the trial said: do not weight
        show_progress=False,
    )

    assert not weight_spy, (
        "The walk-forward re-applied the configured lambda "
        f"{config.sample_weight.lambda_} after the trial had explicitly chosen "
        f"no weighting (observed lambdas: {weight_spy})."
    )


def test_standalone_backtest_still_honours_the_configured_lambda(
    tmp_path, weight_spy
):
    """The fallback exists for a reason: `run_daily_backtest` has no trial to
    read a decision from, so an explicitly configured lambda must still apply.
    Fixing the bug above must not silently disable weighting here.
    """
    df = _frame()
    config = _config(
        tmp_path,
        target_family=TargetFamily.TOTAL_POINTS,
        sample_weight=SampleWeightConfig(enabled=True, lambda_=0.003),
    )
    config.backtest.test_games = 40
    config.backtest.show_progress = False

    backtest_module.run_daily_backtest(config, prepared=_prepared(df))

    assert weight_spy, "run_daily_backtest ignored the configured lambda."
    assert weight_spy == pytest.approx([0.003] * len(weight_spy))
