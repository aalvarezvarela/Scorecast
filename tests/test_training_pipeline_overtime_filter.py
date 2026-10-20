"""Overtime games can be dropped from TRAINING while still being scored.

The asymmetry is the entire point. An overtime game's total is inflated by at
least five minutes of basketball no pre-game feature could have predicted, so
those rows may be teaching the model noise -- but ~5.2% of real games go to
overtime and you are paid or not paid on them, so excluding them from scoring
would measure a world that does not exist.
"""

import numpy as np
import optuna
import pandas as pd
import pytest

from training_pipeline import pipeline as pipeline_module
from training_pipeline.config import (
    LEAKING_TARGET_COLUMNS,
    CleaningConfig,
    DataConfig,
    ExperimentConfig,
    HoldoutConfig,
    OptunaConfig,
    SearchSpaceConfig,
    WalkForwardConfig,
)
from training_pipeline.data import PreparedDataset, training_eligible_mask
from training_pipeline.pipeline import run_experiment
from training_pipeline.splits import apply_training_filter

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _frame(n_games: int = 260, games_per_day: int = 4, ot_rate: float = 0.20):
    rng = np.random.default_rng(0)
    n_days = n_games // games_per_day
    dates = np.repeat(
        pd.date_range("2025-11-01", periods=n_days, freq="D").to_numpy(), games_per_day
    )
    line = rng.uniform(200, 240, size=n_games).round(1)
    is_ot = (rng.random(n_games) < ot_rate).astype(int)
    df = pd.DataFrame(
        {
            "GAME_DATE": dates,
            "SEASON_YEAR": 2025,
            # Overtime really does inflate the total, as in real data.
            "TOTAL_POINTS": (line + rng.normal(0, 12, n_games) + is_ot * 12).round(1),
            "ODDS_TOTAL_LINE_bet365": line,
            "IS_OVERTIME": is_ot,
            "FEATURE_A": rng.normal(size=n_games),
            "FEATURE_B": rng.normal(size=n_games),
        }
    )
    df["LINE_ERROR"] = df["TOTAL_POINTS"] - df["ODDS_TOTAL_LINE_bet365"]
    return df


def _config(tmp_path, **overrides) -> ExperimentConfig:
    kwargs = {
        "experiment_name": "ot",
        "target_family": "total_points",
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
        "save_experiment_artifacts": False,
        "experiment_root_dir": tmp_path / "artifacts",
        "model_output_root": tmp_path / "models",
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


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


# --- it is never a feature --------------------------------------------------


def test_is_overtime_can_never_become_a_feature(tmp_path):
    """A post-game fact: the game already went to overtime and therefore
    already scored more. Retained only so training rows can be filtered on it.
    """
    assert "IS_OVERTIME" in LEAKING_TARGET_COLUMNS
    assert "IS_OVERTIME" in _config(tmp_path).exclude_cols


def test_leak_guard_catches_it_reaching_the_matrix():
    from training_pipeline.data import assert_no_leaking_features

    with pytest.raises(ValueError, match="reached the feature matrix"):
        assert_no_leaking_features(pd.DataFrame({"A": [1.0], "IS_OVERTIME": [1]}))


# --- the mask ---------------------------------------------------------------


def test_default_keeps_every_game(tmp_path):
    df = _frame()
    assert training_eligible_mask(df, _config(tmp_path)).all()


def test_enabling_the_filter_marks_overtime_games_ineligible(tmp_path):
    df = _frame()
    config = _config(
        tmp_path, data=DataConfig(csv_path="x.csv", exclude_overtime_from_training=True)
    )
    mask = training_eligible_mask(df, config)

    assert (~mask).sum() == int(df["IS_OVERTIME"].sum())
    assert not df.loc[mask, "IS_OVERTIME"].any()


def test_unknown_overtime_stays_trainable(tmp_path):
    """NaN means "not yet played", not "went to overtime"."""
    df = _frame().head(4).copy()
    df["IS_OVERTIME"] = [1, 0, np.nan, 1]
    config = _config(
        tmp_path, data=DataConfig(csv_path="x.csv", exclude_overtime_from_training=True)
    )
    assert list(training_eligible_mask(df, config)) == [False, True, True, False]


def test_a_missing_column_explains_how_to_fix_it(tmp_path):
    """Builds predating this option dropped IS_OVERTIME before writing the CSV."""
    config = _config(
        tmp_path, data=DataConfig(csv_path="x.csv", exclude_overtime_from_training=True)
    )
    with pytest.raises(KeyError, match="create_train_data.py"):
        training_eligible_mask(_frame().drop(columns=["IS_OVERTIME"]), config)


# --- training only, never evaluation ----------------------------------------


def test_only_train_indices_are_filtered_never_validation(tmp_path):
    df = _frame()
    config = _config(
        tmp_path, data=DataConfig(csv_path="x.csv", exclude_overtime_from_training=True)
    )
    splits = [(np.arange(0, 200), np.arange(200, 260))]

    (train_idx, valid_idx), = apply_training_filter(df, splits, config)

    assert not df.iloc[train_idx]["IS_OVERTIME"].any()   # training is clean
    assert df.iloc[valid_idx]["IS_OVERTIME"].any()       # scoring is not
    np.testing.assert_array_equal(valid_idx, np.arange(200, 260))


def test_filtering_is_a_no_op_when_disabled(tmp_path):
    df = _frame()
    splits = [(np.arange(0, 200), np.arange(200, 260))]
    assert apply_training_filter(df, splits, _config(tmp_path)) is splits


def test_an_empty_training_fold_is_reported_not_silently_accepted(tmp_path):
    df = _frame(ot_rate=1.0)
    config = _config(
        tmp_path, data=DataConfig(csv_path="x.csv", exclude_overtime_from_training=True)
    )
    with pytest.raises(ValueError, match="no training rows left"):
        apply_training_filter(df, [(np.arange(0, 200), np.arange(200, 260))], config)


def test_the_holdout_still_contains_overtime_games(monkeypatch, tmp_path):
    """End to end: the test period must be scored on every game."""
    df = _frame()
    config = _config(
        tmp_path, data=DataConfig(csv_path="x.csv", exclude_overtime_from_training=True)
    )
    monkeypatch.setattr(pipeline_module, "prepare_dataset", lambda cfg: _prepared(df))

    result = run_experiment(config)

    assert result.df_test["IS_OVERTIME"].any()
    # Every held-out game was predicted, overtime included.
    assert result.walk_forward_result.n_games == len(result.df_test)


def test_cv_folds_train_clean_but_score_everything(monkeypatch, tmp_path):
    df = _frame()
    config = _config(
        tmp_path, data=DataConfig(csv_path="x.csv", exclude_overtime_from_training=True)
    )
    monkeypatch.setattr(pipeline_module, "prepare_dataset", lambda cfg: _prepared(df))

    result = run_experiment(config)

    for train_idx, valid_idx in result.splits:
        assert not result.df_dev.iloc[train_idx]["IS_OVERTIME"].any()
    # Pooled CV scoring still covers overtime games.
    scored = result.df_dev.iloc[
        result.cv_betting.predictions["row_in_dev"].to_numpy()
    ]
    assert scored["IS_OVERTIME"].any()


def test_single_shot_holdout_fit_also_excludes_overtime(monkeypatch, tmp_path):
    """The filter must hold on every training path, not just the default one.

    single_shot fits one model on the dev window instead of retraining daily.
    It previously ignored the flag, so a config asking to exclude overtime
    silently trained on it -- and the run still looked successful.
    """
    from training_pipeline import pipeline as pipeline_module
    from training_pipeline.config import HoldoutEvaluation

    prepared = _prepared(_frame())
    monkeypatch.setattr(pipeline_module, "prepare_dataset", lambda cfg: prepared)

    seen: list[int] = []
    real_fit = pipeline_module.fit_final_model

    def spy(*, X_dev, **kwargs):
        seen.append(len(X_dev))
        return real_fit(X_dev=X_dev, **kwargs)

    monkeypatch.setattr(pipeline_module, "fit_final_model", spy)

    config = _config(
        tmp_path,
        holdout_evaluation=HoldoutEvaluation.SINGLE_SHOT,
        data=DataConfig(csv_path="x.csv", exclude_overtime_from_training=True),
    )
    pipeline_module.run_experiment(config)

    from training_pipeline.splits import build_holdout_split

    df_dev, _ = build_holdout_split(prepared.df_full, config)
    overtime_in_dev = int(
        pd.to_numeric(df_dev["IS_OVERTIME"], errors="coerce").eq(1).sum()
    )
    assert overtime_in_dev > 0, "fixture must contain overtime games in dev"
    assert seen, "fit_final_model was never called"
    assert seen[0] == len(df_dev) - overtime_in_dev
