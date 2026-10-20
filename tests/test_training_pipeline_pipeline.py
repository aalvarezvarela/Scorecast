import pandas as pd
import pytest

from training_pipeline import pipeline as pipeline_module
from training_pipeline.config import DataConfig, ExperimentConfig, TargetFamily
from training_pipeline.data import PreparedDataset
from training_pipeline.naming import build_model_name, resolve_model_output_dir


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "SEASON_YEAR": [2025, 2025],
            "TOTAL_POINTS": [210.0, 220.0],
            "ODDS_TOTAL_LINE_bet365": [205.0, 215.0],
        }
    )


def _config(tmp_path) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="fail_fast_test",
        target_family=TargetFamily.TOTAL_POINTS,
        line_col="ODDS_TOTAL_LINE_bet365",
        data=DataConfig(csv_path="x.csv"),
        window_dir_label="3_seasons",
        window_name_label="three_seasons",
        model_output_root=tmp_path / "models",
        experiment_root_dir=tmp_path / "artifacts",
    )


def test_run_experiment_refuses_existing_bundle_before_running_optuna(monkeypatch, tmp_path):
    """Regression: the overwrite guard used to run only after tuning, so a
    collision discarded a full (potentially hours-long) Optuna run and still
    left an orphan experiment directory behind. It must fail fast instead.
    """
    config = _config(tmp_path)
    df = _tiny_frame()

    # Pre-create the bundle the run would write to.
    out_dir = resolve_model_output_dir(config)
    out_dir.mkdir(parents=True)
    model_name = build_model_name(config, as_of=df["GAME_DATE"].max().date())
    (out_dir / f"{model_name}.json").write_text("{}")

    prepared = PreparedDataset(
        df_full=df,
        X=df[["ODDS_TOTAL_LINE_bet365"]],
        y=df["TOTAL_POINTS"],
        baseline_line_col="ODDS_TOTAL_LINE_bet365",
        target_line_col="ODDS_TOTAL_LINE_bet365",
        feature_names=["ODDS_TOTAL_LINE_bet365"],
    )
    monkeypatch.setattr(pipeline_module, "prepare_dataset", lambda cfg: prepared)
    monkeypatch.setattr(
        pipeline_module, "build_holdout_split", lambda df_full, cfg: (df, df)
    )

    def _explode(*args, **kwargs):
        raise AssertionError("tuning must not start when the bundle path is taken")

    monkeypatch.setattr(pipeline_module, "build_split_provider", _explode)

    with pytest.raises(FileExistsError, match="already exists"):
        pipeline_module.run_experiment(config, save_model=True)

    # No orphan experiment run directory should have been created either.
    assert not (tmp_path / "artifacts").exists()


def test_run_experiment_skips_bundle_check_when_not_saving_model(monkeypatch, tmp_path):
    """save_model=False is the notebook/exploration path -- an existing bundle
    from a previous run must not block it.
    """
    config = _config(tmp_path)
    df = _tiny_frame()

    out_dir = resolve_model_output_dir(config)
    out_dir.mkdir(parents=True)
    model_name = build_model_name(config, as_of=df["GAME_DATE"].max().date())
    (out_dir / f"{model_name}.json").write_text("{}")

    prepared = PreparedDataset(
        df_full=df,
        X=df[["ODDS_TOTAL_LINE_bet365"]],
        y=df["TOTAL_POINTS"],
        baseline_line_col="ODDS_TOTAL_LINE_bet365",
        target_line_col="ODDS_TOTAL_LINE_bet365",
        feature_names=["ODDS_TOTAL_LINE_bet365"],
    )
    monkeypatch.setattr(pipeline_module, "prepare_dataset", lambda cfg: prepared)
    monkeypatch.setattr(
        pipeline_module, "build_holdout_split", lambda df_full, cfg: (df, df)
    )

    reached = {"tuning": False}

    def _marker(*args, **kwargs):
        reached["tuning"] = True
        raise RuntimeError("stop here -- we only care that we got past the guard")

    monkeypatch.setattr(pipeline_module, "build_split_provider", _marker)

    with pytest.raises(RuntimeError):
        pipeline_module.run_experiment(config, save_model=False)

    assert reached["tuning"], "guard must not block runs that don't save a model"
