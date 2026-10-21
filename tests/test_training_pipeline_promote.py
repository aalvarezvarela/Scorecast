import json
from pathlib import Path

import pytest

from training_pipeline.config import (
    DataConfig,
    ExperimentConfig,
    RefitStrategy,
    TargetFamily,
    WalkForwardConfig,
)
from training_pipeline.promote import load_run_config, train_production_model_from_run


def _make_run(
    tmp_path: Path,
    csv_path: str,
    *,
    selected_train_games: int | None = None,
    data: DataConfig | None = None,
    walk_forward: WalkForwardConfig | None = None,
) -> Path:
    run_dir = tmp_path / "src_run_20260101_120000"
    run_dir.mkdir(parents=True)
    config = ExperimentConfig(
        experiment_name="src",
        training_version="2.1",
        target_family=TargetFamily.TOTAL_POINTS,
        line_col="ODDS_TOTAL_LINE_bet365",
        data=data or DataConfig(csv_path=csv_path, expected_checksum="sha256:stale"),
        walk_forward=walk_forward or WalkForwardConfig(train_games=40),
        model_output_root=tmp_path / "models",
    )
    (run_dir / "config.json").write_text(config.model_dump_json())
    (run_dir / "optuna_selected_trial.json").write_text(
        json.dumps(
            {
                "selected_trial": {
                    "number": 4,
                    "value": 13.4,
                    "params": {"max_depth": 2, "learning_rate": 0.03},
                    "user_attrs": {
                        "median_best_iteration": 60,
                        "mean_mae": 13.4,
                        **(
                            {"train_games": selected_train_games}
                            if selected_train_games is not None
                            else {}
                        ),
                    },
                }
            }
        )
    )
    (run_dir / "final_test_metrics.json").write_text(
        json.dumps(
            {
                "cv": {"mae": 13.4},
                "holdout": {"mae": 13.2, "rmse": 17.0, "ou_acc": 0.55},
            }
        )
    )
    return run_dir


@pytest.fixture
def csv(tmp_path):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n = 120
    line = rng.uniform(200, 240, n).round(1)
    df = pd.DataFrame(
        {
            "GAME_ID": [f"00225000{i:02d}" for i in range(n)],
            "GAME_DATE": pd.date_range("2025-11-01", periods=n, freq="D"),
            "SEASON_YEAR": 2025,
            "TOTAL_POINTS": (line + rng.normal(0, 10, n)).round(1),
            "ODDS_TOTAL_LINE_bet365": line,
            "FEATURE_A": rng.normal(size=n),
        }
    )
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    return path


def test_load_run_config_round_trips(tmp_path, csv):
    run_dir = _make_run(tmp_path, str(csv))
    assert load_run_config(run_dir).experiment_name == "src"


def test_load_run_config_requires_a_snapshot(tmp_path):
    empty = tmp_path / "no_config"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="config.json"):
        load_run_config(empty)


def test_promotes_onto_different_data_and_records_the_source_run(tmp_path, csv):
    """The whole point: reuse a run's chosen hyperparameters on newer data."""
    run_dir = _make_run(tmp_path, "data/train_data/old_and_gone.csv")

    result = train_production_model_from_run(run_dir, csv_path=csv)

    assert result.model is not None
    assert result.csv_path == csv
    # Fitted on the configured rolling window, not the whole file.
    assert result.n_train_games == 40
    assert result.hyperparameters.params == {"max_depth": 2, "learning_rate": 0.03}
    assert result.hyperparameters.n_estimators == 60

    meta = json.loads(Path(result.meta_path).read_text())
    # The bundle names the experiment that justified deploying it.
    assert "from_run:src_run_20260101_120000" in meta["model"]["training_code_tag"]
    assert "2.1" in meta["model"]["training_code_tag"]


def test_stale_checksum_is_dropped_when_data_is_overridden(tmp_path, csv):
    """The source run pinned a checksum for its own bytes; keeping it would
    make every promotion onto new data fail.
    """
    run_dir = _make_run(tmp_path, "data/train_data/old_and_gone.csv")
    result = train_production_model_from_run(run_dir, csv_path=csv)
    assert result.dataset_checksum is not None
    assert result.dataset_checksum != "sha256:stale"


def test_full_dataset_strategy_uses_every_game(tmp_path, csv):
    run_dir = _make_run(tmp_path, str(csv))
    result = train_production_model_from_run(
        run_dir, csv_path=csv, refit_strategy=RefitStrategy.FULL_DATASET
    )
    assert result.n_train_games > 40


def test_promotion_uses_selected_window_and_records_games_not_snapshot_rows(
    tmp_path, csv
):
    import pandas as pd

    closing = pd.read_csv(csv, dtype={"GAME_ID": str})
    pooled = pd.concat(
        [
            closing.assign(TIME_TO_MATCH_MIN=720, SNAPSHOT_FEATURE=0.0),
            closing.assign(TIME_TO_MATCH_MIN=360, SNAPSHOT_FEATURE=1.0),
        ],
        ignore_index=True,
    )
    pooled_path = tmp_path / "pooled.csv"
    pooled.to_csv(pooled_path, index=False)

    run_dir = _make_run(
        tmp_path,
        str(pooled_path),
        selected_train_games=25,
        data=DataConfig(
            csv_path=pooled_path,
            dataset_type="intermediate_line",
        ),
        walk_forward=WalkForwardConfig(
            strategy="rolling_origin",
            train_games=40,
            train_games_choices=(25, 40),
            min_train_games=20,
        ),
    )
    result = train_production_model_from_run(run_dir)

    assert result.hyperparameters.train_games == 25
    assert result.n_train_games == 25
    metadata = json.loads(Path(result.meta_path).read_text())
    assert metadata["training_metrics"]["train_games"] == 25


def test_refuses_to_clobber_an_existing_bundle(tmp_path, csv):
    run_dir = _make_run(tmp_path, str(csv))
    train_production_model_from_run(run_dir, csv_path=csv)
    with pytest.raises(FileExistsError):
        train_production_model_from_run(run_dir, csv_path=csv)
    # ...unless explicitly told to.
    train_production_model_from_run(run_dir, csv_path=csv, overwrite=True)
