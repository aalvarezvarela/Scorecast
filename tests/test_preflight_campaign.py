from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts.preflight_campaign import (
    _check_dataset_key_integrity,
    _resolve_config_paths,
)
from training_pipeline.config import DataConfig, DatasetType


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("experiment_name: test\n")
    return path


def test_resolve_config_paths_accepts_an_explicit_subset(tmp_path):
    first = _touch(tmp_path / "first.yaml")
    second = _touch(tmp_path / "second.yml")
    _touch(tmp_path / "not_selected.yaml")

    configs, errors = _resolve_config_paths([first, second, first])

    assert errors == []
    assert configs == [first, second]


def test_resolve_config_paths_expands_directories_and_ignores_base(tmp_path):
    first = _touch(tmp_path / "first.yaml")
    second = _touch(tmp_path / "second.yml")
    _touch(tmp_path / "_base.yaml")
    _touch(tmp_path / "notes.txt")

    configs, errors = _resolve_config_paths([tmp_path])

    assert errors == []
    assert configs == [first, second]


def test_dataset_key_integrity_rejects_invalid_dates_under_fast_preflight(tmp_path):
    csv = tmp_path / "intermediate.csv"
    pd.DataFrame(
        {
            "GAME_ID": ["game-1", "0.0"],
            "GAME_DATE": ["2026-01-01", "0.0"],
            "TIME_TO_MATCH_MIN": [30, None],
        }
    ).to_csv(csv, index=False)
    config = SimpleNamespace(
        data=DataConfig(csv_path=csv, dataset_type=DatasetType.INTERMEDIATE_LINE)
    )

    _, problems = _check_dataset_key_integrity(csv, config)

    assert any("invalid GAME_DATE" in problem for problem in problems)
    assert any("invalid TIME_TO_MATCH_MIN" in problem for problem in problems)


def test_dataset_key_integrity_accepts_unique_intermediate_keys(tmp_path):
    csv = tmp_path / "intermediate.csv"
    pd.DataFrame(
        {
            "GAME_ID": ["game-1", "game-1"],
            "GAME_DATE": ["2026-01-01", "2026-01-01"],
            "TIME_TO_MATCH_MIN": [30, 360],
        }
    ).to_csv(csv, index=False)
    config = SimpleNamespace(
        data=DataConfig(csv_path=csv, dataset_type=DatasetType.INTERMEDIATE_LINE)
    )

    summary, problems = _check_dataset_key_integrity(csv, config)

    assert problems == []
    assert "rows=2" in summary
    assert "unique_keys=2" in summary
