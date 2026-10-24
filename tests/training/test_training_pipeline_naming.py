from datetime import date

import pytest

from training_pipeline.config import DataConfig, ExperimentConfig, TargetFamily
from training_pipeline.naming import (
    DEFAULT_TRAINING_CODE_TAG,
    assert_model_bundle_is_writable,
    build_model_bundle_metadata,
    build_model_name,
    resolve_model_output_dir,
)


def _config(**overrides) -> ExperimentConfig:
    kwargs = {
        "experiment_name": "naming_test",
        "target_family": TargetFamily.TOTAL_POINTS,
        "line_col": "ODDS_TOTAL_LINE_bet365",
        "data": DataConfig(csv_path="data/train_data/example.csv"),
        "window_name_label": "three_seasons",
        "window_dir_label": "3_seasons",
    }
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


def test_build_model_name_uses_supplied_data_date_not_today():
    """Regression: build_model_name used to default to date.today(), which made
    a model retrained on an old snapshot look freshly trained and disagree with
    ModelBundleMetadata.model_version (derived from the training window).
    """
    name = build_model_name(_config(), as_of=date(2026, 3, 18))
    assert name == "three_seasons_xgb_total_points_18_03_26"


def test_build_model_name_requires_as_of():
    with pytest.raises(TypeError):
        build_model_name(_config())  # type: ignore[call-arg]


def _bundle_metadata(config: ExperimentConfig):
    return build_model_bundle_metadata(
        config,
        model_name="m",
        best_params={"max_depth": 3},
        selected_trial_number=1,
        mean_best_iteration=100,
        median_best_iteration=100,
        train_games=2500,
        cv_mae=13.5,
        cv_rmse=17.0,
        cv_ou_acc=0.53,
        final_test_mae=13.4,
        final_test_rmse=16.9,
        final_test_ou_acc=0.55,
        train_date_min=date(2025, 10, 1),
        train_date_max=date(2026, 3, 18),
    )


def test_training_version_is_written_into_the_model_bundle():
    """A saved bundle must be traceable back to the training approach label."""
    metadata = _bundle_metadata(_config(training_version="2.1-style-features"))
    assert metadata.model_info.training_code_tag == "2.1-style-features"


def test_model_bundle_falls_back_to_default_tag_when_version_unset():
    metadata = _bundle_metadata(_config())
    assert metadata.model_info.training_code_tag == DEFAULT_TRAINING_CODE_TAG


def test_resolve_model_output_dir_follows_repo_convention():
    out_dir = resolve_model_output_dir(_config())
    assert out_dir.as_posix().endswith("models/total_points/3_seasons")


def _intermediate_config(snapshot_minutes: int) -> ExperimentConfig:
    return _config(
        data=DataConfig(
            csv_path="data/train_data/example.csv",
            dataset_type="intermediate_line",
            snapshot_minutes=snapshot_minutes,
        ),
    )


def test_intermediate_horizons_on_the_same_window_get_distinct_bundles():
    """Regression: T-30 and T-60, both at 6,275 games and trained to the same
    date, resolved to one bundle path, so promoting the second horizon was
    refused -- or with overwrite would have replaced the first.
    """
    as_of = date(2026, 4, 17)
    t30, t60 = _intermediate_config(30), _intermediate_config(60)

    assert resolve_model_output_dir(t30) != resolve_model_output_dir(t60)
    assert build_model_name(t30, as_of=as_of) != build_model_name(t60, as_of=as_of)
    assert resolve_model_output_dir(t60).as_posix().endswith(
        "models/total_points/t0060/3_seasons"
    )
    assert build_model_name(t60, as_of=as_of) == (
        "three_seasons_t0060_xgb_total_points_17_04_26"
    )


def test_closing_line_bundle_paths_are_unchanged_by_the_horizon_label():
    config = _config()
    assert resolve_model_output_dir(config).as_posix().endswith(
        "models/total_points/3_seasons"
    )
    assert build_model_name(config, as_of=date(2026, 4, 17)) == (
        "three_seasons_xgb_total_points_17_04_26"
    )


def test_assert_model_bundle_is_writable_passes_when_nothing_exists(tmp_path):
    assert_model_bundle_is_writable(
        tmp_path, model_name="some_model", overwrite_existing_model=False
    )


def test_assert_model_bundle_is_writable_raises_on_collision(tmp_path):
    (tmp_path / "some_model.json").write_text("{}")

    with pytest.raises(FileExistsError, match="already exists"):
        assert_model_bundle_is_writable(
            tmp_path, model_name="some_model", overwrite_existing_model=False
        )


def test_assert_model_bundle_is_writable_detects_orphaned_meta_file(tmp_path):
    (tmp_path / "some_model.meta.json").write_text("{}")

    with pytest.raises(FileExistsError):
        assert_model_bundle_is_writable(
            tmp_path, model_name="some_model", overwrite_existing_model=False
        )


def test_assert_model_bundle_is_writable_allows_explicit_overwrite(tmp_path):
    (tmp_path / "some_model.json").write_text("{}")

    assert_model_bundle_is_writable(
        tmp_path, model_name="some_model", overwrite_existing_model=True
    )
