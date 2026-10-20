"""Model save-path/name conventions + ModelBundleMetadata construction.

Matches the existing on-disk convention (models/<target>/<window_dir_label>/
<window_name_label>_xgb_<target>_<DD_MM_YY>.json + .meta.json), built via
nba_ou.modeling.modeling.{ModelBundleMetadata, ModelInfo, TrainingMetrics,
save_model_bundle}.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
from nba_ou.modeling.modeling import ModelBundleMetadata, ModelInfo, TrainingMetrics
from nba_ou.modeling.registry_paths import format_horizon

from training_pipeline.config import DatasetType, ExperimentConfig, TargetFamily

#: Used for ModelInfo.training_code_tag when ExperimentConfig.training_version
#: has not been set by hand.
DEFAULT_TRAINING_CODE_TAG = "training_pipeline-1.0"


def _horizon_segment(config: ExperimentConfig) -> str | None:
    """``t0060``-style label for intermediate-line models, None for closing.

    Without it, every intermediate horizon trained on the same window and data
    end date shares one bundle path -- T-30 and T-60 both at 6,275 games wrote
    ``6275_games_xgb_line_error_17_04_26`` -- and the second is refused (or,
    with overwrite, silently replaces the first). Same label as the registry
    slot, so a local bundle and its slot read alike. Closing-line paths are
    unchanged.
    """
    if config.data.dataset_type is DatasetType.CLOSING_LINE:
        return None
    return format_horizon(config.data.snapshot_minutes)


def resolve_model_output_dir(config: ExperimentConfig) -> Path:
    out_dir = config.model_output_root / config.family.value
    horizon = _horizon_segment(config)
    if horizon is not None:
        out_dir = out_dir / horizon
    return out_dir / config.resolved_window_dir_label


def build_model_name(config: ExperimentConfig, *, as_of: date) -> str:
    """Build the model bundle name.

    ``as_of`` is required and must be the *training data's* latest date, not
    today's date -- this matches the repo's existing convention (see the
    example notebooks, which derive it from ``train_data["GAME_DATE"].max()``)
    and keeps the filename consistent with ``ModelBundleMetadata.model_version``,
    which is also derived from the training window. Defaulting to today's date
    would make a model retrained on an old snapshot look freshly trained.
    """
    date_str = as_of.strftime("%d_%m_%y")
    label = config.resolved_window_name_label
    horizon = _horizon_segment(config)
    if horizon is not None:
        label = f"{label}_{horizon}"
    return f"{label}_xgb_{config.family.value}_{date_str}"


def assert_model_bundle_is_writable(
    out_dir: Path,
    *,
    model_name: str,
    overwrite_existing_model: bool,
) -> None:
    """Refuse to silently clobber an existing model bundle.

    Bundle names are ``<window_label>_xgb_<target>_<DD_MM_YY>``, so two runs
    over the same training window and the same data end date collide. Without
    this guard ``save_model_bundle`` overwrites the previous bundle in place
    and the earlier model is unrecoverable.
    """
    if overwrite_existing_model:
        return

    existing = [
        path
        for path in (out_dir / f"{model_name}.json", out_dir / f"{model_name}.meta.json")
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            f"Model bundle already exists: {', '.join(str(p) for p in existing)}. "
            "Set overwrite_existing_model=True to replace it, or change "
            "window_name_label to keep both."
        )


def _to_datetime(value: pd.Timestamp | datetime) -> datetime:
    return pd.Timestamp(value).to_pydatetime()


def build_model_bundle_metadata(
    config: ExperimentConfig,
    *,
    model_name: str,
    best_params: dict,
    selected_trial_number: int | None,
    mean_best_iteration: int | None,
    median_best_iteration: int | None,
    train_games: int | None,
    cv_mae: float,
    cv_rmse: float | None,
    cv_ou_acc: float | None,
    final_test_mae: float,
    final_test_rmse: float,
    final_test_ou_acc: float,
    train_date_min: pd.Timestamp | datetime,
    train_date_max: pd.Timestamp | datetime,
) -> ModelBundleMetadata:
    sample_weight_lambda_bounds = None
    if config.target_family == TargetFamily.LINE_ERROR and config.sample_weight.enabled:
        sample_weight_lambda_bounds = config.sample_weight.lambda_bounds

    train_date_max_dt = _to_datetime(train_date_max)

    return ModelBundleMetadata(
        model_info=ModelInfo(
            name=model_name,
            model_version=train_date_max_dt.strftime("%d_%m_%y"),
            model_type=f"{config.resolved_window_name_label}_{config.family.value}",
            prediction_source=model_name,
            # ModelBundleMetadata's existing slot for "which training approach
            # produced this". Carry the user's label through when set so a
            # saved bundle can be traced back to it.
            training_code_tag=config.training_version or DEFAULT_TRAINING_CODE_TAG,
        ),
        training_metrics=TrainingMetrics(
            best_params=best_params,
            selected_trial_number=selected_trial_number,
            mean_best_iteration=mean_best_iteration,
            median_best_iteration=median_best_iteration,
            train_games=train_games,
            sample_weight_lambda_bounds=sample_weight_lambda_bounds,
            cv_mae=cv_mae,
            cv_rmse=cv_rmse,
            cv_ou_acc=cv_ou_acc,
            final_test_mae=final_test_mae,
            final_test_rmse=final_test_rmse,
            final_test_ou_acc=final_test_ou_acc,
            nan_threshold=config.cleaning.nan_threshold,
            max_na_per_row=config.cleaning.max_na_per_row,
            train_date_min=_to_datetime(train_date_min),
            train_date_max=train_date_max_dt,
        ),
    )
