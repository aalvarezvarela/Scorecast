from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from nba_ou.config.odds_columns import total_line_col
from nba_ou.config.settings import SETTINGS
from nba_ou.data_processing.missing_data.handle_missing_data import (
    apply_missing_policy,
)
from nba_ou.modeling.modeling import (
    ModelBundleMetadata,
    build_recency_sample_weights,
)
from nba_ou.utils.s3_models import (
    list_s3_objects,
    make_s3_client,
    read_s3_json_object,
    read_s3_object_bytes,
)
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from xgboost import XGBRegressor


@dataclass(frozen=True)
class ProductionArtifacts:
    """Loaded production model artifacts from S3."""

    bucket: str
    production_prefix: str
    model_key: str
    meta_key: str
    model: XGBRegressor
    raw_metadata: dict[str, Any]
    metadata: ModelBundleMetadata


@dataclass(frozen=True)
class RetrainingSettings:
    """All parameters required to retrain one concrete production model."""

    feature_names: list[str]
    target_column: str
    date_column: str
    required_line_col: str | None
    minimum_line_value: float | None
    nan_threshold: float
    max_na_per_row: int
    train_games: int | None
    xgb_params: dict[str, Any]
    sample_weight_lambda: float | None
    sample_weight_lambda_bounds: tuple[float, float] | None
    #: Only set by the legacy chain-refit path, which derived each day's
    #: settings from the previous bundle. A spec-driven refit has a fixed
    #: reference instead and leaves this None.
    source_metadata: ModelBundleMetadata | None = None


def _resolve_column_name(df: pd.DataFrame, desired_column: str) -> str | None:
    """Resolve a dataframe column with a case-insensitive fallback."""
    if desired_column in df.columns:
        return desired_column

    desired_lower = desired_column.lower()
    for column in df.columns:
        if column.lower() == desired_lower:
            return column
    return None


def _nested_value(data: dict[str, Any], *path: str) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _stable_unique_columns(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_columns: list[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        unique_columns.append(column)
    return unique_columns


def _coerce_feature_column_for_xgboost(series: pd.Series) -> pd.Series:
    """Convert object-like feature columns into XGBoost-compatible numeric values."""
    if is_numeric_dtype(series) or is_bool_dtype(series):
        return series

    non_null = series.dropna()
    if non_null.empty:
        return pd.to_numeric(series, errors="coerce")

    normalized = non_null.astype(str).str.strip().str.lower()
    bool_mapping = {
        "true": 1.0,
        "false": 0.0,
        "yes": 1.0,
        "no": 0.0,
        "y": 1.0,
        "n": 0.0,
        "t": 1.0,
        "f": 0.0,
        "0": 0.0,
        "1": 1.0,
    }
    if normalized.isin(bool_mapping).all():
        mapped = series.astype("string").str.strip().str.lower().map(bool_mapping)
        return pd.to_numeric(mapped, errors="coerce")

    numeric = pd.to_numeric(series, errors="coerce")
    if int(numeric.notna().sum()) == int(series.notna().sum()):
        return numeric

    return series


def _metadata_from_raw(raw_metadata: dict[str, Any]) -> ModelBundleMetadata:
    """Normalize S3 metadata keys into the local pydantic schema."""
    normalized_metadata = {
        "model_info": raw_metadata.get("model"),
        "schema_info": raw_metadata.get("schema"),
        "training_metrics": raw_metadata.get("training_metrics"),
        "created_at": raw_metadata.get("created_at"),
    }
    return ModelBundleMetadata.model_validate(normalized_metadata)


def _non_directory_objects(
    *, s3_client, bucket: str, prefix: str
) -> list[dict[str, Any]]:
    return [
        obj
        for obj in list_s3_objects(s3_client=s3_client, bucket=bucket, prefix=prefix)
        if obj["Key"] != prefix and not obj["Key"].endswith("/")
    ]


def _extract_single_bundle_keys(objects: list[dict[str, Any]]) -> tuple[str, str]:
    """Require exactly one model file and one metadata file in the prefix."""
    keys = sorted(str(obj["Key"]) for obj in objects)
    model_keys = [
        key for key in keys if key.endswith(".json") and not key.endswith(".meta.json")
    ]
    meta_keys = [key for key in keys if key.endswith(".meta.json")]

    if len(model_keys) != 1 or len(meta_keys) != 1:
        raise ValueError(
            "Expected exactly one production model file and one metadata file, "
            f"found model_keys={model_keys}, meta_keys={meta_keys}"
        )

    model_key = model_keys[0]
    meta_key = meta_keys[0]
    expected_meta_key = f"{model_key[: -len('.json')]}.meta.json"
    if meta_key != expected_meta_key:
        raise ValueError(
            "Production model and metadata filenames must match. "
            f"Expected metadata {expected_meta_key}, found {meta_key}."
        )

    return model_key, meta_key


def load_production_artifacts_from_s3(
    *,
    production_prefix: str,
    s3_client=None,
    bucket: str | None = None,
) -> ProductionArtifacts:
    """Load the current production model and metadata from S3."""
    if s3_client is None:
        s3_client = make_s3_client(
            profile=SETTINGS.s3_aws_profile,
            region=SETTINGS.s3_aws_region,
        )

    bucket_name = bucket or SETTINGS.s3_bucket
    objects = _non_directory_objects(
        s3_client=s3_client,
        bucket=bucket_name,
        prefix=production_prefix,
    )
    if not objects:
        raise FileNotFoundError(
            f"No production artifacts were found under S3 prefix {production_prefix!r}."
        )

    model_key, meta_key = _extract_single_bundle_keys(objects)

    raw_metadata = read_s3_json_object(
        s3_client=s3_client,
        bucket=bucket_name,
        key=meta_key,
    )
    metadata = _metadata_from_raw(raw_metadata)

    model = XGBRegressor()
    model.load_model(
        bytearray(
            read_s3_object_bytes(
                s3_client=s3_client,
                bucket=bucket_name,
                key=model_key,
            )
        )
    )

    return ProductionArtifacts(
        bucket=bucket_name,
        production_prefix=production_prefix,
        model_key=model_key,
        meta_key=meta_key,
        model=model,
        raw_metadata=raw_metadata,
        metadata=metadata,
    )


def _resolve_n_estimators(
    *,
    training_metrics,
    model: XGBRegressor,
) -> int:
    """Resolve the number of boosting rounds for retraining."""
    if training_metrics.median_best_iteration is not None:
        return int(training_metrics.median_best_iteration)
    if training_metrics.mean_best_iteration is not None:
        return int(training_metrics.mean_best_iteration)

    booster = model.get_booster()
    boosted_rounds = int(booster.num_boosted_rounds())
    if boosted_rounds <= 0:
        raise ValueError(
            "Could not infer n_estimators from the production metadata or model."
        )
    return boosted_rounds


def _infer_target_column_from_metadata(raw_metadata: dict[str, Any]) -> str:
    explicit_paths = (
        ("target",),
        ("target_col",),
        ("target_column",),
        ("model", "target"),
        ("model", "target_col"),
        ("model", "target_column"),
        ("training_metrics", "target"),
        ("training_metrics", "target_col"),
        ("training_metrics", "target_column"),
    )
    for path in explicit_paths:
        value = _nested_value(raw_metadata, *path)
        if isinstance(value, str) and value.strip():
            return value.strip()

    model_type = str(_nested_value(raw_metadata, "model", "model_type") or "")
    prediction_source = str(
        _nested_value(raw_metadata, "model", "prediction_source") or ""
    )
    model_name = str(_nested_value(raw_metadata, "model", "name") or "")
    signature = " ".join(
        [model_type.lower(), prediction_source.lower(), model_name.lower()]
    )

    if "total_points" in signature:
        return "TOTAL_POINTS"
    if (
        "line_error" in signature
        or "error_line" in signature
        or "diff_from_line" in signature
    ):
        return "LINE_ERROR"

    raise ValueError("Could not infer the target column from the production metadata.")


def _infer_required_line_col_from_feature_names(
    feature_names: list[str],
    *,
    target_column: str,
) -> str | None:
    if target_column not in {"TOTAL_POINTS", "LINE_ERROR"}:
        return None

    direct_total_line_features = [
        feature
        for feature in feature_names
        if feature.startswith("ODDS_TOTAL_LINE_")
        and "_LAST_" not in feature
        and "_SEASON_" not in feature
        and "_WMA" not in feature
        and "_MINUS_" not in feature
    ]
    unique_candidates = sorted(set(direct_total_line_features))
    if len(unique_candidates) == 1:
        return unique_candidates[0]

    preferred_candidate = total_line_col()
    if preferred_candidate in unique_candidates:
        return preferred_candidate

    if not unique_candidates:
        raise ValueError(
            "Could not infer any required total-line column from feature_names. "
            f"Candidates: {unique_candidates}"
        )
    raise ValueError(
        "Could not infer a unique required total-line column from feature_names "
        f"or match the configured main sportsbook column {preferred_candidate!r}. "
        f"Candidates: {unique_candidates}"
    )


def _coerce_sample_weight_lambda(value: Any) -> float | None:
    if value is None:
        return None
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    parsed_float = float(parsed)
    if parsed_float < 0:
        raise ValueError("sample_weight_lambda must be >= 0.")
    return parsed_float


def _coerce_sample_weight_lambda_bounds(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None

    low = pd.to_numeric(pd.Series([value[0]]), errors="coerce").iloc[0]
    high = pd.to_numeric(pd.Series([value[1]]), errors="coerce").iloc[0]
    if pd.isna(low) or pd.isna(high):
        return None

    low_float = float(low)
    high_float = float(high)
    if low_float <= 0 or high_float <= 0 or low_float >= high_float:
        return None
    return (low_float, high_float)


def _extract_sample_weight_hyperparameters(
    *,
    artifacts: ProductionArtifacts,
) -> tuple[float | None, tuple[float, float] | None]:
    training_metrics = artifacts.metadata.training_metrics
    best_params = dict(training_metrics.best_params) if training_metrics else {}

    sample_weight_lambda = _coerce_sample_weight_lambda(
        best_params.get("sample_weight_lambda")
    )

    raw_bounds = (
        _nested_value(
            artifacts.raw_metadata, "training_metrics", "sample_weight_lambda_bounds"
        )
        or _nested_value(artifacts.raw_metadata, "model", "sample_weight_lambda_bounds")
        or _nested_value(artifacts.raw_metadata, "sample_weight_lambda_bounds")
    )
    sample_weight_lambda_bounds = _coerce_sample_weight_lambda_bounds(raw_bounds)

    return sample_weight_lambda, sample_weight_lambda_bounds


def build_retraining_settings(
    *,
    artifacts: ProductionArtifacts,
    target_column: str,
    date_column: str,
    nan_threshold: float,
    max_na_per_row: int,
    train_games: int | None,
    required_line_col: str | None,
    minimum_line_value: float | None,
    feature_names: list[str],
    xgb_static_params: dict[str, Any],
) -> RetrainingSettings:
    """Build retraining settings from production artifacts and caller-supplied variables."""
    metadata = artifacts.metadata
    training_metrics = metadata.training_metrics

    if training_metrics is None:
        raise ValueError("Production metadata is missing training_metrics.")
    resolved_feature_names = [str(feature) for feature in feature_names]
    if not resolved_feature_names:
        raise ValueError("feature_names must not be empty.")
    if not xgb_static_params:
        raise ValueError("xgb_static_params must not be empty.")

    n_estimators = _resolve_n_estimators(
        training_metrics=training_metrics,
        model=artifacts.model,
    )
    sample_weight_lambda, sample_weight_lambda_bounds = (
        _extract_sample_weight_hyperparameters(artifacts=artifacts)
    )
    _NON_XGB_PARAM_KEYS = {"sample_weight_lambda"}

    best_params = {
        k: v
        for k, v in (training_metrics.best_params or {}).items()
        if k not in _NON_XGB_PARAM_KEYS
    }

    xgb_params = {
        **xgb_static_params,
        **best_params,
        "n_estimators": n_estimators,
    }

    return RetrainingSettings(
        feature_names=resolved_feature_names,
        target_column=target_column,
        date_column=date_column,
        required_line_col=required_line_col,
        minimum_line_value=minimum_line_value,
        nan_threshold=float(nan_threshold),
        max_na_per_row=int(max_na_per_row),
        train_games=None if train_games is None else int(train_games),
        xgb_params=xgb_params,
        sample_weight_lambda=sample_weight_lambda,
        sample_weight_lambda_bounds=sample_weight_lambda_bounds,
        source_metadata=metadata,
    )


def build_retraining_settings_from_artifacts(
    *,
    artifacts: ProductionArtifacts,
    date_column: str,
    minimum_line_value: float | None,
    xgb_static_params: dict[str, Any],
) -> RetrainingSettings:
    """Build retraining settings by reading the saved production metadata."""
    metadata = artifacts.metadata
    training_metrics = metadata.training_metrics
    schema_info = metadata.schema_info

    if training_metrics is None:
        raise ValueError("Production metadata is missing training_metrics.")
    if schema_info is None or not schema_info.feature_names:
        raise ValueError("Production metadata is missing schema.feature_names.")
    if training_metrics.train_games is None or int(training_metrics.train_games) <= 0:
        raise ValueError("Production metadata is missing a positive train_games value.")

    feature_names = [str(feature) for feature in schema_info.feature_names]
    target_column = _infer_target_column_from_metadata(artifacts.raw_metadata)
    required_line_col = _infer_required_line_col_from_feature_names(
        feature_names,
        target_column=target_column,
    )

    return build_retraining_settings(
        artifacts=artifacts,
        target_column=target_column,
        date_column=date_column,
        nan_threshold=float(training_metrics.nan_threshold),
        max_na_per_row=int(training_metrics.max_na_per_row),
        train_games=int(training_metrics.train_games),
        required_line_col=required_line_col,
        minimum_line_value=minimum_line_value,
        feature_names=feature_names,
        xgb_static_params=xgb_static_params,
    )


def _needs_derived_line_error(
    *,
    target_column: str,
    required_line_col: str | None,
    raw_df: pd.DataFrame,
) -> bool:
    """Return True when LINE_ERROR must be computed from TOTAL_POINTS and the line column."""
    return (
        target_column == "LINE_ERROR"
        and required_line_col is not None
        and _resolve_column_name(raw_df, "LINE_ERROR") is None
        and _resolve_column_name(raw_df, "TOTAL_POINTS") is not None
        and _resolve_column_name(raw_df, required_line_col) is not None
    )


def prepare_retraining_dataframe_from_raw(
    raw_df: pd.DataFrame,
    *,
    settings: RetrainingSettings,
    passthrough_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Prepare retraining rows from the raw dataframe using the selected required columns."""
    if settings.date_column not in raw_df.columns:
        raise KeyError(f"Raw training dataframe must include {settings.date_column!r}.")

    derive_line_error = _needs_derived_line_error(
        target_column=settings.target_column,
        required_line_col=settings.required_line_col,
        raw_df=raw_df,
    )

    required_columns = [settings.date_column, settings.target_column]
    if derive_line_error:
        required_columns.append("TOTAL_POINTS")
    if settings.required_line_col:
        required_columns.append(settings.required_line_col)
    if passthrough_columns:
        required_columns.extend(str(column) for column in passthrough_columns)
    required_columns.extend(settings.feature_names)
    required_columns = _stable_unique_columns(required_columns)

    resolved_columns: dict[str, str] = {}
    missing_columns: list[str] = []

    for column in required_columns:
        resolved = _resolve_column_name(raw_df, column)
        if resolved is None:
            missing_columns.append(column)
        else:
            resolved_columns[column] = resolved

    if derive_line_error:
        missing_columns = [
            col for col in missing_columns if col != settings.target_column
        ]

    if missing_columns:
        raise KeyError(
            "Raw training dataframe is missing required production columns: "
            f"{missing_columns}"
        )

    columns_to_select = [col for col in required_columns if col in resolved_columns]
    prepared = raw_df[[resolved_columns[column] for column in columns_to_select]].copy()
    prepared.columns = columns_to_select
    prepared[settings.date_column] = pd.to_datetime(
        prepared[settings.date_column],
        errors="coerce",
    ).dt.normalize()

    if derive_line_error:
        prepared["TOTAL_POINTS"] = pd.to_numeric(
            prepared["TOTAL_POINTS"], errors="coerce"
        )
        prepared[settings.required_line_col] = pd.to_numeric(
            prepared[settings.required_line_col], errors="coerce"
        )
        prepared[settings.target_column] = (
            prepared["TOTAL_POINTS"] - prepared[settings.required_line_col]
        )
    else:
        prepared[settings.target_column] = pd.to_numeric(
            prepared[settings.target_column],
            errors="coerce",
        )

    if settings.required_line_col is not None:
        prepared[settings.required_line_col] = pd.to_numeric(
            prepared[settings.required_line_col],
            errors="coerce",
        )

    prepared = prepared.dropna(
        subset=[settings.date_column, settings.target_column]
    ).copy()

    prepared = apply_missing_policy(
        prepared,
        current_total_line_col=settings.required_line_col,
        mode="train",
        create_missing_flags=False,
        keep_all_cols=False,
    )

    if (
        settings.required_line_col is not None
        and settings.minimum_line_value is not None
    ):
        prepared = prepared.loc[
            prepared[settings.required_line_col] > settings.minimum_line_value
        ].copy()

    if settings.max_na_per_row >= 0:
        na_per_row = prepared.isna().sum(axis=1)
        prepared = prepared.loc[na_per_row <= settings.max_na_per_row].copy()

    for feature_name in settings.feature_names:
        prepared[feature_name] = _coerce_feature_column_for_xgboost(
            prepared[feature_name]
        )

    invalid_feature_columns = [
        feature_name
        for feature_name in settings.feature_names
        if not (
            is_numeric_dtype(prepared[feature_name])
            or is_bool_dtype(prepared[feature_name])
        )
    ]
    if invalid_feature_columns:
        raise ValueError(
            "Prepared retraining dataframe contains non-numeric feature columns "
            f"that XGBoost cannot consume: {invalid_feature_columns}"
        )

    prepared["_row_order"] = range(len(prepared))
    prepared = prepared.sort_values(
        [settings.date_column, "_row_order"],
        kind="mergesort",
    ).drop(columns="_row_order")
    prepared = prepared.reset_index(drop=True)

    if prepared.empty:
        raise ValueError("No rows remain after retraining filtering.")

    return prepared


def resolve_train_games_to_use(
    *,
    settings: RetrainingSettings,
) -> int | None:
    """Resolve the training window size from the caller-provided settings."""
    return None if settings.train_games is None else int(settings.train_games)


def select_rolling_training_window(
    prepared_df: pd.DataFrame,
    *,
    date_col: str,
    train_games: int | None,
) -> pd.DataFrame:
    """Select either the full history or the latest rolling training window."""
    sorted_df = prepared_df.sort_values(date_col, kind="mergesort").reset_index(
        drop=True
    )

    if train_games is None:
        return sorted_df
    if train_games <= 0:
        raise ValueError("train_games must be greater than zero.")
    if train_games >= len(sorted_df):
        return sorted_df.copy()

    return sorted_df.tail(train_games).reset_index(drop=True)


def retrain_model(
    training_window_df: pd.DataFrame,
    *,
    settings: RetrainingSettings,
) -> XGBRegressor:
    """Refit the production regressor on the selected training window."""
    X_train = training_window_df[settings.feature_names].copy()
    y_train = pd.to_numeric(
        training_window_df[settings.target_column],
        errors="coerce",
    )

    model = XGBRegressor(**settings.xgb_params)
    fit_kwargs: dict[str, Any] = {"verbose": False}

    if settings.sample_weight_lambda is not None:
        fit_kwargs["sample_weight"] = build_recency_sample_weights(
            training_window_df[[settings.date_column]],
            date_col=settings.date_column,
            lambda_=settings.sample_weight_lambda,
        ).to_numpy(dtype=float)

    model.fit(X_train, y_train, **fit_kwargs)
    return model

# ---------------------------------------------------------------------------
# NOTE ON WHAT IS NO LONGER HERE
#
# The functions that wrote a refit into S3 -- save_retrained_bundle_to_staging,
# build_updated_retraining_metadata and their helpers -- are gone. They served
# the previous layout, in which each day's model was derived from the previous
# day's bundle and copied between production/, staging/ and archive/. A refit
# now reads a fixed spec (nba_ou.modeling.refit) and writes an immutable build
# (nba_ou.modeling.registry_store).
#
# The loaders above (load_production_artifacts_from_s3 and friends) still read
# the OLD layout and are kept only for nba_ou.modeling.meta_learner_training_data,
# which has not been migrated. They stop working once the old tree is swept into
# models/retired/.
# ---------------------------------------------------------------------------
