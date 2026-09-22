from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from nba_ou.config.odds_columns import total_line_col
from nba_ou.config.settings import SETTINGS
from nba_ou.modeling.modeling import evaluate_day_by_day_walk_forward
from nba_ou.modeling.retraining_utils import (
    ProductionArtifacts,
    build_retraining_settings_from_artifacts,
    load_production_artifacts_from_s3,
    prepare_retraining_dataframe_from_raw,
    resolve_train_games_to_use,
    retrain_model,
)
from nba_ou.utils.s3_models import make_s3_client
from tqdm.auto import tqdm

DATE_COLUMN = "GAME_DATE"
SEASON_COLUMN = "SEASON_YEAR"
DEFAULT_LAST_N_SEASONS = 2
DEFAULT_TOP_N_FEATURES_PER_MODEL = 50
DEFAULT_PASSTHROUGH_COLUMNS = [
    "GAME_ID",
    DATE_COLUMN,
    SEASON_COLUMN,
    "TEAM_ID_TEAM_HOME",
    "TEAM_ID_TEAM_AWAY",
    "TOTAL_POINTS",
    "LINE_ERROR",
    total_line_col(),
]
XGB_STATIC_PARAMS = {
    "booster": "gbtree",
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "random_state": 16,
    "n_jobs": -1,
    "verbosity": 0,
}


@dataclass(frozen=True)
class MetaLearnerModelSummary:
    production_prefix: str
    model_name: str
    target_column: str
    prediction_column: str
    train_games: int | None
    top_features: list[str]


@dataclass(frozen=True)
class MetaLearnerTrainingDataResult:
    dataframe: pd.DataFrame
    selected_feature_names: list[str]
    prediction_columns: list[str]
    prediction_seasons: list[Any]
    model_summaries: list[MetaLearnerModelSummary]
    output_path: Path | None = None


def _stable_unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def load_training_dataframe(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    header_cols = pd.read_csv(csv_path, nrows=0).columns
    dtype_dict = {col: str for col in header_cols if "ID" in col.upper()}
    df = pd.read_csv(csv_path, dtype=dtype_dict)
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(
            df[DATE_COLUMN], errors="coerce"
        ).dt.normalize()
    return df


def _season_sort_key(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("Season values must not be missing.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)

    text = str(value).strip()
    if not text:
        raise ValueError("Season values must not be empty.")
    if "-" in text:
        text = text.split("-", 1)[0]
    return int(text)


def select_latest_seasons(
    df: pd.DataFrame,
    *,
    season_col: str = SEASON_COLUMN,
    n_seasons: int = DEFAULT_LAST_N_SEASONS,
) -> list[Any]:
    if n_seasons <= 0:
        raise ValueError("n_seasons must be greater than zero.")
    if season_col not in df.columns:
        raise KeyError(f"Missing required season column: {season_col}")

    unique_values = [value for value in df[season_col].dropna().unique().tolist()]
    if not unique_values:
        raise ValueError(f"No season values were found in column {season_col!r}.")

    ordered = sorted(unique_values, key=_season_sort_key)
    return ordered[-n_seasons:]


def derive_prediction_column_name(production_prefix: str) -> str:
    parts = [part for part in production_prefix.strip("/").split("/") if part]
    if len(parts) < 2:
        raise ValueError(f"Unexpected production prefix: {production_prefix}")
    family = parts[-2]
    return f"PRED_{family.upper()}"


def _resolve_feature_importance_pairs(
    artifacts: ProductionArtifacts,
) -> list[tuple[str, float]]:
    schema_info = artifacts.metadata.schema_info
    if schema_info is None or not schema_info.feature_names:
        raise ValueError(
            f"Production metadata for {artifacts.production_prefix!r} is missing feature_names."
        )

    feature_names = [str(name) for name in schema_info.feature_names]
    model = artifacts.model

    feature_importances = getattr(model, "feature_importances_", None)
    if feature_importances is not None and len(feature_importances) == len(
        feature_names
    ):
        return list(
            zip(
                feature_names,
                np.asarray(feature_importances, dtype=float).tolist(),
                strict=True,
            )
        )

    booster = model.get_booster()
    gain_scores = booster.get_score(importance_type="gain")
    if not gain_scores:
        raise ValueError(
            f"Could not read feature importances for {artifacts.production_prefix!r}."
        )

    resolved_scores: list[tuple[str, float]] = []
    for idx, feature_name in enumerate(feature_names):
        score = gain_scores.get(feature_name)
        if score is None:
            score = gain_scores.get(f"f{idx}", 0.0)
        resolved_scores.append((feature_name, float(score)))
    return resolved_scores


def select_top_features_for_model(
    artifacts: ProductionArtifacts,
    *,
    top_n: int = DEFAULT_TOP_N_FEATURES_PER_MODEL,
) -> list[str]:
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")

    importance_pairs = _resolve_feature_importance_pairs(artifacts)
    importance_pairs.sort(key=lambda item: (-item[1], item[0]))
    return [feature_name for feature_name, _ in importance_pairs[:top_n]]


def build_feature_union_from_artifacts(
    artifacts_list: list[ProductionArtifacts],
    *,
    top_n_per_model: int = DEFAULT_TOP_N_FEATURES_PER_MODEL,
) -> tuple[list[str], dict[str, list[str]]]:
    per_model_features: dict[str, list[str]] = {}
    combined: list[str] = []

    for artifacts in artifacts_list:
        top_features = select_top_features_for_model(
            artifacts,
            top_n=top_n_per_model,
        )
        per_model_features[artifacts.production_prefix] = top_features
        combined.extend(top_features)

    return _stable_unique_strings(combined), per_model_features


def _ensure_line_error_column(df: pd.DataFrame) -> pd.DataFrame:
    if "LINE_ERROR" in df.columns:
        return df
    required_line_col = total_line_col()
    if "TOTAL_POINTS" not in df.columns or required_line_col not in df.columns:
        return df

    output = df.copy()
    output["LINE_ERROR"] = pd.to_numeric(
        output["TOTAL_POINTS"], errors="coerce"
    ) - pd.to_numeric(
        output[required_line_col],
        errors="coerce",
    )
    return output


def resolve_merge_keys(df: pd.DataFrame) -> list[str]:
    if "GAME_ID" in df.columns:
        return ["GAME_ID"]

    composite_keys = [DATE_COLUMN, "TEAM_ID_TEAM_HOME", "TEAM_ID_TEAM_AWAY"]
    if all(column in df.columns for column in composite_keys):
        return composite_keys

    raise KeyError(
        "Could not determine merge keys. Expected either ['GAME_ID'] or "
        f"{composite_keys} in the training dataframe."
    )


def build_meta_learner_base_frame(
    raw_df: pd.DataFrame,
    *,
    prediction_seasons: list[Any],
    selected_feature_names: list[str],
    merge_keys: list[str],
    passthrough_columns: list[str],
    season_col: str = SEASON_COLUMN,
) -> pd.DataFrame:
    if season_col not in raw_df.columns:
        raise KeyError(f"Missing required season column: {season_col}")

    base_df = _ensure_line_error_column(raw_df)
    mask = base_df[season_col].isin(prediction_seasons)
    filtered = base_df.loc[mask].copy()
    if filtered.empty:
        raise ValueError("No rows were found for the requested prediction seasons.")

    required_columns = _stable_unique_strings(
        merge_keys + passthrough_columns + selected_feature_names
    )
    missing_columns = [
        column for column in required_columns if column not in filtered.columns
    ]
    if missing_columns:
        raise KeyError(
            f"The raw dataframe is missing required output columns: {missing_columns}"
        )

    base = filtered[required_columns].copy()
    base = base.sort_values([DATE_COLUMN, *merge_keys], kind="mergesort").reset_index(
        drop=True
    )

    if base.duplicated(subset=merge_keys).any():
        duplicated = base.loc[
            base.duplicated(subset=merge_keys, keep=False), merge_keys
        ]
        raise ValueError(
            "Merge keys are not unique in the output base dataframe. "
            f"Duplicated keys sample: {duplicated.head(5).to_dict(orient='records')}"
        )

    return base


def generate_walk_forward_predictions_for_model(
    raw_df: pd.DataFrame,
    *,
    artifacts: ProductionArtifacts,
    prediction_seasons: list[Any],
    merge_keys: list[str],
    top_features: list[str],
    date_column: str = DATE_COLUMN,
    season_col: str = SEASON_COLUMN,
    minimum_line_value: float | None = 100.0,
    xgb_static_params: dict[str, Any] | None = None,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, MetaLearnerModelSummary]:
    """
    Generate day-by-day walk-forward predictions for a single production model.

    For each day in the prediction seasons, this trains the model on all data
    up to the day before (using the model's original features and hyperparameters)
    and predicts for games on that day. This prevents data leakage.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Complete training dataframe with all features and seasons.
    artifacts : ProductionArtifacts
        Loaded production model and metadata from S3.
    prediction_seasons : list
        Seasons to generate predictions for (typically last 2 seasons).
    merge_keys : list[str]
        Columns used to merge predictions back (e.g., ['GAME_ID']).
    top_features : list[str]
        Top N most important features for this model (for documentation only;
        training uses the model's original feature set from metadata).
    date_column : str
        Name of the date column.
    season_col : str
        Name of the season column.
    minimum_line_value : float | None
        Minimum total line value to include in training.
    xgb_static_params : dict | None
        Static XGBoost parameters for training.
    show_progress : bool
        If True, show a tqdm bar for the day-by-day walk-forward loop.

    Returns
    -------
    tuple[pd.DataFrame, MetaLearnerModelSummary]
        Predictions dataframe (merge_keys + prediction column) and model summary.
    """
    xgb_static_params = xgb_static_params or XGB_STATIC_PARAMS
    settings = build_retraining_settings_from_artifacts(
        artifacts=artifacts,
        date_column=date_column,
        minimum_line_value=minimum_line_value,
        xgb_static_params=xgb_static_params,
    )

    passthrough_columns = _stable_unique_strings(merge_keys + [season_col])
    prepared_df = prepare_retraining_dataframe_from_raw(
        raw_df,
        settings=settings,
        passthrough_columns=passthrough_columns,
    )

    prediction_mask = prepared_df[season_col].isin(prediction_seasons)
    df_dev = prepared_df.loc[~prediction_mask].copy()
    df_test_final = prepared_df.loc[prediction_mask].copy().reset_index(drop=True)

    if df_test_final.empty:
        raise ValueError(
            f"No prediction rows remain after preprocessing for {artifacts.production_prefix!r}."
        )
    if df_dev.empty:
        raise ValueError(
            f"No pre-history rows are available to generate walk-forward predictions for {artifacts.production_prefix!r}."
        )

    def fit_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        model = retrain_model(train_df, settings=settings)
        return np.asarray(model.predict(test_df[settings.feature_names]), dtype=float)

    evaluation = evaluate_day_by_day_walk_forward(
        df_dev=df_dev,
        df_test_final=df_test_final,
        fit_and_predict=fit_and_predict,
        metric_fn=lambda y_true, y_pred: 0.0,
        target_col=settings.target_column,
        date_col=date_column,
        max_games=resolve_train_games_to_use(settings=settings),
        metric_name="placeholder_metric",
        show_progress=show_progress,
        progress_desc=f"{artifacts.metadata.model_info.name} days",
    )

    prediction_column = derive_prediction_column_name(artifacts.production_prefix)
    predictions = df_test_final[merge_keys].copy()
    predictions[prediction_column] = np.nan
    row_positions = evaluation.predictions["row_in_test_final"].to_numpy(dtype=int)
    predictions.loc[row_positions, prediction_column] = evaluation.predictions[
        "y_pred"
    ].to_numpy(dtype=float)

    if predictions[prediction_column].isna().any():
        raise ValueError(
            f"Missing walk-forward predictions were produced for {artifacts.production_prefix!r}."
        )

    summary = MetaLearnerModelSummary(
        production_prefix=artifacts.production_prefix,
        model_name=artifacts.metadata.model_info.name,
        target_column=settings.target_column,
        prediction_column=prediction_column,
        train_games=resolve_train_games_to_use(settings=settings),
        top_features=top_features,
    )
    return predictions, summary


def build_meta_learner_training_data(
    raw_df: pd.DataFrame,
    *,
    production_prefixes: list[str] | None = None,
    last_n_seasons: int = DEFAULT_LAST_N_SEASONS,
    top_n_features_per_model: int = DEFAULT_TOP_N_FEATURES_PER_MODEL,
    passthrough_columns: list[str] | None = None,
    drop_rows_missing_any_prediction: bool = True,
    s3_client=None,
    bucket: str | None = None,
    output_path: str | Path | None = None,
    show_progress: bool = True,
) -> MetaLearnerTrainingDataResult:
    """
    Build meta-learner training data from production models.

    This function:
    1. Loads specified production models from S3 (default: 6 models)
    2. Extracts top N most important features from each model
    3. Combines features into a deduplicated union (max N*M features)
    4. Generates day-by-day walk-forward predictions for the last N seasons
    5. Creates a CSV with selected features + predictions from all models

    Parameters
    ----------
    raw_df : pd.DataFrame
        Complete training dataframe with all historical data.
    production_prefixes : list[str] | None
        S3 prefixes for production models. Defaults to configured prefixes
        in settings (typically 6 models: 3 total_points + 3 line_error).
    last_n_seasons : int
        Number of most recent seasons to generate predictions for (default: 2).
    top_n_features_per_model : int
        Number of top features to extract per model (default: 50).
    passthrough_columns : list[str] | None
        Additional columns to include in output (e.g., GAME_ID, TOTAL_POINTS).
    drop_rows_missing_any_prediction : bool
        If True, drop rows where any model prediction is missing.
    s3_client : optional
        Boto3 S3 client. Created automatically if None.
    bucket : str | None
        S3 bucket name. Defaults to configured bucket.
    output_path : str | Path | None
        If provided, save the resulting dataframe to this CSV path.
    show_progress : bool
        If True, display tqdm bars for model-level and day-level progress.

    Returns
    -------
    MetaLearnerTrainingDataResult
        Contains the dataframe, feature names, prediction columns, and model summaries.
    """
    if not production_prefixes:
        # No default from settings any more. This module reads the PRE-REGISTRY
        # layout (one model .json plus one .meta.json directly under a
        # production/ prefix), which the slot layout replaced and which
        # scripts/retire_legacy_models.py sweeps into models/retired/. Falling
        # back to the configured slots would hand it paths it cannot parse, so
        # the caller has to say which old prefixes it means -- and once the
        # sweep has run, those live under models/retired/<date>/.
        raise ValueError(
            "production_prefixes is required. The meta-learner still reads the "
            "pre-registry model layout and has not been migrated to the slot "
            "registry (nba_ou.modeling.registry_store), so it cannot use "
            "[PredictionModels] ENABLED_MODELS. Pass the old-style prefixes "
            "explicitly, e.g. "
            "'models/retired/<YYYYMMDD>/line_error_full_dataset/production/'."
        )

    working_df = raw_df.copy()
    if DATE_COLUMN not in working_df.columns:
        raise KeyError(f"Missing required date column: {DATE_COLUMN}")
    working_df[DATE_COLUMN] = pd.to_datetime(
        working_df[DATE_COLUMN],
        errors="coerce",
    ).dt.normalize()

    if s3_client is None:
        s3_client = make_s3_client(
            profile=SETTINGS.s3_aws_profile,
            region=SETTINGS.s3_aws_region,
        )

    resolved_bucket = bucket or SETTINGS.s3_bucket
    print(f"Loading {len(production_prefixes)} production model bundles from S3...")
    artifacts_list = [
        load_production_artifacts_from_s3(
            production_prefix=production_prefix,
            s3_client=s3_client,
            bucket=resolved_bucket,
        )
        for production_prefix in tqdm(
            production_prefixes,
            desc="Loading models",
            unit="model",
            disable=not show_progress,
        )
    ]

    selected_feature_names, top_features_by_model = build_feature_union_from_artifacts(
        artifacts_list,
        top_n_per_model=top_n_features_per_model,
    )
    prediction_seasons = select_latest_seasons(
        working_df,
        season_col=SEASON_COLUMN,
        n_seasons=last_n_seasons,
    )
    passthrough_columns = _stable_unique_strings(
        list(passthrough_columns or DEFAULT_PASSTHROUGH_COLUMNS)
    )
    merge_keys = resolve_merge_keys(working_df)

    final_df = build_meta_learner_base_frame(
        working_df,
        prediction_seasons=prediction_seasons,
        selected_feature_names=selected_feature_names,
        merge_keys=merge_keys,
        passthrough_columns=passthrough_columns,
    )

    prediction_columns: list[str] = []
    model_summaries: list[MetaLearnerModelSummary] = []

    model_iterator = tqdm(
        artifacts_list,
        desc="Generating model predictions",
        unit="model",
        disable=not show_progress,
    )

    for idx, artifacts in enumerate(model_iterator, 1):
        model_name = artifacts.metadata.model_info.name
        if show_progress:
            model_iterator.set_postfix_str(f"{idx}/{len(artifacts_list)} {model_name}")
        else:
            print(
                f"[{idx}/{len(artifacts_list)}] Generating walk-forward predictions for {model_name}..."
            )

        predictions_df, summary = generate_walk_forward_predictions_for_model(
            working_df,
            artifacts=artifacts,
            prediction_seasons=prediction_seasons,
            merge_keys=merge_keys,
            top_features=top_features_by_model[artifacts.production_prefix],
            show_progress=show_progress,
        )
        final_df = final_df.merge(
            predictions_df,
            on=merge_keys,
            how="left",
            validate="one_to_one",
        )
        prediction_columns.append(summary.prediction_column)
        model_summaries.append(summary)
        if not show_progress:
            print(f"    Added column: {summary.prediction_column}")

    if drop_rows_missing_any_prediction and prediction_columns:
        final_df = final_df.dropna(subset=prediction_columns).reset_index(drop=True)

    saved_output_path: Path | None = None
    if output_path is not None:
        saved_output_path = Path(output_path)
        saved_output_path.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(saved_output_path, index=False)

    return MetaLearnerTrainingDataResult(
        dataframe=final_df,
        selected_feature_names=selected_feature_names,
        prediction_columns=prediction_columns,
        prediction_seasons=prediction_seasons,
        model_summaries=model_summaries,
        output_path=saved_output_path,
    )


def build_meta_learner_training_data_from_csv(
    csv_path: str | Path,
    *,
    production_prefixes: list[str] | None = None,
    last_n_seasons: int = DEFAULT_LAST_N_SEASONS,
    top_n_features_per_model: int = DEFAULT_TOP_N_FEATURES_PER_MODEL,
    passthrough_columns: list[str] | None = None,
    drop_rows_missing_any_prediction: bool = True,
    s3_client=None,
    bucket: str | None = None,
    output_path: str | Path | None = None,
    show_progress: bool = True,
) -> MetaLearnerTrainingDataResult:
    raw_df = load_training_dataframe(csv_path)
    return build_meta_learner_training_data(
        raw_df,
        production_prefixes=production_prefixes,
        last_n_seasons=last_n_seasons,
        top_n_features_per_model=top_n_features_per_model,
        passthrough_columns=passthrough_columns,
        drop_rows_missing_any_prediction=drop_rows_missing_any_prediction,
        s3_client=s3_client,
        bucket=bucket,
        output_path=output_path,
        show_progress=show_progress,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a meta-learner training CSV from the six production models."
    )
    parser.add_argument("csv_path", help="Path to the full training dataframe CSV.")
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path where the meta-learner CSV will be written.",
    )
    parser.add_argument(
        "--last-n-seasons",
        type=int,
        default=DEFAULT_LAST_N_SEASONS,
        help="Number of latest seasons to generate walk-forward predictions for.",
    )
    parser.add_argument(
        "--top-n-features-per-model",
        type=int,
        default=DEFAULT_TOP_N_FEATURES_PER_MODEL,
        help="Number of top-importance features to keep from each production model.",
    )
    parser.add_argument(
        "--keep-missing-predictions",
        action="store_true",
        help="Keep rows even when one or more model predictions are missing after merging.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = build_meta_learner_training_data_from_csv(
        args.csv_path,
        last_n_seasons=args.last_n_seasons,
        top_n_features_per_model=args.top_n_features_per_model,
        drop_rows_missing_any_prediction=not args.keep_missing_predictions,
        output_path=args.output_path,
    )

    print(f"Prediction seasons: {result.prediction_seasons}")
    print(f"Selected feature count: {len(result.selected_feature_names)}")
    print(f"Prediction columns: {result.prediction_columns}")
    print(f"Output rows: {len(result.dataframe)}")
    if result.output_path is not None:
        print(f"Saved CSV: {result.output_path}")


if __name__ == "__main__":
    main()
