import os
import re
from datetime import date, datetime
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from nba_ou.config.odds_columns import total_line_over_col_raw
from nba_ou.config.settings import SETTINGS
from nba_ou.data_processing.missing_data.clean_df_for_training import (
    clean_dataframe_for_training,
)
from nba_ou.postgre_db.predictions.create.create_ou_predictions_db import (
    upload_predictions_to_postgre,
)
from nba_ou.utils.general_utils import get_season_year_from_date
from nba_ou.utils.s3_models import (
    list_s3_objects,
    load_parquet_from_bytes,
    make_s3_client,
    read_s3_object_bytes,
)

PredictionTarget = Literal["PRED_LINE_ERROR", "TOTAL_POINTS"]

PREDICTION_TARGET_LINE_ERROR: PredictionTarget = "PRED_LINE_ERROR"
PREDICTION_TARGET_TOTAL_POINTS: PredictionTarget = "TOTAL_POINTS"
PREDICTION_VALUE_TYPE_TOTAL_POINTS = "TOTAL_POINTS"
# Keep the stored/database value as DIFF_FROM_LINE for backward compatibility,
# but use "line_error" as the canonical name in code and model metadata.
PREDICTION_VALUE_TYPE_LINE_ERROR = "DIFF_FROM_LINE"
PREDICTION_VALUE_TYPE_DIFF_FROM_LINE = PREDICTION_VALUE_TYPE_LINE_ERROR


def _resolve_column_name(df: pd.DataFrame, desired_column: str) -> str | None:
    """Resolve column name with case-insensitive fallback."""
    if desired_column in df.columns:
        return desired_column

    desired_lower = desired_column.lower()
    for col in df.columns:
        if col.lower() == desired_lower:
            return col
    return None


def _select_latest_historical_train_key(s3_client, bucket: str, prefix: str) -> str:
    """Select the latest historical parquet by YYYYMMDD suffix in key name."""
    objects = list_s3_objects(s3_client=s3_client, bucket=bucket, prefix=prefix)
    parquet_objects = [
        obj for obj in objects if obj.get("Key", "").endswith(".parquet")
    ]
    if not parquet_objects:
        raise FileNotFoundError(f"No parquet files found in s3://{bucket}/{prefix}")

    pattern = re.compile(r"historical_training_data_until_(\d{8})\.parquet$")

    with_date: list[tuple[pd.Timestamp, str]] = []
    without_date: list[dict] = []
    for obj in parquet_objects:
        key = obj.get("Key", "")
        match = pattern.search(key)
        if match:
            parsed = pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")
            if pd.notna(parsed):
                with_date.append((parsed, key))
                continue
        without_date.append(obj)

    if with_date:
        with_date.sort(key=lambda x: x[0], reverse=True)
        return with_date[0][1]

    without_date.sort(key=lambda x: x.get("LastModified"), reverse=True)
    return without_date[0]["Key"]


def _init_tabpfn_client() -> type:
    """Initialize tabpfn client auth and return TabPFNRegressor class."""
    try:
        import tabpfn_client
    except ImportError as exc:
        raise ImportError(
            "tabpfn_client is not installed. Install it with: pip install tabpfn-client"
        ) from exc

    token = os.getenv("TABPFN_ACCESS_TOKEN")

    # If we have a token, set it directly
    if token:
        set_token_fn = getattr(tabpfn_client, "set_access_token", None)
        if callable(set_token_fn):
            set_token_fn(token)
        else:
            raise RuntimeError(
                "tabpfn_client.set_access_token not found. "
                "Ensure tabpfn-client is properly installed."
            )
    else:
        # No token provided - try init() for interactive auth
        # This will fail in non-interactive environments (e.g., GitHub Actions)
        init_fn = getattr(tabpfn_client, "init", None)
        if callable(init_fn):
            init_fn()
        else:
            raise RuntimeError(
                "TABPFN_ACCESS_TOKEN environment variable not set and "
                "tabpfn_client.init() not available for interactive auth."
            )

    return tabpfn_client.TabPFNRegressor


def _keep_columns_for_tabpfn(total_points_pick_line_col: str) -> list[str]:
    return [
        "GAME_ID",
        "SEASON_TYPE",
        "GAME_DATE",
        "GAME_TIME",
        "TEAM_NAME_TEAM_HOME",
        "TEAM_NAME_TEAM_AWAY",
        "MATCHUP_TEAM_HOME",
        "TOTAL_POINTS",
        "DIFF_FROM_LINE",
        total_points_pick_line_col,
    ]


def _line_error_target_column_name(total_points_pick_line_col: str) -> str:
    return f"DIFF_FROM_{total_points_pick_line_col.replace('TOTAL_', '')}"


def _prepare_tabpfn_client_datasets(
    df: pd.DataFrame,
    prediction_date: str | datetime | pd.Timestamp,
    *,
    historical_train_s3_key: str | None = None,
    historical_train_prefix: str = "train_data/",
    total_points_pick_line_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, date, str]:
    """Load, merge, and clean the shared train/predict frames for TabPFN."""
    if df.empty:
        raise ValueError("Input dataframe is empty")

    if "GAME_ID" not in df.columns:
        raise ValueError("Input dataframe must include GAME_ID")

    if total_points_pick_line_col is None:
        total_points_pick_line_col = total_line_over_col_raw()

    incoming_df = df.copy()
    incoming_df["GAME_ID"] = incoming_df["GAME_ID"].astype(str)

    prediction_day = pd.to_datetime(prediction_date, errors="coerce")
    if pd.isna(prediction_day):
        raise ValueError(f"Invalid prediction_date: {prediction_date}")
    prediction_day = prediction_day.date()

    incoming_game_dates = pd.to_datetime(
        incoming_df["GAME_DATE"], errors="coerce"
    ).dt.date
    today_mask = incoming_game_dates == prediction_day
    df_to_predict_today = incoming_df[today_mask].copy()
    incoming_df_historical = incoming_df[~today_mask].copy()

    if df_to_predict_today.empty:
        raise ValueError(
            f"No games found in incoming dataframe for prediction_date={prediction_day}"
        )

    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    bucket = SETTINGS.s3_bucket

    selected_key = (
        historical_train_s3_key
        or os.getenv("TABPFN_HISTORICAL_TRAIN_S3_KEY")
        or _select_latest_historical_train_key(
            s3_client=s3, bucket=bucket, prefix=historical_train_prefix
        )
    )

    historical_bytes = read_s3_object_bytes(
        s3_client=s3,
        bucket=bucket,
        key=selected_key,
    )
    historical_df = load_parquet_from_bytes(historical_bytes)

    if "GAME_ID" not in historical_df.columns:
        raise ValueError(
            f"Historical parquet does not contain GAME_ID: s3://{bucket}/{selected_key}"
        )

    historical_df["GAME_ID"] = historical_df["GAME_ID"].astype(str)

    merged_df = pd.concat(
        [historical_df, incoming_df_historical], ignore_index=True, sort=False
    )
    merged_df = merged_df.drop_duplicates(subset=["GAME_ID"], keep="first")

    keep_columns = _keep_columns_for_tabpfn(total_points_pick_line_col)
    exclude_cols_containing = ["fanatics_sportsbook"]

    df_to_predict_today = clean_dataframe_for_training(
        df_to_predict_today,
        nan_threshold=100,
        keep_columns=keep_columns,
        exclude_cols_containing=exclude_cols_containing,
        keep_all_cols=True,
        # In-memory frames carry display columns beyond the declared ones.
        unexpected_text="drop",
        verbose=1,
    )
    cleaned_df = clean_dataframe_for_training(
        merged_df,
        corr_threshold=0.98,
        nan_threshold=50,
        max_na_per_row=80,
        keep_columns=keep_columns,
        exclude_cols_containing=exclude_cols_containing,
        unexpected_text="drop",
        verbose=1,
    )

    if "TOTAL_POINTS" not in cleaned_df.columns:
        raise ValueError("TOTAL_POINTS is required for TabPFN training")

    train_pick_line_col = _resolve_column_name(cleaned_df, total_points_pick_line_col)
    if train_pick_line_col is None:
        raise ValueError(
            f"Column '{total_points_pick_line_col}' is required for TabPFN training."
        )

    cleaned_columns = cleaned_df.columns.tolist()
    missing_predict_columns = [
        col for col in cleaned_columns if col not in df_to_predict_today.columns
    ]
    if missing_predict_columns:
        raise ValueError(
            "Today's dataframe is missing columns required after cleaning: "
            f"{missing_predict_columns[:10]}"
        )
    df_to_predict_today = df_to_predict_today[cleaned_columns]

    full_df = pd.concat(
        [cleaned_df, df_to_predict_today], ignore_index=True, sort=False
    )

    game_dates = pd.to_datetime(full_df["GAME_DATE"], errors="coerce").dt.date
    predict_mask = game_dates == prediction_day

    df_predictable = full_df[predict_mask].copy()
    if df_predictable.empty:
        raise ValueError(
            "No games found for TabPFN prediction after merge/cleaning for "
            f"prediction_date={prediction_day}"
        )

    train_mask = pd.to_numeric(full_df["TOTAL_POINTS"], errors="coerce").notna() & (
        ~predict_mask
    )
    df_train = full_df[train_mask].copy()
    if df_train.empty:
        raise ValueError("No training rows available for TabPFN fit after cleaning")

    prediction_season_year = get_season_year_from_date(str(prediction_day))
    min_train_season_year = prediction_season_year - 3

    if "SEASON_YEAR" not in df_train.columns:
        raise ValueError(
            "SEASON_YEAR is required to limit TabPFN training to 4 seasons"
        )

    df_train["SEASON_YEAR"] = pd.to_numeric(df_train["SEASON_YEAR"], errors="coerce")
    season_mask = df_train["SEASON_YEAR"].between(
        min_train_season_year, prediction_season_year
    )
    df_train = df_train[season_mask].copy()
    if df_train.empty:
        raise ValueError(
            "No training rows available for TabPFN after limiting to the last 4 seasons"
        )

    return df_train, df_predictable, prediction_day, total_points_pick_line_col


def _run_tabpfn_client_prediction(
    *,
    df_train: pd.DataFrame,
    df_predictable: pd.DataFrame,
    prediction_datetime: datetime,
    model_name: str,
    model_version: str,
    total_points_pick_line_col: str,
    prediction_target: PredictionTarget,
) -> pd.DataFrame:
    """Train one TabPFN regressor for the requested target and upload results."""
    drop_feature_cols = {
        "TOTAL_POINTS",
        "DIFF_FROM_LINE",
        _line_error_target_column_name(total_points_pick_line_col),
        "GAME_ID",
        "GAME_DATE",
        "GAME_TIME",
        "SEASON_TYPE",
        "SEASON_YEAR",
        "TEAM_NAME_TEAM_HOME",
        "TEAM_NAME_TEAM_AWAY",
        "MATCHUP",
        "MATCHUP_TEAM_HOME",
        "PRED_LINE_ERROR",
        "PRED_PICK",
        "PRED_TOTAL_POINTS",
    }

    # Raw DIFF_FROM_* and IS_OVER_* columns contain the current game's outcome
    # (TOTAL_POINTS - line) and must be excluded. Columns that merely reference
    # line-error history later in the name, such as injury availability effects,
    # are derived from past games and should be kept.
    _leakage_patterns = ("DIFF_FROM", "IS_OVER")

    def _is_leaky_column(col: str) -> bool:
        if not col.startswith(_leakage_patterns):
            return False
        return "_BEFORE" not in col

    feature_cols = [
        c
        for c in df_train.columns
        if c not in drop_feature_cols
        and not _is_leaky_column(c)
        and (
            pd.api.types.is_numeric_dtype(df_train[c])
            or pd.api.types.is_bool_dtype(df_train[c])
        )
    ]
    if not feature_cols:
        raise ValueError("No feature columns available for TabPFN")

    train_pick_line_col = _resolve_column_name(df_train, total_points_pick_line_col)
    if train_pick_line_col is None:
        raise ValueError(
            f"Column '{total_points_pick_line_col}' is required for TabPFN training."
        )
    train_pick_line = pd.to_numeric(df_train[train_pick_line_col], errors="coerce")

    X_train = df_train[feature_cols].copy()
    if prediction_target == PREDICTION_TARGET_TOTAL_POINTS:
        y_train = pd.to_numeric(df_train["TOTAL_POINTS"], errors="coerce")
    elif prediction_target == PREDICTION_TARGET_LINE_ERROR:
        y_train = (
            pd.to_numeric(df_train["TOTAL_POINTS"], errors="coerce") - train_pick_line
        )
    else:
        raise ValueError(
            "prediction_target must be one of: "
            f"{PREDICTION_TARGET_LINE_ERROR}, {PREDICTION_TARGET_TOTAL_POINTS}"
        )

    valid_train = y_train.notna()
    X_train = X_train.loc[valid_train]
    y_train = y_train.loc[valid_train]
    if X_train.empty:
        raise ValueError("No valid training rows available for TabPFN target")

    missing_features = [
        col for col in feature_cols if col not in df_predictable.columns
    ]
    if missing_features:
        raise ValueError(
            f"Prediction dataframe is missing {len(missing_features)} feature(s) "
            f"present in training data: {missing_features[:10]}"
        )

    X_pred = df_predictable[feature_cols].copy()
    if not X_train.columns.equals(X_pred.columns):
        raise ValueError(
            "Feature column mismatch between X_train and X_pred. "
            f"X_train has {len(X_train.columns)} columns, X_pred has {len(X_pred.columns)}"
        )

    df_predictable = df_predictable.copy()
    na_mask = df_predictable.isna()
    col_names = na_mask.columns.to_numpy()
    df_predictable["NA_COLUMNS_COUNT"] = na_mask.sum(axis=1).astype(int)
    df_predictable["NA_COLUMNS_NAMES"] = [
        ",".join(col_names[row_mask]) if row_mask.any() else None
        for row_mask in na_mask.to_numpy()
    ]

    pred_pick_line_col = _resolve_column_name(
        df_predictable, total_points_pick_line_col
    )
    if pred_pick_line_col is None:
        raise ValueError(
            f"Column '{total_points_pick_line_col}' is required to score TabPFN predictions."
        )
    pick_line = pd.to_numeric(df_predictable[pred_pick_line_col], errors="coerce")

    TabPFNRegressor = _init_tabpfn_client()
    regressor = TabPFNRegressor()

    print(f"Training TabPFN client regressor for target={prediction_target}")
    regressor.fit(X_train, y_train)
    print(f"Predicting with TabPFN client regressor for target={prediction_target}")
    raw_predictions = regressor.predict(X_pred)
    prediction_values = pd.to_numeric(raw_predictions, errors="coerce")

    df_predictable["TOTAL_BET365_LINE_AT_PREDICTION"] = pick_line
    if "TOTAL_OVER_UNDER_LINE" not in df_predictable.columns:
        df_predictable["TOTAL_OVER_UNDER_LINE"] = pick_line
    else:
        df_predictable["TOTAL_OVER_UNDER_LINE"] = pd.to_numeric(
            df_predictable["TOTAL_OVER_UNDER_LINE"], errors="coerce"
        ).fillna(pick_line)

    if prediction_target == PREDICTION_TARGET_TOTAL_POINTS:
        df_predictable["PRED_TOTAL_POINTS"] = prediction_values
        df_predictable["PRED_LINE_ERROR"] = prediction_values - pick_line
        df_predictable["PRED_PICK"] = np.select(
            [
                df_predictable["PRED_TOTAL_POINTS"] > pick_line,
                df_predictable["PRED_TOTAL_POINTS"] < pick_line,
                (df_predictable["PRED_TOTAL_POINTS"] == pick_line)
                & df_predictable["PRED_TOTAL_POINTS"].notna()
                & pick_line.notna(),
            ],
            ["OVER", "UNDER", "PUSH"],
            default=None,
        )
        df_predictable["PREDICTION_VALUE_TYPE"] = PREDICTION_VALUE_TYPE_TOTAL_POINTS
        non_null_pred_col = "PRED_TOTAL_POINTS"
    else:
        df_predictable["PRED_LINE_ERROR"] = prediction_values
        df_predictable["PRED_TOTAL_POINTS"] = pick_line + prediction_values
        df_predictable["PRED_PICK"] = np.select(
            [
                df_predictable["PRED_LINE_ERROR"] > 0,
                df_predictable["PRED_LINE_ERROR"] < 0,
                df_predictable["PRED_LINE_ERROR"] == 0,
            ],
            ["OVER", "UNDER", "PUSH"],
            default=None,
        )
        df_predictable["PREDICTION_VALUE_TYPE"] = PREDICTION_VALUE_TYPE_LINE_ERROR
        non_null_pred_col = "PRED_LINE_ERROR"

    df_predictable.rename(columns={"MATCHUP_TEAM_HOME": "MATCHUP"}, inplace=True)
    if "GAME_DATE" in df_predictable.columns:
        df_predictable["GAME_DATE"] = (
            df_predictable["GAME_DATE"].astype(str).str.split("T").str[0]
        )

    summary_columns = [
        "GAME_ID",
        "SEASON_TYPE",
        "GAME_DATE",
        "GAME_TIME",
        "TEAM_NAME_TEAM_HOME",
        "TEAM_NAME_TEAM_AWAY",
        "PREDICTION_VALUE_TYPE",
        "TOTAL_OVER_UNDER_LINE",
        "TOTAL_BET365_LINE_AT_PREDICTION",
        "PRED_LINE_ERROR",
        "PRED_TOTAL_POINTS",
        "PRED_PICK",
        "NA_COLUMNS_COUNT",
        "NA_COLUMNS_NAMES",
    ]
    missing_summary = [c for c in summary_columns if c not in df_predictable.columns]
    if missing_summary:
        raise ValueError(f"Missing summary columns for upload: {missing_summary}")

    df_summary = df_predictable[summary_columns].copy()
    df_summary["PREDICTION_DATETIME"] = prediction_datetime
    df_summary["PREDICTION_DATE"] = prediction_datetime.strftime("%Y-%m-%d %H:%M:%S")
    df_summary["HOME_PTS"] = None
    df_summary["AWAY_PTS"] = None

    def ensure_timezone_aware(dt_value):
        if pd.isna(dt_value):
            return pd.NaT

        if not isinstance(dt_value, (pd.Timestamp, datetime)):
            try:
                dt_value = pd.to_datetime(dt_value)
            except Exception:
                return pd.NaT

        if hasattr(dt_value, "tzinfo") and dt_value.tzinfo is None:
            return dt_value.tz_localize("US/Pacific")
        if hasattr(dt_value, "tzinfo") and dt_value.tzinfo is not None:
            return dt_value
        return pd.to_datetime(dt_value).tz_localize("US/Pacific")

    game_time_aware = df_summary["GAME_TIME"].apply(ensure_timezone_aware)
    df_summary["TIME_TO_MATCH_MINUTES"] = (
        game_time_aware - df_summary["PREDICTION_DATETIME"]
    ).dt.total_seconds() / 60
    df_summary["TIME_TO_MATCH_MINUTES"] = (
        pd.to_numeric(df_summary["TIME_TO_MATCH_MINUTES"], errors="coerce")
        .fillna(0)
        .round(0)
        .astype(int)
    )

    df_summary["MODEL_NAME"] = model_name
    df_summary["MODEL_TYPE"] = "TabPFNRegressor"
    df_summary["MODEL_VERSION"] = model_version
    df_summary["PREDICTION_SOURCE"] = model_name
    df_summary["TRAINING_CODE_TAG"] = "1.0"
    df_summary["TRAIN_DATE_MIN"] = pd.to_datetime(
        df_train["GAME_DATE"], errors="coerce"
    ).dt.date.min()
    df_summary["TRAIN_DATE_MAX"] = pd.to_datetime(
        df_train["GAME_DATE"], errors="coerce"
    ).dt.date.max()

    df_summary_clean = df_summary.dropna(subset=[non_null_pred_col])
    upload_predictions_to_postgre(df_summary_clean)

    return df_summary_clean


def load_and_predict_tabpfn_client_total_points_for_nba_games(
    df: pd.DataFrame,
    prediction_date: str | datetime | pd.Timestamp,
    prediction_datetime: datetime | None = None,
    historical_train_s3_key: str | None = None,
    historical_train_prefix: str = "train_data/",
    model_name: str = "tabpfn_client_regressor",
    model_version: str = "1.0",
    total_points_pick_line_col: str | None = None,
) -> pd.DataFrame:
    """Generate TabPFN predictions for total points."""
    if prediction_datetime is None:
        prediction_datetime = datetime.now(ZoneInfo("Europe/Madrid"))

    df_train, df_predictable, _, total_points_pick_line_col = (
        _prepare_tabpfn_client_datasets(
            df,
            prediction_date,
            historical_train_s3_key=historical_train_s3_key,
            historical_train_prefix=historical_train_prefix,
            total_points_pick_line_col=total_points_pick_line_col,
        )
    )

    return _run_tabpfn_client_prediction(
        df_train=df_train,
        df_predictable=df_predictable,
        prediction_datetime=prediction_datetime,
        model_name=model_name,
        model_version=model_version,
        total_points_pick_line_col=total_points_pick_line_col,
        prediction_target=PREDICTION_TARGET_TOTAL_POINTS,
    )


def load_and_predict_tabpfn_client_line_error_for_nba_games(
    df: pd.DataFrame,
    prediction_date: str | datetime | pd.Timestamp,
    prediction_datetime: datetime | None = None,
    historical_train_s3_key: str | None = None,
    historical_train_prefix: str = "train_data/",
    model_name: str = "tabpfn_client_regressor_line_error",
    model_version: str = "1.0",
    total_points_pick_line_col: str | None = None,
) -> pd.DataFrame:
    """Generate TabPFN predictions for the main-book line error."""
    if prediction_datetime is None:
        prediction_datetime = datetime.now(ZoneInfo("Europe/Madrid"))

    df_train, df_predictable, _, total_points_pick_line_col = (
        _prepare_tabpfn_client_datasets(
            df,
            prediction_date,
            historical_train_s3_key=historical_train_s3_key,
            historical_train_prefix=historical_train_prefix,
            total_points_pick_line_col=total_points_pick_line_col,
        )
    )

    return _run_tabpfn_client_prediction(
        df_train=df_train,
        df_predictable=df_predictable,
        prediction_datetime=prediction_datetime,
        model_name=model_name,
        model_version=model_version,
        total_points_pick_line_col=total_points_pick_line_col,
        prediction_target=PREDICTION_TARGET_LINE_ERROR,
    )


def load_and_predict_tabpfn_client_for_nba_games(
    df: pd.DataFrame,
    prediction_date: str | datetime | pd.Timestamp,
    prediction_datetime: datetime | None = None,
    historical_train_s3_key: str | None = None,
    historical_train_prefix: str = "train_data/",
    model_name: str = "tabpfn_client_regressor",
    model_version: str = "1.0",
    total_points_pick_line_col: str | None = None,
    line_error_model_name: str = "tabpfn_client_regressor_line_error",
) -> pd.DataFrame:
    """
    Generate both TabPFN outputs for each game:
    - total points
    - difference from the configured main total line
    """
    if prediction_datetime is None:
        prediction_datetime = datetime.now(ZoneInfo("Europe/Madrid"))
    df_train, df_predictable, _, total_points_pick_line_col = (
        _prepare_tabpfn_client_datasets(
            df,
            prediction_date,
            historical_train_s3_key=historical_train_s3_key,
            historical_train_prefix=historical_train_prefix,
            total_points_pick_line_col=total_points_pick_line_col,
        )
    )

    total_points_predictions = _run_tabpfn_client_prediction(
        df_train=df_train,
        df_predictable=df_predictable,
        prediction_datetime=prediction_datetime,
        model_name=model_name,
        model_version=model_version,
        total_points_pick_line_col=total_points_pick_line_col,
        prediction_target=PREDICTION_TARGET_TOTAL_POINTS,
    )
    # line_error_predictions = _run_tabpfn_client_prediction(
    #     df_train=df_train,
    #     df_predictable=df_predictable,
    #     prediction_datetime=prediction_datetime,
    #     model_name=line_error_model_name,
    #     model_version=model_version,
    #     total_points_pick_line_col=total_points_pick_line_col,
    #     prediction_target=PREDICTION_TARGET_LINE_ERROR,
    # )

    return (
        pd.concat(
            [total_points_predictions],
            ignore_index=True,
            sort=False,
        )
        .sort_values(["GAME_ID", "MODEL_NAME", "PREDICTION_VALUE_TYPE"])
        .reset_index(drop=True)
    )
