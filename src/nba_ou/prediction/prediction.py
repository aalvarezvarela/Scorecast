from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import shap
from nba_ou.config.odds_columns import (
    resolve_main_spread_line_col,
    total_line_over_col_raw,
)
from nba_ou.data_processing.missing_data.clean_df_for_training import (
    clean_dataframe_for_training,
)
from nba_ou.postgre_db.predictions.create.create_ou_predictions_db import (
    upload_predictions_to_postgre,
)
from xgboost import XGBRegressor

PredictionTarget = Literal["PRED_LINE_ERROR", "TOTAL_POINTS", "PRED_SPREAD_ERROR"]

PREDICTION_TARGET_LINE_ERROR: PredictionTarget = "PRED_LINE_ERROR"
PREDICTION_TARGET_TOTAL_POINTS: PredictionTarget = "TOTAL_POINTS"
PREDICTION_TARGET_SPREAD_ERROR: PredictionTarget = "PRED_SPREAD_ERROR"
PREDICTION_VALUE_TYPE_TOTAL_POINTS = "TOTAL_POINTS"
# Keep the stored/database value as DIFF_FROM_LINE for backward compatibility,
# but use "line_error" as the canonical name in code and model metadata.
PREDICTION_VALUE_TYPE_LINE_ERROR = "DIFF_FROM_LINE"
PREDICTION_VALUE_TYPE_DIFF_FROM_LINE = PREDICTION_VALUE_TYPE_LINE_ERROR
PREDICTION_VALUE_TYPE_SPREAD_ERROR = "SPREAD_ERROR"

#: Targets priced against the totals market, which therefore require a totals
#: line at prediction time. The spread target requires a spread line instead,
#: so demanding a totals column for every model would reject it for the wrong
#: reason.
TOTALS_TARGETS = frozenset(
    {PREDICTION_TARGET_LINE_ERROR, PREDICTION_TARGET_TOTAL_POINTS}
)

#: How far a model may be used from the horizon it was trained for, in minutes.
#: A T-720 model quietly serving every game at T-60 is invisible without this:
#: the predictions table records when a prediction RAN, never what the model
#: was trained FOR.
DEFAULT_HORIZON_TOLERANCE_MINUTES = 45

#: Map a registry target onto this module's prediction target.
REGISTRY_TARGET_TO_PREDICTION_TARGET: dict[str, PredictionTarget] = {
    "line_error": PREDICTION_TARGET_LINE_ERROR,
    "total_points": PREDICTION_TARGET_TOTAL_POINTS,
    "spread_error": PREDICTION_TARGET_SPREAD_ERROR,
}


def _resolve_column_name(df: pd.DataFrame, desired_column: str) -> str | None:
    """Resolve column name with case-insensitive fallback."""
    if desired_column in df.columns:
        return desired_column

    desired_lower = desired_column.lower()
    for col in df.columns:
        if col.lower() == desired_lower:
            return col
    return None


def _prepare_required_features_for_prediction(
    df: pd.DataFrame,
    required_features: list[str],
    *,
    mandatory_main_book_col: str | None = None,
) -> tuple[list[str], pd.DataFrame, pd.Series, pd.Series]:
    """
    Ensure required model features exist and track NaN values per row.

    Returns:
        - List of required feature names
        - Feature DataFrame aligned to the model's expected column names
        - Per-row count of NaN values in required features
        - Per-row comma-separated list of feature names with NaN values
    """
    normalized_required = [str(feat) for feat in required_features]

    # The anchor line the caller says this market needs. Passing None means
    # "no line is mandatory" -- not "fall back to the totals line", which would
    # reject a spread model for lacking a column its market does not have.
    if mandatory_main_book_col is not None:
        main_book_col = _resolve_column_name(df, mandatory_main_book_col)
        if main_book_col is None:
            raise ValueError(
                f"Column '{mandatory_main_book_col}' is mandatory for prediction."
            )

    # Resolve all required features with case-insensitive fallback
    resolved_feature_columns: list[str] = []
    missing_features: list[str] = []
    for feat in normalized_required:
        resolved_col = _resolve_column_name(df, feat)
        if resolved_col is None:
            missing_features.append(feat)
        else:
            resolved_feature_columns.append(resolved_col)

    if missing_features:
        raise ValueError(
            f"Required model features are missing from the dataframe: {missing_features}"
        )

    # Count NaN values per row in required features
    required_df = df[resolved_feature_columns].copy()
    required_df.columns = normalized_required

    # Convert pd.NA to np.nan for compatibility with model predictions
    # This ensures all NA types are unified before being passed to the model
    required_df = required_df.fillna(np.nan)

    na_mask = required_df.isna()
    col_names = na_mask.columns.to_numpy()

    na_count = na_mask.sum(axis=1).astype(int)
    na_names = pd.Series(
        [
            ",".join(col_names[row_mask]) if row_mask.any() else None
            for row_mask in na_mask.to_numpy()
        ],
        index=df.index,
    )

    return normalized_required, required_df, na_count, na_names


def compute_shap_values(model, X: pd.DataFrame) -> tuple[pd.DataFrame, float | None]:
    """
    Compute per-feature SHAP values for tree-based regressors.

    Returns:
        - DataFrame with one SHAP contribution per feature and row
        - Scalar expected/base value when available
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        if len(shap_values) != 1:
            raise ValueError(
                "Expected a single SHAP output array for regression predictions."
            )
        shap_values = shap_values[0]

    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    shap_array = np.asarray(shap_values)
    if shap_array.ndim != 2:
        raise ValueError(
            f"Expected 2D SHAP values, received shape {shap_array.shape!r}."
        )

    shap_df = pd.DataFrame(
        shap_array,
        columns=X.columns,
        index=X.index,
    )

    expected_value = getattr(explainer, "expected_value", None)
    if expected_value is None:
        base_value: float | None = None
    else:
        expected_array = np.asarray(expected_value).reshape(-1)
        base_value = (
            float(expected_array[0])
            if expected_array.size and pd.notna(expected_array[0])
            else None
        )

    return shap_df, base_value


def compute_shap_confidence_metrics(
    shap_row: pd.Series,
    pred_margin: float,
    top_k: int = 10,
) -> dict:
    """
    Compute SHAP-based confidence metrics for one prediction.

    Measures how decisively the feature contributions support the predicted edge.
    A prediction can have the same margin but very different internal coherence:
    - High confidence: most SHAP mass points in one direction
    - Low confidence: large positive and negative forces cancel out

    Args:
        shap_row: SHAP values for a single row, indexed by feature name
        pred_margin: predicted margin vs line (pred_total - line)
        top_k: number of top absolute SHAP features to inspect

    Returns:
        dict with confidence metrics:
            - shap_directional_confidence: abs(net_margin) / total_abs (0-1)
            - shap_support_ratio: support_sum / opposing_sum
            - shap_top_k_agreement: fraction of top-k features agreeing with pick
            - shap_confidence_score: composite score (0-1)
    """
    shap_values = shap_row.astype(float)

    positive_sum = float(shap_values[shap_values > 0].sum())
    negative_sum = float(np.abs(shap_values[shap_values < 0].sum()))
    total_abs = float(np.abs(shap_values).sum())
    net_margin_from_shap = positive_sum - negative_sum

    if total_abs == 0:
        directional_confidence = 0.0
    else:
        directional_confidence = abs(net_margin_from_shap) / total_abs

    predicted_sign = np.sign(pred_margin)

    if predicted_sign > 0:
        support_sum = positive_sum
        opposing_sum = negative_sum
    elif predicted_sign < 0:
        support_sum = negative_sum
        opposing_sum = positive_sum
    else:
        support_sum = 0.0
        opposing_sum = 0.0

    support_ratio = float(support_sum / (opposing_sum + 1e-9))

    top_features = shap_values.reindex(
        shap_values.abs().sort_values(ascending=False).head(top_k).index
    )
    top_signs = np.sign(top_features.values)

    if predicted_sign == 0 or len(top_signs) == 0:
        top_k_agreement = 0.0
    else:
        top_k_agreement = float(np.mean(top_signs == predicted_sign))

    edge_strength = min(abs(pred_margin) / 3.0, 1.0)

    confidence_score = (
        0.5 * edge_strength + 0.3 * directional_confidence + 0.2 * top_k_agreement
    )

    return {
        "SHAP_DIRECTIONAL_CONFIDENCE": round(directional_confidence, 4),
        "SHAP_SUPPORT_RATIO": round(support_ratio, 4),
        "SHAP_TOP_K_AGREEMENT": round(top_k_agreement, 4),
        "SHAP_CONFIDENCE_SCORE": round(confidence_score, 4),
    }


def top_shap_reasons(
    shap_row: pd.Series,
    *,
    top_n: int = 20,
    direction: Literal["positive", "negative"] = "positive",
) -> str | None:
    """Serialize the strongest positive or negative SHAP contributors."""
    shap_series = pd.to_numeric(shap_row, errors="coerce").dropna()

    if direction == "positive":
        selected = shap_series[shap_series > 0].sort_values(ascending=False).head(top_n)
    elif direction == "negative":
        selected = shap_series[shap_series < 0].sort_values(ascending=True).head(top_n)
    else:
        raise ValueError("direction must be either 'positive' or 'negative'")

    if selected.empty:
        return None

    return ",".join(f"{feature}:{value:+.3f}" for feature, value in selected.items())


def _nested_metadata_value(metadata: dict, *path: str) -> object | None:
    current: object = metadata
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def resolve_prediction_target(registry_target: str) -> PredictionTarget:
    """Map a spec's declared target onto a prediction target.

    Replaces the previous ``_infer_prediction_target_from_metadata``, which
    searched the model's name and type for substrings and **fell back to
    line-error for anything it did not recognise**. A spread model matched
    neither token and would have been served as a line-error model with no
    warning. The target is now declared, so an unknown one raises.
    """
    try:
        return REGISTRY_TARGET_TO_PREDICTION_TARGET[str(registry_target)]
    except KeyError:
        known = ", ".join(sorted(REGISTRY_TARGET_TO_PREDICTION_TARGET))
        raise ValueError(
            f"Unknown model target {registry_target!r}. Known targets: {known}."
        ) from None


def _resolve_feature_names_from_metadata(metadata: dict) -> list[str] | None:
    schema_feature_names = _nested_metadata_value(metadata, "schema", "feature_names")
    if isinstance(schema_feature_names, list) and schema_feature_names:
        return [str(feat) for feat in schema_feature_names]

    return None


def load_and_predict_model_for_nba_games(
    df: pd.DataFrame,
    regressor,
    model_name: str,
    model_type: str,
    model_version: str,
    *,
    required_features: list[str] | None = None,
    prediction_source: str | None = None,
    training_code_tag: str | None = None,
    train_date_min: str | None = None,
    train_date_max: str | None = None,
    prediction_datetime: datetime | None = None,
    prediction_target: PredictionTarget = PREDICTION_TARGET_LINE_ERROR,
    total_points_pick_line_col: str | None = None,
    spread_pick_line_col: str | None = None,
    extra_summary_columns: dict | None = None,
    shap_top_n: int = 20,
) -> pd.DataFrame:
    """
    Predict NBA games using a trained regressor for one of three targets:
    - PRED_LINE_ERROR: model predicts the totals residual directly
    - TOTAL_POINTS: model predicts total points directly
    - PRED_SPREAD_ERROR: model predicts the spread residual (home margin minus
      the anchor book's home line)

    PRED_PICK behavior:
    - PRED_LINE_ERROR mode: OVER if prediction > 0, UNDER if < 0, PUSH if == 0
    - TOTAL_POINTS mode: compares predicted total points against the configured line column
    - PRED_SPREAD_ERROR mode: HOME if the residual > 0 (the home team beats its
      own spread), AWAY if < 0, PUSH if == 0

    Args:
        df: Input DataFrame containing game data with features
        regressor: Trained regressor model.
        model_name: Name of the model being used for predictions
        model_type: Type of the model (e.g., "regression", "classification")
        model_version: Version identifier for the model
        prediction_datetime: Timestamp when predictions are made. If None, uses current time in Europe/Madrid timezone.
        prediction_target: Which target this model predicts.
        total_points_pick_line_col: Line column used to derive OVER/UNDER/PUSH when
            prediction_target is TOTAL_POINTS. Defaults to main sportsbook from config.
        shap_top_n: Number of top positive/negative SHAP contributors to store.

    Returns:
        DataFrame containing predictions with key information
    """
    # Derive total_points_pick_line_col from config if not provided
    if total_points_pick_line_col is None:
        total_points_pick_line_col = total_line_over_col_raw()

    # Step 1: Clean the DataFrame for prediction
    df_predictable = clean_dataframe_for_training(
        df,
        nan_threshold=100,
        keep_columns=[
            "GAME_ID",
            "SEASON_TYPE",
            "GAME_DATE",
            "GAME_TIME",
            "TEAM_NAME_TEAM_HOME",
            "TEAM_NAME_TEAM_AWAY",
            "MATCHUP_TEAM_HOME",
        ],
        keep_all_cols=True,
        verbose=1,
        strict_mode=30,
    )

    # Load the model using joblib
    model = regressor

    # Extract required features from model
    if required_features is None:
        required_features_raw = model.feature_names_in_
        required_features = [str(feat) for feat in required_features_raw]
    else:
        required_features = [str(feat) for feat in required_features]

    # Handle IS_TRAINING_DATA if it's missing - create it with all False values
    if (
        "IS_TRAINING_DATA" in required_features
        and "IS_TRAINING_DATA" not in df_predictable
    ):
        print(
            "⚠️  Warning: 'IS_TRAINING_DATA' column is missing. Creating it with default False values for prediction."
        )
        df_predictable["IS_TRAINING_DATA"] = False

    # Ensure required features exist and track NaN values per row in required features
    (
        required_features,
        X,
        na_count,
        na_names,
    ) = _prepare_required_features_for_prediction(
        df_predictable,
        required_features,
        mandatory_main_book_col=(
            None
            if prediction_target == PREDICTION_TARGET_SPREAD_ERROR
            else total_points_pick_line_col
        ),
    )

    # Add NaN tracking columns (based on required features only)
    df_predictable["NA_COLUMNS_COUNT"] = na_count
    df_predictable["NA_COLUMNS_NAMES"] = na_names

    # Make predictions
    raw_predictions = model.predict(X)
    prediction_values = pd.to_numeric(
        pd.Series(raw_predictions, index=df_predictable.index), errors="coerce"
    )
    pick_line: pd.Series | None = None

    if prediction_target in TOTALS_TARGETS:
        main_book_line_col = _resolve_column_name(
            df_predictable, total_points_pick_line_col
        )
        if main_book_line_col is None:
            raise ValueError(
                f"Main sportsbook line column '{total_points_pick_line_col}' is mandatory for prediction."
            )
        main_book_line = pd.to_numeric(
            df_predictable[main_book_line_col], errors="coerce"
        )
        df_predictable["TOTAL_BET365_LINE_AT_PREDICTION"] = main_book_line
        df_predictable["TOTAL_OVER_UNDER_LINE"] = main_book_line

    if prediction_target == PREDICTION_TARGET_SPREAD_ERROR:
        # The spread residual is defined against the anchor book's line, so
        # there is deliberately no fallback to another book: answering with a
        # different book's spread would change what the target MEANS row by
        # row. See odds_columns.resolve_main_spread_line_col.
        spread_line_col = spread_pick_line_col or resolve_main_spread_line_col(
            df_predictable
        )
        resolved_spread_col = (
            _resolve_column_name(df_predictable, spread_line_col)
            if spread_line_col
            else None
        )
        if resolved_spread_col is None:
            raise ValueError(
                f"Spread line column '{spread_line_col}' is mandatory when "
                f"prediction_target={PREDICTION_TARGET_SPREAD_ERROR}."
            )
        spread_line = pd.to_numeric(
            df_predictable[resolved_spread_col], errors="coerce"
        )
        df_predictable["SPREAD_LINE_AT_PREDICTION"] = spread_line
        df_predictable["PRED_SPREAD_ERROR"] = prediction_values
        # SPREAD_LINE_HOME is the implied home margin, so the model's residual
        # adds to it to give the predicted margin.
        df_predictable["PRED_HOME_MARGIN"] = spread_line + prediction_values

    elif prediction_target == PREDICTION_TARGET_LINE_ERROR:
        df_predictable["PRED_LINE_ERROR"] = prediction_values
        df_predictable["PRED_TOTAL_POINTS"] = (
            df_predictable["TOTAL_OVER_UNDER_LINE"] + df_predictable["PRED_LINE_ERROR"]
        )

    elif prediction_target == PREDICTION_TARGET_TOTAL_POINTS:
        df_predictable["PRED_TOTAL_POINTS"] = prediction_values
        pick_line = df_predictable["TOTAL_OVER_UNDER_LINE"]
        df_predictable["PRED_LINE_ERROR"] = (
            df_predictable["PRED_TOTAL_POINTS"] - pick_line
        )

    else:
        raise ValueError(
            "prediction_target must be one of: "
            f"{PREDICTION_TARGET_LINE_ERROR}, {PREDICTION_TARGET_TOTAL_POINTS}, "
            f"{PREDICTION_TARGET_SPREAD_ERROR}"
        )

    if prediction_target == PREDICTION_TARGET_SPREAD_ERROR:
        pred_spread_error = pd.to_numeric(
            df_predictable["PRED_SPREAD_ERROR"], errors="coerce"
        )
        # A positive residual means the home team beats its own spread.
        df_predictable["PRED_PICK"] = np.select(
            [
                pred_spread_error > 0,
                pred_spread_error < 0,
                pred_spread_error == 0,
            ],
            ["HOME", "AWAY", "PUSH"],
            default=None,
        )
    elif prediction_target == PREDICTION_TARGET_TOTAL_POINTS:
        if pick_line is None:
            pick_line_col = _resolve_column_name(
                df_predictable, total_points_pick_line_col
            )
            if pick_line_col is None:
                raise ValueError(
                    f"Column '{total_points_pick_line_col}' is required to compute PRED_PICK "
                    f"when prediction_target={PREDICTION_TARGET_TOTAL_POINTS}."
                )
            pick_line = pd.to_numeric(df_predictable[pick_line_col], errors="coerce")
        pred_total = pd.to_numeric(df_predictable["PRED_TOTAL_POINTS"], errors="coerce")
        df_predictable["PRED_PICK"] = np.select(
            [
                pred_total > pick_line,
                pred_total < pick_line,
                (pred_total == pick_line) & pred_total.notna() & pick_line.notna(),
            ],
            ["OVER", "UNDER", "PUSH"],
            default=None,
        )
    else:
        pred_line_error = pd.to_numeric(
            df_predictable["PRED_LINE_ERROR"], errors="coerce"
        )
        df_predictable["PRED_PICK"] = np.select(
            [
                pred_line_error > 0,
                pred_line_error < 0,
                pred_line_error == 0,
            ],
            ["OVER", "UNDER", "PUSH"],
            default=None,
        )

    df_predictable["PREDICTION_VALUE_TYPE"] = {
        PREDICTION_TARGET_TOTAL_POINTS: PREDICTION_VALUE_TYPE_TOTAL_POINTS,
        PREDICTION_TARGET_SPREAD_ERROR: PREDICTION_VALUE_TYPE_SPREAD_ERROR,
    }.get(prediction_target, PREDICTION_VALUE_TYPE_LINE_ERROR)

    df_predictable.rename(columns={"MATCHUP_TEAM_HOME": "MATCHUP"}, inplace=True)
    df_predictable["GAME_DATE"] = (
        df_predictable["GAME_DATE"].astype(str).str.split("T").str[0]
    )

    # Sheet 1: Summary DataFrame.
    # The market-specific block differs by target: a spread model never
    # produces a totals line or a totals residual, so asking for those columns
    # would raise rather than simply leave them empty.
    summary_columns = [
        "GAME_ID",
        "SEASON_TYPE",
        "GAME_DATE",
        "GAME_TIME",
        "TEAM_NAME_TEAM_HOME",
        "TEAM_NAME_TEAM_AWAY",
        "PREDICTION_VALUE_TYPE",
    ]
    if prediction_target == PREDICTION_TARGET_SPREAD_ERROR:
        summary_columns += [
            "SPREAD_LINE_AT_PREDICTION",
            "PRED_SPREAD_ERROR",
            "PRED_HOME_MARGIN",
        ]
    else:
        summary_columns += [
            "TOTAL_OVER_UNDER_LINE",
            "TOTAL_BET365_LINE_AT_PREDICTION",
            "PRED_LINE_ERROR",
            "PRED_TOTAL_POINTS",
        ]
    summary_columns += [
        "PRED_PICK",
        "NA_COLUMNS_COUNT",
        "NA_COLUMNS_NAMES",
    ]

    # SHAP confidence columns are added after summary creation
    df_summary = df_predictable[summary_columns].copy()

    # Add prediction timestamp and time to match
    # Use provided prediction_datetime or create timezone-aware timestamp in Madrid time
    if prediction_datetime is None:
        prediction_datetime = datetime.now(ZoneInfo("Europe/Madrid"))

    df_summary["PREDICTION_DATETIME"] = prediction_datetime
    df_summary["PREDICTION_DATE"] = prediction_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # Add score columns (will be None for predictions, filled in later with actual results)
    df_summary["HOME_PTS"] = None
    df_summary["AWAY_PTS"] = None

    # Calculate time to match in minutes
    def ensure_timezone_aware(dt_value):
        """Convert datetime to timezone-aware (US/Pacific) if naive."""
        if pd.isna(dt_value):
            return pd.NaT

        # Convert to datetime if it's not already
        if not isinstance(dt_value, (pd.Timestamp, datetime)):
            try:
                dt_value = pd.to_datetime(dt_value)
            except Exception:
                return pd.NaT

        # Check if timezone-aware
        if hasattr(dt_value, "tzinfo") and dt_value.tzinfo is None:
            # Naive datetime, localize to US/Pacific
            return dt_value.tz_localize("US/Pacific")
        elif hasattr(dt_value, "tzinfo") and dt_value.tzinfo is not None:
            # Already timezone-aware
            return dt_value
        else:
            # Fallback: try to convert and localize
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

    # Add model information from parameters
    df_summary["MODEL_NAME"] = model_name
    df_summary["MODEL_TYPE"] = model_type
    df_summary["MODEL_VERSION"] = model_version
    df_summary["PREDICTION_SOURCE"] = prediction_source
    df_summary["TRAINING_CODE_TAG"] = training_code_tag
    df_summary["TRAIN_DATE_MIN"] = train_date_min
    df_summary["TRAIN_DATE_MAX"] = train_date_max

    shap_df, shap_base_value = compute_shap_values(model, X)
    df_summary["SHAP_BASE_VALUE"] = shap_base_value
    df_summary["SHAP_TOP_POSITIVE_FEATURES"] = shap_df.apply(
        top_shap_reasons,
        axis=1,
        direction="positive",
        top_n=shap_top_n,
    )
    df_summary["SHAP_TOP_NEGATIVE_FEATURES"] = shap_df.apply(
        top_shap_reasons,
        axis=1,
        direction="negative",
        top_n=shap_top_n,
    )

    # Compute SHAP confidence metrics per row
    # The signed edge this model is expressing, whichever market it is in.
    pred_margins = df_summary[
        "PRED_SPREAD_ERROR"
        if prediction_target == PREDICTION_TARGET_SPREAD_ERROR
        else "PRED_LINE_ERROR"
    ]
    confidence_metrics = shap_df.apply(
        lambda row: pd.Series(
            compute_shap_confidence_metrics(
                shap_row=row,
                pred_margin=float(pred_margins.loc[row.name])
                if pd.notna(pred_margins.loc[row.name])
                else 0.0,
                top_k=10,
            )
        ),
        axis=1,
    )
    for col in confidence_metrics.columns:
        df_summary[col] = confidence_metrics[col].values

    # Drop rows with NaN in the model's predicted target before saving to database
    target_column = {
        PREDICTION_TARGET_TOTAL_POINTS: "PRED_TOTAL_POINTS",
        PREDICTION_TARGET_SPREAD_ERROR: "PRED_SPREAD_ERROR",
    }.get(prediction_target, "PRED_LINE_ERROR")
    df_summary_clean = df_summary.dropna(subset=[target_column])

    # Applied BEFORE the upload: these identify which slot, spec and build
    # produced the row, and the upload happens here rather than in the caller.
    if extra_summary_columns:
        for column, value in extra_summary_columns.items():
            df_summary_clean[column] = value

    upload_predictions_to_postgre(df_summary_clean)

    return df_summary_clean


def select_rows_for_horizon(
    df: pd.DataFrame,
    *,
    horizon_minutes: int | None,
    tolerance_minutes: int = DEFAULT_HORIZON_TOLERANCE_MINUTES,
    time_to_match_col: str = "TIME_TO_MATCH_MINUTES",
) -> pd.Series:
    """Which rows this model is entitled to predict.

    A model trained on the T-360 snapshot has only ever seen a market six hours
    from tip; using it at T-60 is a different question asked of the same
    booster. Nothing in the stored output would reveal that, because
    ``TIME_TO_MATCH_MINUTES`` records when the prediction RAN and there was no
    field for what the model was trained FOR -- which is why the fit now
    carries its horizon and this check exists.

    A pooled model (``horizon_minutes is None``) conditions on time-to-tip
    itself and is exempt. A T-0 model is "at or after the last snapshot", so it
    has no upper bound.
    """
    if horizon_minutes is None or time_to_match_col not in df.columns:
        return pd.Series(True, index=df.index)

    minutes = pd.to_numeric(df[time_to_match_col], errors="coerce")
    if horizon_minutes == 0:
        return (minutes <= tolerance_minutes) | minutes.isna()
    within = (minutes - horizon_minutes).abs() <= tolerance_minutes
    return within | minutes.isna()


def load_registry_model_and_predict(
    *,
    s3_client,
    bucket: str,
    slot,
    df: pd.DataFrame,
    prediction_datetime: datetime | None = None,
    total_points_pick_line_col: str | None = None,
    spread_pick_line_col: str | None = None,
    shap_top_n: int = 20,
    horizon_tolerance_minutes: int = DEFAULT_HORIZON_TOLERANCE_MINUTES,
    channel=None,
) -> pd.DataFrame:
    """Predict with whatever a slot's channel currently points at.

    Resolution is ``channels/production.json -> fit -> spec -> model``: the
    served model is interpreted by the spec ITS OWN FIT names, never by the
    slot's current configuration. That is what lets a new configuration be
    adopted without disturbing production.

    Args:
        slot: ``nba_ou.modeling.registry_paths.ModelSlot`` to serve.
        channel: which channel to read; defaults to production.
    """
    from nba_ou.modeling.registry_paths import Channel
    from nba_ou.modeling.registry_store import resolve_channel

    resolved = resolve_channel(
        s3_client=s3_client,
        bucket=bucket,
        slot=slot,
        channel=channel or Channel.PRODUCTION,
    )
    spec = resolved.spec
    fit = resolved.fit

    prediction_target = resolve_prediction_target(spec.identity.target)

    eligible = select_rows_for_horizon(
        df,
        horizon_minutes=spec.identity.horizon_minutes,
        tolerance_minutes=horizon_tolerance_minutes,
    )
    if not bool(eligible.any()):
        print(
            f"  {slot.describe()}: no rows within "
            f"{horizon_tolerance_minutes} min of the model's horizon "
            f"({spec.identity.horizon_minutes}); skipping."
        )
        return pd.DataFrame()
    skipped = int((~eligible).sum())
    if skipped:
        print(
            f"  {slot.describe()}: skipping {skipped} row(s) outside the "
            f"model's horizon (T-{spec.identity.horizon_minutes})."
        )

    regressor = XGBRegressor()
    regressor.load_model(bytearray(resolved.model_bytes))

    predictions = load_and_predict_model_for_nba_games(
        df=df.loc[eligible],
        regressor=regressor,
        model_name=fit.model_name,
        model_type=f"{spec.identity.target}_{slot.horizon_label}",
        model_version=fit.train_date_max.strftime("%d_%m_%y"),
        required_features=list(spec.features.feature_names),
        prediction_source=fit.model_name,
        training_code_tag=spec.provenance.training_version,
        train_date_min=fit.train_date_min.isoformat(),
        train_date_max=fit.train_date_max.isoformat(),
        prediction_datetime=prediction_datetime,
        prediction_target=prediction_target,
        total_points_pick_line_col=total_points_pick_line_col,
        spread_pick_line_col=spread_pick_line_col,
        # Passed in rather than attached to the returned frame: the upload to
        # Postgres happens INSIDE load_and_predict_model_for_nba_games, so
        # anything added here afterwards would never be stored.
        extra_summary_columns={
            "MODEL_TARGET": str(spec.identity.target),
            "MODEL_HORIZON_MINUTES": spec.identity.horizon_minutes,
            "MODEL_SCHEMA_VERSION": spec.identity.schema_version,
            "MODEL_VARIANT": spec.identity.variant,
            "SPEC_ID": spec.spec_id,
            "FIT_ID": fit.fit_id,
        },
        shap_top_n=shap_top_n,
    )
    return predictions
