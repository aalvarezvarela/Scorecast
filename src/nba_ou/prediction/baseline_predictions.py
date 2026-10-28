from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from nba_ou.config.odds_columns import total_line_over_col_raw
from nba_ou.utils.general_utils import get_season_year_from_date

PREDICTION_VALUE_TYPE_LINE_ERROR = "DIFF_FROM_LINE"

RANDOM_LINE_ERROR_BASELINE_MODEL_NAME = "random_line_error_baseline"
HISTORICAL_AVERAGE_LINE_ERROR_BASELINE_MODEL_NAME = (
    "historical_average_line_error_baseline"
)
BASELINE_MODEL_VERSION = "1.0"
BASELINE_TRAINING_CODE_TAG = "1.0"


def _resolve_column_name(df: pd.DataFrame, desired_column: str) -> str | None:
    if desired_column in df.columns:
        return desired_column

    desired_lower = desired_column.lower()
    for col in df.columns:
        if col.lower() == desired_lower:
            return col
    return None


def _series_or_none(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([None] * len(df), index=df.index)


def _ensure_timezone_aware(dt_value):
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


def _prediction_datetime_or_default(
    prediction_datetime: datetime | None,
) -> datetime:
    if prediction_datetime is not None:
        return prediction_datetime
    return datetime.now(ZoneInfo("Europe/Madrid"))


def _ensure_prediction_datetime_aware(prediction_datetime: datetime) -> pd.Timestamp:
    prediction_timestamp = pd.Timestamp(prediction_datetime)
    if prediction_timestamp.tzinfo is None:
        return prediction_timestamp.tz_localize("Europe/Madrid")
    return prediction_timestamp


def _add_time_to_match_minutes(
    df_summary: pd.DataFrame, prediction_datetime: datetime
) -> pd.DataFrame:
    out = df_summary.copy()
    game_time_aware = out["GAME_TIME"].apply(_ensure_timezone_aware)
    prediction_timestamp = _ensure_prediction_datetime_aware(prediction_datetime)
    out["TIME_TO_MATCH_MINUTES"] = (
        game_time_aware - prediction_timestamp
    ).dt.total_seconds() / 60
    out["TIME_TO_MATCH_MINUTES"] = (
        pd.to_numeric(out["TIME_TO_MATCH_MINUTES"], errors="coerce")
        .fillna(0)
        .round(0)
        .astype(int)
    )
    return out


def _historical_game_date_bounds(df_history: pd.DataFrame) -> tuple[object, object]:
    if (
        "TOTAL_POINTS" not in df_history.columns
        or "GAME_DATE" not in df_history.columns
    ):
        return None, None

    total_points = pd.to_numeric(df_history["TOTAL_POINTS"], errors="coerce")
    game_dates = pd.to_datetime(df_history["GAME_DATE"], errors="coerce")
    valid_dates = game_dates[total_points.notna()]
    if valid_dates.empty:
        return None, None
    return valid_dates.dt.date.min(), valid_dates.dt.date.max()


def baseline_history_window(
    history: pd.DataFrame, date_to_predict, n_seasons: int = 2
) -> pd.DataFrame:
    """Games the historical-average baseline averages over.

    The games played before ``date_to_predict`` in its season and the
    ``n_seasons - 1`` before it: the span the prediction build loaded when it
    still built two seasons, so the baseline keeps its meaning now that the
    build loads the full history and returns only tonight's rows. ``history``
    needs GAME_DATE and TOTAL_POINTS (the build's layer inputs carry both).
    """
    date_to_predict = pd.Timestamp(date_to_predict).normalize()
    first_season = int(get_season_year_from_date(date_to_predict)) - (n_seasons - 1)
    game_dates = pd.to_datetime(history["GAME_DATE"], errors="coerce")
    start = pd.Timestamp(year=first_season, month=10, day=1)
    return history[(game_dates >= start) & (game_dates < date_to_predict)]


def _base_prediction_summary(
    df: pd.DataFrame,
    *,
    prediction_datetime: datetime | None,
    model_name: str,
    model_type: str,
    prediction_value_type: str,
    total_points_pick_line_col: str | None,
    train_date_min: object = None,
    train_date_max: object = None,
) -> tuple[pd.DataFrame, pd.Series]:
    if df.empty:
        raise ValueError("Cannot build baseline predictions for an empty dataframe.")
    if "GAME_ID" not in df.columns:
        raise ValueError("Column 'GAME_ID' is required for baseline predictions.")

    if total_points_pick_line_col is None:
        total_points_pick_line_col = total_line_over_col_raw()

    line_col = _resolve_column_name(df, total_points_pick_line_col)
    if line_col is None:
        raise ValueError(
            f"Column '{total_points_pick_line_col}' is required for baseline predictions."
        )

    prediction_datetime = _prediction_datetime_or_default(prediction_datetime)
    pick_line = pd.to_numeric(df[line_col], errors="coerce")

    df_summary = pd.DataFrame(index=df.index)
    df_summary["GAME_ID"] = df["GAME_ID"]
    df_summary["SEASON_TYPE"] = _series_or_none(df, "SEASON_TYPE")
    df_summary["GAME_DATE"] = _series_or_none(df, "GAME_DATE")
    df_summary["GAME_TIME"] = _series_or_none(df, "GAME_TIME")
    df_summary["TEAM_NAME_TEAM_HOME"] = _series_or_none(df, "TEAM_NAME_TEAM_HOME")
    df_summary["TEAM_NAME_TEAM_AWAY"] = _series_or_none(df, "TEAM_NAME_TEAM_AWAY")
    df_summary["PREDICTION_VALUE_TYPE"] = prediction_value_type
    df_summary["TOTAL_OVER_UNDER_LINE"] = pick_line
    df_summary["TOTAL_BET365_LINE_AT_PREDICTION"] = pick_line
    df_summary["NA_COLUMNS_COUNT"] = 0
    df_summary["NA_COLUMNS_NAMES"] = None

    df_summary["PREDICTION_DATETIME"] = prediction_datetime
    df_summary["PREDICTION_DATE"] = prediction_datetime.strftime("%Y-%m-%d %H:%M:%S")
    df_summary["HOME_PTS"] = None
    df_summary["AWAY_PTS"] = None

    df_summary = _add_time_to_match_minutes(df_summary, prediction_datetime)

    df_summary["MODEL_NAME"] = model_name
    df_summary["MODEL_TYPE"] = model_type
    df_summary["MODEL_VERSION"] = BASELINE_MODEL_VERSION
    df_summary["PREDICTION_SOURCE"] = model_name
    df_summary["TRAINING_CODE_TAG"] = BASELINE_TRAINING_CODE_TAG
    df_summary["TRAIN_DATE_MIN"] = train_date_min
    df_summary["TRAIN_DATE_MAX"] = train_date_max

    df_summary["SHAP_BASE_VALUE"] = None
    df_summary["SHAP_TOP_POSITIVE_FEATURES"] = None
    df_summary["SHAP_TOP_NEGATIVE_FEATURES"] = None
    df_summary["SHAP_DIRECTIONAL_CONFIDENCE"] = None
    df_summary["SHAP_SUPPORT_RATIO"] = None
    df_summary["SHAP_TOP_K_AGREEMENT"] = None
    df_summary["SHAP_CONFIDENCE_SCORE"] = None

    if "GAME_DATE" in df_summary.columns:
        df_summary["GAME_DATE"] = (
            df_summary["GAME_DATE"].astype(str).str.split("T").str[0]
        )
        df_summary.loc[
            df_summary["GAME_DATE"].isin(["None", "NaT", "nan"]), "GAME_DATE"
        ] = None

    return df_summary, pick_line


def build_random_line_error_baseline_predictions(
    df: pd.DataFrame,
    *,
    prediction_datetime: datetime | None = None,
    random_seed: int | None = None,
    total_points_pick_line_col: str | None = None,
) -> pd.DataFrame:
    """
    Build a random OVER/UNDER baseline prediction for each game.

    The stored prediction is a +/- 1 point line error, saved with DB value type
    DIFF_FROM_LINE to match the existing line-error model convention.
    """
    df_summary, pick_line = _base_prediction_summary(
        df,
        prediction_datetime=prediction_datetime,
        model_name=RANDOM_LINE_ERROR_BASELINE_MODEL_NAME,
        model_type="random_line_error_baseline",
        prediction_value_type=PREDICTION_VALUE_TYPE_LINE_ERROR,
        total_points_pick_line_col=total_points_pick_line_col,
    )

    rng = np.random.default_rng(random_seed)
    picks = rng.choice(["OVER", "UNDER"], size=len(df_summary))
    pred_line_error = pd.Series(
        np.where(picks == "OVER", 1.0, -1.0),
        index=df_summary.index,
        dtype="float64",
    )

    df_summary["PRED_LINE_ERROR"] = pred_line_error
    df_summary["PRED_TOTAL_POINTS"] = pick_line + pred_line_error
    df_summary["PRED_PICK"] = picks

    return df_summary.dropna(subset=["PRED_LINE_ERROR"])


def build_historical_average_line_error_baseline_predictions(
    df: pd.DataFrame,
    df_history: pd.DataFrame,
    *,
    prediction_datetime: datetime | None = None,
    total_points_pick_line_col: str | None = None,
) -> pd.DataFrame:
    """
    Predict every game with the historical average total score from df_history,
    stored as line error against the current line.
    """
    if "TOTAL_POINTS" not in df_history.columns:
        raise ValueError(
            "Column 'TOTAL_POINTS' is required in df_history for average baseline."
        )

    historical_total_points = pd.to_numeric(
        df_history["TOTAL_POINTS"], errors="coerce"
    ).dropna()
    if historical_total_points.empty:
        raise ValueError(
            "No non-null historical TOTAL_POINTS values available for average baseline."
        )

    average_total_points = float(historical_total_points.mean())
    train_date_min, train_date_max = _historical_game_date_bounds(df_history)
    df_summary, pick_line = _base_prediction_summary(
        df,
        prediction_datetime=prediction_datetime,
        model_name=HISTORICAL_AVERAGE_LINE_ERROR_BASELINE_MODEL_NAME,
        model_type="historical_average_line_error_baseline",
        prediction_value_type=PREDICTION_VALUE_TYPE_LINE_ERROR,
        total_points_pick_line_col=total_points_pick_line_col,
        train_date_min=train_date_min,
        train_date_max=train_date_max,
    )

    pred_total_points = pd.Series(
        average_total_points, index=df_summary.index, dtype="float64"
    )
    pred_line_error = pred_total_points - pick_line

    df_summary["PRED_TOTAL_POINTS"] = pred_total_points
    df_summary["PRED_LINE_ERROR"] = pred_line_error
    df_summary["PRED_PICK"] = np.select(
        [
            pred_total_points > pick_line,
            pred_total_points < pick_line,
            (pred_total_points == pick_line)
            & pred_total_points.notna()
            & pick_line.notna(),
        ],
        ["OVER", "UNDER", "PUSH"],
        default=None,
    )

    return df_summary.dropna(subset=["PRED_LINE_ERROR"])


def load_baseline_predictions_for_nba_games(
    *,
    df: pd.DataFrame,
    df_history: pd.DataFrame,
    prediction_datetime: datetime | None = None,
    random_seed: int | None = None,
    total_points_pick_line_col: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Build the baseline predictions, by baseline model name."""
    random_predictions = build_random_line_error_baseline_predictions(
        df,
        prediction_datetime=prediction_datetime,
        random_seed=random_seed,
        total_points_pick_line_col=total_points_pick_line_col,
    )
    average_predictions = build_historical_average_line_error_baseline_predictions(
        df,
        df_history,
        prediction_datetime=prediction_datetime,
        total_points_pick_line_col=total_points_pick_line_col,
    )

    predictions_by_model = {
        RANDOM_LINE_ERROR_BASELINE_MODEL_NAME: random_predictions,
        HISTORICAL_AVERAGE_LINE_ERROR_BASELINE_MODEL_NAME: average_predictions,
    }

    return predictions_by_model
