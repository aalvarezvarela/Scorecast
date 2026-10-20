from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.config.identity_columns import identifier_dtypes
from nba_ou.config.odds_columns import total_line_col
from nba_ou.modeling.scorers import over_under_betting_accuracy_error_line

try:
    from meta_learner_baselines import (
        BASE_AVG_ALL_6_ERR_COL,
        BASE_MAJORITY_LINE_ERROR_ONLY_ERR_COL,
        BASE_MAJORITY_TOTAL_ONLY_ERR_COL,
        BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL,
        add_default_meta_learner_baselines,
    )
except ModuleNotFoundError:
    from lab.meta_learner.meta_learner_baselines import (
        BASE_AVG_ALL_6_ERR_COL,
        BASE_MAJORITY_LINE_ERROR_ONLY_ERR_COL,
        BASE_MAJORITY_TOTAL_ONLY_ERR_COL,
        BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL,
        add_default_meta_learner_baselines,
    )

DATE_COL = "GAME_DATE"
GAME_ID_COL = "GAME_ID"
SEASON_COL = "SEASON_YEAR"
TARGET_ERROR_COL = "LINE_ERROR"
DEFAULT_ROLLING_WINDOW_DAYS = 5
DEFAULT_LINE_BUCKET_EDGES = [205.0, 215.0, 225.0, 235.0, 245.0]
DEFAULT_CONFIDENCE_BUCKET_EDGES = [1.0, 2.0, 3.0, 5.0]
DEFAULT_MARKET_ROLLING_WINDOWS = (3, 5, 10)
DEFAULT_MODEL_EDGE_THRESHOLDS = (1.0, 2.0, 3.0, 5.0)
DEFAULT_LINE_BUCKET_HISTORY_WINDOW_DAYS = 10
DEFAULT_CONF_BUCKET_HISTORY_WINDOW_DAYS = 10
DEFAULT_SIDE_EDGE_BUCKET_WINDOW_DAYS = 20
DEFAULT_CONSENSUS_TIGHT_BAND = 1.0


@dataclass(frozen=True)
class MetaLearnerFeatureBuildResult:
    dataframe: pd.DataFrame
    line_col: str
    total_model_cols: list[str]
    line_error_model_cols: list[str]
    error_cols: list[str]
    vote_feature_cols: list[str]
    reliability_feature_cols: list[str]
    baseline_cols: list[str]


def load_meta_learner_dataframe(
    csv_path: str | Path,
    *,
    date_col: str = DATE_COL,
    game_id_col: str = GAME_ID_COL,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    header_cols = pd.read_csv(csv_path, nrows=0).columns
    df = pd.read_csv(csv_path, dtype=identifier_dtypes(header_cols))
    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    sort_cols = [column for column in [date_col, game_id_col] if column in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return df


def resolve_prediction_column_groups(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    total_model_cols = [
        column for column in df.columns if column.startswith("PRED_TOTAL_POINTS_")
    ]
    line_error_model_cols = [
        column for column in df.columns if column.startswith("PRED_LINE_ERROR_")
    ]
    if not total_model_cols and not line_error_model_cols:
        raise KeyError("No meta-learner prediction columns were found in the dataframe.")
    return total_model_cols, line_error_model_cols


def resolve_line_column(
    df: pd.DataFrame,
    *,
    preferred_line_col: str | None = None,
) -> str:
    candidates = [
        column
        for column in [preferred_line_col, total_line_col(), "TOTAL_LINE_bet365"]
        if column
    ]
    for column in candidates:
        if column in df.columns:
            return column

    fallback_columns = [column for column in df.columns if column.startswith("TOTAL_LINE_")]
    if fallback_columns:
        return fallback_columns[0]

    raise KeyError("Could not resolve a total-line column from the dataframe.")


def build_error_space_prediction_frame(
    df: pd.DataFrame,
    *,
    total_model_cols: list[str],
    line_error_model_cols: list[str],
    line_col: str,
) -> pd.DataFrame:
    line_values = pd.to_numeric(df[line_col], errors="coerce")
    converted: dict[str, pd.Series] = {}

    for column in total_model_cols:
        converted[f"{column}__ERR"] = pd.to_numeric(df[column], errors="coerce") - line_values

    for column in line_error_model_cols:
        converted[f"{column}__ERR"] = pd.to_numeric(df[column], errors="coerce")

    return pd.DataFrame(converted, index=df.index)


def add_error_space_prediction_columns(
    df: pd.DataFrame,
    *,
    total_model_cols: list[str],
    line_error_model_cols: list[str],
    line_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    output = df.copy()
    error_space_df = build_error_space_prediction_frame(
        output,
        total_model_cols=total_model_cols,
        line_error_model_cols=line_error_model_cols,
        line_col=line_col,
    )
    output = pd.concat([output, error_space_df], axis=1)
    return output, error_space_df.columns.tolist()


def _shannon_entropy_from_counts(count_matrix: np.ndarray) -> np.ndarray:
    totals = count_matrix.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        count_matrix,
        totals,
        out=np.zeros_like(count_matrix, dtype=float),
        where=totals > 0,
    )
    log_probabilities = np.zeros_like(probabilities, dtype=float)
    positive_mask = probabilities > 0.0
    log_probabilities[positive_mask] = np.log2(probabilities[positive_mask])
    return -np.sum(probabilities * log_probabilities, axis=1)


def _safe_mean(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.nan
    return float(valid.mean())


def _safe_median(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.nan
    return float(np.median(valid))


def _safe_std(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.nan
    return float(np.std(valid, ddof=0))


def _safe_abs_mean(values: np.ndarray | pd.Series) -> float:
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.nan
    return float(np.mean(np.abs(valid)))


def _safe_mask_rate(
    mask: np.ndarray | pd.Series,
    *,
    denominator_mask: np.ndarray | pd.Series | None = None,
) -> float:
    resolved_mask = np.asarray(mask, dtype=bool)
    if denominator_mask is None:
        if resolved_mask.size == 0:
            return np.nan
        return float(np.mean(resolved_mask))

    resolved_denominator = np.asarray(denominator_mask, dtype=bool)
    if not np.any(resolved_denominator):
        return np.nan
    return float(np.mean(resolved_mask[resolved_denominator]))


def add_vote_summary_features(
    df: pd.DataFrame,
    *,
    error_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    output = df.copy()
    error_values = output[error_cols].apply(pd.to_numeric, errors="coerce")
    error_array = error_values.to_numpy(dtype=float)
    sign_array = np.sign(error_array)
    valid_mask = np.isfinite(error_array)

    valid_count = valid_mask.sum(axis=1)
    over_count = (sign_array > 0).sum(axis=1)
    under_count = (sign_array < 0).sum(axis=1)
    push_count = ((sign_array == 0) & valid_mask).sum(axis=1)

    vote_feature_cols = [
        "META_VALID_VOTE_COUNT",
        "META_OVER_VOTE_COUNT",
        "META_UNDER_VOTE_COUNT",
        "META_PUSH_VOTE_COUNT",
        "META_VOTE_PROP_OVER",
        "META_VOTE_PROP_UNDER",
        "META_VOTE_PROP_PUSH",
        "META_VOTE_MEAN_ERR",
        "META_VOTE_MEDIAN_ERR",
        "META_VOTE_STD_ERR",
        "META_VOTE_MIN_ERR",
        "META_VOTE_MAX_ERR",
        "META_VOTE_RANGE_ERR",
        "META_VOTE_ENTROPY",
    ]

    output["META_VALID_VOTE_COUNT"] = valid_count
    output["META_OVER_VOTE_COUNT"] = over_count
    output["META_UNDER_VOTE_COUNT"] = under_count
    output["META_PUSH_VOTE_COUNT"] = push_count
    output["META_VOTE_PROP_OVER"] = np.divide(
        over_count,
        valid_count,
        out=np.full(len(output), np.nan),
        where=valid_count > 0,
    )
    output["META_VOTE_PROP_UNDER"] = np.divide(
        under_count,
        valid_count,
        out=np.full(len(output), np.nan),
        where=valid_count > 0,
    )
    output["META_VOTE_PROP_PUSH"] = np.divide(
        push_count,
        valid_count,
        out=np.full(len(output), np.nan),
        where=valid_count > 0,
    )
    output["META_VOTE_MEAN_ERR"] = error_values.mean(axis=1)
    output["META_VOTE_MEDIAN_ERR"] = error_values.median(axis=1)
    output["META_VOTE_STD_ERR"] = error_values.std(axis=1, ddof=0)
    output["META_VOTE_MIN_ERR"] = error_values.min(axis=1)
    output["META_VOTE_MAX_ERR"] = error_values.max(axis=1)
    output["META_VOTE_RANGE_ERR"] = output["META_VOTE_MAX_ERR"] - output["META_VOTE_MIN_ERR"]
    output["META_VOTE_ENTROPY"] = _shannon_entropy_from_counts(
        np.column_stack([over_count, under_count, push_count])
    )

    no_vote_mask = valid_count == 0
    output.loc[no_vote_mask, vote_feature_cols] = np.nan
    return output, vote_feature_cols


def _bucket_codes(values: np.ndarray, inner_edges: list[float]) -> np.ndarray:
    bins = [-np.inf, *inner_edges, np.inf]
    bucket_codes = pd.cut(
        pd.Series(values, copy=False),
        bins=bins,
        labels=False,
        include_lowest=True,
    )
    return bucket_codes.to_numpy(dtype=float)


def _date_group_bounds(dates: pd.Series) -> list[tuple[int, int]]:
    date_days = pd.to_datetime(dates, errors="coerce").values.astype("datetime64[D]")
    if len(date_days) == 0:
        return []
    group_starts = np.flatnonzero(np.r_[True, date_days[1:] != date_days[:-1]])
    group_ends = np.r_[group_starts[1:], len(date_days)]
    return list(zip(group_starts, group_ends, strict=True))


def _compute_accuracy(y_true_error: np.ndarray, y_pred_error: np.ndarray) -> float:
    valid = np.isfinite(y_true_error) & np.isfinite(y_pred_error)
    if not np.any(valid):
        return np.nan

    true_side = np.sign(y_true_error[valid])
    pred_side = np.sign(y_pred_error[valid])
    eligible = (true_side != 0) & (pred_side != 0)
    if not np.any(eligible):
        return np.nan

    return float(np.mean(true_side[eligible] == pred_side[eligible]))


def _compute_mae(y_true_error: np.ndarray, y_pred_error: np.ndarray) -> float:
    valid = np.isfinite(y_true_error) & np.isfinite(y_pred_error)
    if not np.any(valid):
        return np.nan
    return float(np.mean(np.abs(y_pred_error[valid] - y_true_error[valid])))


def _compute_bias(y_true_error: np.ndarray, y_pred_error: np.ndarray) -> float:
    valid = np.isfinite(y_true_error) & np.isfinite(y_pred_error)
    if not np.any(valid):
        return np.nan
    return float(np.mean(y_pred_error[valid] - y_true_error[valid]))


def _compute_side_accuracy(
    y_true_error: np.ndarray,
    y_pred_error: np.ndarray,
    predicted_side: int,
) -> float:
    valid = np.isfinite(y_true_error) & np.isfinite(y_pred_error)
    if predicted_side not in (-1, 1) or not np.any(valid):
        return np.nan

    true_side = np.sign(y_true_error[valid])
    pred_side = np.sign(y_pred_error[valid])
    side_mask = (pred_side == predicted_side) & (true_side != 0)
    if not np.any(side_mask):
        return np.nan
    return float(np.mean(true_side[side_mask] == predicted_side))


def _compute_signed_return_mean(y_true_error: np.ndarray, y_pred_error: np.ndarray) -> float:
    valid = np.isfinite(y_true_error) & np.isfinite(y_pred_error)
    if not np.any(valid):
        return np.nan

    pred_side = np.sign(y_pred_error[valid])
    eligible = pred_side != 0
    if not np.any(eligible):
        return np.nan
    signed_return = pred_side[eligible] * y_true_error[valid][eligible]
    return float(np.mean(signed_return))


def _threshold_suffix(value: float) -> str:
    resolved = float(value)
    if resolved.is_integer():
        return str(int(resolved))
    return str(resolved).replace(".", "_")


def _history_slice_from_date_groups(
    date_groups: list[tuple[int, int]],
    group_idx: int,
    window_days: int,
) -> slice:
    if group_idx == 0:
        return slice(0, 0)
    history_group_start_idx = max(0, group_idx - window_days)
    history_start = date_groups[history_group_start_idx][0]
    history_end = date_groups[group_idx][0]
    return slice(history_start, history_end)


def add_consensus_structure_features(
    df: pd.DataFrame,
    *,
    error_cols: list[str],
    tight_band: float = DEFAULT_CONSENSUS_TIGHT_BAND,
) -> tuple[pd.DataFrame, list[str]]:
    output = df.copy()
    error_values = output[error_cols].apply(pd.to_numeric, errors="coerce")
    error_array = error_values.to_numpy(dtype=float)
    abs_error_array = np.abs(error_array)
    valid_mask = np.isfinite(error_array)
    valid_count = valid_mask.sum(axis=1)
    sign_array = np.sign(error_array)

    structure_feature_cols = [
        "META_MEAN_PAIRWISE_ABS_DIFF_ERR",
        "META_MAX_PAIRWISE_ABS_DIFF_ERR",
        "META_IQR_ERR",
        "META_OVER_UNDER_SPAN_ERR",
        "META_CONSENSUS_TIGHTNESS_RATIO",
        "META_CONSENSUS_SIGN_AGREE_COUNT",
        "META_CONSENSUS_SIGN_AGREE_RATE",
        "META_SIGN_MARGIN_MEAN_EDGE",
        "META_DISSENT_EDGE_GAP",
        "META_UNANIMOUS_FLAG",
    ]

    if len(error_cols) >= 2:
        pairwise_diffs = [
            np.abs(error_array[:, left] - error_array[:, right])
            for left, right in combinations(range(len(error_cols)), 2)
        ]
        pairwise_matrix = np.column_stack(pairwise_diffs)
        pairwise_df = pd.DataFrame(pairwise_matrix, index=output.index)
        output["META_MEAN_PAIRWISE_ABS_DIFF_ERR"] = pairwise_df.mean(axis=1)
        output["META_MAX_PAIRWISE_ABS_DIFF_ERR"] = pairwise_df.max(axis=1)
    else:
        output["META_MEAN_PAIRWISE_ABS_DIFF_ERR"] = np.nan
        output["META_MAX_PAIRWISE_ABS_DIFF_ERR"] = np.nan

    output["META_IQR_ERR"] = error_values.quantile(0.75, axis=1) - error_values.quantile(
        0.25, axis=1
    )

    strongest_over = (
        pd.DataFrame(np.where(error_array > 0.0, error_array, np.nan), index=output.index)
        .max(axis=1)
        .to_numpy(dtype=float)
    )
    strongest_under = (
        pd.DataFrame(np.where(error_array < 0.0, error_array, np.nan), index=output.index)
        .min(axis=1)
        .to_numpy(dtype=float)
    )
    output["META_OVER_UNDER_SPAN_ERR"] = strongest_over - strongest_under

    meta_mean = pd.to_numeric(output["META_VOTE_MEAN_ERR"], errors="coerce").to_numpy(dtype=float)
    within_band = (
        valid_mask
        & np.isfinite(meta_mean)[:, None]
        & (np.abs(error_array - meta_mean[:, None]) <= tight_band)
    )
    output["META_CONSENSUS_TIGHTNESS_RATIO"] = np.divide(
        within_band.sum(axis=1),
        valid_count,
        out=np.full(len(output), np.nan),
        where=valid_count > 0,
    )

    consensus_sign = np.sign(meta_mean)
    agree_with_consensus = (
        valid_mask
        & (consensus_sign[:, None] != 0)
        & (sign_array == consensus_sign[:, None])
    )
    output["META_CONSENSUS_SIGN_AGREE_COUNT"] = agree_with_consensus.sum(axis=1)
    output["META_CONSENSUS_SIGN_AGREE_RATE"] = np.divide(
        agree_with_consensus.sum(axis=1),
        valid_count,
        out=np.full(len(output), np.nan),
        where=valid_count > 0,
    )

    over_edges = np.where(error_array > 0.0, error_array, np.nan)
    under_edges = np.where(error_array < 0.0, np.abs(error_array), np.nan)
    over_edge_mean = pd.DataFrame(over_edges, index=output.index).mean(axis=1).to_numpy(dtype=float)
    under_edge_mean = (
        pd.DataFrame(under_edges, index=output.index).mean(axis=1).to_numpy(dtype=float)
    )
    output["META_SIGN_MARGIN_MEAN_EDGE"] = over_edge_mean - under_edge_mean
    output["META_DISSENT_EDGE_GAP"] = np.abs(over_edge_mean - under_edge_mean)

    unanimous_flag = (
        (output["META_OVER_VOTE_COUNT"] == output["META_VALID_VOTE_COUNT"])
        | (output["META_UNDER_VOTE_COUNT"] == output["META_VALID_VOTE_COUNT"])
        | (output["META_PUSH_VOTE_COUNT"] == output["META_VALID_VOTE_COUNT"])
    )
    output["META_UNANIMOUS_FLAG"] = unanimous_flag.astype(float)

    majority_sign = np.sign(
        pd.to_numeric(output["META_OVER_VOTE_COUNT"], errors="coerce").to_numpy(dtype=float)
        - pd.to_numeric(output["META_UNDER_VOTE_COUNT"], errors="coerce").to_numpy(dtype=float)
    )
    abs_rank_df = pd.DataFrame(abs_error_array, columns=error_cols, index=output.index).rank(
        axis=1,
        ascending=False,
        method="average",
    )

    meta_median = pd.to_numeric(
        output["META_VOTE_MEDIAN_ERR"], errors="coerce"
    ).to_numpy(dtype=float)
    for error_col in error_cols:
        values = pd.to_numeric(output[error_col], errors="coerce").to_numpy(dtype=float)
        signs = np.sign(values)
        same_sign_count = np.sum(sign_array == signs[:, None], axis=1)
        opposite_sign_count = np.sum(sign_array == (-signs)[:, None], axis=1)

        output[f"{error_col}_DISTANCE_TO_META_MEAN_ERR"] = np.abs(values - meta_mean)
        output[f"{error_col}_DISTANCE_TO_META_MEDIAN_ERR"] = np.abs(values - meta_median)
        output[f"{error_col}_MINUS_META_AVG_ERR"] = values - meta_mean
        output[f"{error_col}_MINUS_META_MEDIAN_ERR"] = values - meta_median
        output[f"{error_col}_AGREES_WITH_MAJORITY"] = np.where(
            (majority_sign != 0) & (signs != 0),
            (signs == majority_sign).astype(float),
            np.nan,
        )
        output[f"{error_col}_IS_ONLY_DISSENTER"] = np.where(
            (signs != 0) & (same_sign_count == 1) & (opposite_sign_count >= 1),
            1.0,
            0.0,
        )
        output[f"{error_col}_EDGE_MAGNITUDE_RANK"] = abs_rank_df[error_col].to_numpy(
            dtype=float
        )
        structure_feature_cols.extend(
            [
                f"{error_col}_DISTANCE_TO_META_MEAN_ERR",
                f"{error_col}_DISTANCE_TO_META_MEDIAN_ERR",
                f"{error_col}_MINUS_META_AVG_ERR",
                f"{error_col}_MINUS_META_MEDIAN_ERR",
                f"{error_col}_AGREES_WITH_MAJORITY",
                f"{error_col}_IS_ONLY_DISSENTER",
                f"{error_col}_EDGE_MAGNITUDE_RANK",
            ]
        )

    output.loc[valid_count == 0, structure_feature_cols] = np.nan
    return output, structure_feature_cols


def add_rolling_reliability_features(
    df: pd.DataFrame,
    *,
    error_cols: list[str],
    line_col: str,
    date_col: str = DATE_COL,
    target_error_col: str = TARGET_ERROR_COL,
    rolling_window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
    line_bucket_edges: list[float] | None = None,
    confidence_bucket_edges: list[float] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if rolling_window_days <= 0:
        raise ValueError("rolling_window_days must be greater than zero.")
    if date_col not in df.columns:
        raise KeyError(f"Missing required date column: {date_col}")
    if target_error_col not in df.columns:
        raise KeyError(f"Missing required target error column: {target_error_col}")
    if line_col not in df.columns:
        raise KeyError(f"Missing required line column: {line_col}")

    output = df.copy()
    line_bucket_edges = line_bucket_edges or list(DEFAULT_LINE_BUCKET_EDGES)
    confidence_bucket_edges = confidence_bucket_edges or list(
        DEFAULT_CONFIDENCE_BUCKET_EDGES
    )

    y_true = pd.to_numeric(output[target_error_col], errors="coerce").to_numpy(dtype=float)
    line_values = pd.to_numeric(output[line_col], errors="coerce").to_numpy(dtype=float)
    line_bucket_codes = _bucket_codes(line_values, line_bucket_edges)
    output["META_TOTAL_LINE_BUCKET"] = line_bucket_codes

    date_groups = _date_group_bounds(output[date_col])
    reliability_feature_cols: list[str] = ["META_TOTAL_LINE_BUCKET"]

    for error_col in error_cols:
        y_pred = pd.to_numeric(output[error_col], errors="coerce").to_numpy(dtype=float)
        conf_bucket_col = f"{error_col}_CONF_BUCKET"
        conf_bucket_codes = _bucket_codes(np.abs(y_pred), confidence_bucket_edges)
        output[conf_bucket_col] = conf_bucket_codes
        reliability_feature_cols.append(conf_bucket_col)

        feature_arrays = {
            f"{error_col}_ROLLING_HISTORY_N_{rolling_window_days}D": np.full(
                len(output), np.nan
            ),
            f"{error_col}_ROLLING_MAE_{rolling_window_days}D": np.full(len(output), np.nan),
            f"{error_col}_ROLLING_BETTING_ACC_{rolling_window_days}D": np.full(
                len(output), np.nan
            ),
            f"{error_col}_ROLLING_BIAS_{rolling_window_days}D": np.full(len(output), np.nan),
            f"{error_col}_ROLLING_LINE_BUCKET_N_{rolling_window_days}D": np.full(
                len(output), np.nan
            ),
            f"{error_col}_ROLLING_LINE_BUCKET_MAE_{rolling_window_days}D": np.full(
                len(output), np.nan
            ),
            f"{error_col}_ROLLING_LINE_BUCKET_ACC_{rolling_window_days}D": np.full(
                len(output), np.nan
            ),
            f"{error_col}_ROLLING_CONF_BUCKET_N_{rolling_window_days}D": np.full(
                len(output), np.nan
            ),
            f"{error_col}_ROLLING_CONF_BUCKET_MAE_{rolling_window_days}D": np.full(
                len(output), np.nan
            ),
            f"{error_col}_ROLLING_CONF_BUCKET_ACC_{rolling_window_days}D": np.full(
                len(output), np.nan
            ),
        }

        for group_idx, (start, end) in enumerate(date_groups):
            if group_idx == 0:
                history_slice = slice(0, 0)
            else:
                history_group_start_idx = max(0, group_idx - rolling_window_days)
                history_start = date_groups[history_group_start_idx][0]
                history_slice = slice(history_start, start)

            history_true = y_true[history_slice]
            history_pred = y_pred[history_slice]
            history_line_buckets = line_bucket_codes[history_slice]
            history_conf_buckets = conf_bucket_codes[history_slice]
            history_valid_mask = np.isfinite(history_true) & np.isfinite(history_pred)

            feature_arrays[f"{error_col}_ROLLING_HISTORY_N_{rolling_window_days}D"][
                start:end
            ] = float(history_valid_mask.sum())
            feature_arrays[f"{error_col}_ROLLING_MAE_{rolling_window_days}D"][start:end] = (
                _compute_mae(history_true, history_pred)
            )
            feature_arrays[
                f"{error_col}_ROLLING_BETTING_ACC_{rolling_window_days}D"
            ][start:end] = _compute_accuracy(history_true, history_pred)
            feature_arrays[f"{error_col}_ROLLING_BIAS_{rolling_window_days}D"][start:end] = (
                _compute_bias(history_true, history_pred)
            )

            for row_idx in range(start, end):
                current_line_bucket = line_bucket_codes[row_idx]
                if np.isfinite(current_line_bucket):
                    line_bucket_mask = history_valid_mask & (
                        history_line_buckets == current_line_bucket
                    )
                    feature_arrays[
                        f"{error_col}_ROLLING_LINE_BUCKET_N_{rolling_window_days}D"
                    ][row_idx] = float(line_bucket_mask.sum())
                    feature_arrays[
                        f"{error_col}_ROLLING_LINE_BUCKET_MAE_{rolling_window_days}D"
                    ][row_idx] = _compute_mae(
                        history_true[line_bucket_mask],
                        history_pred[line_bucket_mask],
                    )
                    feature_arrays[
                        f"{error_col}_ROLLING_LINE_BUCKET_ACC_{rolling_window_days}D"
                    ][row_idx] = _compute_accuracy(
                        history_true[line_bucket_mask],
                        history_pred[line_bucket_mask],
                    )

                current_conf_bucket = conf_bucket_codes[row_idx]
                if np.isfinite(current_conf_bucket):
                    conf_bucket_mask = history_valid_mask & (
                        history_conf_buckets == current_conf_bucket
                    )
                    feature_arrays[
                        f"{error_col}_ROLLING_CONF_BUCKET_N_{rolling_window_days}D"
                    ][row_idx] = float(conf_bucket_mask.sum())
                    feature_arrays[
                        f"{error_col}_ROLLING_CONF_BUCKET_MAE_{rolling_window_days}D"
                    ][row_idx] = _compute_mae(
                        history_true[conf_bucket_mask],
                        history_pred[conf_bucket_mask],
                    )
                    feature_arrays[
                        f"{error_col}_ROLLING_CONF_BUCKET_ACC_{rolling_window_days}D"
                    ][row_idx] = _compute_accuracy(
                        history_true[conf_bucket_mask],
                        history_pred[conf_bucket_mask],
                    )

        for column_name, values in feature_arrays.items():
            output[column_name] = values
            reliability_feature_cols.append(column_name)

    return output, reliability_feature_cols


def _build_daily_feature_frame(
    df: pd.DataFrame,
    *,
    error_cols: list[str],
    line_col: str,
    date_col: str,
    target_error_col: str,
    line_bucket_edges: list[float],
    model_edge_thresholds: tuple[float, ...] = DEFAULT_MODEL_EDGE_THRESHOLDS,
) -> pd.DataFrame:
    resolved_dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    date_groups = _date_group_bounds(resolved_dates)

    y_true = pd.to_numeric(df[target_error_col], errors="coerce").to_numpy(dtype=float)
    line_values = pd.to_numeric(df[line_col], errors="coerce").to_numpy(dtype=float)
    meta_mean = pd.to_numeric(df["META_VOTE_MEAN_ERR"], errors="coerce").to_numpy(dtype=float)
    vote_entropy = pd.to_numeric(df["META_VOTE_ENTROPY"], errors="coerce").to_numpy(dtype=float)
    unanimous_flag = pd.to_numeric(
        df["META_UNANIMOUS_FLAG"], errors="coerce"
    ).to_numpy(dtype=float)
    prediction_arrays = {
        error_col: pd.to_numeric(df[error_col], errors="coerce").to_numpy(dtype=float)
        for error_col in error_cols
    }

    low_total_threshold = (
        line_bucket_edges[1] if len(line_bucket_edges) >= 2 else line_bucket_edges[0]
    )
    high_total_threshold = (
        line_bucket_edges[-2] if len(line_bucket_edges) >= 2 else line_bucket_edges[-1]
    )

    records: list[dict[str, float | pd.Timestamp]] = []
    for start, end in date_groups:
        date_value = resolved_dates.iloc[start]
        day_true = y_true[start:end]
        day_line = line_values[start:end]
        day_meta_mean = meta_mean[start:end]
        day_vote_entropy = vote_entropy[start:end]
        day_unanimous = unanimous_flag[start:end]

        valid_true = np.isfinite(day_true)
        valid_line = np.isfinite(day_line)

        record: dict[str, float | pd.Timestamp] = {
            date_col: date_value,
            "DAY_MEAN_LINE_ERROR": _safe_mean(day_true),
            "DAY_MEDIAN_LINE_ERROR": _safe_median(day_true),
            "DAY_STD_LINE_ERROR": _safe_std(day_true),
            "DAY_MAE_LINE_ERROR": _safe_abs_mean(day_true),
            "DAY_OVER_RATE": _safe_mask_rate(day_true > 0.0, denominator_mask=valid_true),
            "DAY_UNDER_RATE": _safe_mask_rate(day_true < 0.0, denominator_mask=valid_true),
            "DAY_PUSH_RATE": _safe_mask_rate(day_true == 0.0, denominator_mask=valid_true),
            "DAY_GAME_COUNT": float(valid_true.sum()),
            "DAY_MEAN_TOTAL_LINE": _safe_mean(day_line),
            "CUR_DAY_GAME_COUNT": float(end - start),
            "CUR_DAY_MEAN_TOTAL_LINE": _safe_mean(day_line),
            "CUR_DAY_STD_TOTAL_LINE": _safe_std(day_line),
            "CUR_DAY_MIN_TOTAL_LINE": float(np.nanmin(day_line))
            if np.any(valid_line)
            else np.nan,
            "CUR_DAY_MAX_TOTAL_LINE": float(np.nanmax(day_line))
            if np.any(valid_line)
            else np.nan,
            "CUR_DAY_HIGH_TOTAL_RATE": _safe_mask_rate(
                day_line >= high_total_threshold,
                denominator_mask=valid_line,
            ),
            "CUR_DAY_LOW_TOTAL_RATE": _safe_mask_rate(
                day_line <= low_total_threshold,
                denominator_mask=valid_line,
            ),
            "CUR_DAY_META_MEAN_EDGE": _safe_mean(day_meta_mean),
            "CUR_DAY_UNANIMOUS_RATE": _safe_mean(day_unanimous),
            "CUR_DAY_MEAN_VOTE_ENTROPY": _safe_mean(day_vote_entropy),
            "META_DAY_BASELINE_ACC": _compute_accuracy(day_true, day_meta_mean),
            "META_DAY_BASELINE_MAE": _compute_mae(day_true, day_meta_mean),
            "META_DAY_MEAN_ABS_CONSENSUS_EDGE": _safe_mean(np.abs(day_meta_mean)),
            "META_DAY_MEAN_VOTE_ENTROPY": _safe_mean(day_vote_entropy),
            "META_DAY_UNANIMOUS_RATE": _safe_mean(day_unanimous),
        }

        for error_col in error_cols:
            day_pred = prediction_arrays[error_col][start:end]
            valid_pred = np.isfinite(day_pred)
            valid_pair = valid_true & valid_pred

            record[f"{error_col}__DAY_MAE"] = _compute_mae(day_true, day_pred)
            record[f"{error_col}__DAY_ACC"] = _compute_accuracy(day_true, day_pred)
            record[f"{error_col}__DAY_BIAS"] = _compute_bias(day_true, day_pred)
            record[f"{error_col}__DAY_MEAN_EDGE"] = _safe_mean(day_pred)
            record[f"{error_col}__DAY_MEAN_ABS_EDGE"] = _safe_mean(np.abs(day_pred))
            record[f"{error_col}__DAY_STD_EDGE"] = _safe_std(day_pred)
            record[f"{error_col}__DAY_OVER_CALL_RATE"] = _safe_mask_rate(
                day_pred > 0.0,
                denominator_mask=valid_pred,
            )
            record[f"{error_col}__DAY_UNDER_CALL_RATE"] = _safe_mask_rate(
                day_pred < 0.0,
                denominator_mask=valid_pred,
            )
            record[f"{error_col}__DAY_PUSH_RATE"] = _safe_mask_rate(
                day_pred == 0.0,
                denominator_mask=valid_pred,
            )
            record[f"{error_col}__DAY_OVER_CALL_ACC"] = _compute_side_accuracy(
                day_true,
                day_pred,
                predicted_side=1,
            )
            record[f"{error_col}__DAY_UNDER_CALL_ACC"] = _compute_side_accuracy(
                day_true,
                day_pred,
                predicted_side=-1,
            )
            record[f"{error_col}__DAY_ELIGIBLE_N"] = float(
                np.sum(valid_pair & (np.sign(day_true) != 0) & (np.sign(day_pred) != 0))
            )
            record[f"CUR_DAY_{error_col}_MEAN_EDGE"] = _safe_mean(day_pred)
            record[f"CUR_DAY_BOARD_OVER_RATE_BY_{error_col}"] = _safe_mask_rate(
                day_pred > 0.0,
                denominator_mask=valid_pred,
            )
            record[f"CUR_DAY_BOARD_UNDER_RATE_BY_{error_col}"] = _safe_mask_rate(
                day_pred < 0.0,
                denominator_mask=valid_pred,
            )

            for threshold in model_edge_thresholds:
                suffix = _threshold_suffix(threshold)
                record[f"{error_col}__DAY_EDGE_GT_{suffix}_RATE"] = _safe_mask_rate(
                    np.abs(day_pred) > threshold,
                    denominator_mask=valid_pred,
                )

        records.append(record)

    return pd.DataFrame.from_records(records).set_index(date_col)


def _add_daily_context_features(
    df: pd.DataFrame,
    *,
    error_cols: list[str],
    line_col: str,
    date_col: str,
    target_error_col: str,
    line_bucket_edges: list[float],
) -> tuple[pd.DataFrame, list[str]]:
    output = df.copy()
    daily_df = _build_daily_feature_frame(
        output,
        error_cols=error_cols,
        line_col=line_col,
        date_col=date_col,
        target_error_col=target_error_col,
        line_bucket_edges=line_bucket_edges,
    )

    date_feature_map: dict[str, pd.Series] = {}
    feature_cols: list[str] = []

    current_day_cols = [column for column in daily_df.columns if column.startswith("CUR_DAY_")]
    for column in current_day_cols:
        date_feature_map[column] = daily_df[column]
    feature_cols.extend(current_day_cols)

    market_daily_specs = {
        "DAY_MEAN_LINE_ERROR": ("MEAN_LINE_ERROR", "mean"),
        "DAY_MEDIAN_LINE_ERROR": ("MEDIAN_LINE_ERROR", "median"),
        "DAY_STD_LINE_ERROR": ("STD_LINE_ERROR", "mean"),
        "DAY_MAE_LINE_ERROR": ("MAE", "mean"),
        "DAY_OVER_RATE": ("OVER_RATE", "mean"),
        "DAY_UNDER_RATE": ("UNDER_RATE", "mean"),
        "DAY_PUSH_RATE": ("PUSH_RATE", "mean"),
        "DAY_GAME_COUNT": ("GAME_COUNT", "mean"),
        "DAY_MEAN_TOTAL_LINE": ("MEAN_TOTAL_LINE", "mean"),
    }

    for raw_column, (suffix, rolling_operation) in market_daily_specs.items():
        source = daily_df[raw_column]
        for lag in (1, 2, 3):
            prefix = "PREV_DAY" if lag == 1 else f"PREV_{lag}DAY"
            feature_name = f"{prefix}_{suffix}"
            date_feature_map[feature_name] = source.shift(lag)
            feature_cols.append(feature_name)

        for avg_window in (2, 3):
            feature_name = f"PREV_{avg_window}DAY_AVG_{suffix}"
            date_feature_map[feature_name] = source.shift(1).rolling(
                avg_window,
                min_periods=1,
            ).mean()
            feature_cols.append(feature_name)

        for window in DEFAULT_MARKET_ROLLING_WINDOWS:
            rolling = source.shift(1).rolling(window, min_periods=1)
            if rolling_operation == "mean":
                values = rolling.mean()
            elif rolling_operation == "median":
                values = rolling.median()
            else:
                raise ValueError(f"Unsupported rolling operation: {rolling_operation}")
            feature_name = f"MARKET_ROLL_{suffix}_{window}D"
            date_feature_map[feature_name] = values
            feature_cols.append(feature_name)

    date_feature_map["MARKET_LINE_ERROR_TREND_3D_MINUS_10D"] = (
        date_feature_map["MARKET_ROLL_MEAN_LINE_ERROR_3D"]
        - date_feature_map["MARKET_ROLL_MEAN_LINE_ERROR_10D"]
    )
    date_feature_map["MARKET_OVER_RATE_TREND_3D_MINUS_10D"] = (
        date_feature_map["MARKET_ROLL_OVER_RATE_3D"]
        - date_feature_map["MARKET_ROLL_OVER_RATE_10D"]
    )
    date_feature_map["MARKET_VOL_SHIFT_3D_MINUS_10D"] = (
        date_feature_map["MARKET_ROLL_STD_LINE_ERROR_3D"]
        - date_feature_map["MARKET_ROLL_STD_LINE_ERROR_10D"]
    )
    feature_cols.extend(
        [
            "MARKET_LINE_ERROR_TREND_3D_MINUS_10D",
            "MARKET_OVER_RATE_TREND_3D_MINUS_10D",
            "MARKET_VOL_SHIFT_3D_MINUS_10D",
        ]
    )

    date_feature_map["META_BASELINE_ACC_5D"] = daily_df["META_DAY_BASELINE_ACC"].shift(1).rolling(
        5,
        min_periods=1,
    ).mean()
    date_feature_map["META_BASELINE_MAE_5D"] = daily_df["META_DAY_BASELINE_MAE"].shift(1).rolling(
        5,
        min_periods=1,
    ).mean()
    date_feature_map["META_ROLL_MEAN_VOTE_ENTROPY_5D"] = daily_df[
        "META_DAY_MEAN_VOTE_ENTROPY"
    ].shift(1).rolling(5, min_periods=1).mean()
    date_feature_map["META_ROLL_UNANIMOUS_RATE_10D"] = daily_df[
        "META_DAY_UNANIMOUS_RATE"
    ].shift(1).rolling(10, min_periods=1).mean()
    date_feature_map["META_ROLL_MEAN_ABS_CONSENSUS_EDGE_5D"] = daily_df[
        "META_DAY_MEAN_ABS_CONSENSUS_EDGE"
    ].shift(1).rolling(5, min_periods=1).mean()
    feature_cols.extend(
        [
            "META_BASELINE_ACC_5D",
            "META_BASELINE_MAE_5D",
            "META_ROLL_MEAN_VOTE_ENTROPY_5D",
            "META_ROLL_UNANIMOUS_RATE_10D",
            "META_ROLL_MEAN_ABS_CONSENSUS_EDGE_5D",
        ]
    )

    for error_col in error_cols:
        prev_day_specs = {
            f"{error_col}__DAY_ACC": f"{error_col}_PREV_DAY_ACC",
            f"{error_col}__DAY_MAE": f"{error_col}_PREV_DAY_MAE",
            f"{error_col}__DAY_BIAS": f"{error_col}_PREV_DAY_BIAS",
            f"{error_col}__DAY_MEAN_ABS_EDGE": f"{error_col}_PREV_DAY_MEAN_ABS_EDGE",
            f"{error_col}__DAY_MEAN_EDGE": f"{error_col}_PREV_DAY_MEAN_EDGE",
            f"{error_col}__DAY_OVER_CALL_ACC": f"{error_col}_PREV_DAY_OVER_CALL_ACC",
            f"{error_col}__DAY_UNDER_CALL_ACC": f"{error_col}_PREV_DAY_UNDER_CALL_ACC",
            f"{error_col}__DAY_ELIGIBLE_N": f"{error_col}_PREV_DAY_ELIGIBLE_N",
        }
        for raw_column, feature_name in prev_day_specs.items():
            date_feature_map[feature_name] = daily_df[raw_column].shift(1)
            feature_cols.append(feature_name)

        rolling_specs = {
            f"{error_col}__DAY_MEAN_ABS_EDGE": [(5, f"{error_col}_ROLL_MEAN_ABS_EDGE_5D")],
            f"{error_col}__DAY_MEAN_EDGE": [(5, f"{error_col}_ROLL_MEAN_EDGE_5D")],
            f"{error_col}__DAY_STD_EDGE": [(5, f"{error_col}_ROLL_STD_EDGE_5D")],
            f"{error_col}__DAY_OVER_CALL_RATE": [(10, f"{error_col}_ROLL_OVER_CALL_RATE_10D")],
            f"{error_col}__DAY_UNDER_CALL_RATE": [(10, f"{error_col}_ROLL_UNDER_CALL_RATE_10D")],
            f"{error_col}__DAY_PUSH_RATE": [(10, f"{error_col}_ROLL_PUSH_RATE_10D")],
            f"{error_col}__DAY_OVER_CALL_ACC": [(10, f"{error_col}_ROLL_ACC_WHEN_OVER_10D")],
            f"{error_col}__DAY_UNDER_CALL_ACC": [(10, f"{error_col}_ROLL_ACC_WHEN_UNDER_10D")],
        }
        for threshold in DEFAULT_MODEL_EDGE_THRESHOLDS:
            suffix = _threshold_suffix(threshold)
            rolling_specs[f"{error_col}__DAY_EDGE_GT_{suffix}_RATE"] = [
                (10, f"{error_col}_ROLL_EDGE_GT_{suffix}_RATE_10D")
            ]

        for raw_column, column_specs in rolling_specs.items():
            for window, feature_name in column_specs:
                date_feature_map[feature_name] = daily_df[raw_column].shift(1).rolling(
                    window,
                    min_periods=1,
                ).mean()
                feature_cols.append(feature_name)

    date_features = pd.DataFrame(date_feature_map, index=daily_df.index)
    mapped_feature_df = pd.DataFrame(
        {
            column: output[date_col].map(date_features[column])
            for column in feature_cols
        },
        index=output.index,
    )
    output = pd.concat([output, mapped_feature_df], axis=1)

    return output, feature_cols


def _build_season_feature_arrays(
    df: pd.DataFrame,
    *,
    line_col: str,
    date_col: str,
    season_col: str,
    target_error_col: str,
) -> tuple[dict[str, np.ndarray], list[str]]:
    feature_names = [
        "SEASON_TO_DATE_OVER_RATE_PRIOR",
        "SEASON_TO_DATE_UNDER_RATE_PRIOR",
        "SEASON_TO_DATE_PUSH_RATE_PRIOR",
        "SEASON_TO_DATE_MEAN_LINE_ERROR_PRIOR",
        "SEASON_TO_DATE_MEDIAN_LINE_ERROR_PRIOR",
        "SEASON_TO_DATE_STD_LINE_ERROR_PRIOR",
        "SEASON_TO_DATE_MAE_PRIOR",
        "SEASON_TO_DATE_GAME_COUNT_PRIOR",
        "TOTAL_LINE_MINUS_SEASON_AVG_PRIOR",
        "TOTAL_LINE_ZSCORE_WITHIN_SEASON_PRIOR",
        "TOTAL_LINE_PERCENTILE_WITHIN_SEASON_PRIOR",
    ]
    arrays = {name: np.full(len(df), np.nan) for name in feature_names}

    if season_col not in df.columns:
        return arrays, []

    resolved_dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    y_true = pd.to_numeric(df[target_error_col], errors="coerce").to_numpy(dtype=float)
    line_values = pd.to_numeric(df[line_col], errors="coerce").to_numpy(dtype=float)

    for _, season_idx in df.groupby(season_col, sort=False).groups.items():
        season_positions = np.asarray(season_idx, dtype=int)
        local_dates = resolved_dates.iloc[season_positions].reset_index(drop=True)
        local_groups = _date_group_bounds(local_dates)

        prior_errors = np.array([], dtype=float)
        prior_lines = np.array([], dtype=float)
        cumulative_sum = 0.0
        cumulative_sq_sum = 0.0
        cumulative_abs_sum = 0.0
        cumulative_count = 0
        cumulative_over = 0
        cumulative_under = 0
        cumulative_push = 0

        for start_local, end_local in local_groups:
            row_positions = season_positions[start_local:end_local]

            if cumulative_count > 0:
                arrays["SEASON_TO_DATE_OVER_RATE_PRIOR"][row_positions] = (
                    cumulative_over / cumulative_count
                )
                arrays["SEASON_TO_DATE_UNDER_RATE_PRIOR"][row_positions] = (
                    cumulative_under / cumulative_count
                )
                arrays["SEASON_TO_DATE_PUSH_RATE_PRIOR"][row_positions] = (
                    cumulative_push / cumulative_count
                )
                arrays["SEASON_TO_DATE_MEAN_LINE_ERROR_PRIOR"][row_positions] = (
                    cumulative_sum / cumulative_count
                )
                arrays["SEASON_TO_DATE_MEDIAN_LINE_ERROR_PRIOR"][row_positions] = float(
                    np.median(prior_errors)
                )
                mean_error = cumulative_sum / cumulative_count
                arrays["SEASON_TO_DATE_STD_LINE_ERROR_PRIOR"][row_positions] = float(
                    np.sqrt(max((cumulative_sq_sum / cumulative_count) - mean_error**2, 0.0))
                )
                arrays["SEASON_TO_DATE_MAE_PRIOR"][row_positions] = (
                    cumulative_abs_sum / cumulative_count
                )
                arrays["SEASON_TO_DATE_GAME_COUNT_PRIOR"][row_positions] = float(
                    cumulative_count
                )

                prior_line_mean = float(np.mean(prior_lines))
                prior_line_std = float(np.std(prior_lines, ddof=0))
                current_lines = line_values[row_positions]
                arrays["TOTAL_LINE_MINUS_SEASON_AVG_PRIOR"][row_positions] = (
                    current_lines - prior_line_mean
                )
                if prior_line_std > 0.0:
                    arrays["TOTAL_LINE_ZSCORE_WITHIN_SEASON_PRIOR"][row_positions] = (
                        current_lines - prior_line_mean
                    ) / prior_line_std

                sorted_prior_lines = np.sort(prior_lines)
                valid_current_lines = np.isfinite(current_lines)
                if np.any(valid_current_lines):
                    percentiles = np.searchsorted(
                        sorted_prior_lines,
                        current_lines[valid_current_lines],
                        side="right",
                    ) / float(sorted_prior_lines.size)
                    arrays["TOTAL_LINE_PERCENTILE_WITHIN_SEASON_PRIOR"][
                        row_positions[valid_current_lines]
                    ] = percentiles

            day_errors = y_true[row_positions]
            day_lines = line_values[row_positions]
            valid_day_errors = day_errors[np.isfinite(day_errors)]
            valid_day_lines = day_lines[np.isfinite(day_lines)]

            if valid_day_errors.size > 0:
                cumulative_count += int(valid_day_errors.size)
                cumulative_sum += float(valid_day_errors.sum())
                cumulative_sq_sum += float(np.square(valid_day_errors).sum())
                cumulative_abs_sum += float(np.abs(valid_day_errors).sum())
                cumulative_over += int(np.sum(valid_day_errors > 0.0))
                cumulative_under += int(np.sum(valid_day_errors < 0.0))
                cumulative_push += int(np.sum(valid_day_errors == 0.0))
                prior_errors = np.concatenate([prior_errors, valid_day_errors])

            if valid_day_lines.size > 0:
                prior_lines = np.concatenate([prior_lines, valid_day_lines])

    return arrays, feature_names


def add_historical_context_features(
    df: pd.DataFrame,
    *,
    error_cols: list[str],
    line_col: str,
    date_col: str = DATE_COL,
    season_col: str = SEASON_COL,
    target_error_col: str = TARGET_ERROR_COL,
    line_bucket_edges: list[float] | None = None,
    confidence_bucket_edges: list[float] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if date_col not in df.columns:
        raise KeyError(f"Missing required date column: {date_col}")
    if target_error_col not in df.columns:
        raise KeyError(f"Missing required target error column: {target_error_col}")
    if line_col not in df.columns:
        raise KeyError(f"Missing required line column: {line_col}")

    output = df.copy()
    line_bucket_edges = line_bucket_edges or list(DEFAULT_LINE_BUCKET_EDGES)
    confidence_bucket_edges = confidence_bucket_edges or list(
        DEFAULT_CONFIDENCE_BUCKET_EDGES
    )

    output, daily_context_cols = _add_daily_context_features(
        output,
        error_cols=error_cols,
        line_col=line_col,
        date_col=date_col,
        target_error_col=target_error_col,
        line_bucket_edges=line_bucket_edges,
    )

    season_feature_arrays, season_feature_cols = _build_season_feature_arrays(
        output,
        line_col=line_col,
        date_col=date_col,
        season_col=season_col,
        target_error_col=target_error_col,
    )
    season_feature_df = pd.DataFrame(season_feature_arrays, index=output.index)

    derived_feature_cols = [
        "PREV_DAY_MEAN_LINE_ERROR_MINUS_SEASON_MEAN",
        "MARKET_ROLL_OVER_RATE_5D_MINUS_SEASON_OVER_RATE",
        "MARKET_ROLL_STD_LINE_ERROR_5D_MINUS_SEASON_STD_LINE_ERROR_PRIOR",
        "TOTAL_LINE_MINUS_PREV_DAY_AVG_LINE",
    ]
    derived_feature_df = pd.DataFrame(
        {
            "PREV_DAY_MEAN_LINE_ERROR_MINUS_SEASON_MEAN": (
                pd.to_numeric(output["PREV_DAY_MEAN_LINE_ERROR"], errors="coerce")
                - pd.to_numeric(
                    season_feature_df["SEASON_TO_DATE_MEAN_LINE_ERROR_PRIOR"],
                    errors="coerce",
                )
            ),
            "MARKET_ROLL_OVER_RATE_5D_MINUS_SEASON_OVER_RATE": (
                pd.to_numeric(output["MARKET_ROLL_OVER_RATE_5D"], errors="coerce")
                - pd.to_numeric(
                    season_feature_df["SEASON_TO_DATE_OVER_RATE_PRIOR"],
                    errors="coerce",
                )
            ),
            "MARKET_ROLL_STD_LINE_ERROR_5D_MINUS_SEASON_STD_LINE_ERROR_PRIOR": (
                pd.to_numeric(output["MARKET_ROLL_STD_LINE_ERROR_5D"], errors="coerce")
                - pd.to_numeric(
                    season_feature_df["SEASON_TO_DATE_STD_LINE_ERROR_PRIOR"],
                    errors="coerce",
                )
            ),
            "TOTAL_LINE_MINUS_PREV_DAY_AVG_LINE": (
                pd.to_numeric(output[line_col], errors="coerce")
                - pd.to_numeric(output["PREV_DAY_MEAN_TOTAL_LINE"], errors="coerce")
            ),
        },
        index=output.index,
    )

    date_groups = _date_group_bounds(output[date_col])
    y_true = pd.to_numeric(output[target_error_col], errors="coerce").to_numpy(dtype=float)
    line_bucket_codes = pd.to_numeric(
        output["META_TOTAL_LINE_BUCKET"], errors="coerce"
    ).to_numpy(dtype=float)

    row_level_feature_arrays: dict[str, np.ndarray] = {
        "MARKET_LINE_BUCKET_HISTORY_N_10D": np.full(len(output), np.nan),
        "MARKET_LINE_BUCKET_MEAN_LINE_ERROR_10D": np.full(len(output), np.nan),
        "MARKET_LINE_BUCKET_OVER_RATE_10D": np.full(len(output), np.nan),
        "MARKET_LINE_BUCKET_MAE_10D": np.full(len(output), np.nan),
    }
    row_level_feature_cols = list(row_level_feature_arrays)

    for error_col in error_cols:
        for feature_name in [
            f"{error_col}_ROLLING_CONF_BUCKET_N_10D",
            f"{error_col}_ROLLING_CONF_BUCKET_ACC_10D",
            f"{error_col}_ROLLING_CONF_BUCKET_BIAS_10D",
            f"{error_col}_ROLLING_CONF_BUCKET_SIGNED_RETURN_10D",
            f"{error_col}_CURRENT_SIDE_EDGE_BUCKET_N_20D",
            f"{error_col}_CURRENT_SIDE_EDGE_BUCKET_ACC_20D",
            f"{error_col}_CURRENT_SIDE_EDGE_BUCKET_SIGNED_RETURN_20D",
        ]:
            row_level_feature_arrays[feature_name] = np.full(len(output), np.nan)
            row_level_feature_cols.append(feature_name)

    for group_idx, (start, end) in enumerate(date_groups):
        market_history_slice = _history_slice_from_date_groups(
            date_groups,
            group_idx,
            DEFAULT_LINE_BUCKET_HISTORY_WINDOW_DAYS,
        )
        history_true_market = y_true[market_history_slice]
        history_bucket_codes = line_bucket_codes[market_history_slice]
        history_true_valid = np.isfinite(history_true_market)

        for row_idx in range(start, end):
            current_bucket = line_bucket_codes[row_idx]
            if np.isfinite(current_bucket):
                bucket_mask = history_true_valid & (history_bucket_codes == current_bucket)
                row_level_feature_arrays["MARKET_LINE_BUCKET_HISTORY_N_10D"][row_idx] = float(
                    bucket_mask.sum()
                )
                row_level_feature_arrays["MARKET_LINE_BUCKET_MEAN_LINE_ERROR_10D"][
                    row_idx
                ] = _safe_mean(history_true_market[bucket_mask])
                row_level_feature_arrays["MARKET_LINE_BUCKET_OVER_RATE_10D"][
                    row_idx
                ] = _safe_mask_rate(history_true_market > 0.0, denominator_mask=bucket_mask)
                row_level_feature_arrays["MARKET_LINE_BUCKET_MAE_10D"][row_idx] = _safe_abs_mean(
                    history_true_market[bucket_mask]
                )

        for error_col in error_cols:
            y_pred = pd.to_numeric(output[error_col], errors="coerce").to_numpy(dtype=float)
            conf_bucket_codes = pd.to_numeric(
                output[f"{error_col}_CONF_BUCKET"], errors="coerce"
            ).to_numpy(dtype=float)

            conf_history_slice = _history_slice_from_date_groups(
                date_groups,
                group_idx,
                DEFAULT_CONF_BUCKET_HISTORY_WINDOW_DAYS,
            )
            side_bucket_history_slice = _history_slice_from_date_groups(
                date_groups,
                group_idx,
                DEFAULT_SIDE_EDGE_BUCKET_WINDOW_DAYS,
            )

            history_true_conf = y_true[conf_history_slice]
            history_pred_conf = y_pred[conf_history_slice]
            history_conf_codes = conf_bucket_codes[conf_history_slice]
            history_conf_valid = np.isfinite(history_true_conf) & np.isfinite(history_pred_conf)

            history_true_side = y_true[side_bucket_history_slice]
            history_pred_side = y_pred[side_bucket_history_slice]
            history_side_codes = conf_bucket_codes[side_bucket_history_slice]
            history_side_valid = np.isfinite(history_true_side) & np.isfinite(history_pred_side)

            for row_idx in range(start, end):
                current_bucket = conf_bucket_codes[row_idx]
                current_side = np.sign(y_pred[row_idx])

                if np.isfinite(current_bucket):
                    conf_mask = history_conf_valid & (history_conf_codes == current_bucket)
                    row_level_feature_arrays[f"{error_col}_ROLLING_CONF_BUCKET_N_10D"][
                        row_idx
                    ] = float(conf_mask.sum())
                    row_level_feature_arrays[f"{error_col}_ROLLING_CONF_BUCKET_ACC_10D"][
                        row_idx
                    ] = _compute_accuracy(
                        history_true_conf[conf_mask],
                        history_pred_conf[conf_mask],
                    )
                    row_level_feature_arrays[f"{error_col}_ROLLING_CONF_BUCKET_BIAS_10D"][
                        row_idx
                    ] = _compute_bias(
                        history_true_conf[conf_mask],
                        history_pred_conf[conf_mask],
                    )
                    row_level_feature_arrays[
                        f"{error_col}_ROLLING_CONF_BUCKET_SIGNED_RETURN_10D"
                    ][row_idx] = _compute_signed_return_mean(
                        history_true_conf[conf_mask],
                        history_pred_conf[conf_mask],
                    )

                    if current_side != 0.0 and np.isfinite(current_side):
                        side_mask = (
                            history_side_valid
                            & (history_side_codes == current_bucket)
                            & (np.sign(history_pred_side) == current_side)
                        )
                        row_level_feature_arrays[
                            f"{error_col}_CURRENT_SIDE_EDGE_BUCKET_N_20D"
                        ][row_idx] = float(side_mask.sum())
                        row_level_feature_arrays[
                            f"{error_col}_CURRENT_SIDE_EDGE_BUCKET_ACC_20D"
                        ][row_idx] = _compute_accuracy(
                            history_true_side[side_mask],
                            history_pred_side[side_mask],
                        )
                        row_level_feature_arrays[
                            f"{error_col}_CURRENT_SIDE_EDGE_BUCKET_SIGNED_RETURN_20D"
                        ][row_idx] = _compute_signed_return_mean(
                            history_true_side[side_mask],
                            history_pred_side[side_mask],
                        )

    row_level_feature_df = pd.DataFrame(row_level_feature_arrays, index=output.index)
    output = pd.concat(
        [output, season_feature_df, derived_feature_df, row_level_feature_df],
        axis=1,
    )

    all_feature_cols = (
        daily_context_cols
        + season_feature_cols
        + derived_feature_cols
        + row_level_feature_cols
    )
    return output, all_feature_cols


def summarize_prediction(
    y_true_error: pd.Series,
    y_pred_error: pd.Series,
    *,
    label: str,
) -> dict[str, float | int | str]:
    y_true = pd.to_numeric(y_true_error, errors="coerce").to_numpy(dtype=float)
    y_pred = pd.to_numeric(y_pred_error, errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    true_side = np.sign(y_true)
    pred_side = np.sign(y_pred)
    eligible = valid & (true_side != 0) & (pred_side != 0)
    actual_non_push = valid & (true_side != 0)
    residual = y_pred - y_true

    accuracy = over_under_betting_accuracy_error_line(
        y_true_error=y_true[valid],
        y_pred_error=y_pred[valid],
    )
    coverage = np.nan
    if actual_non_push.sum() > 0:
        coverage = 100.0 * eligible.sum() / actual_non_push.sum()

    return {
        "model": label,
        "accuracy_pct": 100.0 * accuracy,
        "eligible_games": int(eligible.sum()),
        "coverage_pct_non_push": float(coverage),
        "pred_push_rate_pct": 100.0 * np.mean(pred_side[valid] == 0) if np.any(valid) else np.nan,
        "mean_abs_predicted_edge": float(np.nanmean(np.abs(y_pred[valid]))) if np.any(valid) else np.nan,
        "mean_predicted_edge": float(np.nanmean(y_pred[valid])) if np.any(valid) else np.nan,
        "mae_vs_true_error": float(np.nanmean(np.abs(residual[valid]))) if np.any(valid) else np.nan,
        "rmse_vs_true_error": float(np.sqrt(np.nanmean(np.square(residual[valid])))) if np.any(valid) else np.nan,
        "bias_vs_true_error": float(np.nanmean(residual[valid])) if np.any(valid) else np.nan,
    }


def summarize_prediction_columns(
    df: pd.DataFrame,
    *,
    prediction_cols: list[str],
    target_error_col: str = TARGET_ERROR_COL,
) -> pd.DataFrame:
    rows = [
        summarize_prediction(df[target_error_col], df[column], label=column)
        for column in prediction_cols
    ]
    return pd.DataFrame(rows).sort_values("accuracy_pct", ascending=False).reset_index(
        drop=True
    )


def build_meta_learner_feature_frame(
    df: pd.DataFrame,
    *,
    preferred_line_col: str | None = None,
    date_col: str = DATE_COL,
    game_id_col: str = GAME_ID_COL,
    season_col: str = SEASON_COL,
    target_error_col: str = TARGET_ERROR_COL,
    rolling_window_days: int = DEFAULT_ROLLING_WINDOW_DAYS,
    line_bucket_edges: list[float] | None = None,
) -> MetaLearnerFeatureBuildResult:
    total_model_cols, line_error_model_cols = resolve_prediction_column_groups(df)
    line_col = resolve_line_column(df, preferred_line_col=preferred_line_col)

    working = df.copy()
    if date_col in working.columns:
        working[date_col] = pd.to_datetime(working[date_col], errors="coerce").dt.normalize()
    sort_cols = [column for column in [date_col, game_id_col] if column in working.columns]
    if sort_cols:
        working = working.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    working, error_cols = add_error_space_prediction_columns(
        working,
        total_model_cols=total_model_cols,
        line_error_model_cols=line_error_model_cols,
        line_col=line_col,
    )
    working, vote_feature_cols = add_vote_summary_features(working, error_cols=error_cols)
    working, consensus_structure_cols = add_consensus_structure_features(
        working,
        error_cols=error_cols,
    )
    working, reliability_feature_cols = add_rolling_reliability_features(
        working,
        error_cols=error_cols,
        line_col=line_col,
        date_col=date_col,
        target_error_col=target_error_col,
        rolling_window_days=rolling_window_days,
        line_bucket_edges=line_bucket_edges,
    )
    working = add_default_meta_learner_baselines(
        working,
        total_model_cols=total_model_cols,
        line_error_model_cols=line_error_model_cols,
        line_col=line_col,
    )
    working, historical_context_cols = add_historical_context_features(
        working,
        error_cols=error_cols,
        line_col=line_col,
        date_col=date_col,
        season_col=season_col,
        target_error_col=target_error_col,
        line_bucket_edges=line_bucket_edges,
    )

    baseline_cols = [
        BASE_AVG_ALL_6_ERR_COL,
        BASE_MAJORITY_TOTAL_ONLY_ERR_COL,
        BASE_MAJORITY_LINE_ERROR_ONLY_ERR_COL,
        BASE_MAJORITY_ALL_6_TIE_WITH_PRED_LINE_ERROR_FULL_DATASET_ERR_COL,
    ]

    return MetaLearnerFeatureBuildResult(
        dataframe=working,
        line_col=line_col,
        total_model_cols=total_model_cols,
        line_error_model_cols=line_error_model_cols,
        error_cols=error_cols,
        vote_feature_cols=vote_feature_cols + consensus_structure_cols,
        reliability_feature_cols=reliability_feature_cols + historical_context_cols,
        baseline_cols=baseline_cols,
    )
