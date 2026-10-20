import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, RegressorMixin, clone
from tqdm.auto import tqdm
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Metadata schema
# ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    name: str
    algorithm: str = "xgboost"
    model_version: str
    model_type: str
    prediction_source: str
    training_code_tag: str


class SchemaInfo(BaseModel):
    feature_names: list[str]
    n_features: int


class TrainingMetrics(BaseModel):
    best_params: dict
    selected_trial_number: int | None = None
    mean_best_iteration: int | None = None
    median_best_iteration: int | None = None
    train_games: int | None = None
    sample_weight_lambda_bounds: tuple[float, float] | None = None
    cv_mae: float
    cv_rmse: float | None = None
    cv_ou_acc: float | None = None
    final_test_mae: float
    final_test_rmse: float
    final_test_ou_acc: float
    nan_threshold: float
    max_na_per_row: int
    train_date_min: datetime
    train_date_max: datetime


class ModelBundleMetadata(BaseModel):
    """Structured metadata saved alongside a model bundle.

    Fields
    ------
    model_info
        Identifies the model (serialised as ``"model"`` in JSON).
    schema_info
        Describes the feature schema (serialised as ``"schema"`` in JSON).
        Automatically populated by :func:`save_model_bundle` from the
        ``feature_names`` argument, so callers do not need to fill it in.
    training_metrics
        Optuna study results and hold-out evaluation metrics.
    created_at
        UTC timestamp set automatically at construction time.
    """

    model_info: ModelInfo = Field(serialization_alias="model")
    schema_info: SchemaInfo = Field(
        default=None,  # type: ignore[assignment]
        serialization_alias="schema",
    )
    training_metrics: TrainingMetrics | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    model_config = ConfigDict(populate_by_name=True)


@dataclass
class SplitValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]


@dataclass
class DailyWalkForwardEvaluationResult:
    mean_metric: float
    daily_results: pd.DataFrame
    predictions: pd.DataFrame


def _coerce_datetime_indexed_series(
    data: pd.DataFrame | pd.Series,
    *,
    date_col: str = "GAME_DATE",
    index: pd.Index | None = None,
) -> pd.Series:
    """
    Return a datetime64-normalized series aligned to ``index``.
    """
    if isinstance(data, pd.DataFrame):
        if date_col not in data.columns:
            raise KeyError(f"Missing required column: {date_col}")
        dates = data[date_col]
    else:
        dates = data

    parsed = pd.to_datetime(dates, errors="coerce").dt.normalize()
    if parsed.isna().any():
        raise ValueError("Date series contains missing or invalid values.")

    if index is None:
        return pd.Series(parsed, index=parsed.index, name=date_col)

    aligned = pd.Series(parsed, index=parsed.index, name=date_col).reindex(index)
    if aligned.isna().any():
        raise ValueError("Date series could not be aligned to the requested index.")
    return aligned


def build_recency_sample_weights(
    data: pd.DataFrame | pd.Series,
    *,
    date_col: str = "GAME_DATE",
    lambda_: float = 0.01,
    reference_date: pd.Timestamp | datetime | str | None = None,
) -> pd.Series:
    """
    Build exponentially decayed sample weights from dates.

    The newest row in the provided data gets weight 1.0 when
    ``reference_date`` is omitted. For time-series CV, call this on the
    training subset inside each fold so the decay is anchored only to the
    data visible to that fold.
    """
    if lambda_ < 0:
        raise ValueError("lambda_ must be >= 0")

    dates = _coerce_datetime_indexed_series(data, date_col=date_col)
    resolved_reference_date = (
        dates.max()
        if reference_date is None
        else pd.to_datetime(reference_date, errors="coerce")
    )

    if pd.isna(resolved_reference_date):
        raise ValueError("reference_date could not be parsed.")

    resolved_reference_date = pd.Timestamp(resolved_reference_date).normalize()
    age_days = (resolved_reference_date - dates).dt.days

    if (age_days < 0).any():
        raise ValueError("reference_date is earlier than at least one sample date.")

    weights = np.exp(-float(lambda_) * age_days.to_numpy(dtype=float))
    return pd.Series(weights, index=dates.index, name="sample_weight")


class TemporalDecaySampleWeightRegressor(RegressorMixin, BaseEstimator):
    """
    Wrap a regressor and compute recency weights inside ``fit``.

    This is useful with sklearn CV utilities because each fold fit only sees its
    training rows, so the weight reference date is leakage-safe.
    """

    def __init__(
        self,
        estimator: Any,
        dates: pd.Series,
        *,
        lambda_: float = 0.01,
        date_col: str = "GAME_DATE",
    ) -> None:
        self.estimator = estimator
        self.dates = dates
        self.lambda_ = lambda_
        self.date_col = date_col

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **fit_params,
    ) -> "TemporalDecaySampleWeightRegressor":
        if not hasattr(X, "index"):
            raise TypeError(
                "TemporalDecaySampleWeightRegressor requires X with an index."
            )

        dates = _coerce_datetime_indexed_series(
            self.dates,
            date_col=self.date_col,
            index=X.index,
        )
        temporal_weights = build_recency_sample_weights(dates, lambda_=self.lambda_)

        estimator = clone(self.estimator)
        existing_sample_weight = fit_params.pop("sample_weight", None)

        if existing_sample_weight is not None:
            existing_series = pd.Series(existing_sample_weight, index=X.index)
            temporal_weights = temporal_weights * pd.to_numeric(
                existing_series, errors="coerce"
            ).astype(float)

        estimator.fit(
            X,
            y,
            sample_weight=temporal_weights.to_numpy(dtype=float),
            **fit_params,
        )

        self.estimator_ = estimator
        self.sample_weight_ = temporal_weights
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict(X)

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "estimator_")


def tail_n_games(
    df: pd.DataFrame, n: int, *, group_col: str | None = None
) -> pd.DataFrame:
    """The last ``n`` GAMES of a chronologically ordered training frame.

    ``group_col`` names the column identifying a game. Without it -- and on any
    frame holding one row per game -- this is ``df.tail(n)``, which is what the
    day-by-day walk-forward always did.

    It matters on a frame with several rows per game, such as the
    intermediate-line dataset's one row per (game, pre-game snapshot). There
    ``tail(n)`` keeps n ROWS, so a 3,500-game window silently becomes 350 games
    at ten snapshots each, and it can also slice a game in half -- keeping four
    of its ten snapshots, which is neither a row window nor a game window. Every
    row of a kept game is kept here.
    """
    if group_col is None or group_col not in df.columns:
        return df.tail(n)
    groups = df[group_col].astype(str)
    ordered = groups.drop_duplicates()
    if len(ordered) <= n:
        return df
    return df.loc[groups.isin(set(ordered.tail(n)))]


def evaluate_day_by_day_walk_forward(
    df_dev: pd.DataFrame,
    df_test_final: pd.DataFrame,
    *,
    fit_and_predict: Callable[[pd.DataFrame, pd.DataFrame], np.ndarray],
    metric_fn: Callable[[pd.Series | np.ndarray, np.ndarray], float],
    target_col: str,
    date_col: str = "GAME_DATE",
    max_games: int | None = None,
    max_games_group_col: str | None = None,
    metric_name: str = "metric",
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> DailyWalkForwardEvaluationResult:
    """
    Evaluate a model with a day-by-day walk-forward protocol.

    The evaluation iterates over the unique dates present in ``df_test_final``.
    For each prediction date, the training set is formed from every row strictly
    before that date across the combined ``df_dev`` + ``df_test_final`` history.
    This means earlier test rows become eligible training rows for later dates.

    Parameters
    ----------
    df_dev : pd.DataFrame
        Development data available before the final holdout period.
    df_test_final : pd.DataFrame
        Chronological holdout rows to evaluate day by day.
    fit_and_predict : callable
        Receives ``(train_df, test_df)`` and must return one prediction per row
        in ``test_df``.
    metric_fn : callable
        Receives ``(y_true, y_pred)`` for one day and returns a scalar metric.
    target_col : str
        Target column name.
    date_col : str
        Date column name.
    max_games : int | None
        If provided, keep only the latest ``max_games`` games in each day's
        training set.
    max_games_group_col : str | None
        Column identifying a game, e.g. "GAME_ID". Needed only when the frame
        holds several rows per game, where without it ``max_games`` counts rows
        and the window silently shrinks by the number of rows per game. None
        (the default) keeps the original ``tail(max_games)`` behaviour, which is
        already correct for a one-row-per-game frame.
    metric_name : str
        Column name used in the returned per-day summary.
    show_progress : bool
        If True, display a tqdm progress bar over prediction days.
    progress_desc : str | None
        Optional tqdm description when ``show_progress`` is enabled.

    Returns
    -------
    DailyWalkForwardEvaluationResult
        Mean metric across evaluated days, per-day summary, and row-level
        predictions for the test rows.
    """
    if max_games is not None and max_games <= 0:
        raise ValueError("max_games must be > 0 when provided.")
    if target_col not in df_dev.columns or target_col not in df_test_final.columns:
        raise KeyError(f"Missing required target column: {target_col}")
    if date_col not in df_dev.columns or date_col not in df_test_final.columns:
        raise KeyError(f"Missing required date column: {date_col}")
    if len(df_test_final) == 0:
        raise ValueError("df_test_final must not be empty.")

    dev_part = df_dev.copy()
    dev_part["_walk_source"] = "dev"
    dev_part["_walk_source_order"] = 0
    dev_part["_walk_orig_pos"] = np.arange(len(dev_part))

    test_part = df_test_final.copy()
    test_part["_walk_source"] = "test"
    test_part["_walk_source_order"] = 1
    test_part["_walk_orig_pos"] = np.arange(len(test_part))

    combined = pd.concat([dev_part, test_part], axis=0, ignore_index=True)
    combined["_walk_date"] = pd.to_datetime(combined[date_col], errors="coerce").dt.normalize()

    if combined["_walk_date"].isna().any():
        raise ValueError(f"Invalid dates found in column '{date_col}'.")

    combined = combined.sort_values(
        ["_walk_date", "_walk_source_order", "_walk_orig_pos"],
        kind="mergesort",
    ).reset_index(drop=True)

    test_dates = sorted(
        pd.to_datetime(df_test_final[date_col], errors="coerce").dt.normalize().unique()
    )
    helper_cols = {
        "_walk_source",
        "_walk_source_order",
        "_walk_orig_pos",
        "_walk_date",
    }
    feature_cols = [col for col in combined.columns if col not in helper_cols]

    daily_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    date_iterator = test_dates
    if show_progress:
        date_iterator = tqdm(
            test_dates,
            desc=progress_desc or "Walk-forward days",
            unit="day",
            leave=False,
        )

    for current_day in date_iterator:
        test_mask = (combined["_walk_source"] == "test") & (
            combined["_walk_date"] == current_day
        )
        train_mask = combined["_walk_date"] < current_day

        train_df = combined.loc[train_mask, feature_cols].copy()
        if max_games is not None:
            train_df = tail_n_games(
                train_df, max_games, group_col=max_games_group_col
            ).copy()

        test_positions = combined.loc[test_mask, "_walk_orig_pos"].to_numpy()
        test_df = combined.loc[test_mask, feature_cols].copy()

        if len(test_df) == 0:
            continue
        if len(train_df) == 0:
            raise ValueError(
                f"No training rows are available before test day {pd.Timestamp(current_day).date()}."
            )

        y_true = test_df[target_col].copy()
        y_pred = np.asarray(fit_and_predict(train_df, test_df))
        if y_pred.ndim == 0:
            y_pred = y_pred.reshape(1)

        if len(y_pred) != len(test_df):
            raise ValueError(
                "fit_and_predict must return one prediction for each test row."
            )

        metric_value = float(metric_fn(y_true, y_pred))
        train_dates = pd.to_datetime(train_df[date_col], errors="coerce").dt.normalize()

        daily_rows.append(
            {
                "date": pd.Timestamp(current_day).normalize(),
                "train_n_games": (
                    int(train_df[max_games_group_col].nunique())
                    if max_games_group_col and max_games_group_col in train_df
                    else int(len(train_df))
                ),
                "train_n_rows": int(len(train_df)),
                "test_n_games": (
                    int(test_df[max_games_group_col].nunique())
                    if max_games_group_col and max_games_group_col in test_df
                    else int(len(test_df))
                ),
                "test_n_rows": int(len(test_df)),
                "train_start_date": train_dates.min(),
                "train_end_date": train_dates.max(),
                metric_name: metric_value,
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "date": pd.Timestamp(current_day).normalize(),
                    "row_in_test_final": test_positions,
                    "y_true": y_true.to_numpy(),
                    "y_pred": y_pred,
                }
            )
        )

    if not daily_rows:
        raise ValueError("No daily evaluations were produced.")

    daily_results = pd.DataFrame(daily_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    return DailyWalkForwardEvaluationResult(
        mean_metric=float(daily_results[metric_name].mean()),
        daily_results=daily_results,
        predictions=predictions,
    )


def prepare_time_dataframe(
    df: pd.DataFrame,
    *,
    date_col: str = "GAME_DATE",
    season_col: str | None = "SEASON_YEAR",
    sort_kind: str = "mergesort",
) -> pd.DataFrame:
    """
    Return a copy of the dataframe with:
    - parsed normalized datetime column
    - original row positions stored in _pos
    - stable chronological ordering

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    date_col : str
        Name of the date column.
    season_col : str | None
        Optional season column. If provided, it is preserved in the working copy.
    sort_kind : str
        Sorting method. mergesort is stable and recommended.

    Returns
    -------
    pd.DataFrame
        Chronologically sorted working dataframe with helper columns:
        _date, _pos
    """
    required_cols = [date_col]
    if season_col is not None:
        required_cols.append(season_col)

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    temp = df[required_cols].copy()
    temp["_pos"] = np.arange(len(df))
    temp["_date"] = pd.to_datetime(temp[date_col], errors="coerce").dt.normalize()

    if temp["_date"].isna().any():
        bad_rows = temp[temp["_date"].isna()].index.tolist()[:10]
        raise ValueError(
            f"Invalid dates found in column '{date_col}'. Example bad row indices: {bad_rows}"
        )

    temp = temp.sort_values(["_date", "_pos"], kind=sort_kind).reset_index(drop=True)
    return temp


def _advance_until_n_games(
    date_counts: pd.Series,
    start_pos: int,
    target_games: int,
) -> tuple[int, int]:
    """
    Advance through unique dates until at least target_games have been accumulated.

    Parameters
    ----------
    date_counts : pd.Series
        Index = unique dates, values = number of games on each date.
    start_pos : int
        Starting position inside the unique-date index.
    target_games : int
        Desired minimum number of games.

    Returns
    -------
    tuple[int, int]
        end_pos (exclusive), n_games accumulated
    """
    if target_games <= 0:
        raise ValueError("target_games must be > 0")

    end_pos = start_pos
    n_games = 0

    while end_pos < len(date_counts) and n_games < target_games:
        n_games += int(date_counts.iloc[end_pos])
        end_pos += 1

    return end_pos, n_games


def split_latest_dates_holdout(
    df: pd.DataFrame,
    *,
    date_col: str = "GAME_DATE",
    test_size: float | None = 0.15,
    test_games: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a final holdout using the latest dates only.

    Exactly one of test_size or test_games should be provided.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    date_col : str
        Date column.
    test_size : float | None
        Fraction of rows to include in the final test set.
    test_games : int | None
        Fixed minimum number of games in the final test set.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        df_dev, df_test
    """
    if (test_size is None) == (test_games is None):
        raise ValueError("Provide exactly one of test_size or test_games.")

    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce").dt.normalize()

    if temp[date_col].isna().any():
        raise ValueError(f"Invalid dates found in {date_col}")

    date_counts = temp.groupby(date_col).size().sort_index()

    if test_games is None:
        if not (0 < float(test_size) < 1):
            raise ValueError("test_size must be between 0 and 1")
        target_n_games = int(np.ceil(len(temp) * float(test_size)))
    else:
        if int(test_games) <= 0:
            raise ValueError("test_games must be > 0")
        target_n_games = int(test_games)

    selected_test_dates = []
    running_n = 0

    for dt, n_games_on_date in reversed(list(date_counts.items())):
        selected_test_dates.append(dt)
        running_n += int(n_games_on_date)
        if running_n >= target_n_games:
            break

    selected_test_dates = set(selected_test_dates)
    test_mask = temp[date_col].isin(selected_test_dates)

    df_dev = df.loc[~test_mask].copy().reset_index(drop=True)
    df_test = df.loc[test_mask].copy().reset_index(drop=True)

    if len(df_dev) == 0 or len(df_test) == 0:
        raise ValueError("Holdout split failed: one side is empty.")

    return df_dev, df_test


def make_walk_forward_last_n_seasons_splits(
    df: pd.DataFrame,
    *,
    date_col: str = "GAME_DATE",
    season_col: str = "SEASON_YEAR",
    train_seasons: int = 3,
    test_games: int = 30,
    step_games: int | None = None,
    min_train_games: int = 300,
    max_folds: int | None = None,
    fold_selection: str = "latest",
    exclude_test_months: tuple[int, ...] = (5, 6),
    verbose: int = 0,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """
    Create walk-forward splits where each fold trains on the last N seasons
    available before the first test date.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    date_col : str
        Date column.
    season_col : str
        Season column.
    train_seasons : int
        Number of past seasons to include in training.
    test_games : int
        Minimum number of games in each validation fold.
    step_games : int | None
        Step forward between folds. If None, uses test_games.
    min_train_games : int
        Minimum training rows required to accept a fold.
    max_folds : int | None
        Maximum number of folds to keep.
    fold_selection : str
        One of {"latest", "earliest", "even"} when max_folds is used.
    verbose : int
        If >= 1, print fold summary.

    Returns
    -------
    tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]
        splits, fold_info
    """
    if train_seasons <= 0:
        raise ValueError("train_seasons must be > 0")
    if test_games <= 0:
        raise ValueError("test_games must be > 0")
    if min_train_games <= 0:
        raise ValueError("min_train_games must be > 0")

    if step_games is None:
        step_games = test_games
    if step_games <= 0:
        raise ValueError("step_games must be > 0")

    temp = prepare_time_dataframe(
        df=df,
        date_col=date_col,
        season_col=season_col,
    )

    date_counts = temp.groupby("_date", sort=True).size()
    unique_dates = list(date_counts.index)

    all_splits: list[tuple[np.ndarray, np.ndarray]] = []
    fold_rows: list[dict] = []

    start_date_pos = 0
    fold_num = 1

    while start_date_pos < len(unique_dates):
        first_test_date = unique_dates[start_date_pos]

        past_mask = temp["_date"] < first_test_date
        past_rows = temp.loc[past_mask]

        past_seasons = past_rows[season_col].dropna().drop_duplicates().tolist()
        if len(past_seasons) < train_seasons:
            start_date_pos += 1
            continue

        selected_train_seasons = past_seasons[-train_seasons:]

        train_mask = past_mask & temp[season_col].isin(selected_train_seasons)
        train_idx = temp.loc[train_mask, "_pos"].to_numpy()

        if len(train_idx) < min_train_games:
            start_date_pos += 1
            continue

        first_test_season = temp.loc[temp["_date"] == first_test_date, season_col].iloc[
            0
        ]

        end_pos = start_date_pos
        n_test_games = 0
        test_dates = []
        invalid_test_window = False

        while end_pos < len(unique_dates) and n_test_games < test_games:
            current_date = unique_dates[end_pos]
            current_season = temp.loc[temp["_date"] == current_date, season_col].iloc[0]

            if current_season != first_test_season:
                break

            if current_date.month in exclude_test_months:
                invalid_test_window = True
                break

            test_dates.append(current_date)
            n_test_games += int(date_counts.loc[current_date])
            end_pos += 1

        if invalid_test_window:
            start_date_pos += 1
            continue

        if n_test_games < test_games:
            start_date_pos += 1
            continue

        test_mask = temp["_date"].isin(test_dates)
        test_idx = temp.loc[test_mask, "_pos"].to_numpy()

        if len(test_idx) == 0:
            break

        all_splits.append((train_idx, test_idx))
        fold_rows.append(
            {
                "fold": fold_num,
                "train_n_games": int(len(train_idx)),
                "test_n_games": int(len(test_idx)),
                "train_start_date": temp.loc[train_mask, "_date"].min(),
                "train_end_date": temp.loc[train_mask, "_date"].max(),
                "test_start_date": min(test_dates),
                "test_end_date": max(test_dates),
                "train_seasons": list(selected_train_seasons),
            }
        )

        next_start_pos, _ = _advance_until_n_games(
            date_counts=date_counts,
            start_pos=start_date_pos,
            target_games=step_games,
        )
        start_date_pos = max(start_date_pos + 1, next_start_pos)
        fold_num += 1

    if not all_splits:
        raise ValueError("No valid folds were created.")

    fold_info = pd.DataFrame(fold_rows)

    if max_folds is not None:
        if max_folds <= 0:
            raise ValueError("max_folds must be > 0")

        if max_folds < len(all_splits):
            keep_idx = _select_fold_indices(
                n_folds=len(all_splits),
                max_folds=max_folds,
                strategy=fold_selection,
            )
            all_splits = [all_splits[i] for i in keep_idx]
            fold_info = fold_info.iloc[keep_idx].reset_index(drop=True)

    fold_info["fold"] = np.arange(1, len(fold_info) + 1)

    if verbose >= 1:
        print(f"Created {len(all_splits)} walk-forward folds")
        print(fold_info.to_string(index=False))

    return all_splits, fold_info


def _select_fold_indices(
    n_folds: int,
    max_folds: int,
    strategy: str = "latest",
) -> np.ndarray:
    """
    Deterministically select fold indices when too many folds exist.

    Strategies
    ----------
    latest
        Keep the most recent folds.
    earliest
        Keep the earliest folds.
    even
        Keep evenly spaced folds across the full period.
    """
    if max_folds >= n_folds:
        return np.arange(n_folds)

    strategy = strategy.lower()

    if strategy == "latest":
        return np.arange(n_folds - max_folds, n_folds)

    if strategy == "earliest":
        return np.arange(max_folds)

    if strategy == "even":
        return np.unique(np.round(np.linspace(0, n_folds - 1, max_folds)).astype(int))

    raise ValueError("strategy must be one of {'latest', 'earliest', 'even'}")


def make_test_anchored_walk_forward_splits(
    df: pd.DataFrame,
    *,
    date_col: str = "GAME_DATE",
    season_col: str = "SEASON_YEAR",
    test_games: int = 30,
    step_games_between_tests: int | None = None,
    train_games: int | None = 1000,
    min_train_games: int = 300,
    exclude_test_months: tuple[int, ...] = (5, 6),
    require_same_season_test: bool = True,
    max_folds: int | None = None,
    fold_selection: str = "latest",
    verbose: int = 0,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """
    Create walk-forward splits by building valid test windows first, then
    assigning to each fold the most recent training rows immediately before
    the test window.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    date_col : str
        Date column.
    season_col : str
        Season column.
    test_games : int
        Minimum number of rows/games in each test fold.
    step_games_between_tests : int | None
        Spacing (in games) measured from the end of the accepted test window to
        the start of the next candidate. If None, uses test_games.
    train_games : int | None
        Number of rows/games immediately before the test window to use for
        training. If None, uses all available rows before test start.
    min_train_games : int
        Minimum training rows required to accept a fold.
    exclude_test_months : tuple[int, ...]
        Months that cannot appear in the test window.
    require_same_season_test : bool
        If True, test windows cannot cross season boundaries.
    max_folds : int | None
        Maximum number of folds to keep.
    fold_selection : str
        One of {"latest", "earliest", "even"} when max_folds is used.
    verbose : int
        If >= 1, print fold summary.

    Returns
    -------
    tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]
        splits, fold_info
    """
    if test_games <= 0:
        raise ValueError("test_games must be > 0")
    if min_train_games <= 0:
        raise ValueError("min_train_games must be > 0")
    if train_games is not None and train_games <= 0:
        raise ValueError("train_games must be > 0")

    if step_games_between_tests is None:
        step_games_between_tests = test_games
    if step_games_between_tests <= 0:
        raise ValueError("step_games_between_tests must be > 0")

    temp = prepare_time_dataframe(
        df=df,
        date_col=date_col,
        season_col=season_col,
    )

    date_counts = temp.groupby("_date", sort=True).size()
    unique_dates = list(date_counts.index)
    date_to_season = temp.groupby("_date", sort=True)[season_col].first().to_dict()

    all_splits: list[tuple[np.ndarray, np.ndarray]] = []
    fold_rows: list[dict] = []

    start_date_pos = 0
    fold_num = 1

    while start_date_pos < len(unique_dates):
        first_test_date = unique_dates[start_date_pos]
        first_test_season = date_to_season[first_test_date]

        end_pos = start_date_pos
        n_test_games = 0
        test_dates: list[pd.Timestamp] = []
        invalid_test_window = False

        while end_pos < len(unique_dates) and n_test_games < test_games:
            current_date = unique_dates[end_pos]

            if current_date.month in exclude_test_months:
                invalid_test_window = True
                break

            if (
                require_same_season_test
                and date_to_season[current_date] != first_test_season
            ):
                break

            test_dates.append(current_date)
            n_test_games += int(date_counts.loc[current_date])
            end_pos += 1

        if invalid_test_window or n_test_games < test_games:
            start_date_pos += 1
            continue

        test_mask = temp["_date"].isin(test_dates)
        test_idx = temp.loc[test_mask, "_pos"].to_numpy()

        if len(test_idx) == 0:
            start_date_pos += 1
            continue

        test_start_date = min(test_dates)
        train_pool = temp.loc[temp["_date"] < test_start_date, ["_pos", "_date"]].copy()

        if train_games is None:
            train_idx = train_pool["_pos"].to_numpy()
        else:
            train_idx = train_pool["_pos"].tail(train_games).to_numpy()

        if len(train_idx) < min_train_games:
            start_date_pos += 1
            continue

        train_dates = temp.loc[temp["_pos"].isin(train_idx), "_date"]

        all_splits.append((train_idx, test_idx))
        fold_rows.append(
            {
                "fold": fold_num,
                "train_n_games": int(len(train_idx)),
                "test_n_games": int(len(test_idx)),
                "train_start_date": train_dates.min(),
                "train_end_date": train_dates.max(),
                "test_start_date": min(test_dates),
                "test_end_date": max(test_dates),
                "test_season": first_test_season,
            }
        )

        # Step forward from the end of the accepted test window for cleaner separation.
        next_start_pos, _ = _advance_until_n_games(
            date_counts=date_counts,
            start_pos=end_pos,
            target_games=step_games_between_tests,
        )
        start_date_pos = max(start_date_pos + 1, next_start_pos)
        fold_num += 1

    if not all_splits:
        raise ValueError("No valid folds were created.")

    fold_info = pd.DataFrame(fold_rows)

    if max_folds is not None:
        if max_folds <= 0:
            raise ValueError("max_folds must be > 0")

        if max_folds < len(all_splits):
            keep_idx = _select_fold_indices(
                n_folds=len(all_splits),
                max_folds=max_folds,
                strategy=fold_selection,
            )
            all_splits = [all_splits[i] for i in keep_idx]
            fold_info = fold_info.iloc[keep_idx].reset_index(drop=True)

    fold_info["fold"] = np.arange(1, len(fold_info) + 1)

    if verbose >= 1:
        print(f"Created {len(all_splits)} test-anchored walk-forward folds")
        print(fold_info.to_string(index=False))

    return all_splits, fold_info


def summarize_splits(
    splits: list[tuple[np.ndarray, np.ndarray]],
    df: pd.DataFrame,
    *,
    date_col: str = "GAME_DATE",
) -> pd.DataFrame:
    """
    Build a summary dataframe from already-created splits.

    Parameters
    ----------
    splits : list[tuple[np.ndarray, np.ndarray]]
        CV splits.
    df : pd.DataFrame
        Original dataframe.
    date_col : str
        Date column.

    Returns
    -------
    pd.DataFrame
        One row per fold with date ranges and sizes.
    """
    dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    rows = []
    for i, (train_idx, test_idx) in enumerate(splits, start=1):
        tr_dates = dates.iloc[train_idx]
        te_dates = dates.iloc[test_idx]

        rows.append(
            {
                "fold": i,
                "train_n_games": len(train_idx),
                "test_n_games": len(test_idx),
                "train_start_date": tr_dates.min(),
                "train_end_date": tr_dates.max(),
                "test_start_date": te_dates.min(),
                "test_end_date": te_dates.max(),
            }
        )

    return pd.DataFrame(rows)


def validate_time_splits(
    df: pd.DataFrame,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    date_col: str = "GAME_DATE",
    require_strict_time_order: bool = True,
    require_non_overlapping_test: bool = True,
    require_monotonic_test_windows: bool = True,
) -> SplitValidationResult:
    """
    Validate that time splits are leakage-safe.

    Checks
    ------
    - train and test indices are non-empty
    - no overlap between train and test inside a fold
    - all train dates are before all test dates
    - test windows are monotonic across folds
    - optionally test folds do not overlap one another

    Returns
    -------
    SplitValidationResult
    """
    errors: list[str] = []
    warnings: list[str] = []

    dates = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    if dates.isna().any():
        errors.append(f"Invalid dates found in column '{date_col}'")
        return SplitValidationResult(False, errors, warnings)

    seen_test_idx: set[int] = set()
    prev_test_start = None
    prev_test_end = None

    for fold_num, (train_idx, test_idx) in enumerate(splits, start=1):
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        if len(train_idx) == 0:
            errors.append(f"Fold {fold_num}: empty train split")
            continue

        if len(test_idx) == 0:
            errors.append(f"Fold {fold_num}: empty test split")
            continue

        overlap = np.intersect1d(train_idx, test_idx)
        if len(overlap) > 0:
            errors.append(
                f"Fold {fold_num}: train/test overlap detected ({len(overlap)} rows)"
            )

        train_dates = dates.iloc[train_idx]
        test_dates = dates.iloc[test_idx]

        tr_max = train_dates.max()
        te_min, te_max = test_dates.min(), test_dates.max()

        if require_strict_time_order and not (tr_max < te_min):
            errors.append(
                f"Fold {fold_num}: time leakage detected "
                f"(train_end={tr_max}, test_start={te_min})"
            )

        if require_monotonic_test_windows and prev_test_start is not None:
            if te_min < prev_test_start:
                errors.append(
                    f"Fold {fold_num}: test window starts before previous test window"
                )
            if te_max < prev_test_end:
                errors.append(
                    f"Fold {fold_num}: test window ends before previous test window"
                )

        if require_non_overlapping_test:
            duplicated_test = seen_test_idx.intersection(set(test_idx.tolist()))
            if duplicated_test:
                warnings.append(
                    f"Fold {fold_num}: test indices overlap with previous folds "
                    f"({len(duplicated_test)} rows)"
                )
            seen_test_idx.update(test_idx.tolist())

        prev_test_start = te_min
        prev_test_end = te_max

    return SplitValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def assert_valid_time_splits(
    df: pd.DataFrame,
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    date_col: str = "GAME_DATE",
) -> None:
    """
    Raise a ValueError if validate_time_splits fails.
    """
    result = validate_time_splits(df=df, splits=splits, date_col=date_col)

    if not result.is_valid:
        joined = "\n".join(result.errors)
        raise ValueError(f"Invalid time splits:\n{joined}")


def save_model_bundle(
    model: XGBRegressor,
    feature_names: list[str],
    out_dir: str | Path,
    metadata: ModelBundleMetadata,
) -> tuple[Path, Path]:
    """Save a model and its metadata to *out_dir*.

    The ``schema_info`` inside *metadata* is always overwritten from
    *feature_names*, so callers only need to populate ``model_info``.

    Parameters
    ----------
    model : XGBRegressor
        Fitted model to persist.
    feature_names : list[str]
        Feature names actually used by the model.
    out_dir : str | Path
        Directory to write artefacts into (created if absent).
    metadata : ModelBundleMetadata
        Structured metadata.  ``model_info.name`` is used to derive
        the output file names.

    Returns
    -------
    tuple[Path, Path]
        (model_path, meta_path)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = metadata.model_info.name
    model_path = out_dir / f"{name}.json"
    meta_path = out_dir / f"{name}.meta.json"

    model.save_model(model_path)

    # Always derive the schema from the actual feature list used.
    final_metadata = metadata.model_copy(
        update={
            "schema_info": SchemaInfo(
                feature_names=list(feature_names),
                n_features=len(feature_names),
            )
        }
    )

    meta_path.write_text(final_metadata.model_dump_json(by_alias=True, indent=2))

    return model_path, meta_path


def load_model_bundle(model_path: str | Path, meta_path: str | Path):
    model = XGBRegressor()
    model.load_model(model_path)
    metadata = json.loads(Path(meta_path).read_text())
    return model, metadata
