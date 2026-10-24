"""The day- and game-window aggregations give exactly the per-row versions' values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.merged_home_away_data import global_market_features as gm

from tests.support.frame_identity import assert_identical


def _day_agg_before(values, dates, window_days, func):
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    dates_arr = dates.values.astype("datetime64[D]")
    vals_arr = values.values.astype("float64")
    delta = np.timedelta64(window_days, "D")
    for i in range(len(result)):
        cur_date = dates_arr[i]
        mask = (dates_arr < cur_date) & (dates_arr >= cur_date - delta)
        window_vals = vals_arr[mask]
        window_vals = window_vals[~np.isnan(window_vals)]
        if len(window_vals) == 0:
            continue
        if func == "mean":
            result.iat[i] = np.mean(window_vals)
        elif func == "std":
            result.iat[i] = np.std(window_vals, ddof=0)
        elif func == "median":
            result.iat[i] = np.median(window_vals)
        elif func == "sum":
            result.iat[i] = np.sum(window_vals)
        elif func == "count":
            result.iat[i] = float(len(window_vals))
    return result


def _game_agg_before(series, dates, window, func):
    result = pd.Series(np.nan, index=series.index, dtype="float64")
    date_days = pd.to_datetime(dates, errors="coerce").values.astype("datetime64[D]")
    values = pd.to_numeric(series, errors="coerce").astype("float64").values
    if len(result) == 0:
        return result
    group_starts = np.flatnonzero(np.r_[True, date_days[1:] != date_days[:-1]])
    group_ends = np.r_[group_starts[1:], len(values)]
    for start, end in zip(group_starts, group_ends, strict=True):
        valid = values[max(0, start - window) : start]
        valid = valid[~np.isnan(valid)]
        if func == "count":
            value = float(len(valid))
        elif len(valid) == 0:
            value = np.nan
        elif func == "mean":
            value = float(np.mean(valid))
        elif func == "std":
            value = float(np.std(valid, ddof=0))
        elif func == "median":
            value = float(np.median(valid))
        else:
            value = float(np.sum(valid))
        result.iloc[start:end] = value
    return result


def _games(seed: int, *, sorted_dates: bool, n: int = 900):
    rng = np.random.default_rng(seed)
    days = rng.integers(0, 260, n)
    if sorted_dates:
        days = np.sort(days)
    dates = pd.Series(pd.Timestamp("2023-10-20") + pd.to_timedelta(days, unit="D"))
    values = rng.normal(0, 12, n)
    values[rng.random(n) < 0.1] = np.nan
    index = pd.Index(rng.permutation(n) * 3 + 1)
    return pd.Series(values, index=index), dates.set_axis(index)


@pytest.mark.parametrize("sorted_dates", [True, False])
@pytest.mark.parametrize("func", ["mean", "std", "median", "sum", "count"])
@pytest.mark.parametrize("window_days", [1, 7, 30])
def test_day_window_matches(sorted_dates, func, window_days):
    values, dates = _games(window_days, sorted_dates=sorted_dates)
    assert_identical(
        gm._rolling_day_agg(values, dates, window_days, func),
        _day_agg_before(values, dates, window_days, func),
    )


def test_day_window_with_missing_dates_takes_the_mask_path():
    values, dates = _games(3, sorted_dates=True)
    dates.iloc[[5, 200]] = pd.NaT
    assert_identical(
        gm._rolling_day_agg(values, dates, 7, "mean"),
        _day_agg_before(values, dates, 7, "mean"),
    )


@pytest.mark.parametrize("func", ["mean", "std", "median", "sum", "count"])
@pytest.mark.parametrize("window", [5, 50])
def test_game_window_matches(func, window):
    values, dates = _games(window, sorted_dates=True)
    assert_identical(
        gm._rolling_game_agg(values, dates, window, func),
        _game_agg_before(values, dates, window, func),
    )
