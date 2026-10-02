"""The team rolling-statistics speed-ups must not change a single value.

``compute_all_rolling_statistics`` now runs the rolling stage on a narrow frame
and rebuilds the wide one; ``_compute_all_rolling_statistics_on`` is the stage
itself, run on whatever frame it is given. Running it on the full frame is the
previous behaviour, so the two are compared bit for bit -- rows, row order,
index, column order, dtypes and values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.statistics import statistics
from nba_ou.data_processing.team import rolling
from nba_ou.data_processing.team.rolling import (
    COLS_TO_AVERAGE,
    MAIN_MONEYLINE_COL,
    MAIN_SPREAD_COL,
    MAIN_TOTAL_LINE_COL,
    _compute_all_rolling_statistics_on,
    compute_all_rolling_statistics,
    compute_trend_slope,
)
from nba_ou.data_processing.team.style_matchups import STYLE_SOURCE_COLUMNS

from .frame_identity import assert_identical


def team_frame(seed: int = 0, n_teams: int = 6, games_per_season: int = 24):
    """Two rows per game over three seasons, with same-day games, playoffs,
    NaN statistics, unrelated columns of every dtype and a scrambled index."""
    rng = np.random.default_rng(seed)
    teams = 1610612737 + np.arange(n_teams)
    rows = []
    game_number = 0
    for season in (2021, 2022, 2023):
        start = pd.Timestamp(f"{season}-10-20")
        for game in range(games_per_season * n_teams // 2):
            home, away = rng.choice(teams, 2, replace=False)
            # Few distinct dates, so many games share one.
            date = start + pd.Timedelta(days=int(game // 3))
            season_type = (
                "Playoffs"
                if game > games_per_season * n_teams // 2 - 4
                else ("Regular Season")
            )
            game_number += 1
            game_id = f"00{season - 2000}{game_number:06d}"
            for team, is_home in ((home, 1), (away, 0)):
                rows.append(
                    {
                        "GAME_ID": game_id,
                        "TEAM_ID": int(team),
                        "HOME": is_home,
                        "GAME_DATE": date,
                        "SEASON_YEAR": season,
                        "SEASON_TYPE": season_type,
                    }
                )
    df = pd.DataFrame(rows)
    n = len(df)

    def noisy(mean, sd, missing=0.05):
        values = rng.normal(mean, sd, n)
        values[rng.random(n) < missing] = np.nan
        return values

    for column in dict.fromkeys(COLS_TO_AVERAGE):
        df[column] = noisy(50, 12)
    df[MAIN_TOTAL_LINE_COL] = np.round(noisy(225, 8) * 2) / 2
    df["ODDS_TOTAL_LINE_draftkings"] = np.round(noisy(225, 8, 0.2) * 2) / 2
    df["DIFF_FROM_LINE_bet365"] = noisy(0, 12)
    df["DIFF_FROM_LINE_draftkings"] = noisy(0, 12)
    df["DIFF_FROM_UNRELATED"] = noisy(0, 1)
    df[MAIN_SPREAD_COL] = np.round(noisy(0, 6) * 2) / 2
    df[MAIN_MONEYLINE_COL] = noisy(-110, 80)
    df["total_consensus_pct_over"] = noisy(50, 10)
    df["spread_consensus_pct_home"] = noisy(50, 10)
    df["spread_bet365_pct_bets"] = noisy(50, 15, 0.3)
    df["ml_bet365_pct_money"] = noisy(50, 15, 0.3)
    df["spread_bet365_price"] = noisy(-110, 5)
    df["ml_bet365_price"] = noisy(-120, 60)
    df["total_bet365_price_over"] = noisy(-110, 5)
    for column in STYLE_SOURCE_COLUMNS:
        df[column] = noisy(0.3, 0.05)
    # Integer statistic: the season std stage casts it to float in place.
    df["PF"] = rng.integers(10, 30, n)
    # Columns the stage must carry through untouched.
    df["TEAM_NAME"] = rng.choice(["A", "B", None], n)
    df["UNRELATED_INT"] = rng.integers(0, 100, n)
    df["UNRELATED_NULLABLE"] = pd.array(
        np.where(rng.random(n) < 0.3, None, rng.integers(0, 5, n)), dtype="Int64"
    )
    df["UNRELATED_BOOL"] = rng.random(n) < 0.5
    df["TIPOFF_UTC"] = df["GAME_DATE"].dt.tz_localize("UTC")
    # Already present under names the stage writes: overwritten in place, and
    # a style column's five-game average that the stage drops again.
    df["PTS_LAST_ALL_5_MATCHES_BEFORE"] = -1.0
    df[f"{STYLE_SOURCE_COLUMNS[0]}_LAST_ALL_5_MATCHES_BEFORE"] = -1.0

    order = rng.permutation(n)
    df = df.iloc[order]
    df.index = pd.Index(rng.permutation(n) * 7 + 3, name="row")
    return df


@pytest.mark.parametrize("exclude_yahoo", [False, True])
@pytest.mark.parametrize("seed", [0, 1])
def test_narrow_rolling_stage_matches_the_wide_one(seed, exclude_yahoo):
    df = team_frame(seed)
    expected = _compute_all_rolling_statistics_on(
        df.copy(), exclude_yahoo=exclude_yahoo
    )
    actual = compute_all_rolling_statistics(df.copy(), exclude_yahoo=exclude_yahoo)
    assert_identical(actual, expected)
    assert actual.index.name == expected.index.name


def test_duplicate_column_names_take_the_wide_path():
    df = team_frame(2)
    df = pd.concat([df, df[["UNRELATED_INT"]]], axis=1)
    expected = _compute_all_rolling_statistics_on(df.copy())
    actual = compute_all_rolling_statistics(df.copy())
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_rolling_stage_does_not_modify_its_input():
    df = team_frame(3)
    before = df.copy()
    compute_all_rolling_statistics(df)
    assert_identical(df, before)


# ---- per-window callables -----------------------------------------------------


def _slope_before(series):
    """``calculate_slope`` as it was: the full ``linregress`` fit."""
    from scipy.stats import linregress

    clean_series = [x for x in series if x is not None and not np.isnan(x)]
    if len(clean_series) < 2:
        return 0
    X = np.arange(1, len(clean_series) + 1)
    Y = np.array(clean_series)
    slope, _, _, _, _ = linregress(X, Y)
    return slope


def test_slope_shortcut_matches_linregress_bit_for_bit():
    rng = np.random.default_rng(11)
    for _ in range(3000):
        n = int(rng.integers(2, 11))
        y = rng.normal(rng.normal(0, 200), rng.uniform(0.01, 30), n)
        if rng.random() < 0.3:
            y = np.round(y * 2) / 2
        if rng.random() < 0.1:
            y[:] = y[0]
        expected = np.float64(_slope_before(y))
        actual = np.float64(
            rolling._linregress_slope(np.arange(1, n + 1), np.array(list(y)))
        )
        assert actual.view(np.int64) == expected.view(np.int64)


@pytest.mark.parametrize("parameter", ["PTS", MAIN_TOTAL_LINE_COL])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 5},
        {"window": 10, "include_home_away_relative": False, "relative_to_window": 5},
    ],
)
def test_trend_slope_columns_match_linregress(monkeypatch, parameter, kwargs):
    df = team_frame(4)
    actual = compute_trend_slope(df.copy(), parameter=parameter, **kwargs)
    monkeypatch.setattr(
        rolling, "_linregress_slope", lambda x, y: _slope_before(np.asarray(y))
    )
    expected = compute_trend_slope(df.copy(), parameter=parameter, **kwargs)
    assert_identical(actual, expected)


def _weighted_moving_average_before(x: pd.Series) -> float:
    n = len(x)
    if x.isna().all():
        return np.nan
    w = np.arange(1, n + 1, dtype=float)
    mask = ~x.isna()
    return float((x[mask] * w[mask.to_numpy()]).sum() / w[mask.to_numpy()].sum())


def _compute_rolling_weighted_stats_before(df, param, window, group_by_season=False):
    """The stage's WMA with the Series (``raw=False``) callable it replaced."""
    out = df.copy()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")
    sort_cols = ["TEAM_ID", "GAME_DATE", "GAME_ID"]
    if group_by_season:
        sort_cols = ["TEAM_ID", "SEASON_YEAR", "GAME_DATE", "GAME_ID"]
    out.sort_values(sort_cols, ascending=True, inplace=True)
    series = pd.to_numeric(out[param], errors="coerce")
    keys_any = [out["TEAM_ID"]] + ([out["SEASON_YEAR"]] if group_by_season else [])
    keys_split = keys_any + [out["HOME"]]

    def roll(keys):
        return series.groupby(keys).transform(
            lambda s: (
                s.shift(1)
                .rolling(window, min_periods=1)
                .apply(_weighted_moving_average_before, raw=False)
            )
        )

    out[f"{param}_LAST_{window}_WMA_BEFORE"] = roll(keys_any)
    out[f"{param}_LAST_HOME_AWAY_{window}_WMA_BEFORE"] = (
        roll(keys_split) - out[f"{param}_LAST_{window}_WMA_BEFORE"]
    )
    out.sort_values("GAME_DATE", ascending=False, inplace=True)
    return out


@pytest.mark.parametrize(
    "param", ["PTS", MAIN_TOTAL_LINE_COL, "spread_bet365_pct_bets"]
)
@pytest.mark.parametrize("group_by_season", [False, True])
def test_weighted_moving_average_matches_the_series_version(param, group_by_season):
    df = team_frame(5)
    expected = _compute_rolling_weighted_stats_before(
        df, param, 5, group_by_season=group_by_season
    )
    actual = statistics.compute_rolling_weighted_stats(
        df, param, window=5, group_by_season=group_by_season
    )
    assert_identical(actual, expected)
