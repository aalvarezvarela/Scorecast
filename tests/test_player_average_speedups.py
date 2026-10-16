"""The grouped EWM player averages equal the per-group lambda's values bit for bit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.players import players_statistics
from nba_ou.data_processing.players.players_statistics import (
    _is_fallback_season_type,
    precompute_cumulative_avg_stat,
)

from .frame_identity import assert_identical


def _players(seed: int, n: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    season = rng.choice([2021, 2022, 2023], n)
    game_number = rng.integers(1, 1300, n)
    kind = rng.choice(["002", "004", "001"], n, p=[0.85, 0.1, 0.05])
    df = pd.DataFrame(
        {
            "PLAYER_ID": rng.integers(1, 120, n).astype(float),
            "SEASON_YEAR": season.astype(float),
            "GAME_ID": [
                f"{k}{s % 100:02d}{g:05d}" for k, s, g in zip(kind, season, game_number)
            ],
            "GAME_DATE": pd.Timestamp("2021-10-01")
            + pd.to_timedelta(rng.integers(0, 900, n), unit="D"),
            "PTS": rng.integers(0, 40, n).astype(float),
            "TS_PCT": rng.uniform(0.3, 0.8, n),
            "MIN": np.where(rng.random(n) < 0.2, 0.0, rng.uniform(1, 40, n)),
        },
        index=pd.Index(rng.integers(0, n // 2, n)),  # duplicated labels
    )
    df.loc[rng.random(n) < 0.05, "PTS"] = np.nan
    df.loc[rng.random(n) < 0.02, "PLAYER_ID"] = np.nan
    df.loc[rng.random(n) < 0.02, "SEASON_YEAR"] = np.nan
    return df


def _before(df, stat_col, halflife=players_statistics.EWMA_HALFLIFE_GAMES):
    out = df.copy()
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"], errors="coerce")
    out[stat_col] = pd.to_numeric(out[stat_col], errors="coerce")
    out["MIN"] = pd.to_numeric(out["MIN"], errors="coerce")
    out.sort_values(["SEASON_YEAR", "PLAYER_ID", "GAME_DATE"], ascending=True, inplace=True)
    valid_mask = out["MIN"].fillna(0) > 0
    shifted = (
        out[stat_col]
        .where(valid_mask)
        .groupby([out["SEASON_YEAR"], out["PLAYER_ID"]])
        .shift(1)
    )
    in_season = shifted.groupby(
        [out["SEASON_YEAR"], out["PLAYER_ID"]], group_keys=False
    ).transform(
        lambda s: s.ewm(halflife=halflife, adjust=False, min_periods=1, ignore_na=True).mean()
    )
    regular = valid_mask & out["GAME_ID"].map(_is_fallback_season_type)
    means = (
        out.loc[regular]
        .assign(_stat=pd.to_numeric(out.loc[regular, stat_col], errors="coerce"))
        .groupby(["PLAYER_ID", "SEASON_YEAR"], as_index=False)["_stat"]
        .mean()
    )
    means["SEASON_YEAR"] = means["SEASON_YEAR"] + 1
    means = means.rename(columns={"_stat": "_previous_season_stat"})
    merged = out[["PLAYER_ID", "SEASON_YEAR"]].merge(
        means, on=["PLAYER_ID", "SEASON_YEAR"], how="left"
    )
    previous = pd.Series(merged["_previous_season_stat"].to_numpy(), index=out.index)
    out[f"{stat_col}_CUM_AVG"] = in_season.fillna(previous).fillna(0)
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("stat_col", ["PTS", "TS_PCT"])
def test_same_averages_as_the_per_group_lambda(seed, stat_col):
    df = _players(seed)
    assert_identical(precompute_cumulative_avg_stat(df, stat_col), _before(df, stat_col))
