"""The frame-wide head-to-head lookup gives exactly the per-row function's values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.historic_games.historic_games_statistics import (
    get_last_5_matchup_excluding_current,
    last_5_matchups_for_frame,
)

from .frame_identity import assert_identical


def _games(seed: int, *, ties: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 600
    teams = [1610612737 + i for i in range(6)]
    home = rng.choice(teams, n).astype(object)
    away = rng.choice(teams, n).astype(object)
    home[rng.random(n) < 0.02] = np.nan
    days = rng.integers(0, 120 if ties else 4000, n)
    dates = pd.Series(pd.Timestamp("2018-10-01") + pd.to_timedelta(days, unit="D"))
    dates[rng.random(n) < 0.02] = pd.NaT
    totals = rng.integers(180, 260, n).astype(float)
    totals[rng.random(n) < 0.05] = np.nan  # scheduled games
    return pd.DataFrame(
        {
            "TEAM_ID_TEAM_HOME": home,
            "TEAM_ID_TEAM_AWAY": away,
            "GAME_DATE": dates,
            "TOTAL_POINTS": totals,
        },
        index=rng.permutation(n),
    )


def _reference(df: pd.DataFrame) -> pd.DataFrame:
    rows = df.apply(lambda row: get_last_5_matchup_excluding_current(row, df), axis=1)
    return pd.DataFrame(rows.tolist(), index=df.index)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("ties", [False, True])
def test_same_values_as_the_per_row_lookup(seed, ties):
    df = _games(seed, ties=ties)
    fast = pd.DataFrame(last_5_matchups_for_frame(df), index=df.index)
    assert_identical(fast, _reference(df))


def test_integer_totals_keep_their_type():
    df = _games(3, ties=False).dropna(subset=["TOTAL_POINTS"])
    df["TOTAL_POINTS"] = df["TOTAL_POINTS"].astype("int64")
    fast = pd.DataFrame(last_5_matchups_for_frame(df), index=df.index)
    assert_identical(fast, _reference(df))
