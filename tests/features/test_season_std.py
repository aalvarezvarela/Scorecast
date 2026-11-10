import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.statistics.statistics import compute_season_std


def _home_games(season, points, start):
    return pd.DataFrame(
        {
            "TEAM_ID": 1,
            "SEASON_YEAR": season,
            "HOME": True,
            "GAME_DATE": pd.date_range(start, periods=len(points), freq="3D"),
            "PTS": points,
        }
    )


def test_season_std_is_a_sample_std_and_needs_two_prior_games():
    last = _home_games(2024, [100.0, 110.0, 120.0], "2024-11-01")
    this = _home_games(2025, [118.0, 104.0, 111.0], "2025-11-01")

    out = compute_season_std(pd.concat([last, this]), param="PTS")
    std = out.sort_values("GAME_DATE")["PTS_SEASON_BEFORE_STD"].tolist()

    # No earlier season: nothing to fall back to before two games are played.
    assert np.isnan(std[0]) and np.isnan(std[1])
    assert std[2] == pytest.approx(np.std([100.0, 110.0], ddof=1))
    # First and second game of the next season: last season's sample std,
    # never 0.0 from a single game.
    previous = np.std([100.0, 110.0, 120.0], ddof=1)
    assert std[3] == pytest.approx(previous)
    assert std[4] == pytest.approx(previous)
    assert std[5] == pytest.approx(np.std([118.0, 104.0], ddof=1))
