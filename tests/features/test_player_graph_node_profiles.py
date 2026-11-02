"""As-of node profiles: strictly before D, shrunk to the league, with confidence."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.player_graph.as_of import PointInTimeData
from nba_ou.data_processing.player_graph.node_profiles import (
    PROFILE_COLUMNS,
    ProfileParams,
    node_profiles,
)

D = pd.Timestamp("2024-01-10")


def _line(game, days, player, minutes, pts, start="", usage=0.2):
    return {
        "GAME_ID": game,
        "GAME_DATE": D - pd.Timedelta(days=days),
        "PLAYER_ID": player,
        "MIN": minutes,
        "PTS": pts,
        "FGA": pts / 2,
        "FG3A": 2,
        "FG3M": 1,
        "FTA": 2,
        "OREB": 1,
        "DREB": 4,
        "AST": 3,
        "TOV": 1,
        "STL": 1,
        "BLK": 0,
        "PF": 2,
        "USG_PCT": usage,
        "START_POSITION": start,
    }


def _box():
    rows = []
    for k in range(12):
        game = f"00223{k:05d}"
        rows.append(_line(game, 2 + k, "star", 36.0, 30, start="G", usage=0.32))
        rows.append(_line(game, 2 + k, "bench", 12.0, 4, usage=0.15))
    rows.append(_line("0012300001", 30, "star", 20.0, 50))  # preseason: ignored
    rows.append(_line("0022300999", 0, "star", 40.0, 80))  # on D itself: not seen
    return pd.DataFrame(rows)


def _profiles(params=None):
    data = PointInTimeData.from_frames(pd.DataFrame(), pd.DataFrame(), _box())
    params = params or ProfileParams()
    return node_profiles(data.as_of(D), ["star", "bench", "rookie"], params)


def test_schema_and_games_strictly_before_the_cutoff():
    profiles = _profiles()
    assert list(profiles.columns) == list(PROFILE_COLUMNS)
    assert profiles.loc["star", "games_window"] == 12
    assert profiles.loc["star", "days_since_last_game"] == 2
    assert profiles.loc["star", "games_in_data"] == 12


def test_rates_are_per_36_and_shrunk_toward_the_league():
    raw = _profiles(ProfileParams(prior_minutes=1e-9, half_life_days=1e9))
    assert raw.loc["star", "pts_per36"] == pytest.approx(30.0)
    assert raw.loc["bench", "pts_per36"] == pytest.approx(4 / 12 * 36)
    shrunk = _profiles()
    league = (12 * 30 + 12 * 4) / (12 * 48) * 36
    # Fewer minutes, more shrinkage: the bench player moves further.
    assert abs(shrunk.loc["bench", "pts_per36"] - raw.loc["bench", "pts_per36"]) > abs(
        shrunk.loc["star", "pts_per36"] - raw.loc["star", "pts_per36"]
    )
    assert shrunk.loc["rookie", "pts_per36"] == pytest.approx(league, rel=1e-3)


def test_role_and_confidence_columns():
    profiles = _profiles()
    assert profiles.loc["star", "start_share"] == 1.0
    assert profiles.loc["bench", "start_share"] == 0.0
    assert profiles.loc["star", "recent_minutes"] == 36.0
    assert profiles.loc["star", "usage"] > profiles.loc["bench", "usage"]
    assert profiles.loc["star", "prior_weight"] < profiles.loc["bench", "prior_weight"]
    rookie = profiles.loc["rookie"]
    assert rookie["prior_weight"] == 1.0 and not rookie["has_box_history"]
    assert rookie["minutes_decayed"] == 0 and np.isnan(rookie["days_since_last_game"])
    assert rookie[["p_G", "p_F", "p_C"]].sum() == pytest.approx(1.0)
