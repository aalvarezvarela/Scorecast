import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.merged_home_away_data.add_features_after_merging import (
    add_high_value_features_for_team_points,
)
from nba_ou.data_processing.travel.travel_processing import (
    add_rolling_distances,
    compute_travel_features,
    merge_travel_features,
)


def test_compute_travel_features_adds_recent_jetlag_hours():
    df = pd.DataFrame(
        [
            {
                "GAME_ID": "001",
                "GAME_DATE": "2026-01-01",
                "TEAM_ID_TEAM_HOME": "BOS",
                "TEAM_CITY_TEAM_HOME": "Boston",
                "TEAM_ID_TEAM_AWAY": "LAL",
                "TEAM_CITY_TEAM_AWAY": "Los Angeles",
            },
            {
                "GAME_ID": "002",
                "GAME_DATE": "2026-01-03",
                "TEAM_ID_TEAM_HOME": "LAL",
                "TEAM_CITY_TEAM_HOME": "Los Angeles",
                "TEAM_ID_TEAM_AWAY": "BOS",
                "TEAM_CITY_TEAM_AWAY": "Boston",
            },
            {
                "GAME_ID": "003",
                "GAME_DATE": "2026-01-10",
                "TEAM_ID_TEAM_HOME": "BOS",
                "TEAM_CITY_TEAM_HOME": "Boston",
                "TEAM_ID_TEAM_AWAY": "LAL",
                "TEAM_CITY_TEAM_AWAY": "Los Angeles",
            },
        ]
    )

    result = compute_travel_features(df, log_scale=True)

    assert result.loc[0, "JETLAG_HOURS_FROM_LAST_GAME_HOME_TEAM"] == 0
    assert result.loc[0, "JETLAG_HOURS_FROM_LAST_GAME_AWAY_TEAM"] == 0
    assert result.loc[1, "JETLAG_HOURS_FROM_LAST_GAME_HOME_TEAM"] == 3
    assert result.loc[1, "JETLAG_HOURS_FROM_LAST_GAME_AWAY_TEAM"] == 3
    assert result.loc[2, "JETLAG_HOURS_FROM_LAST_GAME_HOME_TEAM"] == 0
    assert result.loc[2, "JETLAG_HOURS_FROM_LAST_GAME_AWAY_TEAM"] == 0
    # Both teams made two equal Boston/LA trips in 14 days, one in the last 2.
    for side in ("HOME", "AWAY"):
        assert result.loc[
            2, f"TRAVEL_RECENCY_RATIO_{side}_2D_OVER_14D_BEFORE"
        ] == pytest.approx(0.5)


def test_rolling_travel_includes_current_trip_and_left_boundary():
    team_log = pd.DataFrame(
        {
            "GAME_ID": ["g1", "g2", "g3"],
            "GAME_DATE": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "TEAM_ID": ["team-1"] * 3,
            "TRAVEL_KM": [0.0, 100.0, 200.0],
        }
    )

    result = add_rolling_distances(team_log).set_index("GAME_ID")

    assert result.loc["g2", "KM_LAST_1_DAYS"] == pytest.approx(100.0)
    assert result.loc["g3", "KM_LAST_1_DAYS"] == pytest.approx(300.0)


@pytest.mark.parametrize("log_scale", [False, True])
def test_travel_recency_is_a_fraction_of_raw_kilometers(log_scale):
    games = pd.DataFrame(
        {
            "GAME_ID": ["g1", "g2", "g3", "g4"],
            "TEAM_ID_TEAM_HOME": "home",
            "TEAM_ID_TEAM_AWAY": "away",
        }
    )
    rows = []
    for home, short, long in (
        (True, [100.0, 1000.0, 0.0, 0.0], [1000.0, 10000.0, 0.0, 1000.0]),
        (False, [200.0, 2000.0, 0.0, 0.0], [400.0, 4000.0, 0.0, 2000.0]),
    ):
        for game, two_days, fourteen_days in zip(
            games.GAME_ID, short, long, strict=True
        ):
            rows.append(
                {
                    "GAME_ID": game,
                    "TEAM_ID": "home" if home else "away",
                    "IS_HOME": home,
                    "JETLAG_HOURS_FROM_LAST_GAME": 0.0,
                    **{
                        f"KM_LAST_{w}_DAYS": two_days if w <= 2 else fourteen_days
                        for w in (1, 2, 5, 7, 14)
                    },
                }
            )
    # Both dataset builders use this order: travel first, then interactions.
    result = add_high_value_features_for_team_points(
        merge_travel_features(games, pd.DataFrame(rows), log_scale=log_scale)
    )

    for side, fraction in (("HOME", 0.1), ("AWAY", 0.5)):
        ratio = result[f"TRAVEL_RECENCY_RATIO_{side}_2D_OVER_14D_BEFORE"]
        assert ratio.iloc[:2].tolist() == pytest.approx([fraction, fraction])
        assert pd.isna(ratio.iloc[2])  # No travel in either window: undefined.
        assert ratio.iloc[3] == 0.0  # Earlier travel, none in the last two days.
    expected = np.log1p(100.0) if log_scale else 100.0
    assert result.loc[0, "TOTAL_KM_IN_LAST_2_DAYS_HOME_TEAM"] == pytest.approx(expected)
