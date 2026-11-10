import numpy as np
import pandas as pd
import pytest
from nba_ou.config.constants import CITY_TO_LATLON
from nba_ou.data_processing.merged_home_away_data.add_features_after_merging import (
    add_high_value_features_for_team_points,
)
from nba_ou.data_processing.travel.travel_processing import (
    add_rolling_distances,
    compute_travel_features,
    haversine_km,
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
    # The Lakers open away: they flew in from Los Angeles the day before.
    assert result.loc[0, "JETLAG_HOURS_FROM_LAST_GAME_AWAY_TEAM"] == 3
    assert result.loc[1, "JETLAG_HOURS_FROM_LAST_GAME_HOME_TEAM"] == 3
    assert result.loc[1, "JETLAG_HOURS_FROM_LAST_GAME_AWAY_TEAM"] == 3
    assert result.loc[2, "JETLAG_HOURS_FROM_LAST_GAME_HOME_TEAM"] == 0
    assert result.loc[2, "JETLAG_HOURS_FROM_LAST_GAME_AWAY_TEAM"] == 0
    # Equal Boston/LA trips in 14 days, one of them in the last 2: Boston made
    # two, the Lakers three (their away opener counts as a trip from home).
    for side, fraction in (("HOME", 1 / 2), ("AWAY", 1 / 3)):
        assert result.loc[
            2, f"TRAVEL_RECENCY_RATIO_{side}_2D_OVER_14D_BEFORE"
        ] == pytest.approx(fraction)


def test_season_opener_travel_ignores_the_previous_season():
    def game(game_id, day, season, home, away):
        cities = {"BOS": "Boston", "LAL": "Los Angeles", "MIA": "Miami"}
        return {
            "GAME_ID": game_id,
            "GAME_DATE": day,
            "SEASON_YEAR": season,
            "TEAM_ID_TEAM_HOME": home,
            "TEAM_CITY_TEAM_HOME": cities[home],
            "TEAM_ID_TEAM_AWAY": away,
            "TEAM_CITY_TEAM_AWAY": cities[away],
        }

    df = pd.DataFrame(
        [
            # Last season ends with Boston in Los Angeles.
            game("001", "2025-06-10", 2024, "LAL", "BOS"),
            # Boston opens at home, Miami away; then the Lakers open away.
            game("002", "2025-10-21", 2025, "BOS", "MIA"),
            game("003", "2025-10-22", 2025, "MIA", "LAL"),
        ]
    )

    result = compute_travel_features(df, log_scale=False)

    boston_miami = haversine_km(*CITY_TO_LATLON["Boston"], *CITY_TO_LATLON["Miami"])
    la_miami = haversine_km(*CITY_TO_LATLON["Los Angeles"], *CITY_TO_LATLON["Miami"])
    # Home opener: no trip from last season's final city.
    assert result.loc[1, "TOTAL_KM_IN_LAST_1_DAYS_HOME_TEAM"] == 0
    assert result.loc[1, "JETLAG_HOURS_FROM_LAST_GAME_HOME_TEAM"] == 0
    # Away opener: one trip from the team's own city, the day before.
    assert result.loc[1, "TOTAL_KM_IN_LAST_1_DAYS_AWAY_TEAM"] == pytest.approx(
        boston_miami
    )
    assert result.loc[1, "JETLAG_HOURS_FROM_LAST_GAME_AWAY_TEAM"] == 0
    assert result.loc[2, "TOTAL_KM_IN_LAST_1_DAYS_AWAY_TEAM"] == pytest.approx(la_miami)
    assert result.loc[2, "JETLAG_HOURS_FROM_LAST_GAME_AWAY_TEAM"] == 3
    # Miami's second game is its first at home: the opener trip to Boston the
    # day before and the trip back.
    assert result.loc[2, "TOTAL_KM_IN_LAST_1_DAYS_HOME_TEAM"] == pytest.approx(
        2 * boston_miami
    )


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
