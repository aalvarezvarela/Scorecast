import pandas as pd
from nba_ou.data_processing.team.records import (
    FIRST_GAME_REST_DAYS,
    add_team_record_before_game,
    compute_rest_days_before_match,
)


def test_add_team_record_before_game_does_not_shift_across_groups():
    df = pd.DataFrame(
        {
            "GAME_DATE": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-03",
                ]
            ),
            "WL": ["W", "W", "L", "W"],
            "SEASON_TYPE": ["Regular Season"] * 4,
            "SEASON_ID": ["22025", "22025", "22025", "22025"],
            "TEAM_ID": ["A", "A", "B", "B"],
        }
    )

    result = add_team_record_before_game(df)

    team_a = result[result["TEAM_ID"] == "A"].sort_values("GAME_DATE")
    team_b = result[result["TEAM_ID"] == "B"].sort_values("GAME_DATE")

    assert team_a["WINS_BEFORE_THIS_GAME"].tolist() == [0, 1]
    assert team_a["TEAM_RECORD_BEFORE_GAME"].tolist() == [0.0, 1.0]
    assert team_b["WINS_BEFORE_THIS_GAME"].tolist() == [0, 0]
    assert team_b["TEAM_RECORD_BEFORE_GAME"].tolist() == [0.0, 0.0]


def test_first_game_rest_is_seven_days_per_team_and_season():
    df = pd.DataFrame(
        {
            "GAME_ID": ["a2", "b1", "a1", "a3"],
            "GAME_DATE": pd.to_datetime(
                ["2025-10-03", "2025-10-02", "2025-10-01", "2026-10-01"]
            ),
            "TEAM_ID": ["A", "B", "A", "A"],
            "SEASON_YEAR": [2025, 2025, 2025, 2026],
        }
    )

    result = compute_rest_days_before_match(df).set_index("GAME_ID")

    assert result.loc["a1", "REST_DAYS_BEFORE_MATCH"] == FIRST_GAME_REST_DAYS
    assert result.loc["a2", "REST_DAYS_BEFORE_MATCH"] == 2
    assert result.loc["a3", "REST_DAYS_BEFORE_MATCH"] == FIRST_GAME_REST_DAYS
    assert result.loc["b1", "REST_DAYS_BEFORE_MATCH"] == FIRST_GAME_REST_DAYS
