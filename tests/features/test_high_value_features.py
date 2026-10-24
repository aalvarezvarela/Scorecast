import pandas as pd
import pytest
from nba_ou.data_processing.merged_home_away_data.add_features_after_merging import (
    add_high_value_features_for_team_points,
)


def test_points_trend_sum_and_difference_use_merged_source_names():
    df = pd.DataFrame(
        {
            "PTS_TREND_SLOPE_LAST_5_GAMES_BEFORE_TEAM_HOME": [1.5, -2.0],
            "PTS_TREND_SLOPE_LAST_5_GAMES_BEFORE_TEAM_AWAY": [0.5, 3.0],
        }
    )

    result = add_high_value_features_for_team_points(df)

    assert result[
        "PTS_TREND_SLOPE_DIFF_HOME_MINUS_AWAY_BEFORE"
    ].tolist() == pytest.approx([1.0, -5.0])
    assert result[
        "PTS_TREND_SLOPE_SUM_HOME_PLUS_AWAY_BEFORE"
    ].tolist() == pytest.approx([2.0, 1.0])


def test_points_trend_features_are_not_emitted_without_both_teams():
    df = pd.DataFrame(
        {
            "PTS_TREND_SLOPE_LAST_5_GAMES_BEFORE_TEAM_HOME": [1.5],
        }
    )

    result = add_high_value_features_for_team_points(df)

    assert "PTS_TREND_SLOPE_DIFF_HOME_MINUS_AWAY_BEFORE" not in result.columns
    assert "PTS_TREND_SLOPE_SUM_HOME_PLUS_AWAY_BEFORE" not in result.columns
