import numpy as np
import pandas as pd
import pytest
from nba_ou.config.constants import (
    OVERTIME_THRESHOLD_MINUTES,
    REGULATION_GAME_MINUTES,
    TEAM_MINUTES_PER_40,
)
from nba_ou.data_processing.team.cleaning_teams import (
    OVERTIME_NORMALIZED_COUNTING_STATS,
    PTS_PER_40_COLUMN,
    adjust_overtime,
)
from nba_ou.data_processing.team.rolling import (
    IS_OVERTIME_LAST_GAME_BEFORE,
    OVERTIME_FREQUENCY_LAST_5_GAMES_BEFORE,
    OVERTIME_FREQUENCY_SEASON_YEAR_BEFORE,
    add_overtime_history_features,
)
from nba_ou.data_processing.team.totals import compute_total_points_features


def test_overtime_keeps_raw_points_and_actual_total():
    df = pd.DataFrame(
        {
            "GAME_ID": ["game-1", "game-1"],
            "GAME_DATE": ["2026-01-01", "2026-01-01"],
            "TEAM_ID": ["home", "away"],
            "MIN": [265, 265],
            "PTS": [120, 110],
            "ODDS_TOTAL_LINE_bet365": [220.0, 220.0],
        }
    )

    result = compute_total_points_features(adjust_overtime(df)).set_index("TEAM_ID")

    assert result["PTS"].to_dict() == {"home": 120, "away": 110}
    assert result.loc["home", PTS_PER_40_COLUMN] == pytest.approx(120 * 200 / 265)
    assert result.loc["away", PTS_PER_40_COLUMN] == pytest.approx(110 * 200 / 265)
    assert result["TOTAL_POINTS"].eq(230).all()
    assert result["DIFF_FROM_ODDS_LINE_bet365"].eq(10.0).all()


def test_overtime_normalizes_only_allowlisted_additive_statistics():
    original_values = {
        column: float(index + 10)
        for index, column in enumerate(OVERTIME_NORMALIZED_COUNTING_STATS)
    }
    df = pd.DataFrame(
        {
            "GAME_ID": ["game-1"],
            "GAME_DATE": ["2026-01-01"],
            "TEAM_ID": ["home"],
            "MIN": [265],
            "PTS": [120],
            **{column: [value] for column, value in original_values.items()},
        }
    )

    result = adjust_overtime(df).iloc[0]
    factor = REGULATION_GAME_MINUTES / 265

    for column, original_value in original_values.items():
        assert result[column] == pytest.approx(original_value * factor)

    assert result["PTS"] == 120
    assert result[PTS_PER_40_COLUMN] == pytest.approx(120 * 200 / 265)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("FG_PCT", 0.500),
        ("FG3_PCT", 0.375),
        ("FT_PCT", 0.825),
        ("E_OFF_RATING", 116.2),
        ("OFF_RATING", 115.8),
        ("E_DEF_RATING", 110.1),
        ("DEF_RATING", 109.7),
        ("E_NET_RATING", 6.1),
        ("NET_RATING", 6.0),
        ("AST_PCT", 0.625),
        ("AST_TOV", 2.4),
        ("AST_RATIO", 19.5),
        ("OREB_PCT", 0.245),
        ("DREB_PCT", 0.755),
        ("REB_PCT", 0.510),
        ("E_TM_TOV_PCT", 11.8),
        ("TM_TOV_PCT", 11.5),
        ("EFG_PCT", 0.565),
        ("TS_PCT", 0.590),
        ("USG_PCT", 1.0),
        ("E_USG_PCT", 1.0),
        ("E_PACE", 101.4),
        ("PACE", 100.9),
        ("PACE_PER40", 84.1),
        ("PIE", 0.550),
        ("SEASON_YEAR", 2025),
        ("UNRECOGNIZED_NUMERIC_METADATA", 1234.5),
    ],
)
def test_overtime_preserves_rates_ratings_and_numeric_metadata(column, value):
    df = pd.DataFrame(
        {
            "GAME_ID": ["game-1"],
            "GAME_DATE": ["2026-01-01"],
            "TEAM_ID": ["home"],
            "MIN": [265],
            "PTS": [120],
            column: [value],
        }
    )

    result = adjust_overtime(df).iloc[0]

    assert result[column] == value
    assert result["MIN"] == 265
    assert result["PTS"] == 120


def test_regulation_game_keeps_points_without_rescaling_counts():
    df = pd.DataFrame(
        {
            "GAME_ID": ["game-1"],
            "GAME_DATE": ["2026-01-01"],
            "TEAM_ID": ["home"],
            "MIN": [REGULATION_GAME_MINUTES],
            "PTS": [120],
            "FGM": [45],
            "POSS": [100.5],
        }
    )

    result = adjust_overtime(df).iloc[0]

    assert result["IS_OVERTIME"] == 0
    assert result["PTS"] == 120
    assert result[PTS_PER_40_COLUMN] == pytest.approx(100.0)
    assert result["FGM"] == 45
    assert result["POSS"] == 100.5


def test_points_per_40_requires_positive_finite_minutes():
    df = pd.DataFrame(
        {
            "GAME_ID": [
                "short",
                "zero",
                "negative",
                "missing",
                "infinite",
                "regulation",
            ],
            "GAME_DATE": ["2026-01-01"] * 6,
            "TEAM_ID": ["one", "two", "three", "four", "five", "six"],
            "MIN": [160, 0, -10, None, np.inf, REGULATION_GAME_MINUTES],
            "PTS": [100, 101, 102, 103, 104, 105],
        }
    )

    original_points = df.set_index("GAME_ID")["PTS"].to_dict()
    result = adjust_overtime(df).set_index("GAME_ID")

    assert result.loc["short", PTS_PER_40_COLUMN] == pytest.approx(
        100 * TEAM_MINUTES_PER_40 / 160
    )
    assert (
        result.loc[["zero", "negative", "missing", "infinite"], PTS_PER_40_COLUMN]
        .isna()
        .all()
    )
    assert result.loc["regulation", PTS_PER_40_COLUMN] == pytest.approx(87.5)
    assert result["PTS"].to_dict() == original_points
    assert not np.isinf(result[PTS_PER_40_COLUMN]).any()


@pytest.mark.parametrize("minutes", [160.0, 239.0, 240.0, 265.0, 290.0])
def test_points_per_40_preserves_the_same_scoring_rate_at_every_duration(minutes):
    # MIN sums the five players' minutes. A 2.5 points/game-minute rate is
    # always 100 points per 40 minutes, including regulation and overtime.
    points = 2.5 * minutes / 5
    df = pd.DataFrame({"GAME_DATE": ["2026-01-01"], "MIN": [minutes], "PTS": [points]})

    result = adjust_overtime(df).iloc[0]

    assert result[PTS_PER_40_COLUMN] == pytest.approx(100.0)
    assert result["PTS"] == points
    assert result["MIN"] == minutes


def test_overtime_flag_and_normalization_share_one_threshold():
    below_threshold = OVERTIME_THRESHOLD_MINUTES - 1
    at_threshold = OVERTIME_THRESHOLD_MINUTES
    df = pd.DataFrame(
        {
            "GAME_ID": ["below", "at"],
            "GAME_DATE": ["2026-01-01", "2026-01-02"],
            "TEAM_ID": ["one", "two"],
            "MIN": [below_threshold, at_threshold],
            "PTS": [100, 100],
            "FGM": [40, 40],
        }
    )

    result = adjust_overtime(df).set_index("GAME_ID")

    assert result.loc["below", "IS_OVERTIME"] == 0
    assert result.loc["below", "FGM"] == 40
    assert result.loc["at", "IS_OVERTIME"] == 1
    assert result.loc["at", "FGM"] == pytest.approx(
        40 * REGULATION_GAME_MINUTES / at_threshold
    )


def test_overtime_history_is_prior_only_and_season_frequency_resets():
    df = pd.DataFrame(
        {
            "GAME_ID": ["g6", "g3", "g1", "g5", "g4", "g2"],
            "GAME_DATE": [
                "2026-01-08",
                "2025-10-22",
                "2025-04-01",
                "2026-01-07",
                "2025-10-24",
                "2025-04-03",
            ],
            "TEAM_ID": ["team-1"] * 6,
            "SEASON_YEAR": [2025, 2025, 2024, 2025, 2025, 2024],
            "IS_OVERTIME": [None, 1, 0, None, 0, 1],
        }
    )

    result = add_overtime_history_features(df).set_index("GAME_ID")

    assert result.loc["g1", IS_OVERTIME_LAST_GAME_BEFORE] == 0
    assert result.loc["g1", OVERTIME_FREQUENCY_LAST_5_GAMES_BEFORE] == 0
    assert result.loc["g1", OVERTIME_FREQUENCY_SEASON_YEAR_BEFORE] == 0

    assert result.loc["g2", IS_OVERTIME_LAST_GAME_BEFORE] == 0
    assert result.loc["g2", OVERTIME_FREQUENCY_LAST_5_GAMES_BEFORE] == 0
    assert result.loc["g2", OVERTIME_FREQUENCY_SEASON_YEAR_BEFORE] == 0

    assert result.loc["g3", IS_OVERTIME_LAST_GAME_BEFORE] == 1
    assert result.loc["g3", OVERTIME_FREQUENCY_LAST_5_GAMES_BEFORE] == pytest.approx(
        0.5
    )
    assert result.loc["g3", OVERTIME_FREQUENCY_SEASON_YEAR_BEFORE] == 0

    assert result.loc["g4", IS_OVERTIME_LAST_GAME_BEFORE] == 1
    assert result.loc["g4", OVERTIME_FREQUENCY_LAST_5_GAMES_BEFORE] == pytest.approx(
        2 / 3
    )
    assert result.loc["g4", OVERTIME_FREQUENCY_SEASON_YEAR_BEFORE] == 1

    for game_id in ("g5", "g6"):
        assert result.loc[game_id, IS_OVERTIME_LAST_GAME_BEFORE] == 0
        assert result.loc[
            game_id, OVERTIME_FREQUENCY_LAST_5_GAMES_BEFORE
        ] == pytest.approx(0.5)
        assert result.loc[
            game_id, OVERTIME_FREQUENCY_SEASON_YEAR_BEFORE
        ] == pytest.approx(0.5)


def test_overtime_last_five_uses_only_five_completed_games():
    df = pd.DataFrame(
        {
            "GAME_ID": [f"g{index}" for index in range(1, 8)],
            "GAME_DATE": pd.date_range("2026-01-01", periods=7),
            "TEAM_ID": ["team-1"] * 7,
            "SEASON_YEAR": [2025] * 7,
            "IS_OVERTIME": [1, 1, 0, 0, 0, 1, None],
        }
    )

    result = add_overtime_history_features(df).set_index("GAME_ID")

    assert result.loc["g7", OVERTIME_FREQUENCY_LAST_5_GAMES_BEFORE] == pytest.approx(
        0.4
    )


def test_overtime_history_does_not_mix_teams():
    df = pd.DataFrame(
        {
            "GAME_ID": ["a1", "b1", "a2", "b2"],
            "GAME_DATE": ["2026-01-01", "2026-01-01", "2026-01-03", "2026-01-03"],
            "TEAM_ID": ["A", "B", "A", "B"],
            "SEASON_YEAR": [2025] * 4,
            "IS_OVERTIME": [1, 0, 0, 1],
        }
    )

    result = add_overtime_history_features(df).set_index("GAME_ID")

    assert result.loc["a2", IS_OVERTIME_LAST_GAME_BEFORE] == 1
    assert result.loc["b2", IS_OVERTIME_LAST_GAME_BEFORE] == 0


def test_overtime_history_rejects_non_binary_completed_game_flags():
    df = pd.DataFrame(
        {
            "GAME_ID": ["g1"],
            "GAME_DATE": ["2026-01-01"],
            "TEAM_ID": ["team-1"],
            "SEASON_YEAR": [2025],
            "IS_OVERTIME": [2],
        }
    )

    with pytest.raises(ValueError, match="IS_OVERTIME must contain only 0, 1, or null"):
        add_overtime_history_features(df)
