import pandas as pd
import pytest
from nba_ou.config.constants import TEAM_ID_MAP
from nba_ou.data_processing.merged_home_away_data.select_train_columns import (
    select_training_columns,
)
from nba_ou.data_processing.merged_home_away_data.team_one_hot_features import (
    CATEGORICAL_AWAY_COLUMN,
    CATEGORICAL_HOME_COLUMN,
    add_team_one_hot_features,
)


def _slug(team_name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in team_name.upper()).strip(
        "_"
    )


def test_add_team_one_hot_features_adds_home_and_away_columns_for_all_teams():
    df = pd.DataFrame(
        {
            "TEAM_ID_TEAM_HOME": [TEAM_ID_MAP["Boston Celtics"]],
            "TEAM_ID_TEAM_AWAY": [TEAM_ID_MAP["Los Angeles Lakers"]],
        }
    )

    result = add_team_one_hot_features(df)

    home_cols = [col for col in result.columns if col.startswith("TEAM_HOME_")]
    away_cols = [col for col in result.columns if col.startswith("TEAM_AWAY_")]
    assert len(home_cols) == 30
    assert len(away_cols) == 30
    assert result[home_cols].sum(axis=1).iloc[0] == 1
    assert result[away_cols].sum(axis=1).iloc[0] == 1
    assert result.loc[0, "TEAM_HOME_BOSTON_CELTICS_BEFORE"] == 1
    assert result.loc[0, "TEAM_AWAY_LOS_ANGELES_LAKERS_BEFORE"] == 1


def test_team_one_hot_columns_match_team_map_names():
    result = add_team_one_hot_features(
        pd.DataFrame(
            {
                "TEAM_ID_TEAM_HOME": [TEAM_ID_MAP["Boston Celtics"]],
                "TEAM_ID_TEAM_AWAY": [TEAM_ID_MAP["Los Angeles Lakers"]],
            }
        )
    )

    expected_cols = {
        f"TEAM_HOME_{_slug(team_name)}_BEFORE" for team_name in TEAM_ID_MAP
    } | {f"TEAM_AWAY_{_slug(team_name)}_BEFORE" for team_name in TEAM_ID_MAP}

    assert expected_cols.issubset(set(result.columns))


def test_team_one_hot_columns_survive_training_column_selection():
    df = pd.DataFrame(
        {
            "SEASON_ID": ["22025"],
            "IS_OVERTIME": [0],
            "GAME_ID": ["game"],
            "GAME_DATE": [pd.Timestamp("2026-01-01")],
            "SEASON_TYPE": ["Regular Season"],
            "IS_PLAYOFF_GAME_BEFORE": [0],
            "SEASON_YEAR": [2025],
            "TEAM_ID_TEAM_HOME": [TEAM_ID_MAP["Boston Celtics"]],
            "TEAM_ID_TEAM_AWAY": [TEAM_ID_MAP["Los Angeles Lakers"]],
        }
    )
    df = add_team_one_hot_features(df)

    selected = select_training_columns(df, original_columns=[])

    assert "TEAM_HOME_BOSTON_CELTICS_BEFORE" in selected.columns
    assert "TEAM_AWAY_LOS_ANGELES_LAKERS_BEFORE" in selected.columns
    assert selected.loc[0, "TEAM_HOME_BOSTON_CELTICS_BEFORE"] == 1
    assert selected.loc[0, "TEAM_AWAY_LOS_ANGELES_LAKERS_BEFORE"] == 1


def test_add_team_one_hot_features_raises_for_missing_team_id_columns():
    with pytest.raises(ValueError, match="TEAM_ID_TEAM_HOME"):
        add_team_one_hot_features(pd.DataFrame({"TEAM_ID": ["1"]}))


def test_categorical_mode_adds_two_categorical_columns_with_team_name_categories():
    df = pd.DataFrame(
        {
            "TEAM_ID_TEAM_HOME": [
                TEAM_ID_MAP["Boston Celtics"],
                TEAM_ID_MAP["Los Angeles Lakers"],
            ],
            "TEAM_ID_TEAM_AWAY": [
                TEAM_ID_MAP["Los Angeles Lakers"],
                TEAM_ID_MAP["Boston Celtics"],
            ],
        }
    )

    result = add_team_one_hot_features(df, categorical_team_encoding=True)

    one_hot_cols = [
        col
        for col in result.columns
        if col.startswith("TEAM_HOME_") or col.startswith("TEAM_AWAY_")
    ]
    assert set(one_hot_cols) == {CATEGORICAL_HOME_COLUMN, CATEGORICAL_AWAY_COLUMN}
    assert isinstance(result[CATEGORICAL_HOME_COLUMN].dtype, pd.CategoricalDtype)
    assert isinstance(result[CATEGORICAL_AWAY_COLUMN].dtype, pd.CategoricalDtype)
    assert list(result[CATEGORICAL_HOME_COLUMN].cat.categories) == list(
        TEAM_ID_MAP.keys()
    )
    assert result.loc[0, CATEGORICAL_HOME_COLUMN] == "Boston Celtics"
    assert result.loc[0, CATEGORICAL_AWAY_COLUMN] == "Los Angeles Lakers"


def test_categorical_mode_columns_survive_training_column_selection():
    df = pd.DataFrame(
        {
            "SEASON_ID": ["22025"],
            "IS_OVERTIME": [0],
            "GAME_ID": ["game"],
            "GAME_DATE": [pd.Timestamp("2026-01-01")],
            "SEASON_TYPE": ["Regular Season"],
            "IS_PLAYOFF_GAME_BEFORE": [0],
            "SEASON_YEAR": [2025],
            "TEAM_ID_TEAM_HOME": [TEAM_ID_MAP["Boston Celtics"]],
            "TEAM_ID_TEAM_AWAY": [TEAM_ID_MAP["Los Angeles Lakers"]],
        }
    )
    df = add_team_one_hot_features(df, categorical_team_encoding=True)

    selected = select_training_columns(df, original_columns=[])

    assert CATEGORICAL_HOME_COLUMN in selected.columns
    assert CATEGORICAL_AWAY_COLUMN in selected.columns
    assert selected.loc[0, CATEGORICAL_HOME_COLUMN] == "Boston Celtics"
    assert selected.loc[0, CATEGORICAL_AWAY_COLUMN] == "Los Angeles Lakers"
