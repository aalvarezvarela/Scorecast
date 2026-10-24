import nba_ou.data_processing.team.rolling as rolling_module
import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.merged_home_away_data.select_train_columns import (
    select_training_columns,
)
from nba_ou.data_processing.statistics.statistics import compute_rolling_stats
from nba_ou.data_processing.team.rolling import compute_all_rolling_statistics
from nba_ou.data_processing.team.style_matchups import (
    STYLE_SOURCE_COLUMNS,
    add_style_matchup_features,
    add_team_style_source_features,
)


def _team_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": [1, 1, 2, 2],
            "GAME_DATE": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-03", "2026-01-03"]
            ),
            "SEASON_YEAR": [2025, 2025, 2025, 2025],
            "TEAM_ID": [10, 20, 10, 20],
            "HOME": [True, False, False, True],
            "FG3A": [30.0, 40.0, np.nan, np.nan],
            "FGA": [80.0, 100.0, np.nan, np.nan],
            "FTA": [20.0, 10.0, np.nan, np.nan],
            "TOV": [10.0, 20.0, np.nan, np.nan],
            "OREB": [8.0, 12.0, np.nan, np.nan],
            "POSS": [100.0, 100.0, np.nan, np.nan],
        }
    )


def test_team_style_sources_pair_each_team_with_its_opponent():
    result = add_team_style_source_features(_team_rows())
    team_10 = result[(result["GAME_ID"] == 1) & (result["TEAM_ID"] == 10)].iloc[0]

    assert team_10["STYLE_FG3A_RATE"] == pytest.approx(30 / 80)
    assert team_10["STYLE_FTA_RATE"] == pytest.approx(20 / 80)
    assert team_10["STYLE_TOV_RATE"] == pytest.approx(10 / 100)
    assert team_10["STYLE_OREB_RATE"] == pytest.approx(8 / 100)
    assert team_10["STYLE_FG3A_RATE_ALLOWED"] == pytest.approx(40 / 100)
    assert team_10["STYLE_FTA_RATE_ALLOWED"] == pytest.approx(10 / 100)
    assert team_10["STYLE_TOV_FORCED_RATE"] == pytest.approx(20 / 100)
    assert team_10["STYLE_OREB_RATE_ALLOWED"] == pytest.approx(12 / 100)


def test_scheduled_game_style_sources_remain_missing():
    result = add_team_style_source_features(_team_rows())
    scheduled = result[result["GAME_ID"] == 2]

    assert scheduled[list(STYLE_SOURCE_COLUMNS)].isna().all().all()

    rolled = compute_rolling_stats(
        result,
        "STYLE_FG3A_RATE",
        window=5,
        add_extra_season_avg=True,
        add_relative_column=False,
    )
    scheduled_team_10 = rolled[
        (rolled["GAME_ID"] == 2) & (rolled["TEAM_ID"] == 10)
    ].iloc[0]
    assert scheduled_team_10[
        "STYLE_FG3A_RATE_LAST_ALL_5_MATCHES_BEFORE"
    ] == pytest.approx(30 / 80)


def test_current_box_score_cannot_change_its_own_before_feature():
    rows = pd.DataFrame(
        {
            "GAME_ID": [1, 2, 3],
            "GAME_DATE": pd.to_datetime(
                ["2026-01-01", "2026-01-03", "2026-01-05"]
            ),
            "SEASON_YEAR": [2025, 2025, 2025],
            "TEAM_ID": [10, 10, 10],
            "HOME": [True, False, True],
            "STYLE_FG3A_RATE": [0.25, 0.50, 0.75],
        }
    )

    original = compute_rolling_stats(
        rows,
        "STYLE_FG3A_RATE",
        window=5,
        add_extra_season_avg=True,
        add_relative_column=False,
    ).set_index("GAME_ID")

    changed_rows = rows.copy()
    changed_rows.loc[changed_rows["GAME_ID"] == 2, "STYLE_FG3A_RATE"] = 0.99
    changed = compute_rolling_stats(
        changed_rows,
        "STYLE_FG3A_RATE",
        window=5,
        add_extra_season_avg=True,
        add_relative_column=False,
    ).set_index("GAME_ID")

    before_col = "STYLE_FG3A_RATE_LAST_ALL_5_MATCHES_BEFORE"
    assert original.at[2, before_col] == pytest.approx(0.25)
    assert changed.at[2, before_col] == pytest.approx(original.at[2, before_col])
    assert changed.at[3, before_col] != pytest.approx(original.at[3, before_col])


def _matchup_row() -> pd.DataFrame:
    values = {
        "EXPECTED_POSS_FROM_PACE_BEFORE": 100.0,
        "REF_AVG_TOTAL_PF_DIFF_BEFORE": 2.0,
        "STYLE_FG3A_RATE_SEASON_BEFORE_AVG_TEAM_HOME": 0.40,
        "STYLE_FG3A_RATE_ALLOWED_SEASON_BEFORE_AVG_TEAM_HOME": 0.35,
        "STYLE_FG3A_RATE_SEASON_BEFORE_AVG_TEAM_AWAY": 0.30,
        "STYLE_FG3A_RATE_ALLOWED_SEASON_BEFORE_AVG_TEAM_AWAY": 0.25,
        "STYLE_FTA_RATE_SEASON_BEFORE_AVG_TEAM_HOME": 0.25,
        "STYLE_FTA_RATE_ALLOWED_SEASON_BEFORE_AVG_TEAM_HOME": 0.10,
        "STYLE_FTA_RATE_SEASON_BEFORE_AVG_TEAM_AWAY": 0.15,
        "STYLE_FTA_RATE_ALLOWED_SEASON_BEFORE_AVG_TEAM_AWAY": 0.20,
        "STYLE_TOV_RATE_SEASON_BEFORE_AVG_TEAM_HOME": 0.12,
        "STYLE_TOV_FORCED_RATE_SEASON_BEFORE_AVG_TEAM_HOME": 0.13,
        "STYLE_TOV_RATE_SEASON_BEFORE_AVG_TEAM_AWAY": 0.11,
        "STYLE_TOV_FORCED_RATE_SEASON_BEFORE_AVG_TEAM_AWAY": 0.14,
        "STYLE_OREB_RATE_SEASON_BEFORE_AVG_TEAM_HOME": 0.10,
        "STYLE_OREB_RATE_ALLOWED_SEASON_BEFORE_AVG_TEAM_HOME": 0.07,
        "STYLE_OREB_RATE_SEASON_BEFORE_AVG_TEAM_AWAY": 0.09,
        "STYLE_OREB_RATE_ALLOWED_SEASON_BEFORE_AVG_TEAM_AWAY": 0.08,
        "STYLE_FGA_PER_POSS_SEASON_BEFORE_AVG_TEAM_HOME": 0.90,
        "STYLE_FGA_PER_POSS_ALLOWED_SEASON_BEFORE_AVG_TEAM_HOME": 0.84,
        "STYLE_FGA_PER_POSS_SEASON_BEFORE_AVG_TEAM_AWAY": 0.85,
        "STYLE_FGA_PER_POSS_ALLOWED_SEASON_BEFORE_AVG_TEAM_AWAY": 0.86,
    }
    return pd.DataFrame([values])


def test_matchup_features_combine_only_before_offense_and_defense_inputs():
    result_df = add_style_matchup_features(_matchup_row())
    result = result_df.iloc[0]

    assert result["STYLE_EXPECTED_FG3A_RATE_HOME_BEFORE"] == pytest.approx(0.325)
    assert result["STYLE_EXPECTED_FG3A_RATE_AWAY_BEFORE"] == pytest.approx(0.325)
    assert result["STYLE_EXPECTED_FTA_RATE_HOME_BEFORE"] == pytest.approx(0.225)
    assert result["STYLE_EXPECTED_FTA_RATE_AWAY_BEFORE"] == pytest.approx(0.125)
    assert result["STYLE_EXPECTED_TOV_RATE_HOME_BEFORE"] == pytest.approx(0.13)
    assert result["STYLE_EXPECTED_TOV_RATE_AWAY_BEFORE"] == pytest.approx(0.12)
    assert result["STYLE_EXPECTED_OREB_RATE_HOME_BEFORE"] == pytest.approx(0.09)
    assert result["STYLE_EXPECTED_OREB_RATE_AWAY_BEFORE"] == pytest.approx(0.08)
    assert result["STYLE_EXPECTED_TOTAL_FG3A_BEFORE"] == pytest.approx(56.0625)
    assert result["STYLE_EXPECTED_TOTAL_FTA_BEFORE"] == pytest.approx(30.3625)
    assert result["STYLE_EXPECTED_TOTAL_TOV_BEFORE"] == pytest.approx(25.0)
    assert result["STYLE_EXPECTED_TOTAL_OREB_BEFORE"] == pytest.approx(17.0)
    assert result["STYLE_FTA_REFEREE_INTERACTION_BEFORE"] == pytest.approx(0.35)
    assert not any(
        column.startswith(STYLE_SOURCE_COLUMNS)
        for column in result_df.columns
    )
    matchup_columns = [
        column
        for column in result_df.columns
        if column.startswith(("STYLE_EXPECTED_", "STYLE_FTA_REFEREE_"))
    ]
    assert len(matchup_columns) == 13


def test_matchup_builder_rejects_non_lagged_history_suffix():
    with pytest.raises(ValueError, match="BEFORE"):
        add_style_matchup_features(_matchup_row(), history_suffix="SEASON_AVG")


def test_style_sources_receive_only_compact_shifted_rolling_features(monkeypatch):
    style_calls = []

    def fake_rolling_stats(df, parameter, **kwargs):
        if parameter in STYLE_SOURCE_COLUMNS:
            style_calls.append((parameter, kwargs))
            out = df.copy()
            out[f"{parameter}_LAST_ALL_5_MATCHES_BEFORE"] = 0.1
            out[f"{parameter}_SEASON_BEFORE_AVG"] = 0.2
            return out
        return df

    monkeypatch.setattr(rolling_module, "compute_rolling_stats", fake_rolling_stats)
    monkeypatch.setattr(
        rolling_module,
        "compute_rolling_weighted_stats",
        lambda df, parameter, **kwargs: df,
    )
    monkeypatch.setattr(
        rolling_module,
        "compute_season_std",
        lambda df, param: df,
    )
    monkeypatch.setattr(
        rolling_module,
        "compute_trend_slope",
        lambda df, parameter, **kwargs: df,
    )

    result = compute_all_rolling_statistics(
        pd.DataFrame(
            {
                "TEAM_ID": [10],
                "HOME": [True],
                "GAME_DATE": ["2026-01-01"],
                "SEASON_YEAR": [2025],
            }
        )
    )

    assert [parameter for parameter, _ in style_calls] == list(STYLE_SOURCE_COLUMNS)
    assert all(kwargs["window"] == 5 for _, kwargs in style_calls)
    assert all(kwargs["add_extra_season_avg"] for _, kwargs in style_calls)
    assert all(not kwargs["add_relative_column"] for _, kwargs in style_calls)
    assert all(
        f"{source}_SEASON_BEFORE_AVG" in result.columns
        for source in STYLE_SOURCE_COLUMNS
    )
    assert not any(
        f"{source}_LAST_ALL_5_MATCHES_BEFORE" in result.columns
        for source in STYLE_SOURCE_COLUMNS
    )


def test_same_game_style_sources_are_not_selected_for_model_training():
    raw_source = "STYLE_FG3A_RATE_TEAM_HOME"
    lagged_feature = "STYLE_FG3A_RATE_SEASON_BEFORE_AVG_TEAM_HOME"
    selected = select_training_columns(
        pd.DataFrame(
            {
                "GAME_ID": [1],
                raw_source: [0.50],
                lagged_feature: [0.35],
            }
        ),
        original_columns=[],
    )

    assert raw_source not in selected.columns
    assert lagged_feature in selected.columns
