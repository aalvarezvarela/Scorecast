import math

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.all_star_voting.attach_all_star_voting_features import (
    add_all_star_voting_features,
    all_star_season_year_for_game_date,
)
from nba_ou.data_processing.merged_home_away_data.merge_home_away import (
    merge_home_away_data,
)

BOS_ID = "1610612738"
NYK_ID = "1610612752"
LAL_ID = "1610612747"


def _team_row(team_id=BOS_ID, game_date="2026-03-02"):
    return pd.DataFrame(
        {
            "GAME_ID": ["predict"],
            "TEAM_ID": [team_id],
            "SEASON_ID": ["22025"],
            "SEASON_YEAR": [2025],
            "GAME_DATE": [pd.Timestamp(game_date)],
        }
    )


def _players_df(rows=None):
    columns = [
        "GAME_ID",
        "PLAYER_ID",
        "TEAM_ID",
        "SEASON_ID",
        "SEASON_YEAR",
        "GAME_DATE",
        "MIN",
        "PTS",
    ]
    return pd.DataFrame(rows or [], columns=columns)


def _all_star_df(rows):
    return pd.DataFrame(
        rows,
        columns=["season_year", "player_id", "team_name", "fan_votes", "score"],
    )


def test_all_star_season_year_for_game_date_sanity_cases():
    assert all_star_season_year_for_game_date("2026-02-20") == 2024
    assert all_star_season_year_for_game_date("2026-03-02") == 2025
    assert all_star_season_year_for_game_date("2025-11-01") == 2024
    assert all_star_season_year_for_game_date("2024-02-28") == 2022
    assert all_star_season_year_for_game_date("2024-03-01") == 2023


def test_trade_out_removes_all_star_player_from_old_team():
    all_star = _all_star_df(
        [
            [2025, "p1", "Boston Celtics", 40, 1.5],
            [2025, "p2", "Los Angeles Lakers", 60, 2.5],
        ]
    )
    players = _players_df(
        [
            ["g1", "p1", BOS_ID, "22025", 2025, "2026-01-10", 10, 5],
            ["g2", "p1", NYK_ID, "22025", 2025, "2026-02-20", 10, 5],
        ]
    )

    result = add_all_star_voting_features(_team_row(BOS_ID), players, all_star, {})

    assert result.loc[0, "ALL_STAR_CANDIDATE_COUNT_BEFORE"] == 0
    assert result.loc[0, "ALL_STAR_FAN_VOTES_BEFORE"] == 0
    assert result.loc[0, "ALL_STAR_FAN_VOTE_SHARE_BEFORE"] == 0


def test_trade_in_adds_current_roster_player_from_other_all_star_team():
    all_star = _all_star_df(
        [
            [2025, "p1", "Boston Celtics", 40, 1.5],
            [2025, "p2", "Los Angeles Lakers", 60, 2.5],
        ]
    )
    players = _players_df(
        [
            ["g1", "p1", BOS_ID, "22025", 2025, "2026-01-10", 10, 5],
            ["g2", "p1", NYK_ID, "22025", 2025, "2026-02-20", 10, 5],
        ]
    )

    result = add_all_star_voting_features(_team_row(NYK_ID), players, all_star, {})

    assert result.loc[0, "ALL_STAR_CANDIDATE_COUNT_BEFORE"] == 1
    assert result.loc[0, "ALL_STAR_FAN_VOTES_BEFORE"] == 40
    assert result.loc[0, "ALL_STAR_FAN_VOTE_SHARE_BEFORE"] == 0.4


def test_trade_moves_votes_only_after_first_recorded_new_team_game():
    all_star = _all_star_df(
        [
            [2025, "p1", "Boston Celtics", 40, 1.5],
            [2025, "p2", "Los Angeles Lakers", 60, 2.5],
        ]
    )
    players = _players_df(
        [
            ["old-game", "p1", BOS_ID, "22025", 2025, "2026-03-10", 10, 5],
            ["new-game", "p1", NYK_ID, "22025", 2025, "2026-03-20", 10, 5],
        ]
    )
    team_rows = pd.DataFrame(
        {
            "GAME_ID": [
                "before-old",
                "before-new",
                "transfer-old",
                "new-game",
                "after-old",
                "after-new",
            ],
            "TEAM_ID": [BOS_ID, NYK_ID, BOS_ID, NYK_ID, BOS_ID, NYK_ID],
            "SEASON_ID": ["22025"] * 6,
            "SEASON_YEAR": [2025] * 6,
            "GAME_DATE": pd.to_datetime(
                [
                    "2026-03-19",
                    "2026-03-19",
                    "2026-03-20",
                    "2026-03-20",
                    "2026-03-21",
                    "2026-03-21",
                ]
            ),
        }
    )

    result = add_all_star_voting_features(team_rows, players, all_star, {}).set_index(
        "GAME_ID"
    )

    assert result.loc["before-old", "ALL_STAR_FAN_VOTES_BEFORE"] == 40
    assert result.loc["before-new", "ALL_STAR_FAN_VOTES_BEFORE"] == 0
    assert result.loc["transfer-old", "ALL_STAR_FAN_VOTES_BEFORE"] == 40
    assert result.loc["new-game", "ALL_STAR_FAN_VOTES_BEFORE"] == 0
    assert result.loc["after-old", "ALL_STAR_FAN_VOTES_BEFORE"] == 0
    assert result.loc["after-new", "ALL_STAR_FAN_VOTES_BEFORE"] == 40
    assert result.loc["transfer-old", "ALL_STAR_MIN_SCORE_BEFORE"] == 1.5
    assert np.isnan(result.loc["new-game", "ALL_STAR_MIN_SCORE_BEFORE"])


def test_injury_assignment_moves_votes_before_first_new_team_box_score():
    all_star = _all_star_df(
        [
            [2025, "p1", "Boston Celtics", 40, 1.5],
            [2025, "p2", "Los Angeles Lakers", 60, 2.5],
        ]
    )
    players = _players_df(
        [
            ["old-game", "p1", BOS_ID, "22025", 2025, "2026-03-10", 10, 5],
        ]
    )
    injured_dict = {"predict-new": {NYK_ID: ["p1"]}}
    team_rows = pd.DataFrame(
        {
            "GAME_ID": ["predict-old", "predict-new"],
            "TEAM_ID": [BOS_ID, NYK_ID],
            "SEASON_ID": ["22025", "22025"],
            "SEASON_YEAR": [2025, 2025],
            "GAME_DATE": pd.to_datetime(["2026-03-20", "2026-03-20"]),
        }
    )

    result = add_all_star_voting_features(
        team_rows,
        players,
        all_star,
        injured_dict,
    ).set_index("GAME_ID")

    assert result.loc["predict-old", "ALL_STAR_FAN_VOTES_BEFORE"] == 0
    assert result.loc["predict-new", "ALL_STAR_CANDIDATE_COUNT_BEFORE"] == 1
    assert result.loc["predict-new", "ALL_STAR_FAN_VOTES_BEFORE"] == 40
    assert result.loc["predict-new", "ALL_STAR_FAN_VOTE_SHARE_BEFORE"] == 0.4
    assert (
        result.loc["predict-new", "ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE"] == 0.4
    )
    assert result.loc["predict-new", "ALL_STAR_MIN_INJURED_SCORE_BEFORE"] == 1.5


def test_denominator_is_league_wide_when_team_name_changes():
    players = _players_df()
    base = _all_star_df(
        [
            [2025, "p1", "Boston Celtics", 40, 1.5],
            [2025, "p2", "Los Angeles Lakers", 60, 2.5],
        ]
    )
    changed_other_team = _all_star_df(
        [
            [2025, "p1", "Boston Celtics", 40, 1.5],
            [2025, "p2", "New York Knicks", 60, 2.5],
        ]
    )

    base_result = add_all_star_voting_features(_team_row(BOS_ID), players, base, {})
    changed_result = add_all_star_voting_features(
        _team_row(BOS_ID), players, changed_other_team, {}
    )

    assert base_result.loc[0, "ALL_STAR_FAN_VOTE_SHARE_BEFORE"] == 0.4
    assert changed_result.loc[0, "ALL_STAR_FAN_VOTE_SHARE_BEFORE"] == 0.4


def test_empty_missing_all_star_season_raises():
    all_star = _all_star_df([[2024, "p1", "Boston Celtics", 40, 1.5]])

    with pytest.raises(
        ValueError,
        match="missing usable rows for required season_year.*Basketball Reference",
    ):
        add_all_star_voting_features(_team_row(BOS_ID), _players_df(), all_star, {})


def test_player_with_no_prior_games_is_kept_from_all_star_team():
    all_star = _all_star_df(
        [
            [2025, "p1", "Boston Celtics", 40, 1.5],
            [2025, "p2", "Los Angeles Lakers", 60, 2.5],
        ]
    )

    result = add_all_star_voting_features(
        _team_row(BOS_ID), _players_df(), all_star, {}
    )

    assert result.loc[0, "ALL_STAR_CANDIDATE_COUNT_BEFORE"] == 1
    assert result.loc[0, "ALL_STAR_FAN_VOTES_BEFORE"] == 40
    assert result.loc[0, "ALL_STAR_FAN_VOTE_SHARE_BEFORE"] == 0.4
    assert result.loc[0, "ALL_STAR_MIN_SCORE_BEFORE"] == 1.5


def test_share_calculation_with_tiny_fixture():
    all_star = _all_star_df(
        [
            [2025, "p1", "Boston Celtics", 40, 1.5],
            [2025, "p2", "Los Angeles Lakers", 30, 2.5],
            [2025, "p3", "New York Knicks", 30, 3.5],
        ]
    )

    result = add_all_star_voting_features(
        _team_row(BOS_ID), _players_df(), all_star, {}
    )

    assert result.loc[0, "ALL_STAR_CANDIDATE_COUNT_BEFORE"] == 1
    assert result.loc[0, "ALL_STAR_FAN_VOTES_BEFORE"] == 40
    assert math.isclose(result.loc[0, "ALL_STAR_FAN_VOTE_SHARE_BEFORE"], 0.4)
    assert result.loc[0, "ALL_STAR_MIN_SCORE_BEFORE"] == 1.5


def test_min_score_uses_same_candidate_list():
    all_star = _all_star_df(
        [
            [2025, "p1", "Boston Celtics", 40, 3.0],
            [2025, "p2", "Boston Celtics", 20, 1.0],
            [2025, "p3", "Los Angeles Lakers", 40, 0.5],
        ]
    )

    result = add_all_star_voting_features(
        _team_row(BOS_ID), _players_df(), all_star, {}
    )

    assert result.loc[0, "ALL_STAR_CANDIDATE_COUNT_BEFORE"] == 2
    assert result.loc[0, "ALL_STAR_FAN_VOTES_BEFORE"] == 60
    assert result.loc[0, "ALL_STAR_MIN_SCORE_BEFORE"] == 1.0


def test_max_injured_fan_vote_share_is_normalized_by_league_total():
    all_star = _all_star_df(
        [
            [2025, "p1", "Boston Celtics", 40, 3.0],
            [2025, "p2", "Boston Celtics", 20, 1.0],
            [2025, "p3", "Los Angeles Lakers", 40, 0.5],
        ]
    )
    injured_dict = {"predict": {BOS_ID: ["p1", "p2", "not_all_star"]}}

    result = add_all_star_voting_features(
        _team_row(BOS_ID), _players_df(), all_star, injured_dict
    )

    assert result.loc[0, "ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE"] == 0.4


def test_injured_dict_accepts_int_keys_and_int_player_ids():
    all_star = _all_star_df(
        [
            [2025, "p1", "Boston Celtics", 40, 3.0],
            [2025, "p2", "Boston Celtics", 60, 1.0],
        ]
    )
    injured_dict = {"predict": {int(BOS_ID): [1, 2, "p1"]}}

    result = add_all_star_voting_features(
        _team_row(BOS_ID), _players_df(), all_star, injured_dict
    )

    assert result.loc[0, "ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE"] == 0.4
    assert result.loc[0, "ALL_STAR_MIN_INJURED_SCORE_BEFORE"] == 3.0


def test_injured_all_star_features_are_empty_when_no_injured_all_star():
    all_star = _all_star_df(
        [
            [2025, "p1", "Boston Celtics", 40, 3.0],
            [2025, "p2", "Boston Celtics", 60, 1.0],
        ]
    )
    injured_dict = {"predict": {BOS_ID: ["not_all_star"]}}

    result = add_all_star_voting_features(
        _team_row(BOS_ID), _players_df(), all_star, injured_dict
    )

    assert result.loc[0, "ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE"] == 0.0
    assert np.isnan(result.loc[0, "ALL_STAR_MIN_INJURED_SCORE_BEFORE"])


def test_only_share_columns_are_kept_after_home_away_aux_drop():
    df = pd.DataFrame(
        {
            "SEASON_ID": ["22025", "22025"],
            "GAME_ID": ["game", "game"],
            "GAME_DATE": [pd.Timestamp("2026-03-02")] * 2,
            "SEASON_TYPE": ["Regular Season", "Regular Season"],
            "SEASON_YEAR": [2025, 2025],
            "IS_OVERTIME": [0, 0],
            "HOME": [True, False],
            "TEAM_ID": [BOS_ID, LAL_ID],
            "TEAM_CITY": ["Boston", "Los Angeles"],
            "TEAM_ABBREVIATION": ["BOS", "LAL"],
            "TEAM_NAME": ["Celtics", "Lakers"],
            "MATCHUP": ["BOS vs. LAL", "LAL @ BOS"],
            "GAME_NUMBER": [1, 1],
            "OFF_RATING_SEASON_BEFORE_AVG": [100, 100],
            "TOP1_PLAYER_OFF_RATING_BEFORE": [100, 100],
            "TOP1_PLAYER_PTS_BEFORE": [20, 20],
            "PTS_SEASON_BEFORE_AVG": [110, 110],
            "PTS": [100, 90],
            "PF": [20, 18],
            "ALL_STAR_FAN_VOTE_SHARE_BEFORE": [0.4, 0.2],
            "ALL_STAR_MIN_SCORE_BEFORE": [1.0, 2.0],
            "ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE": [0.1, 0.2],
            "ALL_STAR_MIN_INJURED_SCORE_BEFORE": [1.0, 2.0],
            "ALL_STAR_FAN_VOTES_BEFORE": [40, 20],
            "ALL_STAR_CANDIDATE_COUNT_BEFORE": [1, 1],
            "ALL_STAR_SEASON_YEAR_BEFORE": [2025, 2025],
        }
    )

    merged = merge_home_away_data(df)
    all_star_aux_cols = [
        col
        for col in merged.columns
        if col.startswith("ALL_STAR_")
        and not col.startswith("ALL_STAR_FAN_VOTE_SHARE_BEFORE_")
        and not col.startswith("ALL_STAR_MIN_SCORE_BEFORE_")
        and not col.startswith("ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE_")
        and not col.startswith("ALL_STAR_MIN_INJURED_SCORE_BEFORE_")
    ]
    merged = merged.drop(columns=all_star_aux_cols)
    all_star_cols = [col for col in merged.columns if col.startswith("ALL_STAR_")]

    assert set(all_star_cols) == {
        "ALL_STAR_FAN_VOTE_SHARE_BEFORE_TEAM_HOME",
        "ALL_STAR_FAN_VOTE_SHARE_BEFORE_TEAM_AWAY",
        "ALL_STAR_MIN_SCORE_BEFORE_TEAM_HOME",
        "ALL_STAR_MIN_SCORE_BEFORE_TEAM_AWAY",
        "ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE_TEAM_HOME",
        "ALL_STAR_MAX_INJURED_FAN_VOTE_SHARE_BEFORE_TEAM_AWAY",
        "ALL_STAR_MIN_INJURED_SCORE_BEFORE_TEAM_HOME",
        "ALL_STAR_MIN_INJURED_SCORE_BEFORE_TEAM_AWAY",
    }
