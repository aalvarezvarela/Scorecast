import pandas as pd
import pytest
from nba_ou.postgre_db.all_star_voting.process_all_star_voting_data import (
    AllStarVotingPlayerMatcher,
    PlayerMatchError,
    add_fan_votes_pct,
    add_team_names_at_cutoff,
    prepare_all_star_voting_dataset,
    read_voting_csv,
)


def test_add_fan_votes_pct_groups_by_season():
    df = pd.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2025-26"],
            "fan_votes": [75, 25, 50],
        }
    )

    result = add_fan_votes_pct(df)

    assert result["fan_votes_pct"].tolist() == [0.75, 0.25, 1.0]


def test_matcher_matches_suffix_stripped_display_name():
    players_df = pd.DataFrame(
        {
            "season_year": [2024],
            "player_id": ["1629057"],
            "player_name": ["R. Williams III"],
            "firstname": [None],
            "familyname": [None],
        }
    )
    voting_df = pd.DataFrame(
        {
            "season": ["2024-25"],
            "player_name": ["Robert Williams"],
        }
    )

    result = AllStarVotingPlayerMatcher(players_df).match_voting_players(voting_df)

    assert result.tolist() == ["1629057"]


def test_matcher_uses_known_full_names_to_disambiguate_display_names():
    players_df = pd.DataFrame(
        {
            "season_year": [2024, 2024, 2025, 2025],
            "player_id": ["1630598", "203952", "1630598", "203952"],
            "player_name": ["A. Wiggins", "A. Wiggins", "A. Wiggins", "A. Wiggins"],
            "firstname": [None, None, "Aaron", "Andrew"],
            "familyname": [None, None, "Wiggins", "Wiggins"],
        }
    )
    voting_df = pd.DataFrame(
        {
            "season": ["2024-25", "2024-25"],
            "player_name": ["Aaron Wiggins", "Andrew Wiggins"],
        }
    )

    result = AllStarVotingPlayerMatcher(players_df).match_voting_players(voting_df)

    assert result.tolist() == ["1630598", "203952"]


def test_matcher_raises_for_missing_player():
    players_df = pd.DataFrame(
        {
            "season_year": [2024],
            "player_id": ["1"],
            "player_name": ["A. Player"],
            "firstname": [None],
            "familyname": [None],
        }
    )
    voting_df = pd.DataFrame(
        {
            "season": ["2024-25"],
            "player_name": ["Missing Player"],
        }
    )

    with pytest.raises(PlayerMatchError, match="Missing Player"):
        AllStarVotingPlayerMatcher(players_df).match_voting_players(voting_df)


def test_matcher_uses_nba_api_static_fallback_when_supabase_cannot_match():
    players_df = pd.DataFrame(
        {
            "season_year": [2025],
            "player_id": ["1"],
            "player_name": ["A. Player"],
            "firstname": [None],
            "familyname": [None],
        }
    )
    voting_df = pd.DataFrame(
        {
            "season": ["2025-26"],
            "player_name": ["Thomas Sorber"],
        }
    )

    result = AllStarVotingPlayerMatcher(players_df).match_voting_players(voting_df)

    assert result.tolist() == ["1642850"]


def test_matcher_selects_highest_numeric_id_from_two_fallback_candidates(
    monkeypatch,
):
    players_df = pd.DataFrame(
        {
            "season_year": [2016],
            "player_id": ["1"],
            "player_name": ["A. Player"],
            "firstname": [None],
            "familyname": [None],
        }
    )
    voting_df = pd.DataFrame(
        {
            "season": ["2016-17"],
            "player_name": ["Fallback Candidate"],
        }
    )
    matcher = AllStarVotingPlayerMatcher(players_df)
    monkeypatch.setattr(
        matcher,
        "_lookup_nba_static_ids",
        lambda _player_name: {"201945", "76993"},
    )

    result = matcher.match_voting_players(voting_df)

    assert result.tolist() == ["201945"]


def test_matcher_prefers_candidate_observed_closest_to_voting_season(monkeypatch):
    players_df = pd.DataFrame(
        {
            "season_year": [2016],
            "player_id": ["2399"],
            "player_name": ["M. Player"],
            "firstname": [None],
            "familyname": [None],
        }
    )
    voting_df = pd.DataFrame(
        {
            "season": ["2016-17"],
            "player_name": ["Mike Dunleavy"],
        }
    )
    matcher = AllStarVotingPlayerMatcher(players_df)
    monkeypatch.setattr(
        matcher,
        "_lookup_nba_static_ids",
        lambda _player_name: {"2399", "76616"},
    )

    result = matcher.match_voting_players(voting_df)

    assert result.tolist() == ["2399"]


def test_matcher_keeps_more_than_two_fallback_candidates_ambiguous(monkeypatch):
    players_df = pd.DataFrame(
        {
            "season_year": [2016],
            "player_id": ["1"],
            "player_name": ["A. Player"],
            "firstname": [None],
            "familyname": [None],
        }
    )
    voting_df = pd.DataFrame(
        {
            "season": ["2016-17"],
            "player_name": ["Fallback Candidate"],
        }
    )
    matcher = AllStarVotingPlayerMatcher(players_df)
    monkeypatch.setattr(
        matcher,
        "_lookup_nba_static_ids",
        lambda _player_name: {"100", "200", "300"},
    )

    with pytest.raises(PlayerMatchError, match="ambiguous NBA API fallback match"):
        matcher.match_voting_players(voting_df)


def test_matcher_normalizes_serbian_dj_transliteration():
    players_df = pd.DataFrame(
        {
            "season_year": [2025],
            "player_id": ["1"],
            "player_name": ["A. Player"],
            "firstname": [None],
            "familyname": [None],
        }
    )
    voting_df = pd.DataFrame(
        {
            "season": ["2025-26"],
            "player_name": ["Nikola Djurisic"],
        }
    )

    result = AllStarVotingPlayerMatcher(players_df).match_voting_players(voting_df)

    assert result.tolist() == ["1642365"]


def test_read_voting_csv_adds_season_year(tmp_path):
    csv_path = tmp_path / "all_star.csv"
    pd.DataFrame(
        {
            "conference": ["Western Conference"],
            "position": ["Backcourt"],
            "season": ["2019-20"],
            "player_name": ["Test Player"],
            "fan_votes": [100],
            "fan_rank": [1],
            "player_votes": [2],
            "player_rank": [3],
            "media_votes": [4],
            "media_rank": [5],
            "score": [1.0],
        }
    ).to_csv(csv_path, index=False)

    result = read_voting_csv(csv_path)

    assert result.loc[0, "season_year"] == 2019


def test_matcher_uses_known_override_for_kyle_mangas():
    players_df = pd.DataFrame(
        {
            "season_year": [2025],
            "player_id": ["1"],
            "player_name": ["A. Player"],
            "firstname": [None],
            "familyname": [None],
        }
    )
    voting_df = pd.DataFrame(
        {
            "season": ["2025-26"],
            "player_name": ["Kyle Mangas"],
        }
    )

    result = AllStarVotingPlayerMatcher(players_df).match_voting_players(voting_df)

    assert result.tolist() == ["1630667"]


def test_add_team_names_at_cutoff_uses_latest_team_before_february_15():
    voting_df = pd.DataFrame(
        {
            "season": ["2024-25"],
            "season_year": [2024],
            "player_name": ["Test Player"],
            "player_id": ["123"],
        }
    )
    players_df = pd.DataFrame(
        {
            "season_year": [2024, 2024, 2024],
            "player_id": ["123", "123", "123"],
            "game_id": ["old", "cutoff", "late"],
            "team_id": ["1610612737", "1610612738", "1610612752"],
            "team_city": ["Atlanta", "Boston", "New York"],
            "team_name": [None, None, None],
        }
    )
    games_df = pd.DataFrame(
        {
            "game_id": ["old", "cutoff", "late"],
            "team_id": ["1610612737", "1610612738", "1610612752"],
            "game_date": ["2025-01-20", "2025-02-15", "2025-02-16"],
            "season_type": ["Regular Season", "Regular Season", "Regular Season"],
        }
    )

    result = add_team_names_at_cutoff(voting_df, players_df, games_df)

    assert result.loc[0, "team_name"] == "Boston Celtics"


def test_add_team_names_at_cutoff_uses_known_team_override():
    voting_df = pd.DataFrame(
        {
            "season": ["2025-26"],
            "season_year": [2025],
            "player_name": ["Kyle Mangas"],
            "player_id": ["1630667"],
        }
    )
    players_df = pd.DataFrame(
        columns=[
            "season_year",
            "player_id",
            "game_id",
            "team_id",
            "team_city",
            "team_name",
        ]
    )
    games_df = pd.DataFrame(
        columns=["game_id", "team_id", "game_date", "season_type"]
    )

    result = add_team_names_at_cutoff(voting_df, players_df, games_df)

    assert result.loc[0, "team_name"] == "San Antonio Spurs"


def test_add_team_names_at_cutoff_can_skip_unresolved_teams():
    voting_df = pd.DataFrame(
        {
            "season": ["2024-25", "2024-25"],
            "season_year": [2024, 2024],
            "player_name": ["Resolved Player", "Missing Player"],
            "player_id": ["123", "456"],
        }
    )
    players_df = pd.DataFrame(
        {
            "season_year": [2024],
            "player_id": ["123"],
            "game_id": ["game"],
            "team_id": ["1610612738"],
            "team_city": ["Boston"],
            "team_name": ["Celtics"],
        }
    )
    games_df = pd.DataFrame(
        {
            "game_id": ["game"],
            "team_id": ["1610612738"],
            "game_date": ["2025-02-01"],
            "season_type": ["Regular Season"],
        }
    )

    result = add_team_names_at_cutoff(
        voting_df,
        players_df,
        games_df,
        skip_unresolved=True,
    )

    assert result["player_name"].tolist() == ["Resolved Player"]
    assert result["team_name"].tolist() == ["Boston Celtics"]


def test_prepare_skip_unresolved_players_keeps_same_name_in_resolved_season(tmp_path):
    csv_path = tmp_path / "all_star_voting.csv"
    pd.DataFrame(
        {
            "conference": ["Western Conference", "Western Conference"],
            "position": ["Backcourt", "Backcourt"],
            "season": ["2023-24", "2025-26"],
            "player_name": ["Duplicate Name", "Duplicate Name"],
            "fan_votes": [10, 20],
            "fan_rank": [1, 1],
            "player_votes": [0, 0],
            "player_rank": [1, 1],
            "media_votes": [0, 0],
            "media_rank": [1, 1],
            "score": [1.0, 1.0],
        }
    ).to_csv(csv_path, index=False)
    players_df = pd.DataFrame(
        {
            "season_year": [2023],
            "player_id": ["123"],
            "player_name": ["Duplicate Name"],
            "firstname": ["Duplicate"],
            "familyname": ["Name"],
            "game_id": ["game"],
            "team_id": ["1610612738"],
            "team_city": ["Boston"],
            "team_name": ["Celtics"],
        }
    )
    games_df = pd.DataFrame(
        {
            "game_id": ["game"],
            "team_id": ["1610612738"],
            "game_date": ["2024-02-01"],
            "season_type": ["Regular Season"],
        }
    )

    result = prepare_all_star_voting_dataset(
        input_csv=csv_path,
        players_df=players_df,
        games_df=games_df,
        injuries_df=pd.DataFrame(
            columns=["season_year", "player_id", "team_id", "game_date"]
        ),
        skip_unresolved=True,
    )

    assert result["season"].tolist() == ["2023-24"]
    assert result["player_name"].tolist() == ["Duplicate Name"]
