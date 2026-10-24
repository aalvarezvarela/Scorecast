from datetime import datetime, timezone

import pandas as pd
import pytest
from nba_ou.postgre_db.game_time_index.sync_game_time_index import (
    _build_season_snapshot_for_upsert,
    _select_games_to_fetch,
    extract_game_index_record,
    extract_game_time_utc,
    format_season_label,
    parse_season_year,
)


def test_parse_season_year_accepts_start_year_or_label() -> None:
    assert parse_season_year(2024) == 2024
    assert parse_season_year("2024") == 2024
    assert parse_season_year("2024-25") == 2024
    assert parse_season_year("2024-2025") == 2024


def test_parse_season_year_rejects_inconsistent_label() -> None:
    with pytest.raises(ValueError):
        parse_season_year("2024-26")


def test_format_season_label() -> None:
    assert format_season_label(2024) == "2024-25"


def test_extract_game_time_utc_returns_aware_datetime() -> None:
    payload = {"game": {"gameTimeUTC": "2025-01-01T00:30:00Z"}}

    result = extract_game_time_utc(payload)

    assert result == datetime(2025, 1, 1, 0, 30, tzinfo=timezone.utc)


def test_extract_game_index_record_maps_requested_fields() -> None:
    payload = {
        "game": {
            "gameId": "0022300001",
            "gameTimeLocal": "2023-11-03T19:00:00-04:00",
            "gameTimeUTC": "2023-11-03T23:00:00Z",
            "gameTimeHome": "2023-11-03T19:00:00-04:00",
            "gameTimeAway": "2023-11-03T19:00:00-04:00",
            "gameEt": "2023-11-03T19:00:00-04:00",
            "duration": 140,
            "gameCode": "20231103/CLEIND",
            "gameStatusText": "Final",
            "gameStatus": 3,
            "regulationPeriods": 4,
            "period": 4,
            "gameClock": "PT00M00.00S",
            "attendance": 16744,
            "sellout": "0",
            "arena": {
                "arenaId": 1000063,
                "arenaName": "Gainbridge Fieldhouse",
                "arenaCity": "Indianapolis",
                "arenaState": "IN",
                "arenaCountry": "US",
                "arenaTimezone": "America/Indiana/Indianapolis",
            },
            "homeTeam": {
                "teamId": 1610612754,
                "teamName": "Pacers",
                "teamCity": "Indiana",
                "teamTricode": "IND",
                "score": 121,
                "statistics": {"assists": 28},
            },
            "awayTeam": {
                "teamId": 1610612739,
                "teamName": "Cavaliers",
                "teamCity": "Cleveland",
                "teamTricode": "CLE",
                "score": 116,
                "statistics": {"assists": 27, "points": 116},
            },
        }
    }

    result = extract_game_index_record(
        payload,
        fallback_game_id="fallback",
        fallback_game_date="2023-11-03",
        season_year=2023,
    )

    assert result["game_id"] == "0022300001"
    assert result["game_status_text"] == "Final"
    assert result["arena_name"] == "Gainbridge Fieldhouse"
    assert result["away_team_name"] == "Cavaliers"
    assert result["away_team_statistics"] == {"assists": 27, "points": 116}
    assert result["game_time_local"] == "2023-11-03T19:00:00-04:00"
    assert result["game_time_home"] == "2023-11-03T19:00:00-04:00"
    assert result["game_time_utc"] == datetime(2023, 11, 3, 23, 0, tzinfo=timezone.utc)


def test_select_games_to_fetch_only_returns_missing_rows() -> None:
    source_games_df = pd.DataFrame(
        [
            {"game_id": "1", "game_date": "2025-01-01", "season_year": 2024},
            {"game_id": "2", "game_date": "2025-01-02", "season_year": 2024},
        ]
    )
    existing_game_times_df = pd.DataFrame(
        [
            {
                "game_id": "1",
                "game_time_utc": "2025-01-01T00:30:00Z",
                "home_team_statistics": {"points": 100},
                "away_team_statistics": {"points": 99},
                "game_status": 3,
            }
        ]
    )

    result = _select_games_to_fetch(
        source_games_df,
        existing_game_times_df,
        refresh_all=False,
    )

    assert result["game_id"].tolist() == ["2"]


def test_build_season_snapshot_keeps_existing_and_prefers_fetched() -> None:
    source_games_df = pd.DataFrame(
        [
            {"game_id": "1", "game_date": "2025-01-01", "season_year": 2024},
            {"game_id": "2", "game_date": "2025-01-02", "season_year": 2024},
        ]
    )
    existing_game_times_df = pd.DataFrame(
        [
            {
                "game_id": "1",
                "game_date": "2025-01-01",
                "season_year": 2024,
                "game_time_local": "2025-01-01T01:30:00Z",
                "game_time_utc": "2025-01-01T00:30:00Z",
                "source": "existing",
                "home_team_statistics": {"points": 110},
                "away_team_statistics": {"points": 108},
                "game_status": 3,
            }
        ]
    )
    fetched_game_times_df = pd.DataFrame(
        [
            {
                "game_id": "2",
                "game_date": "2025-01-02",
                "season_year": 2024,
                "game_time_local": "2025-01-02T02:00:00Z",
                "game_time_utc": "2025-01-02T01:00:00Z",
                "source": "nba_cdn_boxscore",
                "home_team_statistics": {"points": 111},
                "away_team_statistics": {"points": 109},
                "game_status": 3,
            }
        ]
    )

    result = _build_season_snapshot_for_upsert(
        source_games_df,
        existing_game_times_df,
        fetched_game_times_df,
    )

    assert result["game_id"].tolist() == ["1", "2"]
    assert pd.Timestamp(result.loc[result["game_id"] == "1", "game_time_utc"].iloc[0]) == pd.Timestamp(
        "2025-01-01T00:30:00Z"
    )
    assert pd.Timestamp(result.loc[result["game_id"] == "2", "game_time_utc"].iloc[0]) == pd.Timestamp(
        "2025-01-02T01:00:00Z"
    )
    assert result.loc[result["game_id"] == "2", "away_team_statistics"].iloc[0] == {
        "points": 109
    }
