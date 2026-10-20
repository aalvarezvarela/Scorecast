"""The season schedule falls back to ScheduleLeagueV2 when the static feed is
missing -- as it is for 2026-27 (S3 answers 403 for the absent key)."""

from __future__ import annotations

import nba_api.stats.endpoints as endpoints
import pandas as pd
import requests
from nba_ou.fetch_data.nba_schedule import fetch_nba_schedule as module

PAYLOAD = {
    "leagueSchedule": {
        "gameDates": [
            {
                "games": [
                    {
                        "gameId": "0022600001",
                        "gameStatus": 1,
                        "gameDateEst": "2026-10-20T00:00:00Z",
                        "gameDateTimeEst": "2026-10-20T15:00:00Z",
                        "gameDateTimeUTC": "2026-10-20T19:00:00Z",
                        "arenaName": "Little Caesars Arena",
                        "arenaCity": "Detroit",
                        "homeTeam": {"teamTricode": "DET"},
                        "awayTeam": {"teamTricode": "BOS"},
                    },
                    # A to-be-determined knockout game: no teams yet.
                    {
                        "gameId": "0062600001",
                        "gameStatus": 1,
                        "gameDateEst": "2026-12-16T00:00:00Z",
                        "gameDateTimeEst": "2026-12-16T20:30:00Z",
                        "gameDateTimeUTC": "2026-12-17T01:30:00Z",
                        "homeTeam": {},
                        "awayTeam": {},
                    },
                ]
            }
        ]
    }
}


class _MissingFeed:
    def raise_for_status(self):
        raise requests.HTTPError("403 Client Error: Forbidden")


class _ScheduleLeagueV2:
    def __init__(self, *, season, league_id, timeout):
        assert (season, league_id) == ("2026-27", "00")

    def get_dict(self):
        return PAYLOAD


def test_a_missing_season_feed_falls_back_to_schedule_league_v2(monkeypatch):
    monkeypatch.setattr(module.requests, "get", lambda *a, **k: _MissingFeed())
    monkeypatch.setattr(endpoints, "ScheduleLeagueV2", _ScheduleLeagueV2)

    schedule = module.fetch_season_schedule(2026)

    assert list(schedule.columns) == module.SCHEDULE_COLUMNS
    opener = schedule.iloc[0]
    assert opener["game_id"] == "0022600001"
    assert opener["season_year"] == 2026
    assert str(opener["game_date"]) == "2026-10-20"
    assert opener["tipoff_utc"] == pd.Timestamp("2026-10-20 19:00", tz="UTC")
    # The Eastern field's "Z" is spurious: it is wall-clock ET.
    assert opener["tipoff_et"] == "2026-10-20T15:00:00"
    assert (opener["team_home"], opener["team_away"]) == ("DET", "BOS")
    assert pd.isna(schedule.iloc[1]["team_home"])
