import datetime as dt

import pandas as pd
import pytest
import requests
from nba_ou.fetch_data.nba_schedule import fetch_nba_schedule as feed


def _v2_payload(*games: dict) -> dict:
    return {"leagueSchedule": {"gameDates": [{"gameDate": "x", "games": list(games)}]}}


def _v2_game(**overrides) -> dict:
    game = {
        "gameId": "0022600001",
        "gameDateEst": "2026-10-21T00:00:00Z",
        "gameDateTimeEst": "2026-10-21T19:30:00Z",
        "gameDateTimeUTC": "2026-10-21T23:30:00Z",
        "gameStatus": 1,
        "arenaName": "Madison Square Garden",
        "arenaCity": "New York",
        "homeTeam": {"teamTricode": "NYK"},
        "awayTeam": {"teamTricode": "BOS"},
    }
    game.update(overrides)
    return game


def _legacy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["0022500001"],
            "season_year": [2025],
            "game_date": [dt.date(2025, 10, 21)],
            "tipoff_utc": [pd.Timestamp("2025-10-21T23:30:00Z")],
            "tipoff_et": ["2025-10-21T19:30:00"],
            "team_home": ["NYK"],
            "team_away": ["BOS"],
            "arena_name": ["Madison Square Garden"],
            "arena_city": ["New York"],
            "game_status": ["3"],
        }
    )


class TestScheduleV2Rows:
    def test_matches_legacy_value_formats(self):
        rows = feed._schedule_v2_rows(_v2_payload(_v2_game()), 2026)
        assert rows == [
            {
                "game_id": "0022600001",
                "season_year": 2026,
                "game_date": dt.date(2026, 10, 21),
                "tipoff_utc": pd.Timestamp("2026-10-21T23:30:00Z"),
                # Eastern wall clock without the misleading "Z", like legacy etm.
                "tipoff_et": "2026-10-21T19:30:00",
                "team_home": "NYK",
                "team_away": "BOS",
                "arena_name": "Madison Square Garden",
                "arena_city": "New York",
                # Legacy statuses are strings.
                "game_status": "1",
            }
        ]

    def test_placeholder_fixture_keeps_missing_teams_as_none(self):
        game = _v2_game(
            gameId="0062600001",
            homeTeam={"teamTricode": None},
            awayTeam={"teamTricode": ""},
        )
        (row,) = feed._schedule_v2_rows(_v2_payload(game), 2026)
        assert row["team_home"] is None
        assert row["team_away"] is None

    def test_rows_without_game_id_are_skipped(self):
        assert feed._schedule_v2_rows(_v2_payload(_v2_game(gameId="")), 2026) == []

    def test_empty_payload_gives_schedule_columns(self, monkeypatch):
        monkeypatch.setattr(feed, "_request_schedule_v2", lambda *a: {})
        out = feed.fetch_season_schedule_v2(2026)
        assert out.empty
        assert list(out.columns) == feed.SCHEDULE_COLUMNS


class TestSourceOrder:
    @pytest.fixture
    def legacy_calls(self, monkeypatch):
        calls: list[int] = []

        def fake_legacy(season_year, timeout=30):
            calls.append(season_year)
            return _legacy_frame()

        monkeypatch.setattr(feed, "fetch_legacy_season_schedule", fake_legacy)
        return calls

    def test_v2_is_the_source_when_it_answers(self, monkeypatch, legacy_calls):
        monkeypatch.setattr(
            feed,
            "fetch_season_schedule_v2",
            lambda s, timeout=30: pd.DataFrame(
                feed._schedule_v2_rows(_v2_payload(_v2_game()), s)
            )[feed.SCHEDULE_COLUMNS],
        )
        out = feed.fetch_season_schedule(2026)
        assert out["game_id"].tolist() == ["0022600001"]
        assert legacy_calls == []

    @pytest.mark.parametrize(
        "error",
        [
            requests.ConnectionError("throttled"),
            requests.ReadTimeout("held open"),
            ValueError("Expecting value"),
            KeyError("resultSets"),
        ],
    )
    def test_falls_back_to_legacy_when_v2_fails(self, monkeypatch, legacy_calls, error):
        def broken(season_year, timeout=30):
            raise error

        monkeypatch.setattr(feed, "fetch_season_schedule_v2", broken)
        out = feed.fetch_season_schedule(2025)
        assert out["game_id"].tolist() == ["0022500001"]
        assert legacy_calls == [2025]

    def test_falls_back_to_legacy_when_v2_is_empty(self, monkeypatch, legacy_calls):
        monkeypatch.setattr(
            feed,
            "fetch_season_schedule_v2",
            lambda s, timeout=30: pd.DataFrame(columns=feed.SCHEDULE_COLUMNS),
        )
        feed.fetch_season_schedule(2025)
        assert legacy_calls == [2025]

    def test_legacy_error_propagates_when_both_fail(self, monkeypatch):
        def v2_down(season_year, timeout=30):
            raise requests.ConnectionError("throttled")

        def legacy_gone(season_year, timeout=30):
            raise requests.HTTPError("403 Client Error: Forbidden")

        monkeypatch.setattr(feed, "fetch_season_schedule_v2", v2_down)
        monkeypatch.setattr(feed, "fetch_legacy_season_schedule", legacy_gone)
        with pytest.raises(requests.HTTPError):
            feed.fetch_season_schedule(2026)


class TestRequestScheduleV2:
    def test_retries_then_raises(self, monkeypatch):
        attempts: list[str] = []

        class Failing:
            def __init__(self, *, league_id, season, timeout):
                attempts.append(season)
                raise requests.ConnectionError("throttled")

        import nba_api.stats.endpoints as endpoints

        monkeypatch.setattr(endpoints, "ScheduleLeagueV2", Failing)
        monkeypatch.setattr(feed.time, "sleep", lambda s: None)
        with pytest.raises(requests.ConnectionError):
            feed._request_schedule_v2(2026, timeout=5)
        assert attempts == ["2026-27"] * feed.SCHEDULE_V2_RETRIES
