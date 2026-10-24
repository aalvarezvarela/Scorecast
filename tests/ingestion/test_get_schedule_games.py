from importlib import import_module
from types import SimpleNamespace

import pandas as pd
from nba_ou.fetch_data.scheduled_game.get_schedule_games import (
    DEFAULT_UNKNOWN_GAME_TIME_TEXT,
    EASTERN_TZ,
    _default_unknown_game_times,
    _parse_scoreboard_game_times,
    filter_started_games,
)


def test_parse_scoreboard_game_times_keeps_tbd_as_unknown_time() -> None:
    games = pd.DataFrame(
        [
            {
                "GAME_DATE_EST": "2026-04-29T00:00:00",
                "GAME_STATUS_TEXT": "TBD",
            }
        ]
    )

    result = _parse_scoreboard_game_times(games)

    assert pd.isna(result.iloc[0])


def test_default_unknown_game_times_returns_placeholder_tipoff() -> None:
    games = pd.DataFrame(
        [
            {
                "GAME_DATE_EST": "2026-04-29T00:00:00",
                "GAME_STATUS_TEXT": "TBD",
            }
        ]
    )

    result = _default_unknown_game_times(games)

    assert result.iloc[0] == pd.Timestamp(
        f"2026-04-29 {DEFAULT_UNKNOWN_GAME_TIME_TEXT}",
        tz=EASTERN_TZ,
    )


def test_parse_scoreboard_game_times_extracts_tipoff_time() -> None:
    games = pd.DataFrame(
        [
            {
                "GAME_DATE_EST": "2026-04-29T00:00:00",
                "GAME_STATUS_TEXT": "7:30 PM ET",
            }
        ]
    )

    result = _parse_scoreboard_game_times(games)

    assert result.iloc[0] == pd.Timestamp("2026-04-29 19:30", tz=EASTERN_TZ)


def test_filter_started_games_retains_unknown_times_and_future_known_times() -> None:
    games = pd.DataFrame(
        [
            {"GAME_ID": "unknown", "GAME_TIME": pd.NaT},
            {
                "GAME_ID": "started",
                "GAME_TIME": pd.Timestamp("2026-04-29 18:00", tz=EASTERN_TZ),
            },
            {
                "GAME_ID": "future",
                "GAME_TIME": pd.Timestamp("2026-04-29 21:00", tz=EASTERN_TZ),
            },
        ]
    )

    result = filter_started_games(
        games,
        now_et=pd.Timestamp("2026-04-29 19:00", tz=EASTERN_TZ),
    )

    assert result["GAME_ID"].tolist() == ["unknown", "future"]


def test_get_schedule_games_drops_invalid_status_without_dropping_tbd(
    monkeypatch,
) -> None:
    schedule_module = import_module(
        "nba_ou.fetch_data.scheduled_game.get_schedule_games"
    )
    games = pd.DataFrame(
        [
            {
                "GAME_ID": "tbd",
                "GAME_DATE_EST": "2026-04-29T00:00:00",
                "GAME_STATUS_TEXT": "TBD",
                "HOME_TEAM_ID": "1610612738",
                "VISITOR_TEAM_ID": "1610612752",
            },
            {
                "GAME_ID": "invalid",
                "GAME_DATE_EST": "2026-04-29T00:00:00",
                "GAME_STATUS_TEXT": "Final",
                "HOME_TEAM_ID": "1610612747",
                "VISITOR_TEAM_ID": "1610612744",
            },
        ]
    )
    monkeypatch.setattr(schedule_module, "nba_api_schedule_games", lambda _: games)
    monkeypatch.setattr(schedule_module, "filter_started_games", lambda games: games)

    result = schedule_module.get_schedule_games("2026-04-29")

    assert result["GAME_ID"].tolist() == ["tbd"]
    assert result["GAME_TIME"].iloc[0] == pd.Timestamp(
        f"2026-04-29 {DEFAULT_UNKNOWN_GAME_TIME_TEXT}",
        tz=EASTERN_TZ,
    )


def test_get_schedule_games_defaults_missing_and_tbp_status(monkeypatch) -> None:
    schedule_module = import_module(
        "nba_ou.fetch_data.scheduled_game.get_schedule_games"
    )
    games = pd.DataFrame(
        [
            {
                "GAME_ID": "tbp",
                "GAME_DATE_EST": "2026-04-29T00:00:00",
                "GAME_STATUS_TEXT": "TBP",
                "HOME_TEAM_ID": "1610612738",
                "VISITOR_TEAM_ID": "1610612752",
            },
            {
                "GAME_ID": "blank",
                "GAME_DATE_EST": "2026-04-29T00:00:00",
                "GAME_STATUS_TEXT": "",
                "HOME_TEAM_ID": "1610612747",
                "VISITOR_TEAM_ID": "1610612744",
            },
            {
                "GAME_ID": "missing",
                "GAME_DATE_EST": "2026-04-29T00:00:00",
                "GAME_STATUS_TEXT": None,
                "HOME_TEAM_ID": "1610612743",
                "VISITOR_TEAM_ID": "1610612745",
            },
        ]
    )
    monkeypatch.setattr(schedule_module, "nba_api_schedule_games", lambda _: games)
    monkeypatch.setattr(schedule_module, "filter_started_games", lambda games: games)

    result = schedule_module.get_schedule_games("2026-04-29")

    expected_time = pd.Timestamp(
        f"2026-04-29 {DEFAULT_UNKNOWN_GAME_TIME_TEXT}",
        tz=EASTERN_TZ,
    )
    assert result["GAME_ID"].tolist() == ["tbp", "blank", "missing"]
    assert result["GAME_TIME"].tolist() == [expected_time, expected_time, expected_time]


def test_get_schedule_games_keeps_tbd_and_future_timed_games(monkeypatch) -> None:
    schedule_module = import_module(
        "nba_ou.fetch_data.scheduled_game.get_schedule_games"
    )
    games = pd.DataFrame(
        [
            {
                "GAME_ID": "tbd",
                "GAME_DATE_EST": "2026-04-29T00:00:00",
                "GAME_STATUS_TEXT": "TBD",
                "HOME_TEAM_ID": "1610612738",
                "VISITOR_TEAM_ID": "1610612752",
            },
            {
                "GAME_ID": "timed",
                "GAME_DATE_EST": "2026-04-29T00:00:00",
                "GAME_STATUS_TEXT": "9:30 PM ET",
                "HOME_TEAM_ID": "1610612747",
                "VISITOR_TEAM_ID": "1610612744",
            },
        ]
    )
    monkeypatch.setattr(schedule_module, "nba_api_schedule_games", lambda _: games)
    filter_started_games_original = schedule_module.filter_started_games
    monkeypatch.setattr(
        schedule_module,
        "filter_started_games",
        lambda games: filter_started_games_original(
            games,
            now_et=pd.Timestamp("2026-04-29 19:00", tz=EASTERN_TZ),
        ),
    )

    result = schedule_module.get_schedule_games("2026-04-29")

    assert result["GAME_ID"].tolist() == ["tbd", "timed"]
    assert result.loc[result["GAME_ID"] == "tbd", "GAME_TIME"].iloc[0] == pd.Timestamp(
        f"2026-04-29 {DEFAULT_UNKNOWN_GAME_TIME_TEXT}",
        tz=EASTERN_TZ,
    )
    assert result.loc[result["GAME_ID"] == "timed", "GAME_TIME"].iloc[
        0
    ] == pd.Timestamp(
        "2026-04-29 21:30",
        tz=EASTERN_TZ,
    )


def test_a_failing_scoreboard_v2_falls_back_to_v3(monkeypatch) -> None:
    """2026-10-20: V2 answered the opener with a non-JSON body, V3 had the games."""
    import json

    import nba_api.stats.endpoints as endpoints

    schedule_module = import_module(
        "nba_ou.fetch_data.scheduled_game.get_schedule_games"
    )

    class BrokenV2:
        def __init__(self, **_):
            raise json.JSONDecodeError("Expecting value", "", 0)

    frame = lambda df: SimpleNamespace(get_data_frame=lambda: df)  # noqa: E731

    class V3:
        def __init__(self, **_):
            self.game_header = frame(
                pd.DataFrame(
                    {
                        "gameId": ["0022600001"],
                        "gameCode": ["20261020/BOSDET"],
                        "gameStatus": [1],
                        "gameStatusText": ["3:00 pm ET"],
                        "period": [0],
                        "gameClock": [""],
                    }
                )
            )
            self.line_score = frame(
                pd.DataFrame(
                    {"gameId": ["0022600001"] * 2, "teamId": [1610612765, 1610612738]}
                )
            )

    monkeypatch.setattr(schedule_module, "ScoreboardV2", BrokenV2)
    monkeypatch.setattr(endpoints, "ScoreboardV3", V3)
    BrokenV2.expected_data = {
        "GameHeader": endpoints.ScoreboardV2.expected_data["GameHeader"]
    }

    games = schedule_module.nba_api_schedule_games("2026-10-20")

    assert games["GAME_ID"].tolist() == ["0022600001"]
    assert games["HOME_TEAM_ID"].tolist() == [1610612765]
    assert games["GAME_STATUS_TEXT"].tolist() == ["3:00 pm ET"]


def test_a_game_listed_once_per_broadcaster_is_kept_once(monkeypatch) -> None:
    """2026-10-20: V2 listed every game twice, once per national broadcaster."""
    schedule_module = import_module(
        "nba_ou.fetch_data.scheduled_game.get_schedule_games"
    )
    rows = pd.DataFrame(
        {
            "GAME_ID": ["0022600001", "0022600001", "0022600002"],
            "GAME_STATUS_TEXT": ["3:00 pm ET"] * 2 + ["7:00 pm ET"],
            "NATL_TV_BROADCASTER_ABBREVIATION": ["ABC", "ESPN", "TNT"],
        }
    )

    class V2:
        def __init__(self, **_):
            pass

        def get_data_frames(self):
            return [rows]

    monkeypatch.setattr(schedule_module, "ScoreboardV2", V2)

    games = schedule_module.nba_api_schedule_games("2026-10-20")

    assert games["GAME_ID"].tolist() == ["0022600001", "0022600002"]
