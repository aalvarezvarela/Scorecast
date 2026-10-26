from importlib import import_module

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


def _scoreboard_v3(schedule_module, monkeypatch, games: list[dict]) -> None:
    class V3:
        def __init__(self, **_):
            pass

        def get_dict(self):
            return {"scoreboard": {"games": games}}

    monkeypatch.setattr(schedule_module, "ScoreboardV3", V3)


def _v3_game(game_id: str, status: int, text: str, home: int, away: int) -> dict:
    return {
        "gameId": game_id,
        "gameCode": f"20261026/{game_id}",
        "gameStatus": status,
        "gameStatusText": text,
        "period": 0,
        "gameClock": "",
        "homeTeam": {"teamId": home},
        "awayTeam": {"teamId": away},
    }


def test_schedule_games_read_from_scoreboard_v3(monkeypatch) -> None:
    schedule_module = import_module(
        "nba_ou.fetch_data.scheduled_game.get_schedule_games"
    )
    _scoreboard_v3(
        schedule_module,
        monkeypatch,
        [
            _v3_game("0022600119", 1, "7:00 pm ET", 1610612765, 1610612738),
            _v3_game("0022600120", 1, "10:00 pm ET", 1610612747, 1610612744),
        ],
    )

    games = schedule_module.nba_api_schedule_games("2026-10-26")

    assert games.columns.tolist() == schedule_module.SCOREBOARD_COLUMNS
    assert games["GAME_ID"].tolist() == ["0022600119", "0022600120"]
    assert games["HOME_TEAM_ID"].tolist() == [1610612765, 1610612747]
    assert games["VISITOR_TEAM_ID"].tolist() == [1610612738, 1610612744]
    assert games["GAME_STATUS_TEXT"].tolist() == ["7:00 pm ET", "10:00 pm ET"]
    assert games["SEASON"].tolist() == [2026, 2026]


def test_schedule_games_empty_date(monkeypatch) -> None:
    schedule_module = import_module(
        "nba_ou.fetch_data.scheduled_game.get_schedule_games"
    )
    _scoreboard_v3(schedule_module, monkeypatch, [])

    games = schedule_module.nba_api_schedule_games("2026-07-01")

    assert games.empty
    assert games.columns.tolist() == schedule_module.SCOREBOARD_COLUMNS


def test_get_schedule_games_excludes_started_games_by_status(monkeypatch) -> None:
    schedule_module = import_module(
        "nba_ou.fetch_data.scheduled_game.get_schedule_games"
    )
    _scoreboard_v3(
        schedule_module,
        monkeypatch,
        [
            _v3_game("final", 3, "Final", 1, 2),
            _v3_game("live", 2, "Q2 5:31", 3, 4),
            _v3_game("later", 1, "10:00 pm ET", 5, 6),
        ],
    )
    filter_started_games_original = schedule_module.filter_started_games
    monkeypatch.setattr(
        schedule_module,
        "filter_started_games",
        lambda games: filter_started_games_original(
            games, now_et=pd.Timestamp("2026-10-26 20:00", tz=EASTERN_TZ)
        ),
    )

    games = schedule_module.get_schedule_games("2026-10-26")

    assert games["GAME_ID"].tolist() == ["later"]
