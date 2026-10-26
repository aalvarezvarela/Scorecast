from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from nba_api.stats.endpoints import ScoreboardV3

EASTERN_TZ = ZoneInfo("America/New_York")
UNKNOWN_GAME_TIME_STATUS_TEXT = frozenset({"TBD", "TBA", "TBP"})
DEFAULT_UNKNOWN_GAME_TIME_TEXT = "11:59 PM"
GAME_TIME_PATTERN = r"\b(\d{1,2}:\d{2}\s*[AP]M)\b"
#: ScoreboardV3 ``gameStatus``: 1 scheduled, 2 in progress, 3 final.
SCHEDULED_GAME_STATUS = 1

#: The old ScoreboardV2 ``GameHeader`` columns, which every consumer reads.
SCOREBOARD_COLUMNS = [
    "GAME_DATE_EST",
    "GAME_SEQUENCE",
    "GAME_ID",
    "GAME_STATUS_ID",
    "GAME_STATUS_TEXT",
    "GAMECODE",
    "HOME_TEAM_ID",
    "VISITOR_TEAM_ID",
    "SEASON",
    "LIVE_PERIOD",
    "LIVE_PC_TIME",
    "NATL_TV_BROADCASTER_ABBREVIATION",
    "HOME_TV_BROADCASTER_ABBREVIATION",
    "AWAY_TV_BROADCASTER_ABBREVIATION",
    "LIVE_PERIOD_TIME_BCAST",
    "ARENA_NAME",
    "WH_STATUS",
]


def nba_api_schedule_games(date):
    """
    Fetches NBA scheduled games for a given date from ``ScoreboardV3``.

    ScoreboardV2 is not used: for 2026-27 it keeps every game at status 1 with
    no points, even once final, and at times answers with a non-JSON body.

    Parameters:
    date (str): The date in 'YYYY-MM-DD' format.

    Returns:
    DataFrame: One row per game, in ``SCOREBOARD_COLUMNS``.
    """
    # Validate date format
    try:
        game_date = datetime.strptime(date, "%Y-%m-%d")

    except ValueError as err:
        raise ValueError("Invalid date format. Please use YYYY-MM-DD.") from err

    payload = ScoreboardV3(game_date=date).get_dict()
    season = game_date.year if game_date.month >= 10 else game_date.year - 1
    rows = [
        {
            "GAME_DATE_EST": f"{date}T00:00:00",
            "GAME_SEQUENCE": sequence,
            "GAME_ID": str(game["gameId"]),
            "GAME_STATUS_ID": game.get("gameStatus"),
            "GAME_STATUS_TEXT": game.get("gameStatusText"),
            "GAMECODE": game.get("gameCode"),
            "HOME_TEAM_ID": (game.get("homeTeam") or {}).get("teamId"),
            "VISITOR_TEAM_ID": (game.get("awayTeam") or {}).get("teamId"),
            "SEASON": season,
            "LIVE_PERIOD": game.get("period"),
            "LIVE_PC_TIME": game.get("gameClock"),
        }
        for sequence, game in enumerate(
            payload.get("scoreboard", {}).get("games", []), start=1
        )
        if game.get("gameId")
    ]
    return pd.DataFrame(rows, columns=SCOREBOARD_COLUMNS)


def filter_started_games(
    games: pd.DataFrame, now_et: pd.Timestamp | datetime | None = None
) -> pd.DataFrame:
    """
    Keep only games that have not started yet based on GAME_TIME in Eastern Time.
    Games with unknown start times are retained because the scoreboard can report
    future playoff games as TBD/TBA until the league assigns tip-off times.
    """
    if games.empty or "GAME_TIME" not in games.columns:
        return games

    current_time_et = pd.Timestamp.now(tz=EASTERN_TZ)
    if now_et is not None:
        current_time_et = pd.Timestamp(now_et)
        if current_time_et.tzinfo is None:
            current_time_et = current_time_et.tz_localize(EASTERN_TZ)
        else:
            current_time_et = current_time_et.tz_convert(EASTERN_TZ)

    known_game_time = games["GAME_TIME"].notna()
    upcoming_games = games[
        (~known_game_time) | (games["GAME_TIME"] > current_time_et)
    ].copy()

    filtered_count = (known_game_time & (games["GAME_TIME"] <= current_time_et)).sum()
    if filtered_count > 0:
        print(
            f"Excluded {filtered_count} scheduled game(s) that already started "
            f"as of {current_time_et}."
        )

    return upcoming_games


def _parse_scoreboard_game_times(games: pd.DataFrame) -> pd.Series:
    """Parse tip-off times from the scoreboard's GAME_STATUS_TEXT values."""
    game_dates = pd.to_datetime(games["GAME_DATE_EST"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    game_status_text = (
        games["GAME_STATUS_TEXT"].astype("string").str.strip().str.upper()
    )
    game_time_text = game_status_text.str.extract(GAME_TIME_PATTERN, expand=False)
    parsed_game_times = pd.to_datetime(
        game_dates.astype("string") + " " + game_time_text,
        format="%Y-%m-%d %I:%M %p",
        errors="coerce",
    )
    return parsed_game_times.dt.tz_localize(
        EASTERN_TZ, ambiguous="infer", nonexistent="shift_forward"
    )


def _default_unknown_game_times(games: pd.DataFrame) -> pd.Series:
    """Create a timezone-aware placeholder tip-off for games without a listed time."""
    game_dates = pd.to_datetime(games["GAME_DATE_EST"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    default_game_times = pd.to_datetime(
        game_dates.astype("string") + f" {DEFAULT_UNKNOWN_GAME_TIME_TEXT}",
        format="%Y-%m-%d %I:%M %p",
        errors="coerce",
    )
    return default_game_times.dt.tz_localize(
        EASTERN_TZ, ambiguous="infer", nonexistent="shift_forward"
    )


def get_schedule_games(date_to_predict: str) -> pd.DataFrame:
    """
    Fetch scheduled NBA games for a specific date.
    Args:
        date_to_predict (str): Date in 'YYYY-MM-DD' format
    Returns:
        pd.DataFrame: DataFrame containing scheduled games
    """
    games = nba_api_schedule_games(date_to_predict)
    if games.empty:
        print("No games found for the specified date.")
        return games

    # A started game's status text is its clock or "Final", not a tip-off.
    if "GAME_STATUS_ID" in games.columns:
        status = pd.to_numeric(games["GAME_STATUS_ID"], errors="coerce")
        started = status.notna() & status.ne(SCHEDULED_GAME_STATUS)
        if started.any():
            print(f"Excluded {started.sum()} game(s) already started or final.")
            games = games.loc[~started].copy()
            if games.empty:
                return games

    games["GAME_TIME"] = _parse_scoreboard_game_times(games)

    game_status_text = (
        games["GAME_STATUS_TEXT"].astype("string").str.strip().str.upper()
    )
    unknown_game_time = (
        game_status_text.isin(UNKNOWN_GAME_TIME_STATUS_TEXT)
        | game_status_text.isna()
        | game_status_text.eq("")
    )
    invalid_game_time = games[games["GAME_TIME"].isna() & ~unknown_game_time]
    if not invalid_game_time.empty:
        print("\n" + "==" * 30)
        print("WARNING: Dropping games with invalid GAME_TIME:")
        print("==" * 30)
        print(
            invalid_game_time[
                [
                    "GAME_ID",
                    "GAME_DATE_EST",
                    "GAME_STATUS_TEXT",
                    "HOME_TEAM_ID",
                    "VISITOR_TEAM_ID",
                ]
            ]
        )
        print("==" * 30 + "\n")
        games = games.drop(index=invalid_game_time.index).copy()
        unknown_game_time = unknown_game_time.loc[games.index]

    missing_unknown_time = games["GAME_TIME"].isna() & unknown_game_time
    if missing_unknown_time.any():
        games.loc[missing_unknown_time, "GAME_TIME"] = _default_unknown_game_times(
            games.loc[missing_unknown_time]
        )

    games = filter_started_games(games)
    return games
