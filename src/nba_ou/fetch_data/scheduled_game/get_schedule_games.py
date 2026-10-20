from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from nba_api.stats.endpoints import ScoreboardV2

EASTERN_TZ = ZoneInfo("America/New_York")
UNKNOWN_GAME_TIME_STATUS_TEXT = frozenset({"TBD", "TBA", "TBP"})
DEFAULT_UNKNOWN_GAME_TIME_TEXT = "11:59 PM"
GAME_TIME_PATTERN = r"\b(\d{1,2}:\d{2}\s*[AP]M)\b"


def nba_api_schedule_games(date):
    """
    Fetches NBA scheduled games for a given date.

    Parameters:
    date (str): The date in 'YYYY-MM-DD' format.

    Returns:
    DataFrame: A DataFrame containing scheduled games for the given date.
    """
    # Validate date format
    try:
        datetime.strptime(date, "%Y-%m-%d")

    except ValueError as err:
        raise ValueError("Invalid date format. Please use YYYY-MM-DD.") from err

    # Fetch games. ScoreboardV2 can also fail outright rather than come back
    # empty: on 2026-10-20 (opening night) it answered today's date with a
    # non-JSON body while V3 listed all three games.
    try:
        games = ScoreboardV2(game_date=date).get_data_frames()[0]
    except Exception as exc:  # noqa: BLE001 -- V3 below is the fallback
        print(f"ScoreboardV2 failed for {date} ({type(exc).__name__}: {exc}).")
        games = pd.DataFrame()
    if not games.empty:
        # V2's GameHeader can list a game once per national broadcaster: on
        # 2026-10-20 every game came twice, the rows differing only in
        # NATL_TV_BROADCASTER_ABBREVIATION and LIVE_PERIOD_TIME_BCAST. Every
        # consumer expects one row per game.
        return games.drop_duplicates(subset=["GAME_ID"], keep="first").reset_index(
            drop=True
        )

    from nba_api.stats.endpoints import ScoreboardV3

    scoreboard_v3 = ScoreboardV3(game_date=date)
    game_header = scoreboard_v3.game_header.get_data_frame()
    if game_header.empty:
        return games

    print("ScoreboardV2 gave no games; using ScoreboardV3 fallback.")

    line_score = scoreboard_v3.line_score.get_data_frame()
    team_ids_by_game = pd.DataFrame(
        columns=["gameId", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]
    )
    if not line_score.empty:
        line_score = line_score.copy()
        line_score["_team_position"] = line_score.groupby("gameId").cumcount()
        team_ids_by_game = (
            line_score[line_score["_team_position"].isin([0, 1])]
            .pivot(index="gameId", columns="_team_position", values="teamId")
            .rename(columns={0: "HOME_TEAM_ID", 1: "VISITOR_TEAM_ID"})
            .reset_index()
        )

    fallback_games = game_header.merge(team_ids_by_game, on="gameId", how="left")
    fallback_games["GAME_DATE_EST"] = f"{date}T00:00:00"
    fallback_games["GAME_SEQUENCE"] = range(1, len(fallback_games) + 1)
    fallback_games["GAME_ID"] = fallback_games["gameId"].astype("string")
    fallback_games["GAME_STATUS_ID"] = fallback_games["gameStatus"]
    fallback_games["GAME_STATUS_TEXT"] = fallback_games["gameStatusText"]
    fallback_games["GAMECODE"] = fallback_games["gameCode"]
    game_date = datetime.strptime(date, "%Y-%m-%d")
    fallback_games["SEASON"] = (
        game_date.year if game_date.month >= 10 else game_date.year - 1
    )
    fallback_games["LIVE_PERIOD"] = fallback_games["period"]
    fallback_games["LIVE_PC_TIME"] = fallback_games["gameClock"]
    fallback_games["NATL_TV_BROADCASTER_ABBREVIATION"] = pd.NA
    fallback_games["HOME_TV_BROADCASTER_ABBREVIATION"] = pd.NA
    fallback_games["AWAY_TV_BROADCASTER_ABBREVIATION"] = pd.NA
    fallback_games["LIVE_PERIOD_TIME_BCAST"] = pd.NA
    fallback_games["ARENA_NAME"] = pd.NA
    fallback_games["WH_STATUS"] = pd.NA

    return fallback_games[ScoreboardV2.expected_data["GameHeader"]]


def filter_started_games(
    games: pd.DataFrame, now_et: pd.Timestamp | datetime | None = None
) -> pd.DataFrame:
    """
    Keep only games that have not started yet based on GAME_TIME in Eastern Time.
    Games with unknown start times are retained because ScoreboardV2 can report
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
    """Parse tip-off times from ScoreboardV2 GAME_STATUS_TEXT values."""
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
