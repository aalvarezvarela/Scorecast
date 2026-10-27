import time
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from nba_api.live.nba.endpoints import scoreboard
from nba_api.stats.endpoints import ScoreboardV3

NBA_DATE_TZ = ZoneInfo("America/New_York")
# Rounds over both scoreboards before giving up, and the pause (seconds,
# doubled each round) between them. 2026-10-26: one round failed on a
# ScoreboardV3 read timeout that the next run did not see.
LOOKUP_ROUNDS = 3
RETRY_PAUSE_S = 10


def _current_nba_date() -> str:
    return datetime.now(tz=NBA_DATE_TZ).strftime("%Y-%m-%d")


def _get_live_game_ids_from_live_scoreboard() -> list[str]:
    sb = scoreboard.ScoreBoard()  # defaults to today's games
    games = sb.games.get_dict()  # list[dict]
    live_games = [
        g for g in games if g.get("gameStatus") == 2
    ]  # 1=scheduled, 2=live, 3=final (typical)
    return [str(g["gameId"]) for g in live_games if g.get("gameId") is not None]


def _get_live_game_ids_from_scoreboard_v3(game_date: str) -> list[str]:
    scoreboard_v3 = ScoreboardV3(game_date=game_date)
    game_header = scoreboard_v3.game_header.get_data_frame()
    if game_header.empty or not {"gameId", "gameStatus"}.issubset(game_header.columns):
        return []

    game_status = pd.to_numeric(game_header["gameStatus"], errors="coerce")
    live_games = game_header.loc[game_status.eq(2), "gameId"]
    return live_games.dropna().astype(str).unique().tolist()


def get_live_game_ids():
    """
    Return a list of GAME_IDs for NBA games currently in progress.

    No ScoreboardV2: for 2026-27 it answers with every game at status 1, so it
    reported "no live games" without raising and ScoreboardV3 was never asked.
    """
    game_date = _current_nba_date()
    fallback_fetchers = [
        ("live ScoreBoard", _get_live_game_ids_from_live_scoreboard),
        ("ScoreboardV3", lambda: _get_live_game_ids_from_scoreboard_v3(game_date)),
    ]

    last_error: Optional[Exception] = None
    for attempt in range(1, LOOKUP_ROUNDS + 1):
        for source_name, fetcher in fallback_fetchers:
            try:
                return fetcher()
            except Exception as exc:
                last_error = exc
                print(f"{source_name} live-game lookup failed: {exc}")
        if attempt < LOOKUP_ROUNDS:
            pause = RETRY_PAUSE_S * 2 ** (attempt - 1)
            print(f"Retrying the live-game lookup in {pause} s.")
            time.sleep(pause)

    raise RuntimeError(
        f"Unable to fetch live NBA game IDs for {game_date} from ScoreBoard "
        f"or ScoreboardV3 after {LOOKUP_ROUNDS} rounds."
    ) from last_error


def get_game_ids_for_date(game_date: str) -> list[str]:
    """
    Return all NBA GAME_IDs scheduled on a given date.
    """
    try:
        datetime.strptime(game_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("Invalid date format. Please use YYYY-MM-DD.") from exc

    from nba_ou.fetch_data.scheduled_game.get_schedule_games import (
        nba_api_schedule_games,
    )

    games = nba_api_schedule_games(game_date)
    if games.empty or "GAME_ID" not in games.columns:
        return []

    return games["GAME_ID"].dropna().astype(str).unique().tolist()
