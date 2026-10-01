"""Replay a past date as if it were tonight -- the train/serve parity harness.

The live path builds tonight's rows from history plus in-memory data about
tonight (schedule, line history so far, referees). Replaying a past date means
feeding that same path a date whose rows the training build already made, and
requiring the two to agree. Everything here only *reads*; nothing is stored.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from nba_ou.fetch_data.nba_schedule.fetch_nba_schedule import fetch_schedules
from nba_ou.fetch_data.scheduled_game.get_schedule_games import nba_api_schedule_games
from nba_ou.utils.general_utils import get_season_year_from_date

EASTERN = ZoneInfo("America/New_York")


def replay_scheduled_games(game_date: str) -> pd.DataFrame:
    """``get_schedule_games`` rows for a date that has already been played.

    ``get_schedule_games`` reads tip-off from the scoreboard's status text,
    which says "Final" once a game is over, so it drops every game of a past
    date. Here the scoreboard rows are kept and ``GAME_TIME`` comes from the
    season schedule feed instead -- the same Eastern-time tip-off.
    """
    games = nba_api_schedule_games(game_date)
    if games.empty:
        return games
    season = int(get_season_year_from_date(pd.Timestamp(game_date)))
    feed = fetch_schedules([season])
    tipoff = feed.set_index(feed["game_id"].astype(str))["tipoff_utc"]
    games = games.copy()
    games["GAME_ID"] = games["GAME_ID"].astype(str)
    games["GAME_TIME"] = (
        pd.to_datetime(games["GAME_ID"].map(tipoff), utc=True).dt.tz_convert(EASTERN)
    )
    return games.dropna(subset=["GAME_TIME"]).reset_index(drop=True)
