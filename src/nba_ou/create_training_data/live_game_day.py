"""Everything about tonight that the database does not hold yet, read in memory.

Tonight's games are stored only once they are finished (the rule the closing
odds follow, and ``line_history_aiven.ingest.is_finished`` enforces for line
history). A prediction before tip-off therefore reads them live -- schedule,
line history so far, referee crews if released -- into a ``LiveGameDay`` that
``create_intermediate_line_df(live=...)`` adds to what it reads from the
database. Nothing here writes anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nba_ou.fetch_data.nba_schedule.fetch_nba_schedule import fetch_schedules
from nba_ou.postgre_db.line_history_aiven.live import (
    LiveLineHistory,
    fetch_live_line_history,
    scheduled_team_games,
)
from nba_ou.utils.general_utils import get_season_year_from_date


@dataclass(frozen=True)
class LiveGameDay:
    """Tonight, as of ``as_of``.

    ``scheduled_games`` is ``get_schedule_games``'s frame. ``df_referees_scheduled``
    is the crews frame from ``get_all_info_for_scheduled_games`` (GAME_ID,
    REF_1..REF_3); ``None`` or a missing game means no crew yet, which becomes
    NaN features -- exactly what a pre-09:00-ET snapshot holds in training.
    """

    game_date: pd.Timestamp
    as_of: pd.Timestamp
    scheduled_games: pd.DataFrame
    line_history: LiveLineHistory
    df_referees_scheduled: pd.DataFrame | None = None

    @property
    def history_limit(self) -> pd.Timestamp:
        """The last day the database holds: history stops where tonight begins."""
        return pd.Timestamp(self.game_date).normalize() - pd.Timedelta(days=1)

    @property
    def game_ids(self) -> list[str]:
        return sorted(self.scheduled_games["GAME_ID"].astype(str).unique())


def fetch_live_game_day(
    game_date: str | pd.Timestamp,
    *,
    scheduled_games: pd.DataFrame,
    df_referees_scheduled: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
) -> LiveGameDay:
    """Read tonight's line history for ``scheduled_games`` and bundle it.

    The schedule and crews are passed in because the prediction job already
    fetches them (``get_all_info_for_scheduled_games``); a replay passes
    ``live_replay.replay_scheduled_games`` and the stored crews instead.
    """
    game_date = pd.Timestamp(game_date).normalize()
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    season = int(get_season_year_from_date(game_date))
    line_history = fetch_live_line_history(
        game_date.date(),
        team_games=scheduled_team_games(scheduled_games, season_year=season),
        schedule=fetch_schedules([season]),
        as_of=as_of,
    )
    return LiveGameDay(
        game_date=game_date,
        as_of=line_history.as_of,
        scheduled_games=scheduled_games,
        line_history=line_history,
        df_referees_scheduled=df_referees_scheduled,
    )
