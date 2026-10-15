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
from pathlib import Path

import pandas as pd

from nba_ou.fetch_data.nba_schedule.fetch_nba_schedule import fetch_schedules
from nba_ou.postgre_db.injury_report_aiven.live import LiveInjuryReports
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
    #: Tonight's injury reports published so far; ``None`` reads the store,
    #: which holds a game only once it is played.
    injury_reports: LiveInjuryReports | None = None

    @property
    def history_limit(self) -> pd.Timestamp:
        """The last day the database holds: history stops where tonight begins."""
        return pd.Timestamp(self.game_date).normalize() - pd.Timedelta(days=1)

    @property
    def game_ids(self) -> list[str]:
        return sorted(self.scheduled_games["GAME_ID"].astype(str).unique())


def fetch_live_injury_reports(
    game_date: pd.Timestamp,
    *,
    game_ids: list[str],
    as_of: pd.Timestamp,
    refresh: bool = True,
) -> LiveInjuryReports:
    """Tonight's injury reports published before ``as_of``, read in memory.

    ``refresh`` first fetches the reports published since the last run into the
    S3 archive; a replay of a past date reads the archive as it is.
    """
    from nba_ou.config.settings import SETTINGS
    from nba_ou.fetch_data.injury_reports.archive.live import refresh_recent_reports
    from nba_ou.fetch_data.injury_reports.archive.manifest import ManifestStore
    from nba_ou.fetch_data.injury_reports.archive.storage import S3Storage
    from nba_ou.fetch_data.injury_reports.archive.urls import season_label
    from nba_ou.postgre_db.config.db_config import connect_nba_db
    from nba_ou.postgre_db.injury_report_aiven.live import build_live_injury_reports

    day = pd.Timestamp(game_date).date()
    storage = S3Storage(
        bucket=SETTINGS.s3_bucket,
        profile=SETTINGS.s3_aws_profile,
        region=SETTINGS.s3_aws_region,
    )
    season = season_label(day)
    if refresh:
        manifest = refresh_recent_reports(day, as_of, storage=storage)
    else:
        manifest = ManifestStore(Path("data/injury_reports/manifest")).load(season)
    with connect_nba_db(env="supabase") as source:
        return build_live_injury_reports(
            manifest,
            storage,
            source,
            game_date=day,
            game_ids=game_ids,
            as_of=as_of,
            season=season,
        )


def fetch_live_game_day(
    game_date: str | pd.Timestamp,
    *,
    scheduled_games: pd.DataFrame,
    df_referees_scheduled: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
    injury_reports: bool = True,
) -> LiveGameDay:
    """Read tonight's line history and injury reports for ``scheduled_games``.

    The schedule and crews are passed in because the prediction job already
    fetches them (``get_all_info_for_scheduled_games``); a replay passes
    ``live_replay.replay_scheduled_games`` and the stored crews instead.
    ``injury_reports=False`` leaves tonight's reports to the store, which
    holds them only once the games are played.
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
    game_ids = sorted(scheduled_games["GAME_ID"].astype(str).unique())
    return LiveGameDay(
        game_date=game_date,
        as_of=line_history.as_of,
        scheduled_games=scheduled_games,
        line_history=line_history,
        df_referees_scheduled=df_referees_scheduled,
        injury_reports=(
            fetch_live_injury_reports(
                game_date, game_ids=game_ids, as_of=line_history.as_of
            )
            if injury_reports
            else None
        ),
    )
