"""Everything about tonight that the database does not hold yet, read in memory.

Tonight's games are stored only once they are finished (the rule the closing
odds follow, and ``line_history_aiven.ingest.is_finished`` enforces for line
history). A prediction before tip-off therefore reads them live -- schedule,
line history so far, referee crews if released -- into a ``LiveGameDay`` that
``create_intermediate_line_df(live=...)`` adds to what it reads from the
database. Roster snapshots are archived to S3 and new injury PDFs are synced
to Aiven before a current prediction; line history remains memory-only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nba_ou.fetch_data.injury_reports.sync import synchronize_injury_reports
from nba_ou.fetch_data.live_rosters import load_live_rosters
from nba_ou.fetch_data.nba_schedule.fetch_nba_schedule import fetch_schedules
from nba_ou.postgre_db.injury_report_aiven.live import LiveInjuryReports
from nba_ou.postgre_db.line_history_aiven.live import (
    LiveLineHistory,
    fetch_live_line_history,
    scheduled_team_games,
)
from nba_ou.utils.general_utils import get_season_year_from_date

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveGameDay:
    """Tonight, as of ``as_of``.

    ``scheduled_games`` is ``get_schedule_games``'s frame. ``df_referees_scheduled``
    is the crews frame from ``get_all_info_for_scheduled_games`` (GAME_ID,
    REF_1..REF_3); ``None`` or a missing game means no crew yet, which becomes
    NaN features -- exactly what a pre-09:00-ET snapshot holds in training.
    ``roster_snapshots`` holds official team membership observations with UTC
    timestamps; all horizons use the latest same-day observation at ``as_of``.
    """

    game_date: pd.Timestamp
    as_of: pd.Timestamp
    scheduled_games: pd.DataFrame
    line_history: LiveLineHistory
    df_referees_scheduled: pd.DataFrame | None = None
    roster_snapshots: pd.DataFrame | None = None
    #: Optional archived-PDF overlay for replay diagnostics. Current prediction
    #: reads the synchronized Aiven store, so this stays ``None``.
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
    """Archived injury reports before ``as_of``, for replay diagnostics.

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
    roster_snapshots: pd.DataFrame | None = None,
    injury_reports: bool = True,
) -> LiveGameDay:
    """Read tonight's line history and injury reports for ``scheduled_games``.

    The schedule and crews are passed in because the prediction job already
    fetches them (``get_all_info_for_scheduled_games``); a replay passes
    ``live_replay.replay_scheduled_games`` and the stored crews instead.
    Current injury reports are synchronized to S3/Aiven before feature reads;
    ``injury_reports=False`` skips that refresh and reads the existing store.
    An explicit ``as_of`` replays stored state without refreshing reports.
    """
    game_date = pd.Timestamp(game_date).normalize()
    if roster_snapshots is None:
        roster_snapshots = load_live_rosters(scheduled_games, as_of=as_of)
    refresh = as_of is None
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if refresh and injury_reports:
        try:
            synchronize_injury_reports(
                game_date,
                scheduled_games=scheduled_games,
                roster_snapshots=roster_snapshots,
                as_of=as_of,
            )
        except Exception as exc:
            log.warning(
                "Injury report refresh failed; reading existing Aiven state: %s", exc
            )
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
        roster_snapshots=roster_snapshots,
    )
