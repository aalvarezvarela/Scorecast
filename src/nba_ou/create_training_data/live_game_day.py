"""Everything about tonight that the database does not hold yet, read in memory.

Tonight's games are stored only once they are finished (the rule the closing
odds follow, and ``line_history_aiven.ingest.is_finished`` enforces for line
history). A prediction before tip-off therefore reads them live -- schedule,
line history so far, referee crews if released -- into a ``LiveGameDay`` that
``create_intermediate_line_df(live=...)`` adds to what it reads from the
database. Roster snapshots are archived to S3 and new injury PDFs are synced
to Aiven before a current prediction; line history remains memory-only.

The line history is SBR's, or Covers' when the run's ``LineSource`` says so
(``prediction.line_source``): a game SBR serves no tick for is read from
Covers either way, and the source of each game is left in
``line_source.history_sources``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from nba_ou.fetch_data.injury_reports.sync import synchronize_injury_reports
from nba_ou.fetch_data.live_rosters import load_live_rosters
from nba_ou.fetch_data.nba_schedule.fetch_nba_schedule import fetch_schedules
from nba_ou.postgre_db.injury_report_aiven.live import LiveInjuryReports
from nba_ou.postgre_db.line_history_aiven.alt_lines import drop_alternate_lines
from nba_ou.postgre_db.line_history_aiven.live import (
    LiveLineHistory,
    fetch_live_line_history,
    scheduled_team_games,
)
from nba_ou.postgre_db.line_history_covers.live import (
    covers_live_line_history,
    fill_live_line_history,
    freshest_minutes_before_tip,
    games_with_ticks,
    truncated_game_ids,
)
from nba_ou.prediction.line_source import COVERS, SBR, LineSource, fill_ids
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
    line_source: LineSource | None = None,
) -> LiveGameDay:
    """Read tonight's line history and injury reports for ``scheduled_games``.

    The schedule and crews are passed in because the prediction job already
    fetches them (``get_all_info_for_scheduled_games``); a replay passes
    ``live_replay.replay_scheduled_games`` and the stored crews instead.
    Current injury reports are synchronized to S3/Aiven before feature reads;
    ``injury_reports=False`` skips that refresh and reads the existing store.
    An explicit ``as_of`` replays stored state without refreshing reports.
    ``line_source`` (None: SBR only) brings in Covers' line history.
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
    team_games = scheduled_team_games(scheduled_games, season_year=season)
    schedule = fetch_schedules([season])
    if line_source is None:
        line_history = fetch_live_line_history(
            game_date.date(), team_games=team_games, schedule=schedule, as_of=as_of
        )
    else:
        line_history = _line_history_with_fallback(
            game_date,
            scheduled_games=scheduled_games,
            team_games=team_games,
            schedule=schedule,
            as_of=as_of,
            line_source=line_source,
            # Live, Covers is read again here: the ticks since the closing
            # frame's read belong to this moment.
            refresh=refresh,
        )
    # As the stored reads drop them (fetch_pregame_ticks), so tonight's rows
    # are built from the same kind of ticks as the training rows.
    line_history = replace(line_history, ticks=drop_alternate_lines(line_history.ticks))
    return LiveGameDay(
        game_date=game_date,
        as_of=line_history.as_of,
        scheduled_games=scheduled_games,
        line_history=line_history,
        df_referees_scheduled=df_referees_scheduled,
        roster_snapshots=roster_snapshots,
    )


def _line_history_with_fallback(
    game_date: pd.Timestamp,
    *,
    scheduled_games: pd.DataFrame,
    team_games: pd.DataFrame,
    schedule: pd.DataFrame,
    as_of: pd.Timestamp,
    line_source: LineSource,
    refresh: bool,
) -> LiveLineHistory:
    """SBR's live line history with Covers filling what it does not serve.

    Not served: a game with no SBR tick, or one already tipped whose SBR
    history stops well before tip (``truncated_game_ids``) while Covers'
    goes on later.
    """
    line_history = None
    if not line_source.uses_covers:
        try:
            line_history = fetch_live_line_history(
                game_date.date(), team_games=team_games, schedule=schedule, as_of=as_of
            )
        except Exception as exc:  # noqa: BLE001 -- Covers stands in
            line_source.switch_to_covers(f"SBR live line history failed: {exc}")
        else:
            if not games_with_ticks(line_history):
                line_source.switch_to_covers("SBR live line history has no ticks")
    if line_source.uses_covers:
        served: set[str] = set()
        truncated: set[str] = set()
    else:
        truncated = truncated_game_ids(line_history, as_of)
        served = games_with_ticks(line_history) - truncated
    missing = fill_ids(scheduled_games["GAME_ID"], served)
    sources = dict.fromkeys(sorted(served | truncated), SBR)
    if missing:
        day = line_source.covers_day(
            scheduled_games, game_ids=missing, schedule=schedule, refresh=refresh
        )
        covers = covers_live_line_history(
            day, team_games=team_games, schedule=schedule, game_ids=missing, as_of=as_of
        )
        sbr_fresh = (
            {} if line_history is None else freshest_minutes_before_tip(line_history)
        )
        covers_fresh = freshest_minutes_before_tip(covers)
        # A truncated SBR game moves only if Covers' history goes on later.
        taken = {
            game_id
            for game_id in games_with_ticks(covers)
            if game_id not in truncated
            or covers_fresh[game_id] < sbr_fresh.get(game_id, float("inf"))
        }
        sources.update(dict.fromkeys(sorted(taken), COVERS))
        line_history = (
            covers
            if line_history is None or line_source.uses_covers
            else fill_live_line_history(line_history, covers, taken)
        )
    line_source.history_sources = sources
    return line_history
