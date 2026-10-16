"""Replay a past date as if it were tonight -- the train/serve parity harness.

The live path builds tonight's rows from history plus in-memory data about
tonight (schedule, line history so far, referees). Replaying a past date means
feeding that same path a date whose rows the training build already made, and
requiring the two to agree. Everything here only *reads*; nothing is stored.
"""

from __future__ import annotations

from pathlib import Path
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
    games["GAME_TIME"] = pd.to_datetime(
        games["GAME_ID"].map(tipoff), utc=True
    ).dt.tz_convert(EASTERN)
    return games.dropna(subset=["GAME_TIME"]).reset_index(drop=True)


def stored_crews(game_ids: list[str], season_year: int) -> pd.DataFrame:
    """The stored referee crews of ``game_ids``, shaped like the live feed's
    (``GAME_ID``, ``REF_1``..``REF_3``)."""
    from nba_ou.postgre_db.injuries_refs.fetch_refs_db.get_refs_db import (
        get_refs_data_from_db,
    )

    refs = get_refs_data_from_db([f"{season_year}-{str(season_year + 1)[-2:]}"])
    refs = refs[refs["GAME_ID"].astype(str).isin(set(game_ids))].copy()
    refs["NAME"] = refs["FIRST_NAME"] + " " + refs["LAST_NAME"]
    rows = []
    for game_id, names in refs.groupby(refs["GAME_ID"].astype(str))["NAME"]:
        slots = list(names)[:3] + [None] * (3 - min(3, len(names)))
        rows.append(
            {
                "GAME_ID": game_id,
                "REF_1": slots[0],
                "REF_2": slots[1],
                "REF_3": slots[2],
            }
        )
    return pd.DataFrame(rows, columns=["GAME_ID", "REF_1", "REF_2", "REF_3"])


def replay_live_game_day(game_date: str, as_of: pd.Timestamp):
    """A past date as the live path would have seen it at ``as_of``, from storage.

    Tonight's line history is the stored ticks of the date's games quoted by
    ``as_of`` (the store's read shapes, as the live scrape returns them), the
    injury reports the archived ones published before it, read live, and the
    crews the stored ones. Live and tonight-only builds of it must agree.
    """
    from nba_ou.create_training_data.live_game_day import (
        LiveGameDay,
        fetch_live_injury_reports,
    )
    from nba_ou.postgre_db.line_history_aiven.fetch import (
        fetch_games,
        fetch_pregame_ticks,
    )
    from nba_ou.postgre_db.line_history_aiven.live import LiveLineHistory

    as_of = pd.Timestamp(as_of)
    day = pd.Timestamp(game_date).normalize()
    season = int(get_season_year_from_date(day))
    scheduled = replay_scheduled_games(game_date)
    game_ids = sorted(scheduled["GAME_ID"].astype(str))
    games = fetch_games([season])
    games = games[games["game_id"].astype(str).isin(set(game_ids))].reset_index(
        drop=True
    )
    ticks = fetch_pregame_ticks([season], use_cache=True)
    ticks = ticks[
        ticks["game_id"].astype(str).isin(set(game_ids))
        & (pd.to_datetime(ticks["line_ts"], utc=True) <= as_of)
    ].reset_index(drop=True)
    return LiveGameDay(
        game_date=day,
        as_of=as_of,
        scheduled_games=scheduled,
        line_history=LiveLineHistory(ticks=ticks, games=games, as_of=as_of),
        df_referees_scheduled=stored_crews(game_ids, season),
        injury_reports=fetch_live_injury_reports(
            day, game_ids=game_ids, as_of=as_of, refresh=False
        ),
    )


def replay_scheduled_data(game_date: str, as_of: pd.Timestamp) -> dict:
    """``get_all_info_for_scheduled_games``' dict for a played date, from storage.

    The stored crews and odds rows of the date's games stand in for the live
    feeds, and the archived injury report last published before ``as_of``
    for the live one. A prediction build of it and a full build of the same
    inputs must agree on the date's rows.
    """
    from nba_ou.config.settings import SETTINGS
    from nba_ou.data_processing.scheduled_games.manage_injury_data import (
        process_injury_data,
    )
    from nba_ou.fetch_data.injury_reports.archive import manifest as mf
    from nba_ou.fetch_data.injury_reports.archive.manifest import ManifestStore
    from nba_ou.fetch_data.injury_reports.archive.storage import S3Storage
    from nba_ou.fetch_data.injury_reports.archive.urls import season_label
    from nba_ou.fetch_data.injury_reports.get_latest_injury_report import (
        read_injury_report,
    )
    from nba_ou.postgre_db.odds_sportsbook.fetch_data_from_db.fetch_data_from_odds_sportsbook_db import (
        load_odds_sportsbook_from_db,
    )
    from nba_ou.postgre_db.odds_yahoo.fetch_data_from_db.fetch_data_from_odds_yahoo_db import (
        load_odds_yahoo_from_db,
    )

    as_of = pd.Timestamp(as_of)
    day = pd.Timestamp(game_date).normalize()
    season = int(get_season_year_from_date(day))
    scheduled = replay_scheduled_games(game_date)
    game_ids = sorted(scheduled["GAME_ID"].astype(str))

    def tonight(frame):
        if frame is None or frame.empty:
            return pd.DataFrame()
        column = "game_id" if "game_id" in frame.columns else "GAME_ID"
        return frame[frame[column].astype(str).isin(set(game_ids))].reset_index(
            drop=True
        )

    label = season_label(day.date())
    manifest = ManifestStore(Path("data/injury_reports/manifest")).load(label)
    manifest = manifest[manifest["nba_available"].eq(mf.AVAILABLE_TRUE)]
    published = mf.published_utc(manifest)
    last = manifest[published < as_of].assign(_at=published[published < as_of])
    key = last.sort_values("_at")["s3_key"].iloc[-1]
    storage = S3Storage(
        bucket=SETTINGS.s3_bucket,
        profile=SETTINGS.s3_aws_profile,
        region=SETTINGS.s3_aws_region,
    )
    report = read_injury_report(storage.get(key))
    injury_dict, not_updated = process_injury_data(scheduled, report)
    return {
        "scheduled_games": scheduled,
        "df_referees_scheduled": stored_crews(game_ids, season),
        "injury_dict_scheduled": injury_dict,
        "df_odds_yahoo_scheduled": tonight(load_odds_yahoo_from_db([label])),
        "df_odds_sportsbook_scheduled": tonight(load_odds_sportsbook_from_db([label])),
        "games_not_updated": not_updated,
    }
