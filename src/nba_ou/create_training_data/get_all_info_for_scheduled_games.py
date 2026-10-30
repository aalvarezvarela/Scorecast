import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from nba_ou.config.constants import TEAM_ID_MAP, TEAM_NAME_STANDARDIZATION
from nba_ou.config.odds_columns import (
    drop_history_only_book_columns,
    drop_sportsbook_metadata_columns,
)
from nba_ou.config.settings import SETTINGS
from nba_ou.data_processing.injury_status.report_state import (
    load_prediction_report_state,
    nested_status_dict,
    report_out_overrides,
)
from nba_ou.data_processing.odds.scheduled_opener import OPENER_COLUMNS
from nba_ou.data_processing.referees.process_refs_scheduled_game import (
    process_scheduled_referee_assignments,
)
from nba_ou.fetch_data.injury_reports.sync import synchronize_injury_reports
from nba_ou.fetch_data.live_rosters import load_live_rosters
from nba_ou.fetch_data.odds_covers.day_odds import covers_day_odds
from nba_ou.fetch_data.odds_sportsbook.process_total_lines_data import (
    load_one_day_totals_csv,
)
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import (
    build_sbr_totals_url,
    fetch_sbr_page,
    parse_sbr_odds_table,
    parse_sbr_totals_rows,
    scrape_sportsbook_days,
    season_year_for_date,
)
from nba_ou.fetch_data.odds_yahoo.scrape_yahoo import scrape_yahoo_days
from nba_ou.fetch_data.scheduled_game.get_schedule_games import get_schedule_games
from nba_ou.prediction.line_source import (
    COVERS,
    SBR,
    LineSource,
    complete_game_ids,
    fill_ids,
    priced_game_ids,
)
from nba_ou.utils.ci import annotate
from nba_ou.utils.phase_timer import PhaseTimer


def _normalize_team_series(series: pd.Series) -> pd.Series:
    """Apply ``TEAM_NAME_STANDARDIZATION`` while preserving ``object`` dtype.

    Using ``.map(...).fillna(...)`` silently downcasts an all-NaN object series
    to ``float64``, which then breaks merges against object-typed team columns.
    """
    mapped = series.map(TEAM_NAME_STANDARDIZATION)
    return mapped.where(mapped.notna(), series).astype(object)


def _backfill_missing_teams_from_odds(
    scheduled_subset: pd.DataFrame, odds_df: pd.DataFrame
) -> pd.DataFrame:
    """Fallback: recover missing ``team_home`` / ``team_away`` from odds.

    The NBA schedule endpoint occasionally returns ``None`` for one of the
    teams of an upcoming game (e.g. a play-in/playoff opponent not yet
    determined when the schedule is first published, then later cleared
    before tip-off). The bookmakers usually have both teams correctly listed
    by then, so we use the odds rows as a fallback.

    Only fills when exactly one odds row matches on ``game_date`` plus the
    side that IS known, to avoid wrong assignments. Normal games with both
    teams populated are untouched.
    """
    if odds_df.empty:
        return scheduled_subset

    missing_mask = (
        scheduled_subset["team_home"].isna() | scheduled_subset["team_away"].isna()
    )
    if not missing_mask.any():
        return scheduled_subset

    needed = {"game_date", "team_home", "team_away"}
    if not needed.issubset(odds_df.columns):
        return scheduled_subset

    out = scheduled_subset.copy()
    for idx in out.index[missing_mask]:
        game_date = out.at[idx, "game_date"]
        team_home = out.at[idx, "team_home"]
        team_away = out.at[idx, "team_away"]

        candidates = odds_df[odds_df["game_date"] == game_date]
        if pd.isna(team_home) and pd.notna(team_away):
            candidates = candidates[candidates["team_away"] == team_away]
        elif pd.isna(team_away) and pd.notna(team_home):
            candidates = candidates[candidates["team_home"] == team_home]
        else:
            # Both sides missing — too ambiguous to recover safely.
            continue

        unique = candidates.drop_duplicates(subset=["team_home", "team_away"])
        if len(unique) != 1:
            continue

        if pd.isna(team_home):
            out.at[idx, "team_home"] = unique["team_home"].iloc[0]
        if pd.isna(team_away):
            out.at[idx, "team_away"] = unique["team_away"].iloc[0]
        print(
            f"  ↳ Backfilled missing team(s) for game_id={out.at[idx, 'game_id']} "
            f"on {game_date} from odds: team_home={out.at[idx, 'team_home']}, "
            f"team_away={out.at[idx, 'team_away']}"
        )

    return out


def merge_odds_with_scheduled_games(
    odds_df: pd.DataFrame, scheduled_games_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge odds data with scheduled games to add game_id.

    Merges based on game_date, team_home, and team_away to attach the game_id
    from scheduled games to the odds data.

    Args:
        odds_df (pd.DataFrame): Odds dataframe with game_date, team_home, team_away
        scheduled_games_df (pd.DataFrame): Scheduled games with game_id, game_date, team_home, team_away

    Returns:
        pd.DataFrame: Odds data with game_id added
    """
    # drop game_id cpolumn in odds_df if it exists to avoid duplicates after merge
    if odds_df.empty:
        raise ValueError("Odds dataframe is empty, cannot merge with scheduled games")

    if scheduled_games_df.empty:
        print("No scheduled games data provided for merging")
        raise ValueError(
            "Scheduled games dataframe is empty, cannot merge with odds data"
        )

    odds_df = odds_df.drop(columns=["game_id"], errors="ignore")

    # Normalize team names in both dataframes using TEAM_NAME_STANDARDIZATION
    odds_df = odds_df.copy()
    scheduled_games_df = scheduled_games_df.copy()

    for col in ("team_home", "team_away"):
        if col in odds_df.columns:
            odds_df[col] = _normalize_team_series(odds_df[col])
        if col in scheduled_games_df.columns:
            scheduled_games_df[col] = _normalize_team_series(scheduled_games_df[col])

    # Ensure required columns exist in scheduled_games_df
    required_cols = ["game_id", "game_date", "team_home", "team_away"]
    missing_cols = [
        col for col in required_cols if col not in scheduled_games_df.columns
    ]
    if missing_cols:
        print(f"Missing columns in scheduled_games: {missing_cols}")
        return odds_df

    # Prepare merge keys
    merge_keys = ["game_date", "team_home", "team_away"]

    # Select only needed columns from scheduled_games to avoid duplicates
    scheduled_subset = scheduled_games_df[
        ["game_id", "game_date", "team_home", "team_away"]
    ].copy()

    # Safety net: force object dtype on team merge keys so an all-NaN column on
    # either side does not trigger an object/float64 merge-key dtype mismatch.
    for col in ("team_home", "team_away"):
        odds_df[col] = odds_df[col].astype(object)
        scheduled_subset[col] = scheduled_subset[col].astype(object)

    # Fallback: if the schedule endpoint left a team blank, try to recover it
    # from the odds rows so the merge can still find the game_id.
    scheduled_subset = _backfill_missing_teams_from_odds(scheduled_subset, odds_df)

    # Merge to add game_id
    merged_df = odds_df.merge(
        scheduled_subset,
        on=merge_keys,
        how="left",
    )

    return merged_df


def get_yahoo_prediction_data(
    date_to_predict: str, scheduled_games: pd.DataFrame
) -> pd.DataFrame:
    """Get Yahoo odds data for the prediction date.

    Scrapes Yahoo's odds for the games on ``date_to_predict`` (a US date) and
    merges them with the scheduled games to get the game_id.

    Args:
        date_to_predict (str): Date in format 'YYYY-MM-DD'
        scheduled_games (pd.DataFrame): DataFrame with scheduled games including game_id

    Returns:
        pd.DataFrame: Yahoo odds data merged with game_id for the scheduled games
    """
    base_date = datetime.strptime(date_to_predict, "%Y-%m-%d").date()
    df_yahoo = asyncio.run(scrape_yahoo_days([base_date]))

    if df_yahoo.empty:
        print(f"No Yahoo odds data found for {date_to_predict}")
        return pd.DataFrame()

    # Merge with scheduled_games to get the game_id
    df_yahoo_with_game_id = merge_odds_with_scheduled_games(df_yahoo, scheduled_games)

    # drop NA rows on game ID as they are not shceuled games
    df_yahoo_with_game_id = df_yahoo_with_game_id.dropna(subset=["game_id"])

    return df_yahoo_with_game_id


def _day_odds_with_game_id(
    df_sportsbook: pd.DataFrame, scheduled_games: pd.DataFrame
) -> pd.DataFrame:
    """A day-page frame as the closing builder reads it: scheduled games only."""
    # Same rule as training: stored-only books never reach a prediction frame.
    df_sportsbook = drop_sportsbook_metadata_columns(
        drop_history_only_book_columns(df_sportsbook)
    )
    if df_sportsbook.empty:
        return df_sportsbook

    # Merge with scheduled_games to get the game_id
    df_sportsbook_with_game_id = merge_odds_with_scheduled_games(
        df_sportsbook, scheduled_games
    )
    # Remove duplicates based on game identifiers
    df_sportsbook_with_game_id = df_sportsbook_with_game_id.drop_duplicates(
        subset=["game_date", "team_home", "team_away"], keep="first"
    )
    # drop NA rows on game ID as they are not scheduled games
    return df_sportsbook_with_game_id.dropna(subset=["game_id"])


def get_sportsbook_opener_data(
    date_to_predict: str, scheduled_games: pd.DataFrame
) -> pd.DataFrame:
    """Read only the opening total from SBR's main totals page.

    A frozen current-price feed can still publish the opener training reads.
    This read is independent of the source chosen for current quotes; no history
    popup or synthetic first-book quote is used. Return canonical NBA game IDs
    and raw American opening prices, ready for training's normalization.
    """
    day = datetime.strptime(date_to_predict, "%Y-%m-%d").date()
    with requests.Session() as session:
        model = parse_sbr_odds_table(fetch_sbr_page(session, build_sbr_totals_url(day)))
    if model is None:
        return pd.DataFrame(columns=["game_id", *OPENER_COLUMNS])
    raw = parse_sbr_totals_rows(model, day)
    raw.insert(1, "season", season_year_for_date(day))
    odds = load_one_day_totals_csv(raw)
    games = scheduled_games.copy()
    if not {"game_id", "game_date", "team_home", "team_away"} <= set(games):
        names = {team_id: name for name, team_id in TEAM_ID_MAP.items()}
        games["game_id"] = games["GAME_ID"].astype(str)
        games["game_date"] = pd.to_datetime(games["GAME_DATE_EST"]).dt.date
        games["team_home"] = games["HOME_TEAM_ID"].astype(str).map(names)
        games["team_away"] = games["VISITOR_TEAM_ID"].astype(str).map(names)
    return _day_odds_with_game_id(odds, games).reindex(
        columns=["game_id", *OPENER_COLUMNS]
    )


def get_sportsbook_prediction_data(
    date_to_predict: str,
    scheduled_games: pd.DataFrame,
    *,
    headless: bool,
    line_source: LineSource | None = None,
) -> pd.DataFrame:
    """Get Sportsbook Review odds data for the prediction date.

    Fetches odds (totals, spread, moneyline) for date of prediction, merges with scheduled games to get game_id.

    With a ``line_source``, a game SBR does not price in every market (or
    every game, when the run reads Covers) gets the same columns rebuilt from
    Covers (``odds_covers.day_odds``), and ``line_source.closing_sources``
    records each game's source. An SBR day page with no totals at all switches
    the run to Covers; one missing only its spread or moneyline page sends the
    games without them to Covers and keeps the run on SBR.

    Args:
        date_to_predict (str): Date in format 'YYYY-MM-DD'
        scheduled_games (pd.DataFrame): DataFrame with scheduled games including game_id

    Returns:
        pd.DataFrame: Sportsbook odds data merged with game_id for the scheduled games
    """
    # Parse the date and get date + 1
    base_date = datetime.strptime(date_to_predict, "%Y-%m-%d").date()

    df_sportsbook = pd.DataFrame()
    if line_source is None or not line_source.uses_covers:
        try:
            df_sportsbook = asyncio.run(
                scrape_sportsbook_days([base_date], headless=headless)
            )
        except Exception as exc:
            if line_source is None:
                raise
            print(f"SBR day odds failed for {date_to_predict}: {exc}")
        df_sportsbook = _day_odds_with_game_id(df_sportsbook, scheduled_games)
        if line_source is not None and not priced_game_ids(df_sportsbook):
            line_source.switch_to_covers(
                f"SBR day page has no odds for {date_to_predict}"
            )

    if line_source is not None:
        served = set() if line_source.uses_covers else complete_game_ids(df_sportsbook)
        incomplete = priced_game_ids(df_sportsbook) - served
        if incomplete:
            annotate(
                "warning",
                f"SBR day page lacks spread or moneyline for {len(incomplete)} "
                f"game(s) on {date_to_predict}; reading them from Covers.",
            )
        df_sbr = df_sportsbook
        df_sportsbook = (
            df_sbr[df_sbr["game_id"].astype(str).isin(served)]
            if served
            else pd.DataFrame()
        )
        missing = fill_ids(scheduled_games["game_id"], served)
        sources = dict.fromkeys(sorted(served), SBR)
        if missing:
            day = line_source.covers_day(scheduled_games, game_ids=missing)
            df_covers = _day_odds_with_game_id(
                covers_day_odds(
                    [day.games[g] for g in missing if g in day.games],
                    game_date=base_date,
                    as_of=day.as_of,
                    consensus=day.consensus,
                ),
                scheduled_games,
            )
            sources.update(dict.fromkeys(df_covers.get("game_id", []), COVERS))
            df_sportsbook = (
                df_covers
                if df_sportsbook.empty
                else pd.concat([df_sportsbook, df_covers], ignore_index=True)
            )
        # A game SBR prices only in part keeps that row when Covers lacks it.
        partial = sorted(incomplete - set(map(str, sources)))
        if partial:
            df_partial = df_sbr[df_sbr["game_id"].astype(str).isin(partial)]
            df_sportsbook = (
                df_partial
                if df_sportsbook.empty
                else pd.concat([df_sportsbook, df_partial], ignore_index=True)
            )
            sources.update(dict.fromkeys(partial, SBR))
        line_source.closing_sources = {str(k): v for k, v in sources.items()}

    # Check if we got any data
    if df_sportsbook.empty:
        print(f"No Sportsbook odds data found for {date_to_predict}")
        raise ValueError("No Sportsbook odds data found")

    return df_sportsbook


def get_all_info_for_scheduled_games(
    date_to_predict: str,
    nba_injury_reports_url,
    save_reports_path=None,
    headless: bool | None = None,
    *,
    dataset_type: str = "closing_line",
    line_source: LineSource | None = None,
) -> dict:
    """Get all information needed for scheduled games prediction.

    Fetches scheduled games, referee assignments, injury data, and odds from
    both Yahoo and Sportsbook Review for the specified date.

    Args:
        date_to_predict (str): Date in format 'YYYY-MM-DD'
        nba_injury_reports_url: Legacy argument; timestamped CDN URLs are now used.
        save_reports_path: Deprecated; reports are archived to S3, never local disk.
        headless (bool | None): Playwright mode. If None, uses SETTINGS.headless.
        dataset_type: Closing requires both teams to have filed. Intermediate
            keeps uncovered games, whose snapshot injury features remain NaN.
        line_source: The run's line source (``prediction.line_source``): SBR
            with Covers filling the games it does not price, or Covers. None
            reads SBR only, as before.

    Returns:
        dict: Dictionary containing:
            - scheduled_games (pd.DataFrame): Scheduled games data
            - df_referees_scheduled (pd.DataFrame): Referee assignments
            - injury_dict_scheduled (dict): Injury information
            - df_odds_yahoo_scheduled (pd.DataFrame): Yahoo odds data
            - df_odds_sportsbook_scheduled (pd.DataFrame): Sportsbook odds data
            - games_not_updated: Games to exclude for missing filing coverage;
              always empty for intermediate requests. Every game when no team
              has filed yet, which leaves the closing rows empty.
            - games_missing_injury_reports: Games with at least one uncovered team.
            - line_source: the ``line_source`` passed, shared with the
              intermediate frame's line-history read.
            - line_sources: GAME_ID -> source of its day-page odds
              (``"sbr"``/``"covers"``); empty without a ``line_source``.
    """
    if dataset_type not in {"closing_line", "intermediate_line"}:
        raise ValueError(f"Unknown prediction dataset type: {dataset_type}")
    if not date_to_predict:
        date_to_predict = pd.Timestamp.now(tz=ZoneInfo("US/Pacific")).strftime(
            "%Y-%m-%d"
        )
    if headless is None:
        headless = SETTINGS.headless

    # First Get the games itself
    scheduled_games = get_schedule_games(date_to_predict)

    if scheduled_games.empty:
        print(f"No scheduled games found for {date_to_predict}")
        return {
            "scheduled_games": pd.DataFrame(),
            "df_referees_scheduled": pd.DataFrame(),
            "injury_dict_scheduled": {},
            "df_odds_yahoo_scheduled": pd.DataFrame(),
            "df_odds_sportsbook_scheduled": pd.DataFrame(),
            "games_not_updated": [],
            "games_missing_injury_reports": [],
            "roster_snapshots": pd.DataFrame(),
            "line_source": line_source,
            "line_sources": {},
        }

    # Convert team IDs to team names using TEAM_ID_MAP
    # Create reverse mapping: ID -> Name
    id_to_name = {team_id: name for name, team_id in TEAM_ID_MAP.items()}

    # Create a temporary DataFrame with the additional columns for merging
    scheduled_games_for_merge = scheduled_games.copy()
    scheduled_games_for_merge["team_home"] = (
        scheduled_games_for_merge["HOME_TEAM_ID"].astype(str).map(id_to_name)
    )
    scheduled_games_for_merge["team_away"] = (
        scheduled_games_for_merge["VISITOR_TEAM_ID"].astype(str).map(id_to_name)
    )
    scheduled_games_for_merge["game_date"] = pd.to_datetime(
        scheduled_games_for_merge["GAME_DATE_EST"]
    ).dt.date
    scheduled_games_for_merge["game_id"] = scheduled_games_for_merge["GAME_ID"].astype(
        str
    )

    # Each part prints a [time] line as it ends: the odds in their own
    # thread, the rest in turn.
    timer = PhaseTimer()

    def odds() -> tuple[pd.DataFrame, pd.DataFrame]:
        with timer.phase("step 2 odds (beside the injury reports)"):
            yahoo = get_yahoo_prediction_data(
                date_to_predict, scheduled_games_for_merge
            )
            sportsbook = get_sportsbook_prediction_data(
                date_to_predict,
                scheduled_games_for_merge,
                headless=headless,
                line_source=line_source,
            )
        return yahoo, sportsbook

    # The odds depend on the schedule alone, and nothing below reads them:
    # they are fetched meanwhile and collected at the end.
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="odds")
    odds_future = pool.submit(odds)
    pool.shutdown(wait=False)

    # Then get refs
    with timer.phase("step 2 referees"):
        df_referees_scheduled = process_scheduled_referee_assignments(scheduled_games)
    # Reports are synchronized before any feature reads. Aiven is the sole
    # prediction source; a failed refresh may use its previous eligible state.
    with timer.phase("step 2 live rosters"):
        roster_snapshots = load_live_rosters(scheduled_games)
    injury_as_of = pd.Timestamp.now(tz="UTC")
    with timer.phase("step 2 injury report sync"):
        try:
            synced = synchronize_injury_reports(
                date_to_predict,
                scheduled_games=scheduled_games,
                as_of=injury_as_of,
                roster_snapshots=roster_snapshots,
            )
            print(
                f"Injury report sync: {synced.pending} pending, {synced.ingested} "
                f"ingested, {synced.unavailable} unavailable, {synced.failed} failed"
            )
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Injury report refresh failed; reading existing Aiven state: %s", exc
            )
    with timer.phase("step 2 injury report state"):
        injury_report_state = load_prediction_report_state(
            scheduled_games.GAME_ID.astype(str).tolist(), injury_as_of
        )
    # Tonight's games only. The state also holds every earlier covered game,
    # and this dict is merged into the realized-absence history, which must
    # keep what actually happened in those games, not their pre-game reports.
    tonight_ids = set(scheduled_games.GAME_ID.astype(str))
    injury_dict_scheduled = nested_status_dict(
        {
            key: players
            for key, players in report_out_overrides(injury_report_state).items()
            if key[0] in tonight_ids
        }
    )
    # Intermediate training keeps unknown availability as NaN, so it must not
    # inherit the closing job's filing requirement. Keep diagnostics separate
    # from the exclusion list consumed by the closing feature builder.
    games_missing_injury_reports = [
        str(row.GAME_ID)
        for row in scheduled_games.itertuples()
        if any(
            (str(row.GAME_ID), str(team)) not in injury_report_state.covered
            for team in (row.HOME_TEAM_ID, row.VISITOR_TEAM_ID)
        )
    ]
    games_not_updated = (
        games_missing_injury_reports if dataset_type == "closing_line" else []
    )

    if len(games_not_updated) == len(scheduled_games):
        # Not an error: early in the day no team may have filed yet. The
        # closing rows are then empty and their slots skip; the caller decides.
        print("No game has both teams' injury reports filed yet: no closing rows")

    print("Fetched and processed scheduled games, referees, and injuries")
    # A failed odds fetch raises here, as it did when it ran last.
    df_odds_yahoo_scheduled, df_odds_sportsbook_scheduled = odds_future.result()

    return {
        "scheduled_games": scheduled_games,
        "df_referees_scheduled": df_referees_scheduled,
        "injury_dict_scheduled": injury_dict_scheduled,
        "injury_report_state": injury_report_state,
        "injury_as_of_utc": injury_as_of,
        "roster_snapshots": roster_snapshots,
        "df_odds_yahoo_scheduled": df_odds_yahoo_scheduled,
        "df_odds_sportsbook_scheduled": df_odds_sportsbook_scheduled,
        "games_not_updated": games_not_updated,
        "games_missing_injury_reports": games_missing_injury_reports,
        "line_source": line_source,
        "line_sources": (
            {} if line_source is None else dict(line_source.closing_sources)
        ),
    }
