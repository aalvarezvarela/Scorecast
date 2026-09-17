import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from nba_ou.config.constants import TEAM_ID_MAP, TEAM_NAME_STANDARDIZATION
from nba_ou.config.odds_columns import (
    drop_history_only_book_columns,
    drop_sportsbook_metadata_columns,
)
from nba_ou.config.settings import SETTINGS
from nba_ou.data_processing.referees.process_refs_scheduled_game import (
    process_scheduled_referee_assignments,
)
from nba_ou.data_processing.scheduled_games.manage_injury_data import (
    process_injury_data,
)
from nba_ou.fetch_data.injury_reports.get_latest_injury_report import (
    retrieve_injury_report_as_df,
)
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import scrape_sportsbook_days
from nba_ou.fetch_data.odds_yahoo.process_yahoo_day import yahoo_one_row_per_game
from nba_ou.fetch_data.odds_yahoo.scrape_yahoo import scrape_yahoo_days
from nba_ou.fetch_data.scheduled_game.get_schedule_games import get_schedule_games


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
    date_to_predict: str, scheduled_games: pd.DataFrame, *, headless: bool
) -> pd.DataFrame:
    """Get Yahoo odds data for the prediction date.

    Fetches odds for both the date and date+1 to account for timezone differences
    (Yahoo uses local time in Europe). Merges with scheduled games to get game_id.

    Args:
        date_to_predict (str): Date in format 'YYYY-MM-DD'
        scheduled_games (pd.DataFrame): DataFrame with scheduled games including game_id

    Returns:
        pd.DataFrame: Yahoo odds data merged with game_id for the scheduled games
    """
    # Parse the date and get date + 1
    base_date = datetime.strptime(date_to_predict, "%Y-%m-%d").date()
    next_date = base_date + timedelta(days=1)

    # Scrape both dates to account for timezone differences
    days_to_scrape = [next_date, base_date]
    df_yahoo = asyncio.run(
        scrape_yahoo_days(
            days_to_scrape, headless=headless, target_dates=[base_date] * 2
        )
    )

    # Check if we got any data
    if df_yahoo.empty:
        print(f"No Yahoo odds data found for {date_to_predict}")
        return pd.DataFrame()

    # Convert from 2 rows per game to 1 row per game
    df_yahoo_processed = yahoo_one_row_per_game(df_yahoo)

    # Merge with scheduled_games to get the game_id
    df_yahoo_with_game_id = merge_odds_with_scheduled_games(
        df_yahoo_processed, scheduled_games
    )
    # Remove duplicates from scraping multiple dates for timezone coverage
    df_yahoo_with_game_id = df_yahoo_with_game_id.drop_duplicates(
        subset=["game_date", "team_home", "team_away"], keep="first"
    )

    # drop NA rows on game ID as they are not shceuled games
    df_yahoo_with_game_id = df_yahoo_with_game_id.dropna(subset=["game_id"])

    return df_yahoo_with_game_id


def get_sportsbook_prediction_data(
    date_to_predict: str, scheduled_games: pd.DataFrame, *, headless: bool
) -> pd.DataFrame:
    """Get Sportsbook Review odds data for the prediction date.

    Fetches odds (totals, spread, moneyline) for date of prediction, merges with scheduled games to get game_id.

    Args:
        date_to_predict (str): Date in format 'YYYY-MM-DD'
        scheduled_games (pd.DataFrame): DataFrame with scheduled games including game_id

    Returns:
        pd.DataFrame: Sportsbook odds data merged with game_id for the scheduled games
    """
    # Parse the date and get date + 1
    base_date = datetime.strptime(date_to_predict, "%Y-%m-%d").date()

    # Scrape both dates to account for timezone differences
    days_to_scrape = [base_date]
    df_sportsbook = asyncio.run(
        scrape_sportsbook_days(days_to_scrape, headless=headless)
    )

    # Same rule as training: stored-only books never reach a prediction frame.
    df_sportsbook = drop_sportsbook_metadata_columns(
        drop_history_only_book_columns(df_sportsbook)
    )

    # Check if we got any data
    if df_sportsbook.empty:
        print(f"No Sportsbook odds data found for {date_to_predict}")
        raise ValueError("No Sportsbook odds data found")

    # Merge with scheduled_games to get the game_id
    df_sportsbook_with_game_id = merge_odds_with_scheduled_games(
        df_sportsbook, scheduled_games
    )
    # Remove duplicates based on game identifiers
    df_sportsbook_with_game_id = df_sportsbook_with_game_id.drop_duplicates(
        subset=["game_date", "team_home", "team_away"], keep="first"
    )
    # drop NA rows on game ID as they are not scheduled games
    df_sportsbook_with_game_id = df_sportsbook_with_game_id.dropna(subset=["game_id"])

    return df_sportsbook_with_game_id


def get_all_info_for_scheduled_games(
    date_to_predict: str,
    nba_injury_reports_url,
    save_reports_path=None,
    headless: bool | None = None,
) -> dict:
    """Get all information needed for scheduled games prediction.

    Fetches scheduled games, referee assignments, injury data, and odds from
    both Yahoo and Sportsbook Review for the specified date.

    Args:
        date_to_predict (str): Date in format 'YYYY-MM-DD'
        nba_injury_reports_url: URL for NBA injury reports
        save_reports_path: Path to save injury reports
        headless (bool | None): Playwright mode. If None, uses SETTINGS.headless.

    Returns:
        dict: Dictionary containing:
            - scheduled_games (pd.DataFrame): Scheduled games data
            - df_referees_scheduled (pd.DataFrame): Referee assignments
            - injury_dict_scheduled (dict): Injury information
            - df_odds_yahoo_scheduled (pd.DataFrame): Yahoo odds data
            - df_odds_sportsbook_scheduled (pd.DataFrame): Sportsbook odds data
    """
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

    # Then get refs
    df_referees_scheduled = process_scheduled_referee_assignments(scheduled_games)
    # Then Injuries
    injury_report_df = retrieve_injury_report_as_df(
        nba_injury_reports_url, reports_path=save_reports_path
    )

    injury_dict_scheduled, games_not_updated = process_injury_data(
        scheduled_games, injury_report_df
    )

    if len(games_not_updated) == len(scheduled_games):
        raise ValueError("No games were updated with injury data")

    print("Fetched and processed scheduled games, referees, and injuries")
    print("Processing odds data for scheduled games...")
    # Fetch Yahoo odds data for the scheduled games using the temp dataframe
    df_odds_yahoo_scheduled = get_yahoo_prediction_data(
        date_to_predict, scheduled_games_for_merge, headless=headless
    )
    df_odds_sportsbook_scheduled = get_sportsbook_prediction_data(
        date_to_predict, scheduled_games_for_merge, headless=headless
    )

    return {
        "scheduled_games": scheduled_games,
        "df_referees_scheduled": df_referees_scheduled,
        "injury_dict_scheduled": injury_dict_scheduled,
        "df_odds_yahoo_scheduled": df_odds_yahoo_scheduled,
        "df_odds_sportsbook_scheduled": df_odds_sportsbook_scheduled,
        "games_not_updated": games_not_updated,
    }
