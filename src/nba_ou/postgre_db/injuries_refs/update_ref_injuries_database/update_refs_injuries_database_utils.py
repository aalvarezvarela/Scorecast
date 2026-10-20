"""
NBA Over/Under Predictor - Referees and Injuries Database Utilities

This module contains utility functions for fetching and updating NBA referee
and injury data from the NBA API.
"""

import random
import re
import time

import pandas as pd
from nba_api.stats.endpoints import BoxScoreSummaryV3, LeagueGameFinder
from nba_ou.config.constants import SEASON_TYPE_MAP, TEAM_ID_MAP
from nba_ou.fetch_data.nba_api_session import reset_nba_http_session
from nba_ou.postgre_db.config.db_config import (
    connect_injuries_db,
    connect_refs_db,
    get_schema_name_injuries,
    get_schema_name_refs,
)
from nba_ou.postgre_db.injuries_refs.mapping_v3_v2_injuries_refs import (
    V3_TO_V2_INJURIES_MAP,
    V3_TO_V2_OFFICIALS_MAP,
)
from psycopg import sql
from tqdm import tqdm


def get_existing_injury_game_ids_from_db(season_year: str, db_connection=None) -> set:
    """Query existing injury game IDs from PostgreSQL database for a specific season.

    Args:
        season_year: The first 4 digits of the season (e.g., '2024')
        db_connection: Optional database connection. If None, attempts to create one.

    Returns:
        set: Set of existing game IDs in the injuries database for the given season
    """
    close_conn = False
    if db_connection is None:
        try:
            db_connection = connect_injuries_db()
            close_conn = True
        except Exception as e:
            print(f"Could not connect to injuries database: {e}")
            return set()

    cursor = db_connection.cursor()
    schema_name = get_schema_name_injuries()

    query = sql.SQL(
        """
        SELECT DISTINCT game_id
        FROM {}.nba_injuries
        WHERE season_year = %s
    """
    ).format(sql.Identifier(schema_name))

    cursor.execute(query, (int(season_year),))
    game_ids = {row[0] for row in cursor.fetchall()}

    cursor.close()
    if close_conn:
        db_connection.close()

    print(
        f"Found {len(game_ids)} existing games in injuries database for season {season_year}"
    )
    return game_ids


def get_existing_ref_game_ids_from_db(season_year: str, db_connection=None) -> set:
    """Query existing ref game IDs from PostgreSQL database for a specific season.

    Args:
        season_year: The first 4 digits of the season (e.g., '2024')
        db_connection: Optional database connection. If None, attempts to create one.

    Returns:
        set: Set of existing game IDs in the refs database for the given season
    """
    close_conn = False
    if db_connection is None:
        try:
            db_connection = connect_refs_db()
            close_conn = True
        except Exception as e:
            print(f"Could not connect to refs database: {e}")
            return set()

    cursor = db_connection.cursor()
    schema_name = get_schema_name_refs()

    query = sql.SQL(
        """
        SELECT DISTINCT game_id
        FROM {}.nba_refs
        WHERE season_year = %s
    """
    ).format(sql.Identifier(schema_name))

    cursor.execute(query, (int(season_year),))
    game_ids = {row[0] for row in cursor.fetchall()}

    cursor.close()
    if close_conn:
        db_connection.close()

    print(
        f"Found {len(game_ids)} existing games in refs database for season {season_year}"
    )
    return game_ids


def calculate_season_year(date):
    """
    Calculate NBA season year from game date.

    NBA season logic: Jan-Jul games belong to previous year's season,
    Aug-Dec games belong to current year's season.

    Args:
        date: Date object or parseable date string

    Returns:
        Season year (int) or None if date is invalid
    """
    if pd.isna(date):
        return None

    if hasattr(date, "month"):
        month = date.month
        year = date.year
    else:
        dt = pd.to_datetime(date)
        month = dt.month
        year = dt.year

    return year - 1 if month in [1, 2, 3, 4, 5, 6, 7] else year


def fetch_box_score_summary(game_id: str, season_id: str, game_date, n_tries: int = 3):
    """
    Fetches boxscore summary data for a single game, extracting referee and injury info.
    Uses V3 API and applies mapping to maintain V2 column compatibility.

    Args:
        game_id: NBA game ID
        season_id: Season ID (e.g., '12024')
        game_date: Game date
        n_tries: Number of retry attempts

    Returns:
        tuple: (df_refs, df_injuries, limit_reached)
    """
    limit_reached = False
    df_refs = None
    df_injuries = None

    attempts = 0
    time.sleep(random.uniform(0.05, 0.1))

    while attempts < n_tries:
        try:
            boxscore = BoxScoreSummaryV3(game_id=game_id)
            game_info = boxscore.get_data_frames()

            df_refs = game_info[3].copy()
            df_refs.rename(columns=V3_TO_V2_OFFICIALS_MAP, inplace=True)

            df_refs["GAME_ID"] = game_id
            df_refs["SEASON_ID"] = season_id
            df_refs["GAME_DATE"] = game_date

            df_injuries = game_info[5].copy()
            df_injuries.rename(columns=V3_TO_V2_INJURIES_MAP, inplace=True)

            df_injuries["GAME_ID"] = game_id
            df_injuries["SEASON_ID"] = season_id
            df_injuries["GAME_DATE"] = game_date

            if (
                "TEAM_NAME" in df_injuries.columns
                and "TEAM_ID" not in df_injuries.columns
            ):
                df_injuries["TEAM_ID"] = df_injuries["TEAM_NAME"].map(TEAM_ID_MAP)

            if "TEAM_ID" not in df_injuries.columns:
                df_injuries["TEAM_ID"] = None
            if "TEAM_CITY" not in df_injuries.columns:
                df_injuries["TEAM_CITY"] = None
            if "TEAM_NAME" not in df_injuries.columns:
                df_injuries["TEAM_NAME"] = None
            if "TEAM_ABBREVIATION" not in df_injuries.columns:
                df_injuries["TEAM_ABBREVIATION"] = None

            break

        except AttributeError as e:
            print(f"Rate limit likely reached for game {game_id}: {e}")
            limit_reached = False
            break

        except Exception as e:
            attempts += 1
            if attempts < n_tries:
                print(
                    f"Error fetching boxscore summary for game {game_id} (attempt {attempts}/{n_tries}): {e}"
                )
                time.sleep(random.uniform(1, 3))
                reset_nba_http_session()
            else:
                print(
                    f"Failed to fetch boxscore summary for game {game_id} after {n_tries} attempts: {e}"
                )

    if df_refs is None or df_injuries is None:
        print(
            f"Warning: Missing data for game_id {game_id}. Refs: {df_refs is not None}, Injuries: {df_injuries is not None}"
        )

    return df_refs, df_injuries, limit_reached


def fetch_refs_injuries_data(
    season_nullable: str,
    existing_game_ids: set = None,
    n_tries: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Fetches and processes NBA referee and injury data for a specified season.

    Args:
        season_nullable: The NBA season in the format 'YYYY-YY' (e.g., '2023-24').
        existing_game_ids: A set of game IDs already present in the database.
        n_tries: The number of attempts to fetch each game's data in case of failures.

    Returns:
        tuple:
            - pd.DataFrame: Referee data
            - pd.DataFrame: Injury data
            - bool: Flag indicating if the rate limit was reached
    """
    limit_reached = False

    if not re.match(r"^\d{4}-\d{2}$", season_nullable):
        raise ValueError("Invalid season format. Expected format: 'YYYY-YY'")

    for attempt in range(1, 4):
        try:
            time.sleep(random.uniform(0.1, 0.3))
            games = LeagueGameFinder(season_nullable=season_nullable).get_data_frames()[
                0
            ]
            break
        except Exception as e:
            if attempt < 3:
                print(
                    f"Error fetching LeagueGameFinder for {season_nullable} (attempt {attempt}/3): {e}"
                )
                time.sleep(random.uniform(2, 5))
                reset_nba_http_session()
            else:
                print(
                    f"Failed to fetch LeagueGameFinder for {season_nullable} after 3 attempts: {e}"
                )
                return pd.DataFrame(), pd.DataFrame(), False

    games_unique = games.drop_duplicates(subset=["GAME_ID"]).copy()

    if existing_game_ids is not None:
        existing_ids = existing_game_ids
    else:
        existing_ids = set()

    game_ids_to_fetch = set(games_unique["GAME_ID"]) - existing_ids

    if not game_ids_to_fetch:
        print(
            f"No new games found for season {season_nullable}. Skipping data fetch..."
        )
        return pd.DataFrame(), pd.DataFrame(), False

    game_info_map = games_unique.set_index("GAME_ID")[
        ["SEASON_ID", "GAME_DATE"]
    ].to_dict("index")

    all_refs = []
    all_injuries = []
    fetched_counter = 0
    excluded_types = {
        code
        for code, name in SEASON_TYPE_MAP.items()
        if name in ("Preseason", "All Star")
    }
    game_ids_to_fetch = [
        game_id
        for game_id in game_ids_to_fetch
        if str(game_id)[:3] not in excluded_types
    ]
    for game_id in tqdm(game_ids_to_fetch, desc="Fetching Refs/Injuries Data"):
        time.sleep(random.uniform(0.1, 0.5))
        game_info = game_info_map.get(game_id)
        if not game_info:
            print(f"Warning: Could not find game info for {game_id}")
            continue

        season_id = game_info["SEASON_ID"]
        game_date = game_info["GAME_DATE"]

        df_refs, df_injuries, limit_reached = fetch_box_score_summary(
            game_id, season_id, game_date, n_tries
        )

        if df_refs is not None and not df_refs.empty:
            all_refs.append(df_refs)

        if df_injuries is not None and not df_injuries.empty:
            all_injuries.append(df_injuries)

        fetched_counter += 1
        if fetched_counter == 299:
            print("\n⚠️ Reached 299 games. Pausing to avoid rate limit...")
            time.sleep(5)
            reset_nba_http_session()
            break

        if limit_reached:
            break

    refs_df = pd.concat(all_refs, ignore_index=True) if all_refs else pd.DataFrame()
    injuries_df = (
        pd.concat(all_injuries, ignore_index=True) if all_injuries else pd.DataFrame()
    )

    if not refs_df.empty:
        refs_df["GAME_DATE"] = pd.to_datetime(
            refs_df["GAME_DATE"], format="mixed", errors="coerce"
        )
        refs_df["SEASON_YEAR"] = refs_df["GAME_DATE"].apply(calculate_season_year)
        refs_df = refs_df.dropna(subset=["SEASON_YEAR"])
        refs_df["SEASON_YEAR"] = refs_df["SEASON_YEAR"].astype(int)

    if not injuries_df.empty:
        injuries_df["GAME_DATE"] = pd.to_datetime(
            injuries_df["GAME_DATE"], format="mixed", errors="coerce"
        )
        injuries_df["SEASON_YEAR"] = injuries_df["GAME_DATE"].apply(
            calculate_season_year
        )
        injuries_df = injuries_df.dropna(subset=["SEASON_YEAR"])
        injuries_df["SEASON_YEAR"] = injuries_df["SEASON_YEAR"].astype(int)

    return refs_df, injuries_df, limit_reached
