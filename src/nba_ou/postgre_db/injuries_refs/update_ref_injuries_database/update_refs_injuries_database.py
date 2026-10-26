"""
NBA Over/Under Predictor - Referees and Injuries Database Update Script

This script manages the update of the NBA referees and injuries databases by fetching
new data from the NBA API and loading it to PostgreSQL.
"""

import pandas as pd
from nba_ou.fetch_data.fetch_nba_data.inactive_rows import (
    FIRST_BOX_SCORE_INACTIVES_SEASON,
)
from nba_ou.postgre_db.injuries_refs.creation.create_injuries_db import (
    upload_injuries_data_to_db,
)
from nba_ou.postgre_db.injuries_refs.creation.create_refs_db import (
    upload_refs_data_to_db,
)
from nba_ou.postgre_db.injuries_refs.update_ref_injuries_database.update_refs_injuries_database_utils import (
    fetch_refs_injuries_data,
    get_existing_injury_game_ids_from_db,
    get_existing_ref_game_ids_from_db,
)
from nba_ou.utils.general_utils import get_season_nullable_from_year


def load_existing_data(filepath: str, dtype: dict):
    """Attempts to load existing CSV data, returns None if file not found."""
    try:
        df = pd.read_csv(filepath, dtype=dtype)
        print(f"Existing data found: {filepath}")
        return df
    except FileNotFoundError:
        print(f"No existing data found: {filepath}. Starting from scratch...")
        return None


def upload_refs_to_postgresql(refs_df: pd.DataFrame):
    """Uploads referee data to PostgreSQL database.

    Args:
        refs_df: DataFrame containing referee data

    Returns:
        bool: True if successful, False otherwise
    """
    if refs_df is None or refs_df.empty:
        print("No referee data to upload to PostgreSQL.")
        return False

    print("\nUploading referee data to PostgreSQL database...")
    success = upload_refs_data_to_db(refs_df, if_exists="append")
    if success:
        print("✅ Successfully uploaded referee data to PostgreSQL!")
    else:
        print("❌ Failed to upload referee data to PostgreSQL.")
    return success


def upload_injuries_to_postgresql(injuries_df: pd.DataFrame):
    """Uploads injury data to PostgreSQL database.

    Args:
        injuries_df: DataFrame containing injury data

    Returns:
        bool: True if successful, False otherwise
    """
    if injuries_df is None or injuries_df.empty:
        print("No injury data to upload to PostgreSQL.")
        return False

    print("\nUploading injury data to PostgreSQL database...")
    success = upload_injuries_data_to_db(injuries_df, if_exists="append")
    if success:
        print("✅ Successfully uploaded injury data to PostgreSQL!")
    else:
        print("❌ Failed to upload injury data to PostgreSQL.")
    return success


def update_refs_injuries_database(
    season_year,
    exclude_game_ids: list[str] | None = None,
):
    """
    Main function to update referees and injuries databases.

    Fetches the box-score summary of every game not done yet: its referees
    are not stored or, up to 2025-26, its inactive list is not. The summary's
    inactive list, when it has one, goes to the injuries table as before.

    Args:
        season_year: Year to determine the season (e.g., 2023 for 2023-24)
        exclude_game_ids: Optional list of game IDs to exclude from uploads

    Returns:
        bool: True if rate limit was reached, False otherwise
    """

    season_nullable = get_season_nullable_from_year(season_year)

    print(f"Updating referees and injuries for season: {season_nullable}")

    existing_game_ids = get_existing_ref_game_ids_from_db(season_year)
    if int(season_year) < FIRST_BOX_SCORE_INACTIVES_SEASON:
        # The summary is these seasons' only source of the inactive list, so a
        # game is done once both its crew and its inactives are stored.
        existing_game_ids &= get_existing_injury_game_ids_from_db(season_year)
    # From 2026-27 the summary's inactive list is empty and the box-score
    # update stores it (``fetch_data.fetch_nba_data.inactive_rows``): the crew
    # alone says a game is done, or every game would be fetched daily.
    print(f"Games already done: {len(existing_game_ids)}")

    refs_df, injuries_df, limit_reached = fetch_refs_injuries_data(
        season_nullable,
        existing_game_ids=existing_game_ids,
        n_tries=3,
    )

    if exclude_game_ids:
        exclude_set = {str(game_id) for game_id in exclude_game_ids if game_id}
        if refs_df is not None and not refs_df.empty:
            refs_df = refs_df[~refs_df["GAME_ID"].astype(str).isin(exclude_set)]
        if injuries_df is not None and not injuries_df.empty:
            injuries_df = injuries_df[
                ~injuries_df["GAME_ID"].astype(str).isin(exclude_set)
            ]

    if refs_df is not None and not refs_df.empty:
        upload_refs_to_postgresql(refs_df)

    if injuries_df is not None and not injuries_df.empty:
        upload_injuries_to_postgresql(injuries_df)

    print(f"Completed season: {season_nullable}\n" + "-" * 50)
    return limit_reached


if __name__ == "__main__":
    from nba_ou.postgre_db.season_tables import current_season_year

    update_refs_injuries_database(season_year=current_season_year())
