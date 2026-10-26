import pandas as pd
import psycopg
from nba_ou.fetch_data.fetch_nba_data.inactive_rows import is_inactive_comment
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_players,
)
from nba_ou.postgre_db.players.create_nba_players_db import (
    schema_exists,
)
from psycopg import sql


def filter_players_by_team_minutes(
    df: pd.DataFrame, min_minutes_threshold: int = 180
) -> pd.DataFrame:
    """
    Filter out games where any team's total player minutes is below the threshold.

    Groups by GAME_ID and TEAM_ID, calculates sum of minutes for each team.
    If any team in a game doesn't meet the minimum threshold, removes ALL rows
    with that GAME_ID (entire game, not just the problematic team).

    Args:
        df (pd.DataFrame): DataFrame with GAME_ID, TEAM_ID, and MIN columns
        min_minutes_threshold (int): Minimum total minutes required per team (default: 180)

    Returns:
        pd.DataFrame: Filtered DataFrame with invalid games removed
    """
    initial_count = len(df)
    initial_games = df["GAME_ID"].nunique()

    # Convert MIN from "MM:SS" format to numeric minutes if needed
    if df["MIN"].dtype == object:

        def parse_minutes(min_str):
            if pd.isna(min_str) or min_str == "" or min_str is None:
                return 0
            try:
                if ":" in str(min_str):
                    parts = str(min_str).split(":")
                    return int(parts[0]) + int(parts[1]) / 60
                else:
                    return float(min_str)
            except Exception:
                return 0

        df["MIN_NUMERIC"] = df["MIN"].apply(parse_minutes)
    else:
        df["MIN_NUMERIC"] = pd.to_numeric(df["MIN"], errors="coerce").fillna(0)

    # Group by GAME_ID and TEAM_ID, sum the minutes
    team_minutes = df.groupby(["GAME_ID", "TEAM_ID"])["MIN_NUMERIC"].sum().reset_index()
    team_minutes.columns = ["GAME_ID", "TEAM_ID", "TOTAL_MINUTES"]

    # Find teams with insufficient minutes
    insufficient_teams = team_minutes[
        team_minutes["TOTAL_MINUTES"] < min_minutes_threshold
    ]

    # Get all GAME_IDs that have at least one team with insufficient minutes
    invalid_game_ids = insufficient_teams["GAME_ID"].unique()

    if len(invalid_game_ids) > 0:
        print(f"\nFound {len(invalid_game_ids)} games with insufficient team minutes:")
        print(f"  Games with issues: {len(invalid_game_ids)}")
        print(f"  Teams below threshold: {len(insufficient_teams)}")

        # Show some examples
        if len(insufficient_teams) > 0:
            print("\nExample insufficient teams:")
            print(insufficient_teams.head(10).to_string(index=False))

        # Remove all rows with those game IDs
        df = df[~df["GAME_ID"].isin(invalid_game_ids)].copy()

        final_games = df["GAME_ID"].nunique()
        filtered_count = initial_count - len(df)
        filtered_games = initial_games - final_games

        print(f"\nFiltered out {filtered_count} rows from {initial_count} total rows")
        print(f"Removed {filtered_games} games from {initial_games} total games")
    else:
        print(f"\nAll teams meet minimum {min_minutes_threshold} minutes threshold")

    # Drop the temporary numeric column
    if "MIN_NUMERIC" in df.columns:
        df = df.drop(columns=["MIN_NUMERIC"])

    return df


def upload_players_data_to_db(
    df: pd.DataFrame,
    conn: psycopg.Connection | None = None,
    exclude_game_ids: list = None,
):
    """Load players dataframe into schema SCHEMA_NAME_PLAYERS table nba_players.

    Args:
        df (pd.DataFrame): DataFrame with player data
        conn (psycopg.Connection, optional): Database connection
        exclude_game_ids (list, optional): List of GAME_IDs to exclude from upload
    """
    close_conn = False
    schema = get_schema_name_players()
    table = schema  # convention: schema == table

    # Remove unnecessary slug columns
    columns_to_remove = ["teamSlug", "playerSlug"]
    for col in columns_to_remove:
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Removed '{col}' column")

    # Exclude specified game IDs if provided
    if exclude_game_ids is not None and len(exclude_game_ids) > 0:
        initial_count = len(df)
        df = df[~df["GAME_ID"].isin(exclude_game_ids)].copy()
        excluded_count = initial_count - len(df)
        print(f"Excluded {excluded_count} rows with {len(exclude_game_ids)} game IDs")

    # From 2026-27 the box score lists inactive players as rows; they belong in
    # nba_injuries (fetch_data.fetch_nba_data.inactive_rows), never here.
    if "COMMENT" in df.columns:
        inactive = is_inactive_comment(df["COMMENT"])
        if inactive.any():
            raise ValueError(
                f"{int(inactive.sum())} inactive-player rows reached the nba_players "
                "upload; split them out with split_inactive_rows first."
            )

    if conn is None:
        conn = connect_nba_db()
        close_conn = True


    if not schema_exists(schema):
        raise ValueError(
            f"Schema '{schema}' does not exist. Please create it first using create_players_schema()."
        )

    # Filter out games with insufficient team minutes
    df = filter_players_by_team_minutes(df)
    
    if df.empty:
        print("No valid data to upload after filtering. Exiting upload.")
        if close_conn:
            conn.close()
        return False

    with conn.cursor() as cur:
        print("Converting data types...")

        integer_cols = [
            "FGM",
            "FGA",
            "FG3M",
            "FG3A",
            "FTM",
            "FTA",
            "OREB",
            "DREB",
            "REB",
            "AST",
            "STL",
            "BLK",
            "TOV",
            "PF",
            "PTS",
        ]
        for col in integer_cols:
            if col in df.columns:
                df[col] = (
                    pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
                )
                df[col] = df[col].astype(object).where(df[col].notna(), None)

        float_cols = [
            "FG_PCT",
            "FG3_PCT",
            "FT_PCT",
            "PLUS_MINUS",
            "E_OFF_RATING",
            "OFF_RATING",
            "E_DEF_RATING",
            "DEF_RATING",
            "E_NET_RATING",
            "NET_RATING",
            "AST_PCT",
            "AST_TOV",
            "AST_RATIO",
            "OREB_PCT",
            "DREB_PCT",
            "REB_PCT",
            "TM_TOV_PCT",
            "EFG_PCT",
            "TS_PCT",
            "USG_PCT",
            "E_USG_PCT",
            "E_PACE",
            "PACE",
            "PACE_PER40",
            "POSS",
            "PIE",
        ]
        for col in float_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert pandas NA/NaT to None
        df = df.where(pd.notna(df), None)

        print(f"Loading {len(df)} rows into database...")

        columns = [col for col in df.columns if col not in columns_to_remove]
        column_names = ", ".join([col.lower() for col in columns])
        placeholders = ", ".join(["%s"] * len(columns))

        insert_query = sql.SQL(
            """
            INSERT INTO {}.{} ({})
            VALUES ({})
            ON CONFLICT (game_id, team_id, player_id, season_year) DO NOTHING
            """
        ).format(
            sql.Identifier(schema),
            sql.Identifier(table),
            sql.SQL(column_names),
            sql.SQL(placeholders),
        )

        batch_size = 1000
        total_inserted = 0
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            values = [tuple(row) for row in batch[columns].values]
            cur.executemany(insert_query, values)
            conn.commit()
            total_inserted += len(batch)
            print(f"Inserted {total_inserted}/{len(df)} rows...")

        print(f"\nSuccessfully loaded {total_inserted} rows into the database!")

        cur.execute(
            sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                sql.Identifier(schema), sql.Identifier(table)
            )
        )
        count = cur.fetchone()[0]
        print(f"Total rows in {schema}.{table}: {count}")

    if close_conn:
        conn.close()
    return True
