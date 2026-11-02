import numpy as np
import pandas as pd
import psycopg
from nba_ou.data_processing.team.cleaning_teams import fix_home_away_parsing_errors
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_games,
)
from nba_ou.postgre_db.games.creation.create_nba_games_db import (
    schema_exists,
)
from psycopg import sql


def filter_before_upload(df: pd.DataFrame, return_invalid: bool = False) -> pd.DataFrame:
    """
    Filter out invalid games before uploading to database.

    Removes games where:
    - WL is None, empty, or NaN
    - PTS is None or less than 50
    - MIN is None or less than 120

    Args:
        df (pd.DataFrame): DataFrame to filter

    Returns:
        pd.DataFrame: Filtered DataFrame with invalid games removed
    """
    initial_count = len(df)

    # Find invalid game IDs
    invalid_condition = (
        (df["WL"].isna())
        | (df["WL"] == "")
        | (df["WL"].isnull())
        | (df["PTS"].isna())
        | (df["PTS"] < 50)
        | (df["MIN"].isna())
        | (df["MIN"] < 120)
    )

    invalid_game_ids = df.loc[invalid_condition, "GAME_ID"].unique()

    if len(invalid_game_ids) > 0:
        print(f"Found {len(invalid_game_ids)} invalid games to filter out")
        print(
            f"Invalid GAME_IDs: {invalid_game_ids[:10]}..."
            if len(invalid_game_ids) > 10
            else f"Invalid GAME_IDs: {invalid_game_ids}"
        )

        # Remove all rows with those game IDs
        df = df[~df["GAME_ID"].isin(invalid_game_ids)].copy()

        filtered_count = initial_count - len(df)
        print(f"Filtered out {filtered_count} rows from {initial_count} total rows")
    else:
        print("No invalid games found")

    if return_invalid:
        return df, invalid_game_ids
    return df


def upload_games_data_to_db(
    df: pd.DataFrame,
    conn: psycopg.Connection | None = None,
    exclude_game_ids: list = None,
) -> bool:
    """Upload games data to database.

    Args:
        df (pd.DataFrame): DataFrame with games data
        conn (psycopg.Connection, optional): Database connection
        exclude_game_ids (list, optional): List of GAME_IDs to exclude from upload

    Returns:
        bool: True if successful
    """
    close_conn = False
    schema = get_schema_name_games()

    if "teamSlug" in df.columns:
        df = df.drop(columns=["teamSlug"])

    # Exclude specified game IDs if provided
    if exclude_game_ids is not None and len(exclude_game_ids) > 0:
        initial_count = len(df)
        df = df[~df["GAME_ID"].isin(exclude_game_ids)].copy()
        excluded_count = initial_count - len(df)
        print(f"Excluded {excluded_count} rows with {len(exclude_game_ids)} game IDs")

    if conn is None:
        conn = connect_nba_db()
        close_conn = True

    # # Check if database and schema exist
    # if not database_exists():
    #     raise ValueError(
    #         "Database does not exist. Please create it first using create_database()."
    #     )

    if not schema_exists( schema):
        raise ValueError(
            f"Schema '{schema}' does not exist. Please create it first using create_games_schema()."
        )

    # 1) Create SEASON_YEAR
    if "SEASON_ID" in df.columns:
        df["SEASON_YEAR"] = df["SEASON_ID"].astype(str).str[-4:].astype("Int64")

    # 2) Clean WL / HOME / GAME_DATE
    if "WL" in df.columns:
        df["WL"] = df["WL"].replace("", None)
        df["WL"] = df["WL"].where(df["WL"].notna(), None)

    if "HOME" in df.columns:
        df["HOME"] = df["HOME"].map(
            {"True": True, "False": False, True: True, False: False}
        )

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

    # Apply safety measure to fix home/away parsing errors
    df = fix_home_away_parsing_errors(df)

    # Filter out invalid games before upload
    df = filter_before_upload(df)

    if df.empty:
        print("No valid data to upload after filtering. Exiting upload.")
        if close_conn:
            conn.close()
        return False

    # 3) Enforce exact column set in the table
    insert_cols = [
        "SEASON_ID",
        "SEASON_YEAR",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "TEAM_NAME",
        "GAME_ID",
        "GAME_DATE",
        "MATCHUP",
        "WL",
        "MIN",
        "PTS",
        "FGM",
        "FGA",
        "FG_PCT",
        "FG3M",
        "FG3A",
        "FG3_PCT",
        "FTM",
        "FTA",
        "FT_PCT",
        "OREB",
        "DREB",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TOV",
        "PF",
        "PLUS_MINUS",
        "SEASON_TYPE",
        "HOME",
        "TEAM_CITY",
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
        "E_TM_TOV_PCT",
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

    missing = [c for c in insert_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for insert: {missing}")

    df = df[insert_cols].copy()

    # 4) Cast INTEGER columns including PLUS_MINUS (table defines plus_minus INTEGER)
    integer_cols = [
        "SEASON_YEAR",
        "MIN",
        "PTS",
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
        "PLUS_MINUS",
    ]
    for col in integer_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    float_cols = [
        "FG_PCT",
        "FG3_PCT",
        "FT_PCT",
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
        "E_TM_TOV_PCT",
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
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # TEAM_ID is TEXT in DB; enforce string to avoid ambiguous casts
    df["TEAM_ID"] = df["TEAM_ID"].astype(str)
    df["PLUS_MINUS"] = df["PLUS_MINUS"].replace({np.nan: None})
    df["PLUS_MINUS"] = df["PLUS_MINUS"].replace(pd.NA, None)
    # NaN -> None for psycopg
    df = df.where(pd.notna(df), None)

    # 5) Build a safe INSERT with identifiers for each column
    col_idents = [sql.Identifier(c.lower()) for c in insert_cols]
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in insert_cols)

    insert_query = sql.SQL("""
        INSERT INTO {}.{} ({})
        VALUES ({})
        ON CONFLICT (game_id, team_id, season_id, season_year) DO NOTHING
    """).format(
        sql.Identifier(schema),
        sql.Identifier(schema),
        sql.SQL(", ").join(col_idents),
        placeholders,
    )

    batch_size = 1000
    with conn.cursor() as cur:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            values = [tuple(r) for r in batch.to_numpy()]
            try:
                cur.executemany(insert_query, values)
                conn.commit()
            except Exception as e:
                conn.rollback()

                # Print one representative row to debug quickly
                r0 = batch.iloc[0]
                print(f"Error inserting batch starting at index {i}: {e}")
                print(
                    "Example row keys:",
                    {
                        "GAME_ID": r0.get("GAME_ID"),
                        "TEAM_ID": r0.get("TEAM_ID"),
                        "SEASON_ID": r0.get("SEASON_ID"),
                        "SEASON_YEAR": r0.get("SEASON_YEAR"),
                    },
                )
                raise

        # Verify count
        cur.execute(
            sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                sql.Identifier(schema), sql.Identifier(schema)
            )
        )
        count = cur.fetchone()[0]
        print(f"Total rows in {schema}.{schema}: {count}")

    if close_conn:
        conn.close()

    return True
