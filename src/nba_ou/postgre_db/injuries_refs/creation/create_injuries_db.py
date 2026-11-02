from pathlib import Path

import pandas as pd
import psycopg
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    connect_postgres_db,
    get_db_credentials,
    get_schema_name_injuries,
)
from psycopg import sql


def database_exists() -> bool:
    """Check if the database exists."""
    try:
        db_name = get_db_credentials()["dbname"]
        conn = connect_postgres_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,)
            )
            exists = cur.fetchone()
        conn.close()
        return exists is not None
    except Exception as e:
        print(f"Error checking database existence: {e}")
        return False


def schema_exists(schema_name: str = None) -> bool:
    """Check if the schema exists in the database."""
    try:
        if schema_name is None:
            schema_name = get_schema_name_injuries()

        conn = connect_nba_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
                (schema_name,),
            )
            exists = cur.fetchone()
        conn.close()
        return exists is not None
    except Exception as e:
        print(f"Error checking schema existence: {e}")
        return False


def create_schema_if_not_exists(conn: psycopg.Connection, schema: str) -> None:
    """Create schema if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
    conn.commit()


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

    # Convert to datetime if needed
    if hasattr(date, "month"):
        month = date.month
        year = date.year
    else:
        dt = pd.to_datetime(date)
        month = dt.month
        year = dt.year

    # January to July → season_year = year - 1
    # August to December → season_year = year
    return year - 1 if month in [1, 2, 3, 4, 5, 6, 7] else year


def create_injuries_schema(drop_existing: bool = True):
    """Create the nba_injuries table in schema SCHEMA_NAME_INJURIES."""
    try:
        schema = get_schema_name_injuries()
        conn = connect_nba_db()

        create_schema_if_not_exists(conn, schema)

        with conn.cursor() as cur:
            if drop_existing:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                        sql.Identifier(schema),
                        sql.Identifier("nba_injuries"),
                    )
                )

            create_table_query = sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.{} (
                    player_id TEXT NOT NULL,
                    first_name VARCHAR(100),
                    last_name VARCHAR(100),
                    jersey_num VARCHAR(10),
                    team_id TEXT NOT NULL,
                    team_city VARCHAR(100),
                    team_name VARCHAR(100),
                    team_abbreviation VARCHAR(10),
                    game_id TEXT NOT NULL,
                    season_id TEXT NOT NULL,
                    season_year INTEGER NOT NULL,
                    game_date DATE NOT NULL,
                    PRIMARY KEY (player_id, game_id),
                    CHECK (season_year >= 2000 AND season_year <= 2100)
                )
            """
            ).format(sql.Identifier(schema), sql.Identifier("nba_injuries"))

            cur.execute(create_table_query)

            # Create indexes for common query patterns
            cur.execute(
                sql.SQL(
                    """
                CREATE INDEX IF NOT EXISTS idx_injuries_season_year 
                ON {}.{} (season_year)
            """
                ).format(sql.Identifier(schema), sql.Identifier("nba_injuries"))
            )

            cur.execute(
                sql.SQL(
                    """
                CREATE INDEX IF NOT EXISTS idx_injuries_game_date 
                ON {}.{} (game_date)
            """
                ).format(sql.Identifier(schema), sql.Identifier("nba_injuries"))
            )

            cur.execute(
                sql.SQL(
                    """
                CREATE INDEX IF NOT EXISTS idx_injuries_player_id 
                ON {}.{} (player_id)
            """
                ).format(sql.Identifier(schema), sql.Identifier("nba_injuries"))
            )

            cur.execute(
                sql.SQL(
                    """
                CREATE INDEX IF NOT EXISTS idx_injuries_game_id 
                ON {}.{} (game_id)
            """
                ).format(sql.Identifier(schema), sql.Identifier("nba_injuries"))
            )

        conn.commit()
        conn.close()

        print(f"Schema '{schema}' and table 'nba_injuries' created successfully!")
        return True

    except Exception as e:
        print(f"Error creating injuries schema: {e}")
        import traceback

        traceback.print_exc()
        return False


def upload_injuries_data_to_db(df: pd.DataFrame, if_exists: str = "append") -> bool:
    """
    Load injuries data from DataFrame to PostgreSQL database.

    Args:
        df: DataFrame with columns PLAYER_ID, FIRST_NAME, LAST_NAME, JERSEY_NUM,
            TEAM_ID, TEAM_CITY, TEAM_NAME, TEAM_ABBREVIATION, GAME_ID,
            SEASON_ID, GAME_DATE
        if_exists: 'append' or 'replace' - how to behave if table exists

    Returns:
        True if successful, False otherwise
    """
    if df is None or df.empty:
        print("No data to load")
        return False

    try:
        schema = get_schema_name_injuries()
        conn = connect_nba_db()

        # Make a copy to avoid modifying original
        df_load = df.copy()

        # Normalize column names to lowercase for database
        df_load.columns = df_load.columns.str.lower()

        # Ensure GAME_DATE is datetime (handle both date-only and datetime formats)
        if "game_date" in df_load.columns:
            df_load["game_date"] = pd.to_datetime(
                df_load["game_date"], format="mixed", errors="coerce"
            ).dt.date

        # Calculate season_year from game_date
        df_load["season_year"] = df_load["game_date"].apply(calculate_season_year)

        # Remove any rows where season_year couldn't be calculated
        before_count = len(df_load)
        df_load = df_load.dropna(subset=["season_year"])
        after_count = len(df_load)

        if before_count != after_count:
            print(f"Dropped {before_count - after_count} rows with invalid dates")

        # Convert season_year to int
        df_load["season_year"] = df_load["season_year"].astype(int)

        # Select only the columns we need in the correct order
        required_columns = [
            "player_id",
            "first_name",
            "last_name",
            "jersey_num",
            "team_id",
            "team_city",
            "team_name",
            "team_abbreviation",
            "game_id",
            "season_id",
            "season_year",
            "game_date",
        ]

        df_load = df_load[required_columns]

        # Load to database
        with conn.cursor() as cur:
            if if_exists == "replace":
                # Clear table first
                cur.execute(
                    sql.SQL("DELETE FROM {}.{}").format(
                        sql.Identifier(schema), sql.Identifier("nba_injuries")
                    )
                )

            # Use INSERT with ON CONFLICT DO NOTHING to skip duplicates
            columns = ", ".join(required_columns)
            placeholders = ", ".join(["%s"] * len(required_columns))

            insert_sql = sql.SQL(
                "INSERT INTO {}.{} ({}) VALUES ({}) ON CONFLICT (player_id, game_id) DO NOTHING"
            ).format(
                sql.Identifier(schema),
                sql.Identifier("nba_injuries"),
                sql.SQL(columns),
                sql.SQL(placeholders),
            )

            # Execute batch insert
            data_tuples = [tuple(row) for row in df_load.values]
            cur.executemany(insert_sql.as_string(conn), data_tuples)

        conn.commit()
        conn.close()

        print(f"Successfully loaded {len(df_load)} injury records to database")
        return True

    except Exception as e:
        print(f"Error loading injuries to database: {e}")
        import traceback

        traceback.print_exc()
        return False


def load_all_injuries_from_csv_folder(
    folder_path: str = None, if_exists: str = "replace"
) -> bool:
    """
    Load all injury CSV files from a folder into the database.

    Args:
        folder_path: Path to folder containing injury CSV files.
                    If None, uses default data/injury_data folder.
        if_exists: 'append' or 'replace' - how to behave if table exists

    Returns:
        True if successful, False otherwise
    """
    try:
        # Use default path if not specified
        if folder_path is None:
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent.parent
            folder_path = project_root / "data" / "injury_data"
        else:
            folder_path = Path(folder_path)

        if not folder_path.exists():
            print(f"Error: Folder not found: {folder_path}")
            return False

        # Find all CSV files
        csv_files = sorted(folder_path.glob("nba_injuries_*.csv"))

        if not csv_files:
            print(f"No injury CSV files found in {folder_path}")
            return False

        print(f"Found {len(csv_files)} injury CSV files")

        # Load and combine all CSV files
        all_dfs = []
        for csv_file in csv_files:
            print(f"Reading {csv_file.name}...")
            # Load all columns as strings to avoid parsing issues
            df = pd.read_csv(csv_file, dtype=str)

            # Strip whitespace from all string columns
            for col in df.columns:
                df[col] = df[col].str.strip()

            all_dfs.append(df)

        # Combine all dataframes
        df_combined = pd.concat(all_dfs, ignore_index=True)
        print(f"\nTotal records to load: {len(df_combined)}")

        # Remove duplicates if any (based on player_id and game_id)
        before_dedup = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=["PLAYER_ID", "GAME_ID"])
        after_dedup = len(df_combined)

        if before_dedup != after_dedup:
            print(f"Removed {before_dedup - after_dedup} duplicate records")

        # Load to database
        success = upload_injuries_data_to_db(df_combined, if_exists=if_exists)

        if success:
            print("\n✓ Successfully loaded all injury data to database")
        else:
            print("\n✗ Failed to load injury data to database")

        return success

    except Exception as e:
        print(f"Error loading injuries from CSV folder: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function to create injuries database schema and load data."""
    print("=" * 60)
    print("NBA Injuries Database Setup")
    print("=" * 60)

    # Step 1: Create schema
    print("\n1. Creating injuries database schema...")
    if not create_injuries_schema(drop_existing=True):
        print("✗ Failed to create injuries schema")
        return False
    print("✓ Injuries schema created successfully")

    # Step 2: Load data from CSV files
    print("\n2. Loading injury data from CSV files...")
    if not load_all_injuries_from_csv_folder(if_exists="replace"):
        print("✗ Failed to load injury data")
        return False

    print("\n" + "=" * 60)
    print("✓ Database setup completed successfully!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()
