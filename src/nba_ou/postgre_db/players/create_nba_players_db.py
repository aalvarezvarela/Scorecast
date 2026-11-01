import psycopg
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    connect_postgres_db,
    get_db_credentials,
    get_schema_name_players,
)
from psycopg import sql


def create_database():
    """Create the single PostgreSQL database (DB_NAME) if it doesn't exist."""
    try:
        db_name = get_db_credentials()["dbname"]

        conn = connect_postgres_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_name,)
            )
            exists = cur.fetchone()

            if not exists:
                cur.execute(
                    sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
                )
                print(f"Database '{db_name}' created successfully!")
            else:
                print(f"Database '{db_name}' already exists.")

        conn.close()
        return True
    except Exception as e:
        print(f"Error creating database: {e}")
        return False


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


def schema_exists( schema_name: str = None, ) -> bool:
    """Check if the schema exists in the database."""
    try:
        if schema_name is None:
            schema_name = get_schema_name_players()

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
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
    conn.commit()


def create_players_table(drop_existing: bool = True):
    """Create the nba_players table inside schema SCHEMA_NAME_PLAYERS."""
    try:
        schema = get_schema_name_players()
        conn = connect_nba_db()

        create_schema_if_not_exists(conn, schema)

        with conn.cursor() as cur:
            if drop_existing:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                        sql.Identifier(schema),
                        sql.Identifier(schema),
                    )
                )

            create_table_query = sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.{} (
                    game_id TEXT NOT NULL,
                    season_year INTEGER NOT NULL,
                    team_id TEXT NOT NULL,
                    team_abbreviation VARCHAR(100),
                    team_city VARCHAR(100),
                    team_name VARCHAR(100),
                    player_id TEXT NOT NULL,
                    player_name VARCHAR(100),
                    nickname VARCHAR(100),
                    firstname VARCHAR(100),
                    familyname VARCHAR(100),
                    start_position VARCHAR(100),
                    comment TEXT,
                    jerseynum VARCHAR(10),
                    min VARCHAR(20),
                    fgm INTEGER,
                    fga INTEGER,
                    fg_pct NUMERIC(5, 3),
                    fg3m INTEGER,
                    fg3a INTEGER,
                    fg3_pct NUMERIC(5, 3),
                    ftm INTEGER,
                    fta INTEGER,
                    ft_pct NUMERIC(5, 3),
                    oreb INTEGER,
                    dreb INTEGER,
                    reb INTEGER,
                    ast INTEGER,
                    stl INTEGER,
                    blk INTEGER,
                    tov INTEGER,
                    pf INTEGER,
                    pts INTEGER,
                    plus_minus NUMERIC(8, 3),
                    e_off_rating NUMERIC(8, 3),
                    off_rating NUMERIC(8, 3),
                    e_def_rating NUMERIC(8, 3),
                    def_rating NUMERIC(8, 3),
                    e_net_rating NUMERIC(8, 3),
                    net_rating NUMERIC(8, 3),
                    ast_pct NUMERIC(8, 3),
                    ast_tov NUMERIC(8, 3),
                    ast_ratio NUMERIC(8, 3),
                    oreb_pct NUMERIC(8, 3),
                    dreb_pct NUMERIC(8, 3),
                    reb_pct NUMERIC(8, 3),
                    tm_tov_pct NUMERIC(8, 3),
                    efg_pct NUMERIC(8, 3),
                    ts_pct NUMERIC(8, 3),
                    usg_pct NUMERIC(8, 3),
                    e_usg_pct NUMERIC(8, 3),
                    e_pace NUMERIC(8, 3),
                    pace NUMERIC(8, 3),
                    pace_per40 NUMERIC(8, 3),
                    poss NUMERIC(8, 3),
                    pie NUMERIC(8, 3),
                    PRIMARY KEY (game_id, team_id, player_id, season_year)
                )
                """
            ).format(sql.Identifier(schema), sql.Identifier(schema))

            cur.execute(create_table_query)

            # Indexes
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(game_id)").format(
                    sql.Identifier("idx_player_game_id"),
                    sql.Identifier(schema),
                    sql.Identifier(schema),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(team_id)").format(
                    sql.Identifier("idx_player_team_id"),
                    sql.Identifier(schema),
                    sql.Identifier(schema),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}(player_id)").format(
                    sql.Identifier("idx_player_id"),
                    sql.Identifier(schema),
                    sql.Identifier(schema),
                )
            )

        conn.commit()
        conn.close()
        print(f"Table '{schema}.{schema}' created successfully!")
        return True

    except Exception as e:
        print(f"Error creating table: {e}")
        return False



if __name__ == "__main__":
    print("Step 1: Creating database...")
    if not create_database():
        exit(1)

    print("\nStep 2: Creating table...")
    if not create_players_table():
        exit(1)

    print("\nStep 3: Loading combined data...")
    from combine_games_data import combine_all_nba_games
    from nba_ou.postgre_db.players.upload_players_data_to_db import (
        upload_players_data_to_db,
    )

    data_dir = (
        "/home/adrian_alvarez/Projects/NBA_over_under_predictor/data/season_games_data"
    )
    df_all_players = combine_all_nba_games(data_dir, file_prefix="nba_players")

    if df_all_players is not None:
        print("\nStep 4: Inserting data into PostgreSQL...")
        upload_players_data_to_db(df_all_players)
    else:
        print("Failed to load combined dataframe!")
        exit(1)

    print("\n✅ Database setup complete!")
