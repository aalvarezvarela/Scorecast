import pandas as pd
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_injuries,
    get_schema_name_refs,
)
from nba_ou.postgre_db.games.fetch_data_from_db.fetch_data_from_games_db import (
    load_games_from_db,
)
from nba_ou.postgre_db.players.fetch_players_data_from_db import (
    load_players_from_db,
)
from psycopg import sql


def load_injuries_from_db(seasons=None) -> pd.DataFrame | None:
    """
    Load NBA injuries data from Postgres.

    Args:
        seasons: list like ['2023-24', '2024-25'] or None for all seasons.

    Returns:
        DataFrame or None if failure.
    """
    schema = get_schema_name_injuries()
    table = "nba_injuries"

    conn = None
    try:
        conn = connect_nba_db()

        if seasons is not None:
            season_years = [int(s.split("-")[0]) for s in seasons]

            query_obj = sql.SQL("""
                SELECT *
                FROM {}.{}
                WHERE season_year = ANY(%s)
                ORDER BY game_date DESC
            """).format(sql.Identifier(schema), sql.Identifier(table))

            query = query_obj.as_string(conn)
            df = pd.read_sql_query(query, conn, params=(season_years,))
        else:
            query_obj = sql.SQL("""
                SELECT *
                FROM {}.{}
                ORDER BY game_date DESC
            """).format(sql.Identifier(schema), sql.Identifier(table))

            query = query_obj.as_string(conn)
            df = pd.read_sql_query(query, conn)

        print(f"Loaded {len(df)} injury records from database")
        return df

    except Exception as e:
        print(f"Error loading injuries from database: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        if conn is not None:
            conn.close()


def load_refs_from_db(seasons=None) -> pd.DataFrame | None:
    """
    Load NBA referees data from Postgres.

    Args:
        seasons: list like ['2023-24', '2024-25'] or None for all seasons.

    Returns:
        DataFrame or None if failure.
    """
    schema = get_schema_name_refs()
    table = "nba_refs"

    conn = None
    try:
        conn = connect_nba_db()

        if seasons is not None:
            season_years = [int(s.split("-")[0]) for s in seasons]

            query_obj = sql.SQL("""
                SELECT *
                FROM {}.{}
                WHERE season_year = ANY(%s)
                ORDER BY game_date DESC
            """).format(sql.Identifier(schema), sql.Identifier(table))

            query = query_obj.as_string(conn)
            df = pd.read_sql_query(query, conn, params=(season_years,))
        else:
            query_obj = sql.SQL("""
                SELECT *
                FROM {}.{}
                ORDER BY game_date DESC
            """).format(sql.Identifier(schema), sql.Identifier(table))

            query = query_obj.as_string(conn)
            df = pd.read_sql_query(query, conn)

        print(f"Loaded {len(df)} referee records from database")
        return df

    except Exception as e:
        print(f"Error loading refs from database: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        if conn is not None:
            conn.close()


def load_all_nba_data_from_db(seasons=None, extra_game_ids=None):
    """Load games and players from Postgres with uppercase column names.

    Args:
        seasons (list[str], optional): Seasons such as ["2023-24", "2024-25"].
            None or an empty list leaves the season filter unset.
        extra_game_ids (list, optional): Game IDs included in addition to the
            selected seasons. With no seasons, these IDs restrict the query.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Game rows and player rows.

    Raises:
        ValueError: If either database loader returns None.
    """
    print("Loading NBA data from PostgreSQL...")

    df_games = load_games_from_db(seasons=seasons, extra_game_ids=extra_game_ids)
    df_players = load_players_from_db(seasons=seasons, extra_game_ids=extra_game_ids)

    if df_games is None or df_players is None:
        print("Failed to load data from databases")
        raise ValueError("Failed to load data from databases")

    # Match your previous pipeline expectations (uppercase columns)
    df_games.columns = df_games.columns.str.upper()
    df_players.columns = df_players.columns.str.upper()

    print(
        f"Successfully loaded {len(df_games)} games and {len(df_players)} player records"
    )
    return df_games, df_players
