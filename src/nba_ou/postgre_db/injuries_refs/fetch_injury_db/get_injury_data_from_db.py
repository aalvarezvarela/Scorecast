import pandas as pd
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_injuries,
)
from psycopg import sql


def _normalize_game_ids(game_ids) -> list[str]:
    """Normalize GAME_ID inputs into a unique, non-empty string list."""
    if game_ids is None:
        return []

    normalized = []
    seen = set()
    for game_id in game_ids:
        if pd.isna(game_id):
            continue
        game_id_str = str(game_id).strip()
        if not game_id_str or game_id_str in seen:
            continue
        seen.add(game_id_str)
        normalized.append(game_id_str)
    return normalized


def get_injury_data_from_db(seasons=None, extra_game_ids=None):
    """Load injury data from database for the specified seasons.

    Args:
        seasons (list): List of seasons to load (e.g., ["2023-24", "2022-23"])
        extra_game_ids (list, optional): Game IDs included in addition to the
            selected seasons. With None or empty seasons, these IDs restrict the
            query; without either filter, all rows are loaded.

    Returns:
        pd.DataFrame: Combined injury data for all seasons
    """

    schema = get_schema_name_injuries()
    table = schema

    conn = None
    try:
        conn = connect_nba_db()

        where_parts = []
        query_params = []

        if seasons is not None and len(seasons) > 0:
            # Convert season format from "2023-24" to 2023 (year only)
            season_years = [int(s.split("-")[0]) for s in seasons]
            where_parts.append(sql.SQL("season_year = ANY(%s)"))
            query_params.append(season_years)

        normalized_extra_game_ids = _normalize_game_ids(extra_game_ids)
        if normalized_extra_game_ids:
            where_parts.append(sql.SQL("game_id = ANY(%s)"))
            query_params.append(normalized_extra_game_ids)

        where_clause = sql.SQL("")
        if where_parts:
            where_clause = sql.SQL("WHERE ") + sql.SQL(" OR ").join(where_parts)

        query_obj = sql.SQL("""
            SELECT *
            FROM {}.{}
            {}
        """).format(
            sql.Identifier(schema),
            sql.Identifier(table),
            where_clause,
        )

        query = query_obj.as_string(conn)
        if query_params:
            df_injuries = pd.read_sql_query(query, conn, params=tuple(query_params))
        else:
            df_injuries = pd.read_sql_query(query, conn)

        # Convert column names to uppercase to match expected format
        df_injuries.columns = df_injuries.columns.str.upper()

        # Ensure TEAM_ID and PLAYER_ID are strings
        if "TEAM_ID" in df_injuries.columns:
            df_injuries["TEAM_ID"] = df_injuries["TEAM_ID"].astype(str)
        if "PLAYER_ID" in df_injuries.columns:
            df_injuries["PLAYER_ID"] = df_injuries["PLAYER_ID"].astype(str)

        # Remove duplicates
        df_injuries = df_injuries.drop_duplicates()

        print(f"Loaded {len(df_injuries)} injury records from database")
        return df_injuries

    except Exception as e:
        print(f"Error loading injury data from database: {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()

    finally:
        if conn is not None:
            conn.close()
