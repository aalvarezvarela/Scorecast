import pandas as pd
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_odds_yahoo,
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


def load_odds_yahoo_from_db(seasons=None, extra_game_ids=None) -> pd.DataFrame | None:
    """Load Yahoo odds rows from Postgres.

    Args:
        seasons (list[str], optional): Seasons such as ["2023-24", "2024-25"].
            None or an empty list leaves the season filter unset.
        extra_game_ids (list, optional): Game IDs included in addition to the
            selected seasons. With no seasons, these IDs restrict the query.

    Returns:
        pd.DataFrame | None: Matching rows in descending game-date order, or
            None if loading fails.
    """
    schema = get_schema_name_odds_yahoo()
    table = schema  # convention: schema == table

    conn = None
    try:
        conn = connect_nba_db()
        where_parts = []
        query_params = []

        if seasons is not None and len(seasons) > 0:
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
            ORDER BY game_date DESC
        """).format(
            sql.Identifier(schema),
            sql.Identifier(table),
            where_clause,
        )

        query = query_obj.as_string(conn)
        if query_params:
            df = pd.read_sql_query(query, conn, params=tuple(query_params))
        else:
            df = pd.read_sql_query(query, conn)

        print(f"Loaded {len(df)} Yahoo odds records from database")
        return df

    except Exception as e:
        print(f"Error loading Yahoo odds from database: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        if conn is not None:
            conn.close()
