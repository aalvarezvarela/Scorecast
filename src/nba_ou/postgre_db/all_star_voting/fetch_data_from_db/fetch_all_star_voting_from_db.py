from __future__ import annotations

import pandas as pd
from psycopg import sql

from nba_ou.postgre_db.config.db_config import (
    connect_all_star_voting_db,
    get_schema_name_all_star_voting,
)
from nba_ou.postgre_db.row_order import read_sql_canonical
from nba_ou.postgre_db.season_tables import read_through
from nba_ou.utils.history_cache import cached


def _normalize_optional_list(values) -> list:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]

    normalized = []
    seen = set()
    for value in values:
        if pd.isna(value):
            continue
        key = str(value).strip()
        if not key or key in seen:
            continue
        normalized.append(key)
        seen.add(key)
    return normalized


@cached()
def load_all_star_voting_from_db(
    player_ids=None,
    player_names=None,
    season_years=None,
) -> pd.DataFrame | None:
    """
    Load all-star voting rows from Postgres.

    Args:
        player_ids: optional player ID or list of player IDs.
        player_names: optional player name or list of player names.
        season_years: optional season year or list of start years, e.g. 2019.
    """
    schema = get_schema_name_all_star_voting()
    table = schema
    conn = None

    try:
        conn = connect_all_star_voting_db()
        where_parts = []
        query_params = []

        normalized_player_ids = _normalize_optional_list(player_ids)
        if normalized_player_ids:
            where_parts.append(sql.SQL("player_id = ANY(%s)"))
            query_params.append(normalized_player_ids)

        normalized_player_names = _normalize_optional_list(player_names)
        if normalized_player_names:
            where_parts.append(sql.SQL("player_name = ANY(%s)"))
            query_params.append(normalized_player_names)

        if season_years is not None:
            if isinstance(season_years, int):
                season_years = [season_years]
            normalized_season_years = [
                int(value) for value in season_years if not pd.isna(value)
            ]
            if normalized_season_years:
                where_parts.append(sql.SQL("season_year = ANY(%s)"))
                query_params.append(normalized_season_years)

        where_clause = sql.SQL("")
        if where_parts:
            where_clause = sql.SQL("WHERE ") + sql.SQL(" AND ").join(where_parts)

        query_obj = sql.SQL(
            """
            SELECT *
            FROM {}.{}
            {}
            ORDER BY season_year, conference, position, fan_rank NULLS LAST
            """
        ).format(
            sql.Identifier(schema),
            sql.Identifier(table),
            where_clause,
        )

        query = query_obj.as_string(conn)
        if query_params:
            if season_years is not None and not (
                normalized_player_ids or normalized_player_names
            ):
                # Seasons are the only filter: past seasons from the S3 input
                # cache, the rest from here.
                cached = read_through(
                    "nba_all_star_voting.nba_all_star_voting",
                    normalized_season_years,
                    None,
                    conn,
                )
                if cached is not None:
                    return cached
            return read_sql_canonical(query, conn, params=tuple(query_params))
        return read_sql_canonical(query, conn)

    except Exception as e:
        print(f"Error loading all-star voting data from database: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        if conn is not None:
            conn.close()
