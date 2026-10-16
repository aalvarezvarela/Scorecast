"""Read validated lineup stints with resolved five-player IDs."""

from __future__ import annotations

import pandas as pd
import psycopg
from psycopg import sql

from nba_ou.postgre_db.config.db_config import connect_nba_db

from .schema import SCHEMA


def fetch_game_statuses(
    *, conn: psycopg.Connection | None = None
) -> dict[str, tuple[str, str | None]]:
    """Return every stored game status, creating the schema when necessary."""
    from .schema import create_schema

    if conn is None:
        with connect_nba_db() as owned_conn:
            return fetch_game_statuses(conn=owned_conn)
    create_schema(conn)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT game_id, status, reason FROM {}.lu_game_status").format(
                sql.Identifier(SCHEMA)
            )
        )
        return {
            str(game_id): (str(status), reason)
            for game_id, status, reason in cur.fetchall()
        }


def fetch_stints(
    season_years: list[int], *, conn: psycopg.Connection | None = None
) -> pd.DataFrame:
    """Return only games with a successful validation status."""
    if not season_years:
        return pd.DataFrame()
    owned = conn is None
    conn = conn or connect_nba_db()
    try:
        query = sql.SQL("""
            SELECT s.*, g.game_date,
                   h.team_id AS home_team_id,
                   ARRAY[h.p1,h.p2,h.p3,h.p4,h.p5] AS home_lineup,
                   a.team_id AS away_team_id,
                   ARRAY[a.p1,a.p2,a.p3,a.p4,a.p5] AS away_lineup
            FROM {}.lu_stint AS s
            JOIN {}.lu_game_status AS g ON g.game_id=s.game_id AND g.status='ok'
            JOIN {}.lu_lineup AS h ON h.lineup_id=s.home_lineup_id
            JOIN {}.lu_lineup AS a ON a.lineup_id=s.away_lineup_id
            WHERE s.season_year = ANY(%s)
            ORDER BY g.game_date, s.game_id, s.seg_idx
        """).format(*(sql.Identifier(SCHEMA) for _ in range(4)))
        with conn.cursor() as cur:
            cur.execute(query, (season_years,))
            rows = cur.fetchall()
            return pd.DataFrame(
                rows, columns=[column.name for column in cur.description]
            )
    finally:
        if owned:
            conn.close()
