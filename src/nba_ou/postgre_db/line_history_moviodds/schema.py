"""MoviOdds bet365 store: the unchanged SBR DDL in a separate schema."""

from __future__ import annotations

import psycopg
from psycopg import sql

from nba_ou.postgre_db.line_history_aiven.schema import (
    DDL_STATEMENTS,
    MARKETS,
    partition_name,
)

SCHEMA = "line_history_moviodds"
BOOKS = {1: ("bet365", "bet365")}


def create_schema(conn: psycopg.Connection) -> None:
    """Create the schema and its tables, and register markets and books."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(SCHEMA))
        )
        for statement in DDL_STATEMENTS:
            cur.execute(sql.SQL(statement).format(schema=sql.Identifier(SCHEMA)))
        for market_id, code in MARKETS:
            cur.execute(
                sql.SQL(
                    "INSERT INTO {}.lh_market (market_id, code) VALUES (%s, %s) "
                    "ON CONFLICT (market_id) DO NOTHING"
                ).format(sql.Identifier(SCHEMA)),
                (market_id, code),
            )
    conn.commit()
    register_books(conn)


def register_books(conn: psycopg.Connection) -> None:
    """Insert every ``BOOKS`` entry under its MoviOdds id."""
    with conn.cursor() as cur:
        for book_id, (slug, name) in sorted(BOOKS.items()):
            cur.execute(
                sql.SQL(
                    "INSERT INTO {}.lh_book (book_id, slug, name) VALUES (%s, %s, %s) "
                    "ON CONFLICT (book_id) DO NOTHING"
                ).format(sql.Identifier(SCHEMA)),
                (int(book_id), slug, name),
            )
    conn.commit()


def book_ids() -> dict[str, int]:
    """Store slug -> ``lh_book.book_id``."""
    return {slug: book_id for book_id, (slug, _) in BOOKS.items()}


def create_season_partition(conn: psycopg.Connection, season_year: int) -> None:
    # Partition bounds are DDL and cannot be parameterised.
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {}.{} "
                "PARTITION OF {}.lh_line FOR VALUES IN ({})"
            ).format(
                sql.Identifier(SCHEMA),
                sql.Identifier(partition_name(season_year)),
                sql.Identifier(SCHEMA),
                sql.Literal(int(season_year)),
            )
        )
    conn.commit()


def schema_exists(conn: psycopg.Connection) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
            (SCHEMA,),
        )
        return cur.fetchone() is not None
