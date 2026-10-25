"""Aiven schema of the Covers line-history store.

The tables are the SBR store's (``line_history_aiven.schema.DDL_STATEMENTS``,
used verbatim), so the stores are read the same way and a fallback can swap
one for another per game. Two differences, both in content:

* ``lh_book.book_id`` is the Covers sportsbook id (``books.COVERS_BOOKS``),
  plus a synthetic id for the rebuilt ``consensus_opener``.
* ``lh_game.event_id`` is the Covers game id, and ``tipoff_utc`` the NBA
  schedule feed's (Covers' finished games carry no tip time).

One table is Covers' own (``COVERS_DDL_STATEMENTS``): ``lh_consensus``, each
stored game's Covers Consensus pick counts, the counts SBR's
``*_consensus_pct_*`` shares are taken from.

The SBR modules are not parameterised for another schema on purpose: they are
in the code closure that keys the training-data caches, and any edit there
invalidates them. This module reuses their constants and repeats their few
statements instead, as ``line_history_oddslogic`` does.
"""

from __future__ import annotations

import psycopg
from psycopg import sql

from nba_ou.fetch_data.odds_covers.books import COVERS_BOOKS
from nba_ou.postgre_db.line_history_aiven.schema import (
    DDL_STATEMENTS,
    MARKETS,
    partition_name,
)

SCHEMA = "line_history_covers"

#: Tables of this store only, created after the SBR store's.
COVERS_DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS {schema}.lh_consensus (
        game_id           TEXT PRIMARY KEY REFERENCES {schema}.lh_game (game_id),
        season_year       SMALLINT    NOT NULL,
        spread_away_picks INTEGER     NOT NULL,
        spread_home_picks INTEGER     NOT NULL,
        total_over_picks  INTEGER     NOT NULL,
        total_under_picks INTEGER     NOT NULL,
        fetched_at        TIMESTAMPTZ NOT NULL
    )
    """,
)


def create_schema(conn: psycopg.Connection) -> None:
    """Create the schema and its tables, and register markets and books."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(SCHEMA))
        )
        for statement in (*DDL_STATEMENTS, *COVERS_DDL_STATEMENTS):
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
    """Insert every ``COVERS_BOOKS`` entry under its Covers id."""
    with conn.cursor() as cur:
        for book_id, (slug, name) in sorted(COVERS_BOOKS.items()):
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
    return {slug: book_id for book_id, (slug, _) in COVERS_BOOKS.items()}


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


def consensus_table_exists(conn: psycopg.Connection) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = 'lh_consensus'",
            (SCHEMA,),
        )
        return cur.fetchone() is not None
