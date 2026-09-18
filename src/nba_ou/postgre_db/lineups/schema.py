"""Season-partitioned storage for validated five-on-five stints."""

from __future__ import annotations

import psycopg
from psycopg import sql

SCHEMA = "lineups"
COUNT_FIELDS = ("fga", "fg3a", "fta", "oreb", "dreb", "tov")


def create_schema(conn: psycopg.Connection) -> None:
    """Create only the tables; the caller chooses and verifies the DB first."""
    with conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(SCHEMA)))
        cur.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {}.lu_lineup (
                lineup_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                team_id BIGINT NOT NULL,
                p1 BIGINT NOT NULL, p2 BIGINT NOT NULL, p3 BIGINT NOT NULL,
                p4 BIGINT NOT NULL, p5 BIGINT NOT NULL,
                UNIQUE (team_id, p1, p2, p3, p4, p5)
            )
        """).format(sql.Identifier(SCHEMA)))
        cur.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {}.lu_stint (
                game_id TEXT NOT NULL,
                seg_idx SMALLINT NOT NULL,
                season_year SMALLINT NOT NULL,
                period SMALLINT NOT NULL,
                start_ds INTEGER NOT NULL,
                end_ds INTEGER NOT NULL,
                home_lineup_id INTEGER NOT NULL REFERENCES {}.lu_lineup,
                away_lineup_id INTEGER NOT NULL REFERENCES {}.lu_lineup,
                home_pts SMALLINT NOT NULL, away_pts SMALLINT NOT NULL,
                home_fga SMALLINT NOT NULL, home_fg3a SMALLINT NOT NULL,
                home_fta SMALLINT NOT NULL, home_oreb SMALLINT NOT NULL,
                home_dreb SMALLINT NOT NULL, home_tov SMALLINT NOT NULL,
                away_fga SMALLINT NOT NULL, away_fg3a SMALLINT NOT NULL,
                away_fta SMALLINT NOT NULL, away_oreb SMALLINT NOT NULL,
                away_dreb SMALLINT NOT NULL, away_tov SMALLINT NOT NULL,
                PRIMARY KEY (game_id, seg_idx, season_year),
                CHECK (end_ds > start_ds)
            ) PARTITION BY LIST (season_year)
        """).format(sql.Identifier(SCHEMA), sql.Identifier(SCHEMA), sql.Identifier(SCHEMA)))
        cur.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {}.lu_game_status (
                game_id TEXT PRIMARY KEY,
                season_year SMALLINT NOT NULL,
                game_date DATE NOT NULL,
                tipoff_utc TIMESTAMPTZ,
                status TEXT NOT NULL CHECK (status IN ('ok', 'failed')),
                reason TEXT,
                built_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """).format(sql.Identifier(SCHEMA)))


def ensure_partition(conn: psycopg.Connection, season_year: int) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            CREATE TABLE IF NOT EXISTS {}.{} PARTITION OF {}.lu_stint
            FOR VALUES IN ({})
        """).format(
            sql.Identifier(SCHEMA),
            sql.Identifier(f"lu_stint_{season_year}"),
            sql.Identifier(SCHEMA),
            sql.Literal(season_year),
        ))
