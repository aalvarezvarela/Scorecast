from __future__ import annotations

import psycopg
from psycopg import sql

from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_all_star_voting,
)


def schema_exists(schema_name: str | None = None) -> bool:
    """Check if the all-star voting schema exists."""
    try:
        if schema_name is None:
            schema_name = get_schema_name_all_star_voting()

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


def create_all_star_voting_schema_if_not_exists(
    conn: psycopg.Connection, schema: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
    conn.commit()


def _all_star_voting_columns() -> list[tuple[str, str]]:
    return [
        ("conference", "VARCHAR(50) NOT NULL"),
        ("position", "VARCHAR(50) NOT NULL"),
        ("season", "VARCHAR(7) NOT NULL"),
        ("season_year", "INTEGER NOT NULL"),
        ("player_name", "VARCHAR(120) NOT NULL"),
        ("player_id", "TEXT NOT NULL"),
        ("team_name", "VARCHAR(100)"),
        ("fan_votes", "INTEGER"),
        ("fan_votes_pct", "NUMERIC(13, 12)"),
        ("fan_rank", "INTEGER"),
        ("player_votes", "INTEGER"),
        ("player_rank", "INTEGER"),
        ("media_votes", "INTEGER"),
        ("media_rank", "INTEGER"),
        ("score", "NUMERIC(10, 4)"),
    ]


def _migrate_existing_all_star_voting_table(
    cur: psycopg.Cursor,
    schema: str,
    table: str,
) -> None:
    cur.execute(
        sql.SQL(
            """
            ALTER TABLE IF EXISTS {}.{}
            ALTER COLUMN fan_votes_pct TYPE NUMERIC(13, 12)
            USING fan_votes_pct::NUMERIC(13, 12)
            """
        ).format(sql.Identifier(schema), sql.Identifier(table))
    )
    for index_name in (
        "idx_all_star_voting_player_id",
        "idx_all_star_voting_player_name",
    ):
        cur.execute(
            sql.SQL("DROP INDEX IF EXISTS {}.{}").format(
                sql.Identifier(schema),
                sql.Identifier(index_name),
            )
        )


def create_all_star_voting_table(drop_existing: bool = False) -> bool:
    """Create the all-star voting table inside SCHEMA_NAME_ALL_STAR_VOTING."""
    try:
        schema = get_schema_name_all_star_voting()
        table = schema
        conn = connect_nba_db()

        create_all_star_voting_schema_if_not_exists(conn, schema)

        with conn.cursor() as cur:
            if drop_existing:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                        sql.Identifier(schema),
                        sql.Identifier(table),
                    )
                )

            column_defs = _all_star_voting_columns()
            column_sql = ",\n".join(
                [f"{name} {dtype}" for name, dtype in column_defs]
                + [
                    (
                        "PRIMARY KEY "
                        "(season_year, conference, position, player_id)"
                    )
                ]
            )

            create_table_query = sql.SQL(
                f"""
                CREATE TABLE IF NOT EXISTS {{}}.{{}} (
                    {column_sql}
                )
                """
            ).format(sql.Identifier(schema), sql.Identifier(table))
            cur.execute(create_table_query)
            _migrate_existing_all_star_voting_table(cur, schema, table)

            index_specs = [
                ("idx_all_star_voting_season_year", "season_year"),
                (
                    "idx_all_star_voting_player_id_season_year",
                    "player_id, season_year",
                ),
            ]
            for index_name, columns in index_specs:
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{}({})").format(
                        sql.Identifier(index_name),
                        sql.Identifier(schema),
                        sql.Identifier(table),
                        sql.SQL(columns),
                    )
                )

        conn.commit()
        conn.close()
        print(f"Table '{schema}.{table}' created successfully!")
        return True
    except Exception as e:
        print(f"Error creating all-star voting table: {e}")
        return False


if __name__ == "__main__":
    if not create_all_star_voting_table(drop_existing=False):
        raise SystemExit(1)
