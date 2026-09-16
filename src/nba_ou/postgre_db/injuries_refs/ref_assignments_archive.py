"""Archive of the daily official.nba.com referee assignments, with roles.

The historical ``nba_refs`` table stores crews without roles: its row order
only started to follow Crew Chief / Referee / Umpire in 2021-22, and the
primary key does not preserve order at all. The assignments page is the one
source that states roles and is published before tip, but it only ever shows
the current day. Archiving each scrape is therefore the only way to build a
pre-game, role-labelled history.

Every distinct crew seen for a game is kept with the time it was first seen,
so a late replacement is recorded as a second row rather than overwriting the
original assignment.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
import psycopg
from nba_ou.postgre_db.config.db_config import connect_nba_db, get_schema_name_refs
from psycopg import sql

ASSIGNMENTS_TABLE = "nba_ref_assignments"
_ET = ZoneInfo("America/New_York")

_COLUMNS = [
    "assignment_date",
    "game",
    "crew_chief",
    "referee",
    "umpire",
    "alternate",
    "first_seen_utc",
]


def ensure_ref_assignments_table(conn: psycopg.Connection) -> None:
    schema = get_schema_name_refs()
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.{} (
                    assignment_date DATE NOT NULL,
                    game TEXT NOT NULL,
                    crew_chief TEXT NOT NULL DEFAULT '',
                    referee TEXT NOT NULL DEFAULT '',
                    umpire TEXT NOT NULL DEFAULT '',
                    alternate TEXT,
                    first_seen_utc TIMESTAMPTZ NOT NULL,
                    PRIMARY KEY (assignment_date, game, crew_chief, referee, umpire)
                )
                """
            ).format(sql.Identifier(schema), sql.Identifier(ASSIGNMENTS_TABLE))
        )
    conn.commit()


def assignments_to_rows(
    df_assignments: pd.DataFrame, fetched_at_utc: pd.Timestamp
) -> pd.DataFrame:
    """Shape the scraped table (``Game``, ``Crew Chief``, ``Referee``, ``Umpire``, ``Alternate``)."""
    fetched_at_utc = pd.Timestamp(fetched_at_utc)
    if fetched_at_utc.tzinfo is None:
        fetched_at_utc = fetched_at_utc.tz_localize("UTC")

    def text(column: str) -> pd.Series:
        if column not in df_assignments.columns:
            return pd.Series("", index=df_assignments.index)
        return df_assignments[column].fillna("").astype(str).str.strip()

    rows = pd.DataFrame(
        {
            "assignment_date": fetched_at_utc.tz_convert(_ET).date(),
            "game": text("Game"),
            "crew_chief": text("Crew Chief"),
            "referee": text("Referee"),
            "umpire": text("Umpire"),
            "alternate": text("Alternate").replace("", None),
            "first_seen_utc": fetched_at_utc.to_pydatetime(),
        }
    )
    return rows[rows["game"] != ""][_COLUMNS]


def archive_referee_assignments(
    df_assignments: pd.DataFrame, fetched_at_utc: pd.Timestamp | None = None
) -> int:
    """Insert newly seen crews; returns the number of rows sent."""
    if df_assignments is None or df_assignments.empty:
        return 0
    rows = assignments_to_rows(
        df_assignments, fetched_at_utc or pd.Timestamp.now(tz="UTC")
    )
    if rows.empty:
        return 0

    schema = get_schema_name_refs()
    conn = connect_nba_db()
    try:
        ensure_ref_assignments_table(conn)
        insert = sql.SQL(
            "INSERT INTO {}.{} ({}) VALUES ({}) "
            "ON CONFLICT (assignment_date, game, crew_chief, referee, umpire) DO NOTHING"
        ).format(
            sql.Identifier(schema),
            sql.Identifier(ASSIGNMENTS_TABLE),
            sql.SQL(", ").join(map(sql.Identifier, _COLUMNS)),
            sql.SQL(", ").join([sql.Placeholder()] * len(_COLUMNS)),
        )
        with conn.cursor() as cur:
            cur.executemany(insert, [tuple(r) for r in rows.itertuples(index=False)])
        conn.commit()
    finally:
        conn.close()
    return len(rows)
