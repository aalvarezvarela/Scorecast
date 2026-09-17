"""Rewrite stored SBR closing lines from the line history.

The ``odds_sportsbook`` table was filled from SBR day pages, which show each
book's *last* number -- for 27% of games an in-play line. This applies
``nba_ou.data_processing.odds.closing_line_repair`` to the stored rows and
writes the result back, so the table itself holds bettable closes and every
consumer, not only the training loader, reads the same numbers.

Three tables are involved, all in the ``odds_sportsbook`` schema:

* ``odds_sportsbook`` -- per-book closing columns rewritten; ``closes_repaired_at``
  stamped on every game that had at least one line-history quote. Games with
  none are left unstamped, so a later run retries them once their line history
  lands.
* ``odds_sportsbook_sbr_raw`` -- the row exactly as SBR stored it, copied the
  first time a game is repaired and never touched again. ``--redo`` repairs
  from here, so reruns never compound.
* ``odds_sportsbook_close_repair`` -- one audit row per (game, market, book):
  what was done and the levels involved.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import psycopg
from psycopg import sql

from nba_ou.data_processing.odds.closing_line_repair import (
    ACTION_UNVERIFIED,
    audit_repaired_closes,
    closing_quote_columns,
    print_repair_summary,
    repair_closing_lines,
    summarize_repair,
)
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_odds_sportsbook,
)
from nba_ou.postgre_db.line_history_aiven.fetch import fetch_closing_ticks
from nba_ou.postgre_db.odds_sportsbook.create_db.create_odds_sportsbook_db import (
    create_odds_sportsbook_table,
)

RAW_TABLE = "odds_sportsbook_sbr_raw"
AUDIT_TABLE = "odds_sportsbook_close_repair"

#: Above this many games, ticks are fetched per season instead of per game id.
_GAME_ID_FETCH_LIMIT = 1500

_AUDIT_TYPES: dict[str, str] = {
    "game_id": "TEXT NOT NULL",
    "season_year": "INTEGER",
    "market": "TEXT NOT NULL",
    "book": "TEXT NOT NULL",
    "action": "TEXT NOT NULL",
    "sbr_level": "DOUBLE PRECISION",
    "history_level": "DOUBLE PRECISION",
    "final_level": "DOUBLE PRECISION",
    "history_minutes_before_tip": "DOUBLE PRECISION",
    "others_median": "DOUBLE PRECISION",
    "others_range": "DOUBLE PRECISION",
    "n_others": "DOUBLE PRECISION",
    "deviation": "DOUBLE PRECISION",
}


def current_season_year(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 8 else today.year - 1


def _native(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
        if isinstance(value, float) and np.isnan(value):
            return None
    return value


def _column_types(cur: psycopg.Cursor, schema: str, table: str) -> dict[str, str]:
    """Column name -> SQL type (with precision) for an existing table."""
    cur.execute(
        """
        SELECT a.attname, format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = %s AND c.relname = %s
          AND a.attnum > 0 AND NOT a.attisdropped
        ORDER BY a.attnum
        """,
        (schema, table),
    )
    return dict(cur.fetchall())


def ensure_repair_tables(conn: psycopg.Connection, schema: str, table: str) -> None:
    """Create the raw-backup and audit tables, and keep the backup's columns in sync."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {}.{} (LIKE {}.{} INCLUDING DEFAULTS, "
                "PRIMARY KEY (game_id))"
            ).format(
                sql.Identifier(schema),
                sql.Identifier(RAW_TABLE),
                sql.Identifier(schema),
                sql.Identifier(table),
            )
        )
        source = _column_types(cur, schema, table)
        present = set(_column_types(cur, schema, RAW_TABLE))
        for name, dtype in source.items():
            if name not in present:
                cur.execute(
                    sql.SQL("ALTER TABLE {}.{} ADD COLUMN {} {}").format(
                        sql.Identifier(schema),
                        sql.Identifier(RAW_TABLE),
                        sql.Identifier(name),
                        sql.SQL(dtype),
                    )
                )

        columns = ", ".join(f"{name} {dtype}" for name, dtype in _AUDIT_TYPES.items())
        cur.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {}.{} ("
                + columns
                + ", repaired_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
                "PRIMARY KEY (game_id, market, book))"
            ).format(sql.Identifier(schema), sql.Identifier(AUDIT_TABLE))
        )
    conn.commit()


def _read(
    conn: psycopg.Connection, query: sql.Composed, params: tuple
) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(query, params)
        columns = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=columns)


def load_rows_to_repair(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    *,
    season_years: list[int] | None,
    redo: bool,
) -> pd.DataFrame:
    """Stored rows needing repair, with raw SBR values restored when ``redo``."""
    where = [sql.SQL("TRUE")]
    params: list = []
    if season_years is not None:
        where.append(sql.SQL("season_year = ANY(%s)"))
        params.append([int(s) for s in season_years])
    with conn.cursor() as cur:
        has_stamp = "closes_repaired_at" in _column_types(cur, schema, table)
    if not redo and has_stamp:
        where.append(sql.SQL("closes_repaired_at IS NULL"))
    rows = _read(
        conn,
        sql.SQL("SELECT * FROM {}.{} WHERE {} ORDER BY game_date, game_id").format(
            sql.Identifier(schema), sql.Identifier(table), sql.SQL(" AND ").join(where)
        ),
        tuple(params),
    )
    if rows.empty or not redo:
        return rows

    with conn.cursor() as cur:
        if not _column_types(cur, schema, RAW_TABLE):
            return rows
    raw = _read(
        conn,
        sql.SQL("SELECT * FROM {}.{} WHERE game_id = ANY(%s)").format(
            sql.Identifier(schema), sql.Identifier(RAW_TABLE)
        ),
        (rows["game_id"].astype(str).tolist(),),
    )
    if raw.empty:
        return rows
    quote_columns = [c for c in closing_quote_columns(rows) if c in raw.columns]
    restored = rows.set_index("game_id")
    raw = raw.set_index("game_id")[quote_columns]
    restored.loc[raw.index, quote_columns] = raw
    return restored.reset_index()


def _numeric_quotes(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    for column in closing_quote_columns(out):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    out["game_id"] = out["game_id"].astype(str)
    return out


def _fetch_ticks(rows: pd.DataFrame) -> pd.DataFrame:
    game_ids = rows["game_id"].astype(str).unique().tolist()
    if len(game_ids) <= _GAME_ID_FETCH_LIMIT:
        return fetch_closing_ticks(game_ids=game_ids)
    seasons = sorted(int(s) for s in rows["season_year"].dropna().unique())
    return fetch_closing_ticks(season_years=seasons)


def repair_stored_closes(
    *,
    season_years: list[int] | None,
    redo: bool = False,
    dry_run: bool = False,
    conn: psycopg.Connection | None = None,
) -> dict[str, int]:
    """Repair stored closes for ``season_years`` (``None`` = every season)."""
    schema = get_schema_name_odds_sportsbook()
    table = schema
    if not dry_run and not create_odds_sportsbook_table(drop_existing=False):
        raise RuntimeError("Failed to create/validate odds_sportsbook table.")

    owned = conn is None
    conn = conn or connect_nba_db()
    try:
        if not dry_run:
            ensure_repair_tables(conn, schema, table)
        rows = load_rows_to_repair(
            conn, schema, table, season_years=season_years, redo=redo
        )
        if rows.empty:
            print("No stored games need a closing-line repair.")
            return {"games": 0, "repaired_games": 0, "cells": 0}

        rows = _numeric_quotes(rows)
        print(f"Repairing closes for {len(rows)} stored game(s)...")
        ticks = _fetch_ticks(rows)
        repaired, audit = repair_closing_lines(
            rows,
            ticks,
            game_tick_loader=lambda ids: fetch_closing_ticks(game_ids=ids, last_n=None),
        )
        audit_repaired_closes(repaired, audit)
        print_repair_summary(audit)

        verified_games = set(
            audit.loc[audit["action"].ne(ACTION_UNVERIFIED), "game_id"].astype(str)
        )
        waiting = rows["game_id"].nunique() - len(verified_games)
        summary = {
            "games": int(rows["game_id"].nunique()),
            "repaired_games": len(verified_games),
            "games_without_line_history": int(waiting),
            "cells": int(len(audit)),
        }
        if dry_run:
            with pd.option_context("display.width", 200):
                print(summarize_repair(audit))
            print(f"Dry run: would repair {len(verified_games)} game(s); nothing written.")
            return summary

        write = repaired[repaired["game_id"].isin(verified_games)]
        _write_repair(conn, schema, table, write, audit[audit["game_id"].isin(verified_games)])
        print(
            f"Repaired {len(verified_games)} game(s); "
            f"{waiting} left for a later run (no line history yet)."
        )
        return summary
    finally:
        if owned:
            conn.close()


def _write_repair(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    repaired: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    """Back up raw rows, rewrite quotes and store the audit in one transaction."""
    if repaired.empty:
        return
    game_ids = repaired["game_id"].astype(str).tolist()
    quote_columns = closing_quote_columns(repaired)

    with conn.cursor() as cur:
        table_columns = list(_column_types(cur, schema, table))
        quote_columns = [c for c in quote_columns if c in table_columns]
        column_list = sql.SQL(", ").join(map(sql.Identifier, table_columns))

        # First repair only: the raw row is kept exactly as SBR stored it.
        cur.execute(
            sql.SQL(
                "INSERT INTO {schema}.{raw} ({cols}) SELECT {cols} FROM {schema}.{tbl} "
                "WHERE game_id = ANY(%s) ON CONFLICT (game_id) DO NOTHING"
            ).format(
                schema=sql.Identifier(schema),
                raw=sql.Identifier(RAW_TABLE),
                tbl=sql.Identifier(table),
                cols=column_list,
            ),
            (game_ids,),
        )

        update = sql.SQL(
            "UPDATE {}.{} SET {}, closes_repaired_at = now() WHERE game_id = %s"
        ).format(
            sql.Identifier(schema),
            sql.Identifier(table),
            sql.SQL(", ").join(
                sql.SQL("{} = %s").format(sql.Identifier(c)) for c in quote_columns
            ),
        )
        cur.executemany(
            update,
            [
                tuple(_native(v) for v in values) + (str(game_id),)
                for game_id, values in zip(
                    repaired["game_id"],
                    repaired[quote_columns].itertuples(index=False, name=None),
                    strict=True,
                )
            ],
        )

        audit_columns = list(_AUDIT_TYPES)
        upsert = sql.SQL(
            "INSERT INTO {}.{} ({cols}, repaired_at) VALUES ({vals}, now()) "
            "ON CONFLICT (game_id, market, book) DO UPDATE SET {sets}, "
            "repaired_at = now()"
        ).format(
            sql.Identifier(schema),
            sql.Identifier(AUDIT_TABLE),
            cols=sql.SQL(", ").join(map(sql.Identifier, audit_columns)),
            vals=sql.SQL(", ").join(sql.Placeholder() * len(audit_columns)),
            sets=sql.SQL(", ").join(
                sql.SQL("{c} = EXCLUDED.{c}").format(c=sql.Identifier(c))
                for c in audit_columns
                if c not in ("game_id", "market", "book")
            ),
        )
        cur.executemany(
            upsert,
            [
                tuple(_native(v) for v in row)
                for row in audit[audit_columns]
                .astype(object)
                .itertuples(index=False, name=None)
            ],
        )
    conn.commit()
