#!/usr/bin/env python3
"""Validate past seasons in the S3 input cache, uploading the ones that changed.

    python scripts/refresh_input_cache.py                       # every table, every past season
    python scripts/refresh_input_cache.py --tables nba_players.nba_players --seasons 2023 2024
    python scripts/refresh_input_cache.py --force               # refetch even if unchanged
    python scripts/refresh_input_cache.py --list                # what the store holds
    python scripts/refresh_input_cache.py --check               # compare only, write nothing

Each past season's server fingerprint (row count and a hash over every
column) is compared with the stored one. An unchanged season only has its
``validated_at`` renewed; a changed or missing one is refetched and uploaded
as Parquet (``nba_ou.postgre_db.season_store``). The current season is never
stored: loaders always query it.

Run it daily after the games are final, and after any repair or backfill of a
past season (with ``--seasons``): loaders serve every stored finished season
whatever its age (``season_tables.servable``), so a repaired season is stale in
S3 until this replaces it. ``--check`` reports, per season, whether the stored
entry still matches the server, without writing anything; ``CHANGED`` lines
are the seasons a refresh would replace.

``line_history_aiven.closing_ticks`` and ``line_history_aiven.pregame_ticks`` (in
the default ``--tables``) do the same for the Aiven ticks the odds repair and the
intermediate builder read, fingerprinted with ``line_history_season_fingerprint``; the ``season_queries`` entries (box-score
history, injury-report reads) store each season's result of the loader's own
query, fingerprinted by a server-side hash over that result.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd
from nba_ou.postgre_db.config.db_config import connect_nba_db
from nba_ou.postgre_db.line_history_aiven.fetch import (
    available_seasons,
    refresh_season_ticks,
)
from nba_ou.postgre_db.season_queries import (
    refresh_season_query,
    registered_season_queries,
)
from nba_ou.postgre_db.season_store import default_store
from nba_ou.postgre_db.season_tables import (
    SEASON_TABLES,
    current_season_year,
    refresh_season_table,
)
from psycopg import sql

CLOSING_TICKS = "line_history_aiven.closing_ticks"
PREGAME_TICKS = "line_history_aiven.pregame_ticks"
TICK_TABLES = {CLOSING_TICKS: "closing_ticks", PREGAME_TICKS: "pregame_ticks"}


def _stored_seasons(conn, spec) -> list[int]:
    query = sql.SQL("SELECT DISTINCT season_year FROM {}.{}").format(
        sql.Identifier(spec.schema), sql.Identifier(spec.table)
    )
    with conn.cursor() as cur:
        cur.execute(query)
        return sorted(int(row[0]) for row in cur.fetchall() if row[0] is not None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    queries = registered_season_queries()
    parser.add_argument(
        "--tables",
        nargs="*",
        default=sorted(SEASON_TABLES) + sorted(queries) + sorted(TICK_TABLES),
    )
    parser.add_argument("--seasons", nargs="*", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true", help="show the stored seasons")
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the stored seasons with the server; write nothing",
    )
    args = parser.parse_args()

    unknown = sorted(
        set(args.tables) - set(SEASON_TABLES) - set(queries) - set(TICK_TABLES)
    )
    if unknown:
        parser.error(f"unknown tables {unknown}; known: {sorted(SEASON_TABLES)}")
    store = default_store()
    if store is None:
        print("S3 is not configured (S3.BUCKET); nothing to do.")
        return 1
    current = current_season_year()
    print(
        f"s3://{store.bucket}/{store.prefix}/  (current season {current}, never stored)"
    )

    for table in [t for t in args.tables if t in TICK_TABLES]:
        seasons = args.seasons or [
            s for s in available_seasons(covers_fallback=False) if s < current
        ]
        if args.list:
            print(f"{table}: seasons {seasons} (see the S3 prefix for entries)")
            continue
        started = time.perf_counter()
        outcome = refresh_season_ticks(
            store,
            seasons,
            force=args.force,
            name=TICK_TABLES[table],
            check_only=args.check,
        )
        for season, result in outcome.items():
            print(f"{table:42s} {season}: {result}")
        print(f"{table:42s} done in {time.perf_counter() - started:.1f} s")

    for name in [t for t in args.tables if t in queries]:
        spec, seasons_sql = queries[name]
        with connect_nba_db(spec.database) as conn:
            with conn.cursor() as cur:
                cur.execute(seasons_sql)
                listed = sorted(int(r[0]) for r in cur.fetchall() if r[0] is not None)
            seasons = args.seasons or [s for s in listed if s < current]
            if args.list:
                print(f"{name:42s} seasons {seasons}")
                continue
            started = time.perf_counter()
            outcome = refresh_season_query(
                spec,
                seasons,
                store=store,
                conn=conn,
                force=args.force,
                check_only=args.check,
            )
        for season, result in outcome.items():
            print(f"{name:42s} {season}: {result}")
        print(f"{name:42s} done in {time.perf_counter() - started:.1f} s")

    with connect_nba_db() as conn:
        for name in [t for t in args.tables if t in SEASON_TABLES]:
            spec = SEASON_TABLES[name]
            seasons = args.seasons or [
                s for s in _stored_seasons(conn, spec) if s < current
            ]
            if args.list:
                key = spec.code_key()
                for season in seasons:
                    entry = store.entry(spec.dataset, key, season)
                    print(
                        f"{name:42s} {season}: "
                        + (
                            f"{entry.rows:,} rows, validated {entry.validated_at:%Y-%m-%d %H:%M} UTC"
                            if entry
                            else "not stored"
                        )
                    )
                continue
            started = time.perf_counter()
            outcome = refresh_season_table(
                spec,
                seasons,
                store=store,
                conn=conn,
                force=args.force,
                check_only=args.check,
            )
            for season, result in outcome.items():
                print(f"{name:42s} {season}: {result}")
            print(f"{name:42s} done in {time.perf_counter() - started:.1f} s")
    print(f"finished {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
