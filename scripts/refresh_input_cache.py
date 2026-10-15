#!/usr/bin/env python3
"""Validate past seasons in the S3 input cache, uploading the ones that changed.

    python scripts/refresh_input_cache.py                       # every table, every past season
    python scripts/refresh_input_cache.py --tables nba_players.nba_players --seasons 2023 2024
    python scripts/refresh_input_cache.py --force               # refetch even if unchanged
    python scripts/refresh_input_cache.py --list                # what the store holds

Each past season's server fingerprint (row count and a hash over every
column) is compared with the stored one. An unchanged season only has its
``validated_at`` renewed; a changed or missing one is refetched and uploaded
as Parquet (``nba_ou.postgre_db.season_store``). The current season is never
stored: loaders always query it.

Run it daily after the games are final, and after any repair or backfill of a
past season (with ``--seasons``), so loaders never read a stale season: one not
validated within ``season_tables.MAX_VALIDATION_AGE`` is queried instead.
"""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd
from nba_ou.postgre_db.config.db_config import connect_nba_db
from nba_ou.postgre_db.season_store import default_store
from nba_ou.postgre_db.season_tables import (
    SEASON_TABLES,
    current_season_year,
    refresh_season_table,
)
from psycopg import sql


def _stored_seasons(conn, spec) -> list[int]:
    query = sql.SQL("SELECT DISTINCT season_year FROM {}.{}").format(
        sql.Identifier(spec.schema), sql.Identifier(spec.table)
    )
    with conn.cursor() as cur:
        cur.execute(query)
        return sorted(int(row[0]) for row in cur.fetchall() if row[0] is not None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tables", nargs="*", default=sorted(SEASON_TABLES))
    parser.add_argument("--seasons", nargs="*", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true", help="show the stored seasons")
    args = parser.parse_args()

    unknown = sorted(set(args.tables) - set(SEASON_TABLES))
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

    with connect_nba_db() as conn:
        for name in args.tables:
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
                spec, seasons, store=store, conn=conn, force=args.force
            )
            for season, result in outcome.items():
                print(f"{name:42s} {season}: {result}")
            print(f"{name:42s} done in {time.perf_counter() - started:.1f} s")
    print(f"finished {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
