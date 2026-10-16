#!/usr/bin/env python3
"""Live check: S3 season pieces + database == the single query, and how fast.

For each registered table, reads the same seasons (and extra game ids, where
the table takes them) twice:

* direct: the loader's single multi-season query through ``read_sql_canonical``;
* cached: ``season_tables.read_season_table`` (S3 past seasons + one query).

Both must give identical frames; the time of each is printed. Read-only on
the database and on S3.

    python scripts/perf/check_season_tables_live.py [--seasons 2022 2023 2024 2025 2026]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src")]

import warnings  # noqa: E402

from nba_ou.postgre_db.config.db_config import connect_nba_db  # noqa: E402
from nba_ou.postgre_db.row_order import read_sql_canonical  # noqa: E402
from nba_ou.postgre_db.season_store import default_store, frame_differences  # noqa: E402
from nba_ou.postgre_db.season_tables import SEASON_TABLES, read_season_table  # noqa: E402
from psycopg import sql  # noqa: E402

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="*", type=int, default=[2023, 2024, 2025, 2026])
    args = parser.parse_args()
    store = default_store()
    failures = 0
    with connect_nba_db() as conn:
        for spec in SEASON_TABLES.values():
            takes_extras = not spec.order_by or bool(spec.sort_keys)
            extras = []
            if takes_extras and spec.dataset != "nba_all_star_voting.nba_all_star_voting":
                # A few older games, as the head-to-head lookups add.
                q = sql.SQL(
                    "SELECT DISTINCT game_id FROM {}.{} WHERE season_year = 2019 "
                    "ORDER BY game_id LIMIT 5"
                ).format(sql.Identifier(spec.schema), sql.Identifier(spec.table))
                with conn.cursor() as cur:
                    cur.execute(q)
                    extras = [r[0] for r in cur.fetchall()]
            where = [sql.SQL("season_year = ANY(%s)")]
            params = [args.seasons]
            if extras:
                where.append(sql.SQL("game_id = ANY(%s)"))
                params.append(extras)
            order = sql.SQL(" ORDER BY " + spec.order_by) if spec.order_by else sql.SQL("")
            query = sql.SQL("SELECT * FROM {}.{} WHERE {}{}").format(
                sql.Identifier(spec.schema),
                sql.Identifier(spec.table),
                sql.SQL(" OR ").join(where),
                order,
            )
            started = time.perf_counter()
            direct = read_sql_canonical(query.as_string(conn), conn, params=tuple(params))
            t_direct = time.perf_counter() - started
            started = time.perf_counter()
            cached, summary = read_season_table(
                spec, args.seasons, extras, store=store, conn=conn
            )
            t_cached = time.perf_counter() - started
            problems = frame_differences(cached, direct)
            failures += bool(problems)
            print(
                f"{'EXACT' if not problems else 'DIFFERENT':9s} {spec.dataset:40s} "
                f"{len(direct):>8,} rows  supabase {t_direct:5.1f} s | "
                f"s3+db {t_cached:5.1f} s (s3 {summary['s3']}, db {summary['db']}, "
                f"{summary['extra_ids']} extra ids) {problems[:3] if problems else ''}"
            )
    print("RESULT:", "EQUAL" if not failures else f"DIFFERENT ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
