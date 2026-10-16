#!/usr/bin/env python3
"""Live check: season queries through S3 == the loader's single query, and how fast.

    python scripts/perf/check_season_queries_live.py [--refresh]

``--refresh`` first stores every finished season of each registered query
(writes to S3). Then each query runs twice for every season (``None``):

* direct: the loader's own query, through ``read_sql_canonical``;
* stored: ``read_season_query`` (S3 seasons plus one query for the rest).

Both must be identical; the time of each is printed. Read-only otherwise.
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
from nba_ou.postgre_db.season_queries import (  # noqa: E402
    read_season_query,
    refresh_season_query,
    registered_season_queries,
)
from nba_ou.postgre_db.season_store import default_store, frame_differences  # noqa: E402
from nba_ou.postgre_db.season_tables import current_season_year  # noqa: E402

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    store = default_store()
    current = current_season_year()
    failures = 0
    for name, (spec, seasons_sql) in registered_season_queries().items():
        with connect_nba_db(spec.database) as conn:
            if args.refresh:
                with conn.cursor() as cur:
                    cur.execute(seasons_sql)
                    seasons = sorted(int(r[0]) for r in cur.fetchall() if r[0] is not None)
                started = time.perf_counter()
                outcome = refresh_season_query(
                    spec, [s for s in seasons if s < current], store=store, conn=conn
                )
                print(f"{name}: refresh {time.perf_counter() - started:.1f} s {outcome}")
            # The loader's query exactly: its filter with season_years=None.
            original = spec.query(
                spec.season_column.join(
                    ["(%(season_years)s::int[] IS NULL OR ", " = ANY(%(season_years)s))"]
                )
            )
            started = time.perf_counter()
            direct = read_sql_canonical(
                original, conn, params={**dict(spec.params), "season_years": None}
            )
            t_direct = time.perf_counter() - started
            started = time.perf_counter()
            stored, summary = read_season_query(spec, None, store=store, conn=conn)
            t_stored = time.perf_counter() - started
        problems = frame_differences(stored, direct)
        failures += bool(problems)
        print(
            f"{'EXACT' if not problems else 'DIFFERENT':9s} {name:40s} {len(direct):>9,} rows "
            f"direct {t_direct:5.1f} s | stored {t_stored:5.1f} s (S3 {summary['s3']}) "
            f"{problems[:3] if problems else ''}"
        )
    print("RESULT:", "EQUAL" if not failures else f"DIFFERENT ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
