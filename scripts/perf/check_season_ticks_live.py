#!/usr/bin/env python3
"""Live check: closing ticks through S3 == fetched from Aiven, and how fast.

    python scripts/perf/check_season_ticks_live.py [--refresh]

``--refresh`` first runs the refresher for every finished season (writes to
S3). Then, for all seasons the store holds plus the current one:

* direct: ``fetch_closing_ticks`` without any cache (Aiven only);
* cached: the cached path with an empty local cache, so finished seasons
  come from S3 and only the rest from Aiven.

Both must be identical; the time of each is printed.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src")]

from nba_ou.postgre_db import season_tables  # noqa: E402
from nba_ou.postgre_db.line_history_aiven import fetch  # noqa: E402
from nba_ou.postgre_db.season_store import default_store, frame_differences  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    seasons = fetch.available_seasons(covers_fallback=False)
    current = season_tables.current_season_year()
    print(f"seasons {seasons}, current {current}")
    if args.refresh:
        started = time.perf_counter()
        outcome = fetch.refresh_season_ticks(
            default_store(), [s for s in seasons if s < current]
        )
        for season, result in outcome.items():
            print(f"  refresh {season}: {result}")
        print(f"refresh: {time.perf_counter() - started:.1f} s")

    started = time.perf_counter()
    direct = fetch.fetch_closing_ticks(season_years=seasons, use_cache=False)
    t_direct = time.perf_counter() - started

    os.environ["NBA_OU_INPUT_CACHE_DIR"] = tempfile.mkdtemp()
    os.environ["NBA_OU_S3_SEASONS"] = "1"
    started = time.perf_counter()
    cached = fetch.fetch_closing_ticks(season_years=seasons, use_cache=True)
    t_cached = time.perf_counter() - started

    problems = frame_differences(cached, direct)
    print(
        f"{len(direct):,} ticks; Aiven only {t_direct:.1f} s, "
        f"S3 + Aiven {t_cached:.1f} s"
    )
    print("RESULT:", "EQUAL" if not problems else f"DIFFERENT {problems[:5]}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
