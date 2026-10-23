"""Archive OddsTrader NBA line history to S3 (raw JSON, insert-only).

Writes ``s3://<bucket>/train_data/odds_sources/oddstrader/raw/v1/<date>/``. Resumable: a
date or game already stored is not fetched again, so an interrupted run is
continued by running the same command. Nothing goes to Postgres.

Seasons run in ``PRIORITY_SEASONS`` order unless ``--seasons`` or a date range
is given. The default end is two days ago (Eastern), so every game is settled.

Usage:
    python scripts/line_history/oddstrader/backfill_oddstrader_raw.py
    python scripts/line_history/oddstrader/backfill_oddstrader_raw.py --seasons 2023 2024
    python scripts/line_history/oddstrader/backfill_oddstrader_raw.py \
        --start 2024-01-15 --end 2024-01-16
    python scripts/line_history/oddstrader/backfill_oddstrader_raw.py --dry-run

Long runs (about 12 h for every season since 2014-15): start them with
``systemd-run --user --unit oddstrader-backfill <python> <this script>``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta

from nba_ou.fetch_data.odds_oddstrader import backfill as bf
from nba_ou.fetch_data.odds_oddstrader.client import EASTERN_TZ, OddsTraderClient
from nba_ou.fetch_data.odds_oddstrader.store import events_key, make_store


def _dates(args: argparse.Namespace, last_day: date) -> list[date]:
    if args.start or args.end:
        start = args.start or date(bf.FIRST_SEASON, 10, 1)
        end = min(args.end or last_day, last_day)
        return [start + timedelta(days=i) for i in range((end - start).days + 1)]
    seasons = args.seasons or [
        s for s in bf.PRIORITY_SEASONS if date(s, 10, 1) <= last_day
    ]
    return [day for s in seasons for day in bf.season_dates(s, last_day=last_day)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", help="season start years")
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report how many dates are stored and still to fetch, then stop",
    )
    args = parser.parse_args()

    last_day = datetime.now(EASTERN_TZ).date() - timedelta(days=2)
    days = _dates(args, last_day)
    store = make_store()
    existing = store.list_raw_keys()
    todo = [d for d in days if events_key(d) not in existing]
    games = sum(1 for k in existing if not k.endswith("/events.json.gz"))
    print(
        f"{len(days)} dates requested ({days[0] if days else '-'} .. "
        f"{days[-1] if days else '-'} in run order); {len(days) - len(todo)} already "
        f"have a slate file; {games} games stored in all"
    )
    if args.dry_run:
        print(
            f"[DRY RUN] would fetch {len(todo)} slates (2 requests each) plus one "
            f"request per unstored game on every requested date"
        )
        return 0

    with OddsTraderClient() as client:
        result = bf.backfill_days(days, store=store, client=client, existing=existing)
    print(result.summary())
    for line in result.failed_dates + result.failed_events:
        print(f"  ! {line}")
    return 1 if result.failed_dates or result.failed_events else 0


if __name__ == "__main__":
    sys.exit(main())
