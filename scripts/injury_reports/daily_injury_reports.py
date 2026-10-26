#!/usr/bin/env python3
"""Archive every injury report published so far, and store the finished games.

    python scripts/injury_reports/daily_injury_reports.py            # catch up
    python scripts/injury_reports/daily_injury_reports.py --days 5   # rebuild >= 5 days
    python scripts/injury_reports/daily_injury_reports.py --dry-run  # parse, no writes

Run it after the night's games are final (the daily database job). Two steps:

1. ``backfill_injury_reports.py`` discovers and downloads every report of the
   season published up to now (today's included) into the S3 archive, with the
   manifests mirrored there too. Slots already resolved are not probed again
   and slots not yet due are left for a later run, so this is cheap to repeat
   and nothing is lost when a run is skipped.
2. ``ingest_season(write_game_dates=...)`` rebuilds the spans of the games
   played from the day after the last game the store holds to yesterday --
   never fewer than the last ``--days`` days -- replacing what it held for them.

Both steps share a local PDF cache (``--cache-dir``), so newly archived reports
are parsed from disk instead of downloaded again from S3. Older reports are
cached on their first S3 read. Ephemeral CI runners keep it only for this job.

Only finished games are written -- the store holds a game once it is played,
the rule the line history and closing odds follow. Today's reports already list
tonight's games (most spans start the day before the game), so they are
archived but not written; they are tomorrow's run's. A game's spans read every
report that lists it, which all fall in the ingested window.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd  # noqa: E402
import psycopg  # noqa: E402
from nba_ou.config.settings import SETTINGS  # noqa: E402
from nba_ou.fetch_data.injury_reports.archive import manifest as mf  # noqa: E402
from nba_ou.fetch_data.injury_reports.archive import urls as U  # noqa: E402
from nba_ou.fetch_data.injury_reports.archive.manifest import (  # noqa: E402
    ManifestStore,
)
from nba_ou.fetch_data.injury_reports.archive.storage import S3Storage  # noqa: E402
from nba_ou.postgre_db.config.db_config import connect_nba_db  # noqa: E402
from nba_ou.postgre_db.injury_report_aiven.ingest import ingest_season  # noqa: E402

ET = ZoneInfo("America/New_York")
DEFAULT_MANIFEST_ROOT = "data/injury_reports/manifest"
DEFAULT_CACHE_DIR = "data/injury_reports/cache"


def last_stored_game_date(aiven: psycopg.Connection, season_year: int) -> date | None:
    """Latest game date of ``season_year`` the store holds spans for.

    Every game a report lists gets filing spans, so a game without any was
    never written (preseason games, which the reports do not list, never are).
    """
    with aiven.cursor() as cur:
        cur.execute(
            "SELECT max(g.game_date) FROM injury_report.ir_game g "
            "WHERE g.season_year = %s AND EXISTS ("
            "  SELECT 1 FROM injury_report.ir_filing_span f "
            "  WHERE f.game_id = g.game_id AND f.season_year = %s)",
            (season_year, season_year),
        )
        return cur.fetchone()[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="rebuild at least this many game days, even if already stored",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--manifest-root", default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    # Same resolution as the download step (backfill_injury_reports.py), so
    # both steps of one run use the same S3 identity.
    parser.add_argument("--bucket", default=SETTINGS.s3_bucket)
    parser.add_argument("--profile", default=SETTINGS.s3_aws_profile)
    parser.add_argument("--region", default=SETTINGS.s3_aws_region)
    parser.add_argument(
        "--today", default=None, help="run as of this ET date (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    today = (
        datetime.strptime(args.today, "%Y-%m-%d").date()
        if args.today
        else datetime.now(ET).date()
    )
    last = today - timedelta(days=1)
    season = U.season_label(last)

    # Opened in a dry run too: it is read for the last stored game, and only
    # handed to the ingest (which writes) outside a dry run.
    aiven = connect_nba_db(env="aiven")
    source = connect_nba_db(env="supabase")
    try:
        # Rebuild from the day after the last game stored, so a missed run (or
        # the gap since the season opener) is caught up rather than lost.
        stored = last_stored_game_date(aiven, int(season[:4]))
        season_start = U.season_date_range(season, end=today)[0]
        caught_up = stored + timedelta(days=1) if stored else season_start
        first = min(caught_up, today - timedelta(days=args.days))
        report_start = first - timedelta(days=1)
        download_start = min(season_start, report_start)
        print(
            f"Last stored game: {stored or 'none this season'}; "
            f"game days {first}..{last}; reports {report_start}..{last}; "
            f"archive {download_start}..{today}"
        )

        if not args.skip_download:
            command = [
                sys.executable,
                str(_ROOT / "scripts/injury_reports/backfill_injury_reports.py"),
                "--start-date",
                str(download_start),
                "--end-date",
                str(today),
                "--manifest-root",
                args.manifest_root,
                "--cache-dir",
                args.cache_dir,
            ] + (["--dry-run"] if args.dry_run else [])
            if subprocess.run(command, check=False).returncode not in (0, 2):
                print("report download failed; nothing ingested")
                return 1

        store = ManifestStore(Path(args.manifest_root))
        storage = S3Storage(
            bucket=args.bucket,
            profile=args.profile,
            region=args.region,
            cache_dir=Path(args.cache_dir),
        )
        seasons = sorted({U.season_label(d) for d in U.date_range(first, last)})
        for season in seasons:
            manifest = store.load(season)
            published = mf.published_utc(manifest).dt.tz_convert(ET).dt.date
            window = manifest[(published >= report_start) & (published <= last)]
            if window.empty:
                print(f"{season}: no reports in the window")
                continue
            summary = ingest_season(
                season,
                manifest=window,
                storage=storage,
                aiven=None if args.dry_run else aiven,
                source=source,
                dry_run=args.dry_run,
                quiet=True,
                write_game_dates=(first, last),
            )
            print(summary.describe())
    finally:
        source.close()
        aiven.close()
    print(f"finished {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
