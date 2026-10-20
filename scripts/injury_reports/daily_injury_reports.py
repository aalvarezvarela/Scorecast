#!/usr/bin/env python3
"""Store the injury reports of the days just finished, every day.

    python scripts/injury_reports/daily_injury_reports.py            # the last 2 days
    python scripts/injury_reports/daily_injury_reports.py --days 5   # catch up
    python scripts/injury_reports/daily_injury_reports.py --dry-run  # parse, no writes

Run it after the night's games are final (the daily database job). Two steps:

1. ``backfill_injury_reports.py`` discovers and downloads the reports published
   from the day before the first game date to yesterday, into the S3 archive
   (manifests mirrored there too).
2. ``ingest_season(write_game_dates=...)`` rebuilds the spans of the games
   played in those days only, replacing what the store held for them.

Only finished games are written -- the store holds a game once it is played,
the rule the line history and closing odds follow. Yesterday's late reports
already list tonight's games (most spans start the day before the game), so
those are parsed but not written; they are tomorrow's run's. A game's spans
read every report that lists it, which all fall in the downloaded window.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd  # noqa: E402
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--days", type=int, default=2, help="game days to rebuild")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--manifest-root", default=DEFAULT_MANIFEST_ROOT)
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
    first = today - timedelta(days=args.days)
    report_start = first - timedelta(days=1)
    print(f"Game days {first}..{last}; reports {report_start}..{last}")

    if not args.skip_download:
        command = [
            sys.executable,
            str(_ROOT / "scripts/injury_reports/backfill_injury_reports.py"),
            "--start-date",
            str(report_start),
            "--end-date",
            str(last),
            "--manifest-root",
            args.manifest_root,
        ] + (["--dry-run"] if args.dry_run else [])
        if subprocess.run(command, check=False).returncode not in (0, 2):
            print("report download failed; nothing ingested")
            return 1

    store = ManifestStore(Path(args.manifest_root))
    storage = S3Storage(bucket=args.bucket, profile=args.profile, region=args.region)
    seasons = sorted({U.season_label(d) for d in U.date_range(first, last)})
    aiven = None if args.dry_run else connect_nba_db(env="aiven")
    source = connect_nba_db(env="supabase")
    try:
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
                aiven=aiven,
                source=source,
                dry_run=args.dry_run,
                quiet=True,
                write_game_dates=(first, last),
            )
            print(summary.describe())
    finally:
        source.close()
        if aiven is not None:
            aiven.close()
    print(f"finished {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC")
    return 0


if __name__ == "__main__":
    sys.exit(main())
