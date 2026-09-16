#!/usr/bin/env python3
"""Load archived injury-report PDFs into the Aiven ``injury_report`` schema.

Prerequisite: the PDFs must already be in S3 (or a local root). Run
``backfill_injury_reports.py`` first -- discovery alone is not enough, the
download phase has to have run.

    # create the schema, then load one season
    python scripts/injury_reports/load_injury_reports_to_aiven.py --create-schema
    python scripts/injury_reports/load_injury_reports_to_aiven.py --season 2023-24

    # shape everything without writing (no DB changes, still hits the network)
    python scripts/injury_reports/load_injury_reports_to_aiven.py \
        --season 2023-24 --dry-run

Seasons are loaded whole because a span may only be built from a contiguous run
of reports; see ``src/nba_ou/postgre_db/injury_report_aiven/ingest.py``.

    # reload a season after new reports were downloaded into it
    python scripts/injury_reports/load_injury_reports_to_aiven.py \
        --season 2025-26 --replace-season

Without ``--replace-season`` a reload keeps the old spans (inserts are
``ON CONFLICT DO NOTHING``), and their end times go stale around every added
report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from nba_ou.fetch_data.injury_reports.archive.manifest import (  # noqa: E402
    ManifestStore,
)
from nba_ou.fetch_data.injury_reports.archive.storage import (  # noqa: E402
    LocalStorage,
    S3Storage,
)
from nba_ou.postgre_db.config.db_config import connect_nba_db  # noqa: E402
from nba_ou.postgre_db.injury_report_aiven.ingest import ingest_season  # noqa: E402
from nba_ou.postgre_db.injury_report_aiven.schema import (  # noqa: E402
    create_schema,
    drop_schema,
)

DEFAULT_BUCKET = "adrian-nba-model-registry-eu-west-1"
DEFAULT_REGION = "eu-west-1"
DEFAULT_PROFILE = "adrian-personal"
DEFAULT_PREFIX = "injury_reports"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--season", action="append", default=[], help="e.g. 2023-24; repeatable"
    )
    p.add_argument("--all-seasons", action="store_true")
    p.add_argument("--list-seasons", action="store_true")
    p.add_argument("--create-schema", action="store_true")
    p.add_argument(
        "--reset",
        action="store_true",
        help="DROP the whole injury_report schema first. Destroys all loaded spans.",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="shape everything, write nothing"
    )
    p.add_argument(
        "--replace-season",
        action="store_true",
        help=(
            "delete each season's spans, filing spans, reports and unresolved "
            "counts before loading it. Required when reports were added to an "
            "already-loaded season (spans insert with ON CONFLICT DO NOTHING)."
        ),
    )
    p.add_argument("--limit", type=int, help="parse at most N reports per season")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--manifest-root", default="data/injury_reports/manifest")
    p.add_argument("--local-root", help="read PDFs from here instead of S3")
    p.add_argument("--bucket", default=DEFAULT_BUCKET)
    p.add_argument("--s3-prefix", default=DEFAULT_PREFIX)
    p.add_argument("--profile", default=DEFAULT_PROFILE)
    p.add_argument("--region", default=DEFAULT_REGION)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.replace_season and args.limit:
        print(
            "--replace-season with --limit would delete a season and reload part of it"
        )
        return 2
    store = ManifestStore(Path(args.manifest_root))

    if args.list_seasons:
        for season in store.known_seasons():
            frame = store.load(season)
            available = int((frame["nba_available"] == "true").sum())
            downloaded = int(frame["sha256"].notna().sum())
            print(f"  {season}  discovered={available:>6}  downloaded={downloaded:>6}")
        return 0

    if args.reset:
        conn = connect_nba_db(env="aiven")
        drop_schema(conn)
        create_schema(conn)
        conn.close()
        print("schema reset")
        if not (args.season or args.all_seasons):
            return 0
    elif args.create_schema:
        conn = connect_nba_db(env="aiven")
        create_schema(conn)
        conn.close()
        print("schema created")
        if not (args.season or args.all_seasons):
            return 0

    seasons = store.known_seasons() if args.all_seasons else list(args.season)
    if not seasons:
        print("nothing to do: pass --season, --all-seasons or --list-seasons")
        return 2

    storage = (
        LocalStorage(Path(args.local_root))
        if args.local_root
        else S3Storage(bucket=args.bucket, profile=args.profile, region=args.region)
    )

    aiven = None if args.dry_run else connect_nba_db(env="aiven")
    source = connect_nba_db(env="supabase")
    try:
        for season in seasons:
            manifest = store.load(season)
            if manifest.empty:
                print(f"{season}: no manifest, skipping")
                continue
            summary = ingest_season(
                season,
                manifest=manifest,
                storage=storage,
                aiven=aiven,
                source=source,
                limit=args.limit,
                dry_run=args.dry_run,
                quiet=args.quiet,
                replace=args.replace_season,
            )
            print(summary.describe())
            if summary.resolution.unresolved and not args.quiet:
                print(
                    f"  {len(summary.resolution.unresolved)} unresolved names recorded"
                )
    finally:
        source.close()
        if aiven is not None:
            aiven.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to continue.")
        sys.exit(130)
