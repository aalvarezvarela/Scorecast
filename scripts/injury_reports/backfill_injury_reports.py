"""Back up the NBA official injury-report PDF archive to S3 (or a local mirror).

Runs in two phases, either together or separately:

* **discover** -- HEAD every candidate URL and record what exists in a per-season
  Parquet manifest. No bodies are fetched, so this is cheap and safe to repeat.
* **download** -- fetch only the URLs discovery confirmed, validate each one
  against its own in-PDF header, and store it under a canonical ET-timestamped
  key.

Both phases are resumable: the manifest is the only state, a candidate with a
terminal verdict is never re-probed, and a stored object is never re-fetched.
Stop it with Ctrl-C at any point and re-run the same command to continue.

The CDN answers "missing" and "throttled" with the same ``403``, so a canary URL
adjudicates every 403 and the run stops rather than record a false absence
(see ``docs/injury_report_archive_plan.md`` §4.2).

Examples::

    # what seasons exist, and what is already collected
    python scripts/injury_reports/backfill_injury_reports.py --list-seasons

    # plan a season without touching the network
    python scripts/injury_reports/backfill_injury_reports.py \
        --season 2023-24 --report-only

    # one season, end to end, into the local mirror first
    python scripts/injury_reports/backfill_injury_reports.py \
        --season 2023-24 --local-root data/injury_reports

    # one season into S3
    python scripts/injury_reports/backfill_injury_reports.py --season 2023-24

    # everything, oldest first (the full backfill)
    python scripts/injury_reports/backfill_injury_reports.py --all-seasons

    # discovery only, capped, to sample how a season looks
    python scripts/injury_reports/backfill_injury_reports.py \
        --season 2025-26 --phase discover --max-requests 500

    # an explicit window, ignoring season boundaries
    python scripts/injury_reports/backfill_injury_reports.py \
        --start-date 2026-03-01 --end-date 2026-03-31
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from nba_ou.config.settings import SETTINGS
from nba_ou.fetch_data.injury_reports.archive import discovery, download
from nba_ou.fetch_data.injury_reports.archive import manifest as mf
from nba_ou.fetch_data.injury_reports.archive import urls as U
from nba_ou.fetch_data.injury_reports.archive.client import (
    DEFAULT_COOLDOWN_S,
    DEFAULT_DELAY_S,
    ArchiveClient,
)
from nba_ou.fetch_data.injury_reports.archive.storage import (
    LocalStorage,
    ManifestSync,
    S3Storage,
    Storage,
)

DEFAULT_MANIFEST_ROOT = Path("data/injury_reports/manifest")
DEFAULT_S3_PREFIX = "injury_reports"


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _resolve_windows(args: argparse.Namespace) -> list[tuple[str, date, date]]:
    """Return ``[(label, start, end), ...]`` for whatever the user asked for."""
    if args.start_date or args.end_date:
        start = args.start_date or U.CDN_START
        end = args.end_date or U.archive_end()
        return [(f"{start}..{end}", start, end)]

    seasons = args.season or (U.all_seasons() if args.all_seasons else [])
    if not seasons:
        raise SystemExit(
            "Nothing to do: pass --season, --all-seasons, or --start-date/--end-date."
        )

    known = set(U.all_seasons())
    unknown = [s for s in seasons if s not in known]
    if unknown:
        raise SystemExit(
            f"Unknown season(s): {', '.join(unknown)}\n"
            f"Available: {', '.join(sorted(known))}"
        )

    out = []
    for s in sorted(seasons):
        lo, hi = U.season_date_range(s)
        out.append((s, lo, hi))
    return out


def _make_storage(args: argparse.Namespace) -> Storage:
    if args.local_root:
        return LocalStorage(Path(args.local_root))
    return S3Storage(
        bucket=args.bucket or SETTINGS.s3_bucket,
        profile=SETTINGS.s3_aws_profile,
        region=SETTINGS.s3_aws_region,
    )


def _list_seasons(store: mf.ManifestStore) -> int:
    print(f"{'season':9s} {'dates':23s} {'known':>7s} {'exists':>7s} {'stored':>7s}")
    for s in U.all_seasons():
        lo, hi = U.season_date_range(s)
        df = store.load(s)
        exists = int((df.nba_available == mf.AVAILABLE_TRUE).sum()) if len(df) else 0
        stored = int((df.download_status == mf.DOWNLOAD_STORED).sum()) if len(df) else 0
        print(f"{s:9s} {str(lo)} .. {str(hi)} {len(df):7d} {exists:7d} {stored:7d}")
    return 0


def _report_only(store: mf.ManifestStore, windows: list[tuple[str, date, date]]) -> int:
    grand_c = grand_known = grand_exists = grand_stored = 0
    for label, lo, hi in windows:
        candidates = sum(len(U.candidates_for_date(d)) for d in U.date_range(lo, hi))
        season = label if label in set(U.all_seasons()) else None
        df = store.load(season) if season else mf.empty_frame()
        known = len(df)
        exists = int((df.nba_available == mf.AVAILABLE_TRUE).sum()) if known else 0
        stored = int((df.download_status == mf.DOWNLOAD_STORED).sum()) if known else 0
        print(
            f"{label:12s} {lo} .. {hi}  candidates={candidates:6d}  "
            f"probed={known:6d}  exists={exists:6d}  stored={stored:6d}  "
            f"to probe={max(candidates - known, 0):6d}"
        )
        grand_c += candidates
        grand_known += known
        grand_exists += exists
        grand_stored += stored
    print(
        f"\nTOTAL candidates={grand_c:,}  probed={grand_known:,}  "
        f"exists={grand_exists:,}  stored={grand_stored:,}"
    )
    print("(no network requests were made)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    scope = parser.add_argument_group("what to collect")
    scope.add_argument(
        "--season",
        action="append",
        default=None,
        metavar="YYYY-YY",
        help="season to collect, e.g. 2023-24. Repeatable.",
    )
    scope.add_argument(
        "--all-seasons", action="store_true", help="every season in the archive"
    )
    scope.add_argument("--start-date", type=_parse_date, default=None)
    scope.add_argument("--end-date", type=_parse_date, default=None)

    mode = parser.add_argument_group("how to run")
    mode.add_argument(
        "--phase",
        choices=("discover", "download", "both"),
        default="both",
        help="default: both",
    )
    mode.add_argument("--list-seasons", action="store_true")
    mode.add_argument(
        "--report-only",
        action="store_true",
        help="print the plan and current coverage; makes no requests",
    )
    mode.add_argument(
        "--dry-run", action="store_true", help="probe/fetch but write nothing"
    )
    mode.add_argument("--max-requests", type=int, default=None)
    mode.add_argument("--max-files", type=int, default=None)
    mode.add_argument("--quiet", action="store_true")

    dest = parser.add_argument_group("where it goes")
    dest.add_argument(
        "--local-root",
        default=None,
        help="write PDFs to this directory instead of S3 (e.g. data/injury_reports)",
    )
    dest.add_argument("--bucket", default=None, help="override the configured bucket")
    dest.add_argument("--manifest-root", default=str(DEFAULT_MANIFEST_ROOT))
    dest.add_argument(
        "--no-manifest-sync",
        action="store_true",
        help="do not mirror manifests to S3 (they stay local only)",
    )

    net = parser.add_argument_group("politeness")
    net.add_argument("--delay", type=float, default=DEFAULT_DELAY_S)
    net.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN_S)

    args = parser.parse_args()
    store = mf.ManifestStore(root=Path(args.manifest_root))

    if args.list_seasons:
        return _list_seasons(store)

    windows = _resolve_windows(args)
    if args.report_only:
        return _report_only(store, windows)

    client = ArchiveClient(
        delay_s=args.delay, cooldown_s=args.cooldown, verbose=not args.quiet
    )
    # Storage is needed for downloads, and also for mirroring manifests when the
    # destination is S3 -- otherwise a discovery-only run leaves its resume state
    # stranded on one machine.
    needs_storage = args.phase in ("download", "both") or not args.local_root
    storage = _make_storage(args) if needs_storage else None
    sync = (
        ManifestSync(storage=storage)
        if isinstance(storage, S3Storage) and not args.no_manifest_sync
        else None
    )

    print(f"Manifest : {store.root}" + ("  (mirrored to S3)" if sync else ""))
    if storage is not None:
        print(f"Storage  : {storage.describe()}")
    print(f"Windows  : {', '.join(w[0] for w in windows)}")
    if args.dry_run:
        print("DRY RUN - nothing will be written")
    print()

    exit_code = 0
    for label, lo, hi in windows:
        print(f"=== {label}  ({lo} .. {hi}) ===")
        window_seasons = sorted({U.season_label(d) for d in U.date_range(lo, hi)})

        if sync and not args.dry_run:
            for season in window_seasons:
                if sync.pull(season, store.path_for(season)):
                    print(f"  restored manifest for {season} from S3")

        if args.phase in ("discover", "both"):
            stats = discovery.discover(
                store=store,
                client=client,
                start=lo,
                end=hi,
                max_requests=args.max_requests,
                dry_run=args.dry_run,
                verbose=not args.quiet,
            )
            print(stats.render())
            if stats.stopped_early:
                exit_code = 2

        if args.phase in ("download", "both") and storage is not None:
            pending = [store.pending_downloads(s) for s in window_seasons]
            pending = [p for p in pending if len(p)]
            if not pending:
                print("Nothing pending to download.")
            else:
                import pandas as pd

                rows = pd.concat(pending, ignore_index=True).sort_values(
                    "report_datetime_utc", ignore_index=True
                )
                dstats = download.download_rows(
                    rows=rows,
                    store=store,
                    client=client,
                    storage=storage,
                    max_files=args.max_files,
                    dry_run=args.dry_run,
                    verbose=not args.quiet,
                )
                print(dstats.render())
                if dstats.stopped_early:
                    exit_code = 2

        if sync and not args.dry_run:
            for season in window_seasons:
                key = sync.push(season, store.path_for(season))
                if key:
                    print(f"  manifest -> s3://{storage.bucket}/{key}")
        print()

    print(f"HTTP requests made: {client.requests_made}")
    if client.cooldowns_taken:
        print(f"Throttle cooldowns: {client.cooldowns_taken}")
    if exit_code == 2:
        print("\nRun stopped early - re-run the same command to continue.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
