"""Import seasons of PlayByPlayV3 from the shufinskiy/nba_data archive.

One ~8 MB download replaces a season of rate-limited calls. After this runs,
the backfill only owes gamerotation:

    python scripts/lineups/import_pbp_archive.py --min-season 2018 --max-season 2025
    python scripts/lineups/backfill_lineup_raw.py --endpoints gamerotation
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from nba_ou.fetch_data.nba_lineups.archive import RawArchive, S3JsonStorage
from nba_ou.fetch_data.nba_lineups.manifest import Manifest
from nba_ou.fetch_data.nba_lineups.pbp_archive import (
    archive_urls,
    download_season_csv,
    games_in_csv,
    season_datasets,
)
from nba_ou.fetch_data.nba_lineups.run_lock import (
    BackfillAlreadyRunning,
    lineup_run_lock,
)
from tqdm import tqdm

ENDPOINT = "playbyplayv3"


def import_season(
    season_year: int,
    *,
    archive: RawArchive,
    manifest: Manifest,
    urls: dict[str, str],
    playoffs: bool = True,
    force: bool = False,
) -> dict[str, int]:
    counts = {"imported": 0, "skipped": 0, "missing_dataset": 0}
    for name in season_datasets(season_year, playoffs=playoffs):
        url = urls.get(name)
        if url is None:
            counts["missing_dataset"] += 1
            tqdm.write(f"  {name}: not published, skipping")
            continue
        with tempfile.TemporaryDirectory() as scratch:
            csv_path = download_season_csv(url, Path(scratch))
            rows: list[tuple[str, str, str, int]] = []
            bar = tqdm(
                games_in_csv(csv_path), desc=f"import {name}", unit="game", disable=None
            )
            for game_id, payload in bar:
                if not force and manifest.is_ok(
                    season_year, game_id, ENDPOINT
                ) and archive.exists(ENDPOINT, season_year, game_id):
                    counts["skipped"] += 1
                    continue
                nbytes = archive.put(ENDPOINT, season_year, game_id, payload)
                rows.append((game_id, ENDPOINT, "ok", nbytes))
                counts["imported"] += 1
            bar.close()
            # One write per dataset rather than one per game.
            manifest.record_many(season_year, rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--s3", action="store_true")
    parser.add_argument("--min-season", type=int, default=2018)
    parser.add_argument("--max-season", type=int, required=True)
    parser.add_argument(
        "--no-playoffs", action="store_true", help="Regular season only"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-import games already archived"
    )
    args = parser.parse_args()
    storage = None
    if args.s3:
        from nba_ou.config.settings import SETTINGS

        storage = S3JsonStorage(
            SETTINGS.s3_bucket,
            profile=SETTINGS.s3_aws_profile,
            region=SETTINGS.s3_aws_region,
        )
        archive = RawArchive(storage)
    else:
        archive = RawArchive(root=args.local_root)
    manifest = Manifest(args.local_root / "nba_api_raw" / "manifest", mirror=storage)
    urls = archive_urls()
    totals = {"imported": 0, "skipped": 0, "missing_dataset": 0}
    # The importer writes the same per-season manifest the fetcher does, so the
    # two must not run at once or one would overwrite the other's rows.
    try:
        with lineup_run_lock(args.local_root):
            for season_year in range(args.min_season, args.max_season + 1):
                counts = import_season(
                    season_year,
                    archive=archive,
                    manifest=manifest,
                    urls=urls,
                    playoffs=not args.no_playoffs,
                    force=args.force,
                )
                print(f"{season_year}: {counts}", flush=True)
                for key, value in counts.items():
                    totals[key] += value
    except BackfillAlreadyRunning as exc:
        manifest.sync_all()
        parser.exit(2, f"{exc}\n")
    manifest.sync_all()
    print(totals)


if __name__ == "__main__":
    main()
