"""Import who-guarded-whom (BoxScoreMatchupsV3) into per-season Parquet.

History is one ~4 MB download per season from the shufinskiy/nba_data archive,
with no NBA API calls. ``--fill-missing`` then asks the API for finished games
the archive lacks; ``--no-import`` skips the download, which is how a season the
archive has not published yet gets filled:

    python scripts/lineups/import_matchups.py --min-season 2017 --max-season 2025
    python scripts/lineups/import_matchups.py --min-season 2017 --max-season 2025 \\
        --fill-missing --no-import
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_ou.fetch_data.nba_lineups.client import CircuitOpen, LineupClient
from nba_ou.fetch_data.nba_lineups.matchups import (
    FIRST_SEASON,
    MatchupStore,
    fill_from_api,
    import_season,
)
from nba_ou.fetch_data.nba_lineups.pbp_archive import archive_urls
from nba_ou.fetch_data.nba_lineups.run_lock import (
    BackfillAlreadyRunning,
    lineup_run_lock,
)


def owed_games(
    store: MatchupStore, seasons: range, *, playoffs: bool = True
) -> dict[int, list[str]]:
    """Finished games per season that the store neither holds nor knows are empty."""
    from scripts.lineups.backfill_lineup_raw import finished_games

    settled = {season_year: store.settled_games(season_year) for season_year in seasons}
    owed: dict[int, list[str]] = {}
    for season_year, game_id in finished_games(seasons.start):
        season_year, game_id = int(season_year), str(game_id).zfill(10)
        if season_year not in seasons or (not playoffs and game_id.startswith("004")):
            continue
        if game_id not in settled[season_year]:
            owed.setdefault(season_year, []).append(game_id)
    return owed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--min-season", type=int, default=FIRST_SEASON)
    parser.add_argument("--max-season", type=int, required=True)
    parser.add_argument("--no-playoffs", action="store_true", help="Regular season only")
    parser.add_argument(
        "--no-import", action="store_true", help="Skip the archive download"
    )
    parser.add_argument(
        "--fill-missing",
        action="store_true",
        help="Fetch finished games the store lacks from the NBA API (needs the DB)",
    )
    parser.add_argument("--limit", type=int, help="Maximum API calls this run")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument(
        "--s3", action="store_true", help="Mirror the Parquet files to the S3 bucket"
    )
    args = parser.parse_args()
    if args.min_season < FIRST_SEASON:
        parser.error(f"Matchup tracking starts in {FIRST_SEASON}")
    seasons = range(args.min_season, args.max_season + 1)
    store = MatchupStore(args.local_root)
    touched: set[int] = set()

    if not args.no_import:
        urls = archive_urls()
        for season_year in seasons:
            counts = import_season(
                season_year, store=store, urls=urls, playoffs=not args.no_playoffs
            )
            print(f"{season_year} archive: {counts}", flush=True)
            if counts["games"]:
                touched.add(season_year)

    if args.fill_missing:
        # The API budget is per IP, so this must not run beside the lineup
        # backfill: two paced processes together would pace at half the gap.
        try:
            with lineup_run_lock(args.local_root):
                owed = owed_games(store, seasons, playoffs=not args.no_playoffs)
                budget = args.limit
                client = LineupClient(timeout=args.timeout)
                for season_year, game_ids in sorted(owed.items()):
                    if budget is not None:
                        game_ids = game_ids[:budget]
                        budget -= len(game_ids)
                    if not game_ids:
                        continue
                    counts = fill_from_api(
                        season_year, game_ids, store=store, client=client
                    )
                    touched.add(season_year)
                    print(
                        f"{season_year} api: {counts} ({len(game_ids)} owed)", flush=True
                    )
        except BackfillAlreadyRunning as exc:
            parser.exit(2, f"{exc}\nThe archive import above is kept; rerun the fill later.\n")
        except CircuitOpen as exc:
            print(f"Stopped early, progress kept: {exc}", flush=True)

    if args.s3 and touched:
        from nba_ou.config.settings import SETTINGS
        from nba_ou.fetch_data.nba_lineups.archive import S3JsonStorage

        storage = S3JsonStorage(
            SETTINGS.s3_bucket, profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region
        )
        for season_year in sorted(touched):
            for key, data in store.parquet_bytes(season_year).items():
                storage.put(key, data, {"content_type": "application/octet-stream"})
        print(f"Mirrored seasons {sorted(touched)} to {storage.describe()}")


if __name__ == "__main__":
    main()
