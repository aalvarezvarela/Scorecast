"""Fetch and validate newly finished games into the local stint store.

The Parquet store under ``data/lineup_stints/`` is the default destination.
Loading into Postgres is opt-in (``--load-db``) until the lineup features have
earned a place in the training pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_ou.data_processing.lineups.stint_store import read_game_statuses
from nba_ou.fetch_data.nba_lineups.archive import RawArchive
from nba_ou.fetch_data.nba_lineups.client import CircuitOpen, LineupClient
from nba_ou.fetch_data.nba_lineups.manifest import Manifest
from nba_ou.fetch_data.nba_lineups.run_lock import (
    BackfillAlreadyRunning,
    lineup_run_lock,
)
from nba_ou.postgre_db.lineups.fetch import fetch_game_statuses

from scripts.lineups.backfill_lineup_raw import backfill, finished_games
from scripts.lineups.build_lineup_stints import build_archived
from scripts.lineups.load_lineup_stints import load_season


def _validated_game_ids(local_root: Path, *, load_to_db: bool) -> set[str]:
    """Which games are already built, according to the destination in use."""
    if load_to_db:
        stored = fetch_game_statuses()
        return {game_id for game_id, (status, _) in stored.items() if status == "ok"}
    statuses = read_game_statuses(local_root=local_root)
    if statuses.empty:
        return set()
    return set(statuses.loc[statuses.status.eq("ok"), "game_id"])


def update(
    local_root: Path,
    *,
    min_season: int = 2018,
    archive: RawArchive | None = None,
    manifest: Manifest | None = None,
    load_to_db: bool = False,
) -> dict:
    archive = archive or RawArchive(root=local_root)
    manifest = manifest or Manifest(local_root / "nba_api_raw" / "manifest")
    games = [
        (int(season), str(game_id).zfill(10))
        for season, game_id in finished_games(min_season)
    ]
    # A game recorded as failed is retried: the raw archive is already paid for,
    # so a parser fix or a corrected feed self-heals on the next daily run.
    done = _validated_game_ids(local_root, load_to_db=load_to_db)
    pending = [(season, game_id) for season, game_id in games if game_id not in done]
    blocked: CircuitOpen | None = None
    counts = {"ok": 0, "empty": 0, "failed": 0, "skipped": 0}
    try:
        counts = backfill(
            pending,
            archive=archive,
            manifest=manifest,
            client=LineupClient(),
        )
    except CircuitOpen as exc:
        # Preserve and load everything completed before the breaker opened.
        blocked = exc
    pending_by_season = {
        season: {
            str(game_id).zfill(10)
            for item_season, game_id in pending
            if item_season == season
        }
        for season in sorted({season for season, _ in pending})
    }
    built = {
        season: build_archived(
            season,
            archive=archive,
            manifest=manifest,
            output_root=local_root / "lineup_stints",
            game_ids=game_ids,
        )
        for season, game_ids in pending_by_season.items()
    }
    result = {"fetched": counts, "built": built}
    if load_to_db:
        result["loaded"] = {
            season: load_season(season, local_root=local_root)
            for season in pending_by_season
        }
    if blocked is not None:
        raise CircuitOpen(f"{blocked}; completed work: {result}") from blocked
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--min-season", type=int, default=2018)
    parser.add_argument(
        "--s3", action="store_true", help="Use the configured S3 raw archive"
    )
    parser.add_argument(
        "--load-db",
        action="store_true",
        help="Also load the validated stints into Postgres (off by default)",
    )
    args = parser.parse_args()
    archive = None
    manifest = None
    if args.s3:
        from nba_ou.config.settings import SETTINGS
        from nba_ou.fetch_data.nba_lineups.archive import S3JsonStorage

        storage = S3JsonStorage(
            SETTINGS.s3_bucket,
            profile=SETTINGS.s3_aws_profile,
            region=SETTINGS.s3_aws_region,
        )
        archive = RawArchive(storage)
        manifest = Manifest(
            args.local_root / "nba_api_raw" / "manifest", mirror=storage
        )
    try:
        with lineup_run_lock(args.local_root):
            result = update(
                args.local_root,
                min_season=args.min_season,
                archive=archive,
                manifest=manifest,
                load_to_db=args.load_db,
            )
            print(result)
    except (CircuitOpen, BackfillAlreadyRunning) as exc:
        parser.exit(2, f"{exc}\n")
    finally:
        if manifest is not None:
            manifest.sync_all()


if __name__ == "__main__":
    main()
