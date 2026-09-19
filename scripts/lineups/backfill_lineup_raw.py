"""Fetch finished NBA games' rotation and play-by-play, resumably."""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_ou.fetch_data.nba_lineups.archive import RawArchive, S3JsonStorage
from nba_ou.fetch_data.nba_lineups.client import (
    CircuitOpen,
    EmptyResponse,
    LineupClient,
)
from nba_ou.fetch_data.nba_lineups.manifest import Manifest
from nba_ou.fetch_data.nba_lineups.run_lock import (
    BackfillAlreadyRunning,
    lineup_run_lock,
)
from nba_ou.postgre_db.games.fetch_data_from_db.fetch_data_from_games_db import (
    load_games_from_db,
)


def finished_games(min_season: int = 2018) -> list[tuple[int, str]]:
    games = load_games_from_db()
    if games is None:
        raise RuntimeError("Cannot read the finished-games database")
    games.columns = games.columns.str.upper()
    games = games.loc[
        games["SEASON_YEAR"].ge(min_season)
        & games["SEASON_TYPE"].isin(["Regular Season", "Playoffs"])
        & games["PTS"].notna()
    ]
    games = games.sort_values(["GAME_DATE", "GAME_ID"])
    return list(
        games[["SEASON_YEAR", "GAME_ID"]]
        .drop_duplicates("GAME_ID")
        .itertuples(index=False, name=None)
    )


def backfill(
    games: list[tuple[int, str]],
    *,
    archive: RawArchive,
    manifest: Manifest,
    client: LineupClient,
    limit: int | None = None,
) -> dict[str, int]:
    counts = {"ok": 0, "empty": 0, "failed": 0, "skipped": 0}
    attempted = 0
    for season, game_id in games:
        game_id = str(game_id).zfill(10)
        for endpoint in ("gamerotation", "playbyplayv3"):
            if manifest.is_ok(season, game_id, endpoint):
                # A manifest without its object is not a successful archive.
                if archive.exists(endpoint, season, game_id):
                    counts["skipped"] += 1
                    continue
            if limit is not None and attempted >= limit:
                return counts
            attempted += 1
            try:
                raw = client.fetch(endpoint, game_id)
                nbytes = archive.put(endpoint, season, game_id, raw)
            except EmptyResponse:
                manifest.record(season, game_id, endpoint, "empty")
                counts["empty"] += 1
            except CircuitOpen:
                manifest.record(season, game_id, endpoint, "failed")
                counts["failed"] += 1
                raise
            else:
                manifest.record(season, game_id, endpoint, "ok", nbytes)
                counts["ok"] += 1
                if counts["ok"] % 100 == 0:
                    print(f"Archived {counts['ok']} responses through {game_id}", flush=True)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--s3", action="store_true", help="Store raw JSON in configured S3 bucket")
    parser.add_argument("--min-season", type=int, default=2018)
    parser.add_argument("--limit", type=int, default=None, help="Maximum API calls this run")
    args = parser.parse_args()
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
    manifest = Manifest(
        args.local_root / "nba_api_raw" / "manifest",
        mirror=storage if args.s3 else None,
    )
    try:
        with lineup_run_lock(args.local_root):
            counts = backfill(
                finished_games(args.min_season),
                archive=archive,
                manifest=manifest,
                client=LineupClient(),
                limit=args.limit,
            )
    except (CircuitOpen, BackfillAlreadyRunning) as exc:
        manifest.sync_all()
        parser.exit(2, f"{exc}\n")
    manifest.sync_all()
    print(counts)


if __name__ == "__main__":
    main()
