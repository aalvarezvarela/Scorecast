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
from tqdm import tqdm


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


ENDPOINT_ORDER = ("gamerotation", "playbyplayv3")


def pending_calls(
    games: list[tuple[int, str]],
    *,
    archive: RawArchive,
    manifest: Manifest,
    endpoints: tuple[str, ...] = ENDPOINT_ORDER,
) -> tuple[list[tuple[int, str, str]], int]:
    """Split the work into calls still owed and calls already archived."""
    pending, skipped = [], 0
    for season, game_id in games:
        game_id = str(game_id).zfill(10)
        for endpoint in endpoints:
            # A manifest without its object is not a successful archive.
            if manifest.is_ok(season, game_id, endpoint) and archive.exists(
                endpoint, season, game_id
            ):
                skipped += 1
                continue
            pending.append((season, game_id, endpoint))
    return pending, skipped


def backfill(
    games: list[tuple[int, str]],
    *,
    archive: RawArchive,
    manifest: Manifest,
    client: LineupClient,
    limit: int | None = None,
    endpoints: tuple[str, ...] = ENDPOINT_ORDER,
) -> dict[str, int]:
    # Resolving what is owed up front costs under a second locally and keeps
    # already-archived calls out of the progress bar, so its rate and ETA
    # describe the API calls that actually take ~5 s each.
    pending, skipped = pending_calls(
        games, archive=archive, manifest=manifest, endpoints=endpoints
    )
    counts = {"ok": 0, "empty": 0, "failed": 0, "skipped": skipped}
    if limit is not None:
        pending = pending[:limit]
    # disable=None silences the bar when stdout is not a terminal, which is how
    # an overnight nohup run behaves; that run gets periodic log lines instead.
    with tqdm(
        pending,
        desc=f"fetch ({skipped:,} archived)",
        unit="call",
        disable=None,
    ) as bar:
        for season, game_id, endpoint in bar:
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
            bar.set_postfix(counts, refresh=False)
            if bar.disable and counts["ok"] and counts["ok"] % 100 == 0:
                print(
                    f"Archived {counts['ok']} of {len(pending)} owed responses "
                    f"through {game_id}",
                    flush=True,
                )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--s3", action="store_true", help="Store raw JSON in configured S3 bucket")
    parser.add_argument("--min-season", type=int, default=2018)
    parser.add_argument(
        "--max-season", type=int, help="Stop after this season, to run in chunks"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request read timeout; a hung socket costs this much before "
        "the session is reset and retried",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum API calls this run")
    parser.add_argument(
        "--endpoints",
        nargs="+",
        choices=ENDPOINT_ORDER,
        default=list(ENDPOINT_ORDER),
        help="Endpoints to fetch. Pass 'gamerotation' alone once "
        "scripts/lineups/import_pbp_archive.py has supplied the play-by-play.",
    )
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
            games = finished_games(args.min_season)
            if args.max_season is not None:
                games = [item for item in games if item[0] <= args.max_season]
            counts = backfill(
                games,
                archive=archive,
                manifest=manifest,
                client=LineupClient(timeout=args.timeout),
                limit=args.limit,
                endpoints=tuple(args.endpoints),
            )
    except (CircuitOpen, BackfillAlreadyRunning) as exc:
        manifest.sync_all()
        parser.exit(2, f"{exc}\n")
    manifest.sync_all()
    print(counts)


if __name__ == "__main__":
    main()
