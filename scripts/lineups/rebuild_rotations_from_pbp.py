"""Fill GameRotation gaps from archived PlayByPlayV2, with no NBA API calls.

For every finished game still owed a rotation, rebuild it from the season's
``nbastats_YYYY`` file (shufinskiy/nba_data, through 2024-25), check it, and
archive it where a fetched rotation would go, marked in the manifest with
``source=rebuilt_from_playbyplayv2``. Games that fail the check are left owed.
Then build stints as usual:

    python scripts/lineups/rebuild_rotations_from_pbp.py --season 2020 --verify
    python scripts/lineups/rebuild_rotations_from_pbp.py --season 2020
    python scripts/lineups/build_lineup_stints.py --season 2020

``--verify`` writes nothing: it rebuilds games that already have an API
rotation and reports how often the two agree. ``--replace-rejected`` rebuilds
the games whose *fetched* rotation the stint builder rejected (the NBA feed
sometimes has stints that end before they start), overwriting that file.
"""

from __future__ import annotations

import argparse
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd
from nba_ou.fetch_data.nba_lineups.archive import RawArchive, S3JsonStorage
from nba_ou.fetch_data.nba_lineups.manifest import REBUILT_SOURCE, Manifest
from nba_ou.fetch_data.nba_lineups.pbp_archive import archive_urls, download_season_csv
from nba_ou.fetch_data.nba_lineups.rotation_rebuild import (
    RotationRebuildError,
    lineup_agreement,
    rebuild_rotation,
    rotation_from_payload,
    rotation_payload,
)
from nba_ou.fetch_data.nba_lineups.run_lock import (
    BackfillAlreadyRunning,
    lineup_run_lock,
)
from tqdm import tqdm

from scripts.lineups.backfill_lineup_raw import finished_games, pending_calls
from scripts.lineups.build_lineup_stints import game_context

ENDPOINT = "gamerotation"
#: Lineups must agree on this share of seconds for --verify to call a game a match.
MATCH_THRESHOLD = 0.99


def load_pbp_v2(season_year: int, urls: dict[str, str], *, playoffs: bool) -> pd.DataFrame:
    frames = []
    for name in [f"nbastats_{season_year}"] + ([f"nbastats_po_{season_year}"] if playoffs else []):
        url = urls.get(name)
        if url is None:
            tqdm.write(f"  {name}: not published")
            continue
        with tempfile.TemporaryDirectory() as scratch:
            frames.append(pd.read_csv(download_season_csv(url, Path(scratch)), low_memory=False))
    if not frames:
        return pd.DataFrame()
    pbp = pd.concat(frames, ignore_index=True)
    pbp["GAME_ID"] = pbp["GAME_ID"].map(lambda value: str(int(value)).zfill(10))
    return pbp


def api_rotation_games(
    season_year: int, games: list[str], *, archive: RawArchive, manifest: Manifest
) -> list[str]:
    """Games whose archived rotation came from the API, for --verify."""
    entries = manifest.load(season_year)
    fetched = entries.loc[
        entries.endpoint.eq(ENDPOINT) & entries.status.eq("ok") & entries.source.isna(),
        "game_id",
    ]
    fetched = set(fetched)
    return [
        game_id for game_id in games
        if game_id in fetched and archive.exists(ENDPOINT, season_year, game_id)
    ]


def rejected_games(season_year: int, local_root: Path, games: list[str]) -> list[str]:
    """Games whose stints failed validation, per the stint store's status file."""
    path = local_root / "lineup_stints" / f"season={season_year}" / "game_status.parquet"
    if not path.exists():
        return []
    statuses = pd.read_parquet(path)
    failed = set(statuses.loc[statuses.status.eq("failed"), "game_id"])
    return [game_id for game_id in games if game_id in failed]


def rebuild_season(
    season_year: int,
    game_ids: list[str],
    *,
    urls: dict[str, str],
    playoffs: bool,
    archive: RawArchive,
    manifest: Manifest,
    verify: bool,
) -> dict:
    pbp = load_pbp_v2(season_year, urls, playoffs=playoffs)
    if pbp.empty:
        return {"no_pbp_dataset": len(game_ids)}
    by_game = dict(tuple(pbp.groupby("GAME_ID", sort=False)))
    games, box = game_context(game_ids)
    reasons: Counter[str] = Counter()
    agreements: list[float] = []
    rows: list[tuple[str, str, str, int]] = []
    for game_id in tqdm(game_ids, desc=f"rebuild {season_year}", unit="game", disable=None):
        game_pbp = by_game.get(game_id)
        teams = games.loc[games.GAME_ID.eq(game_id)]
        home = teams.loc[teams.HOME.eq(True), "TEAM_ID"]
        if game_pbp is None:
            reasons["no_pbp"] += 1
            continue
        if len(teams) != 2 or len(home) != 1:
            reasons["missing_game_teams"] += 1
            continue
        team_ids = tuple(int(team) for team in teams.TEAM_ID)
        try:
            rotation = rebuild_rotation(
                game_pbp, box.loc[box.GAME_ID.eq(game_id)], team_ids
            )
        except RotationRebuildError as exc:
            reasons[exc.reason] += 1
            continue
        reasons["ok"] += 1
        if verify:
            fetched = rotation_from_payload(archive.get(ENDPOINT, season_year, game_id))
            agreements.append(lineup_agreement(rotation, fetched))
            continue
        payload = rotation_payload(
            game_id, rotation, home_team_id=int(home.iloc[0]), pbp=game_pbp
        )
        nbytes = archive.put(ENDPOINT, season_year, game_id, payload)
        rows.append((game_id, ENDPOINT, "ok", nbytes))
    if rows:
        manifest.record_many(season_year, rows, source=REBUILT_SOURCE)
    summary: dict = dict(reasons)
    if agreements:
        series = pd.Series(agreements)
        summary["median_agreement"] = round(float(series.median()), 4)
        summary[f"share_>={MATCH_THRESHOLD}"] = round(float(series.ge(MATCH_THRESHOLD).mean()), 4)
        summary["worst_agreement"] = round(float(series.min()), 4)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--season", type=int, help="One season; or use --min/--max-season")
    parser.add_argument("--min-season", type=int, default=2018)
    parser.add_argument("--max-season", type=int)
    parser.add_argument("--no-playoffs", action="store_true", help="Regular season only")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Rebuild games that have an API rotation and report agreement; write nothing",
    )
    parser.add_argument(
        "--replace-rejected",
        action="store_true",
        help="Also rebuild games whose fetched rotation the stint builder rejected",
    )
    parser.add_argument("--limit", type=int, help="At most this many games per season")
    parser.add_argument("--s3", action="store_true", help="Archive in the configured S3 bucket")
    args = parser.parse_args()
    if args.verify and args.replace_rejected:
        parser.error("--verify writes nothing; drop --replace-rejected")
    if args.season is not None:
        seasons = range(args.season, args.season + 1)
    elif args.max_season is not None:
        seasons = range(args.min_season, args.max_season + 1)
    else:
        parser.error("Pass --season, or --max-season with --min-season")

    storage = None
    if args.s3:
        from nba_ou.config.settings import SETTINGS

        storage = S3JsonStorage(
            SETTINGS.s3_bucket, profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region
        )
        archive = RawArchive(storage)
    else:
        archive = RawArchive(root=args.local_root)
    manifest = Manifest(args.local_root / "nba_api_raw" / "manifest", mirror=storage)
    try:
        # Writes the per-season manifest the fetcher writes, so never beside it.
        with lineup_run_lock(args.local_root):
            urls = archive_urls()
            all_games = finished_games(seasons.start)
            for season_year in seasons:
                games = [
                    str(game_id).zfill(10) for season, game_id in all_games
                    if int(season) == season_year
                    and (not args.no_playoffs or not str(game_id).zfill(10).startswith("004"))
                ]
                if args.verify:
                    targets = api_rotation_games(
                        season_year, games, archive=archive, manifest=manifest
                    )
                else:
                    owed, _ = pending_calls(
                        [(season_year, game_id) for game_id in games],
                        archive=archive, manifest=manifest, endpoints=(ENDPOINT,),
                    )
                    targets = [game_id for _, game_id, _ in owed]
                    if args.replace_rejected:
                        targets += [
                            game_id
                            for game_id in rejected_games(season_year, args.local_root, games)
                            if game_id not in targets
                        ]
                if args.limit is not None:
                    targets = targets[: args.limit]
                if not targets:
                    print(f"{season_year}: nothing to {'verify' if args.verify else 'rebuild'}")
                    continue
                summary = rebuild_season(
                    season_year, targets, urls=urls, playoffs=not args.no_playoffs,
                    archive=archive, manifest=manifest, verify=args.verify,
                )
                print(f"{season_year} ({len(targets)} games): {summary}", flush=True)
    except BackfillAlreadyRunning as exc:
        parser.exit(2, f"{exc}\n")
    manifest.sync_all()


if __name__ == "__main__":
    main()
