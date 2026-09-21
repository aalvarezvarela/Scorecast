"""Validate archived games and write local stint/status Parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.lineups.stints import (
    StintValidationError,
    build_game_stints,
    decode_raw_game,
    validate_game_stints,
)
from nba_ou.fetch_data.nba_lineups.archive import RawArchive
from nba_ou.fetch_data.nba_lineups.manifest import Manifest
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_games,
    get_schema_name_players,
)
from psycopg import sql
from tqdm import tqdm


def game_context(game_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only the scores and box minutes needed to validate these games."""
    with connect_nba_db() as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT game_id, team_id, home, pts, game_date FROM {}.{} WHERE game_id = ANY(%s)"
            ).format(
                sql.Identifier(get_schema_name_games()),
                sql.Identifier(get_schema_name_games()),
            ),
            (game_ids,),
        )
        games = pd.DataFrame(
            cur.fetchall(), columns=["GAME_ID", "TEAM_ID", "HOME", "PTS", "GAME_DATE"]
        )
        cur.execute(
            sql.SQL(
                "SELECT game_id, team_id, player_id, min FROM {}.{} WHERE game_id = ANY(%s)"
            ).format(
                sql.Identifier(get_schema_name_players()),
                sql.Identifier(get_schema_name_players()),
            ),
            (game_ids,),
        )
        box = pd.DataFrame(
            cur.fetchall(), columns=["GAME_ID", "TEAM_ID", "PLAYER_ID", "MIN"]
        )
    return games, box


def build_archived(
    season_year: int,
    *,
    archive: RawArchive,
    manifest: Manifest,
    output_root: Path,
    limit: int | None = None,
    force: bool = False,
    game_ids: set[str] | None = None,
) -> dict[str, int]:
    entries = manifest.load(season_year)
    ok = entries.loc[entries.status.eq("ok")]
    complete = sorted(
        set(ok.loc[ok.endpoint.eq("gamerotation"), "game_id"])
        & set(ok.loc[ok.endpoint.eq("playbyplayv3"), "game_id"])
    )
    if game_ids is not None:
        complete = [game_id for game_id in complete if game_id in game_ids]
    if limit is not None:
        complete = complete[:limit]
    if not complete:
        return {"ok": 0, "failed": 0}
    target = output_root / f"season={season_year}"
    target.mkdir(parents=True, exist_ok=True)
    status_path = target / "game_status.parquet"
    statuses = (
        pd.read_parquet(status_path)
        if status_path.exists()
        else pd.DataFrame(columns=["game_id", "status", "reason"])
    )
    if not force:
        validated = set(statuses.loc[statuses.status.eq("ok"), "game_id"])
        complete = [game_id for game_id in complete if game_id not in validated]
    if not complete:
        return {"ok": 0, "failed": 0}
    games, box = game_context(complete)
    counts = {"ok": 0, "failed": 0}
    bar = tqdm(complete, desc=f"build {season_year}", unit="game", disable=None)
    for game_id in bar:
        try:
            rotation = archive.get("gamerotation", season_year, game_id)
            pbp_raw = archive.get("playbyplayv3", season_year, game_id)
            if rotation is None or pbp_raw is None:
                raise StintValidationError("missing_archive_object")
            home, away, pbp = decode_raw_game(rotation, pbp_raw)
            stints = build_game_stints(home, away, pbp)
            score = games.loc[games.GAME_ID.eq(game_id)]
            if len(score) != 2:
                raise StintValidationError("missing_game_score")
            home_score = score.loc[score.HOME.eq(True), "PTS"]
            away_score = score.loc[score.HOME.eq(False), "PTS"]
            if len(home_score) != 1 or len(away_score) != 1:
                raise StintValidationError("missing_game_score")
            validate_game_stints(
                stints,
                home,
                away,
                home_points=int(home_score.iloc[0]),
                away_points=int(away_score.iloc[0]),
                box_minutes=box.loc[box.GAME_ID.eq(game_id)],
            )
            stints.insert(0, "game_id", game_id)
            stints.insert(1, "season_year", season_year)
            # The Parquet store must stand on its own: ratings are fitted
            # walk-forward by date, so the date travels with the stints.
            stints.insert(2, "game_date", pd.Timestamp(score.GAME_DATE.iloc[0]))
            path = target / f"{game_id}.parquet"
            tmp = path.with_suffix(".parquet.tmp")
            stints.to_parquet(tmp, index=False)
            tmp.replace(path)
            verdict = "ok", ""
        except (StintValidationError, KeyError, ValueError) as exc:
            verdict = (
                "failed",
                exc.reason
                if isinstance(exc, StintValidationError)
                else type(exc).__name__,
            )
        statuses = statuses.loc[statuses.game_id.ne(game_id)]
        statuses = pd.concat(
            [
                statuses,
                pd.DataFrame(
                    [{"game_id": game_id, "status": verdict[0], "reason": verdict[1]}]
                ),
            ],
            ignore_index=True,
        )
        tmp = status_path.with_suffix(".parquet.tmp")
        statuses.to_parquet(tmp, index=False)
        tmp.replace(status_path)
        counts[verdict[0]] += 1
        bar.set_postfix(counts, refresh=False)
        if verdict[0] == "failed":
            tqdm.write(f"  {game_id} rejected: {verdict[1]}")
    bar.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--force", action="store_true", help="Rebuild even validated games"
    )
    args = parser.parse_args()
    print(
        build_archived(
            args.season,
            archive=RawArchive(root=args.local_root),
            manifest=Manifest(args.local_root / "nba_api_raw" / "manifest"),
            output_root=args.local_root / "lineup_stints",
            limit=args.limit,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
