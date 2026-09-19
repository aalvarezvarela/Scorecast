"""Load locally validated stint Parquet into the configured default database."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_games,
)
from nba_ou.postgre_db.lineups.load import load_game, mark_failed
from nba_ou.postgre_db.lineups.schema import SCHEMA, create_schema
from psycopg import sql


def _status_key(status: object, reason: object) -> tuple[str, str | None]:
    normalized_status = str(status)
    if normalized_status == "ok" or pd.isna(reason) or reason == "":
        return normalized_status, None
    return normalized_status, str(reason)


def load_season(
    season_year: int,
    *,
    local_root: Path,
    limit: int | None = None,
    force: bool = False,
) -> int:
    """Idempotently load locally validated results for one season."""
    base = local_root / "lineup_stints" / f"season={season_year}"
    status_path = base / "game_status.parquet"
    if not status_path.exists():
        return 0
    statuses = pd.read_parquet(status_path)
    if limit is not None:
        statuses = statuses.head(limit)
    if statuses.empty:
        return 0
    game_ids = statuses.game_id.astype(str).tolist()
    with connect_nba_db() as conn:
        create_schema(conn)
        with conn.cursor() as cur:
            if not force:
                cur.execute(
                    sql.SQL(
                        "SELECT game_id, status, reason FROM {}.lu_game_status "
                        "WHERE season_year=%s"
                    ).format(sql.Identifier(SCHEMA)),
                    (season_year,),
                )
                stored = {
                    str(game_id): _status_key(status, reason)
                    for game_id, status, reason in cur.fetchall()
                }
                statuses = statuses.loc[
                    [
                        stored.get(str(game_id)) != _status_key(status, reason)
                        for game_id, status, reason in statuses[
                            ["game_id", "status", "reason"]
                        ].itertuples(index=False, name=None)
                    ]
                ]
                if statuses.empty:
                    return 0
                game_ids = statuses.game_id.astype(str).tolist()
            cur.execute(
                sql.SQL(
                    "SELECT game_id, game_date FROM {}.{} WHERE game_id = ANY(%s)"
                ).format(
                    sql.Identifier(get_schema_name_games()),
                    sql.Identifier(get_schema_name_games()),
                ),
                (game_ids,),
            )
            dates = dict(cur.fetchall())
        for game_id, status, reason in statuses[
            ["game_id", "status", "reason"]
        ].itertuples(index=False, name=None):
            if game_id not in dates:
                raise ValueError(f"Missing game date for {game_id}")
            if status == "ok":
                frame = pd.read_parquet(base / f"{game_id}.parquet")
                load_game(
                    conn,
                    frame,
                    game_id=game_id,
                    season_year=season_year,
                    game_date=dates[game_id],
                )
            else:
                mark_failed(
                    conn,
                    game_id=game_id,
                    season_year=season_year,
                    game_date=dates[game_id],
                    reason=_status_key(status, reason)[1] or "unknown",
                )
            conn.commit()
    return len(statuses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    loaded = load_season(
        args.season, local_root=args.local_root, limit=args.limit, force=args.force
    )
    print(f"Loaded {loaded} game statuses for {args.season}")


if __name__ == "__main__":
    main()
