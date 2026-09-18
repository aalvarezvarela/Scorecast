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
from nba_ou.postgre_db.lineups.schema import create_schema
from psycopg import sql


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    base = args.local_root / "lineup_stints" / f"season={args.season}"
    statuses = pd.read_parquet(base / "game_status.parquet")
    if args.limit is not None:
        statuses = statuses.head(args.limit)
    with connect_nba_db() as conn:
        create_schema(conn)
        for game_id, status, reason in statuses[
            ["game_id", "status", "reason"]
        ].itertuples(index=False, name=None):
            with conn.cursor() as cur:
                cur.execute(sql.SQL("""
                    SELECT game_date FROM {}.{} WHERE game_id=%s LIMIT 1
                """).format(
                    sql.Identifier(get_schema_name_games()),
                    sql.Identifier(get_schema_name_games()),
                ), (game_id,))
                row = cur.fetchone()
            if row is None:
                raise ValueError(f"Missing game date for {game_id}")
            if status == "ok":
                frame = pd.read_parquet(base / f"{game_id}.parquet")
                load_game(conn, frame, game_id=game_id, season_year=args.season,
                          game_date=row[0])
            else:
                mark_failed(conn, game_id=game_id, season_year=args.season,
                            game_date=row[0], reason=reason)
            conn.commit()
    print(f"Loaded {len(statuses)} game statuses for {args.season}")


if __name__ == "__main__":
    main()
