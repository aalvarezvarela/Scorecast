"""Build the observed attacker-defender ``pair_game`` table, one Parquet a season.

Reads the local stint and matchup stores; no database, no API:

    python scripts/player_graph/build_pair_game.py --min-season 2017 --max-season 2025

Output: ``data/player_graph/pair_game/season=YYYY.parquet``. See
``nba_ou.data_processing.player_graph.pair_game`` for the definitions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_ou.data_processing.lineups.stint_store import read_stints
from nba_ou.data_processing.player_graph.pair_game import (
    build_pair_game,
    season_path,
)
from nba_ou.fetch_data.nba_lineups.matchups import FIRST_SEASON, MatchupStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--min-season", type=int, default=FIRST_SEASON)
    parser.add_argument("--max-season", type=int, required=True)
    args = parser.parse_args()
    store = MatchupStore(args.local_root)
    for season in range(args.min_season, args.max_season + 1):
        pairs = build_pair_game(
            read_stints([season], local_root=args.local_root), store.read(season)
        )
        path = season_path(season, local_root=args.local_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_parquet(path, index=False)
        no_cofloor = int(pairs["cofloor_seconds"].eq(0).sum())
        no_row = int((~pairs["has_matchup_row"]).sum())
        clipped = int(pairs["guard_rate"].eq(1.0).sum())
        print(
            f"{season}: {pairs['game_id'].nunique():,} games, {len(pairs):,} pairs "
            f"({no_row:,} without a matchup row, {no_cofloor} without co-floor "
            f"time, {clipped} rates clipped) -> {path}"
        )


if __name__ == "__main__":
    main()
