"""Run the Phase C ratings-only total-points acceptance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.lineups.rating_gate import (
    ratings_only_game_projections,
    score_rating_gate,
)
from nba_ou.data_processing.players.attach_player_features import (
    _parse_minutes_series,
)
from nba_ou.postgre_db.games.fetch_data_from_db.fetch_data_from_games_db import (
    load_games_from_db,
)
from nba_ou.postgre_db.players.fetch_players_data_from_db import load_players_from_db


def _seasons(first: int, last: int) -> list[str]:
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(first, last + 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-season", type=int, default=2018)
    parser.add_argument("--last-season", type=int, required=True)
    parser.add_argument("--validation-from", required=True)
    parser.add_argument("--validation-to", required=True)
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--recent-games", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    seasons = _seasons(args.first_season, args.last_season)
    games = load_games_from_db(seasons=seasons)
    players = load_players_from_db(seasons=seasons)
    if games is None or players is None:
        raise RuntimeError("Could not load games and players for the rating gate")
    games.columns = games.columns.str.upper()
    players.columns = players.columns.str.upper()
    players["MIN"] = _parse_minutes_series(players.MIN)
    ratings = pd.read_parquet(args.ratings)
    projections = ratings_only_game_projections(
        games,
        players,
        ratings,
        validation_from=args.validation_from,
        validation_to=args.validation_to,
        recent_games=args.recent_games,
    )
    result = score_rating_gate(projections)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    projections.to_parquet(args.output_dir / "rating_gate_games.parquet", index=False)
    (args.output_dir / "rating_gate_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(result)


if __name__ == "__main__":
    main()
