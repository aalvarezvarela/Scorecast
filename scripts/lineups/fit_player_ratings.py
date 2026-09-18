"""Fit date-filtered player ratings from validated lineup stints."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.lineups.player_ratings import walk_forward_player_ratings
from nba_ou.postgre_db.lineups.fetch import fetch_stints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-season", type=int, default=2018)
    parser.add_argument("--last-season", type=int, required=True)
    parser.add_argument("--lambda-offdef", type=float, required=True)
    parser.add_argument("--lambda-pace", type=float, required=True)
    parser.add_argument("--half-life-days", type=float, default=180.0)
    parser.add_argument("--as-of-date", help="Also emit ratings for this future date")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seasons = list(range(args.first_season, args.last_season + 1))
    stints = fetch_stints(seasons)
    if stints.empty:
        parser.error("No validated stints in the lineup database for those seasons")
    dates = sorted(set(stints.game_date.unique()) |
                   ({pd.Timestamp(args.as_of_date)} if args.as_of_date else set()))
    ratings = walk_forward_player_ratings(
        stints, dates,
        lambda_offdef=args.lambda_offdef,
        lambda_pace=args.lambda_pace,
        half_life_days=args.half_life_days,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    ratings.to_parquet(tmp, index=False)
    tmp.replace(args.output)
    print(f"Wrote {len(ratings):,} as-of player ratings to {args.output}")


if __name__ == "__main__":
    main()
