"""Write walk-forward player lineup ratings and their fit metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nba_ou.data_processing.lineups.rating_cache import build_player_rating_cache
from nba_ou.postgre_db.lineups.fetch import fetch_stints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-season", type=int, default=2018)
    parser.add_argument("--last-season", type=int, required=True)
    parser.add_argument("--as-of-from", required=True)
    parser.add_argument("--as-of-to", required=True)
    parser.add_argument("--lambda-offdef", type=float, required=True)
    parser.add_argument("--lambda-pace", type=float, required=True)
    parser.add_argument("--half-life-days", type=float, default=180.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/lineup_ratings/player_ratings.parquet"),
    )
    args = parser.parse_args()
    stints = fetch_stints(list(range(args.first_season, args.last_season + 1)))
    ratings = build_player_rating_cache(
        stints,
        as_of_from=args.as_of_from,
        as_of_to=args.as_of_to,
        lambda_offdef=args.lambda_offdef,
        lambda_pace=args.lambda_pace,
        half_life_days=args.half_life_days,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    ratings.to_parquet(temporary, index=False)
    temporary.replace(args.output)
    metadata = {
        "first_season": args.first_season,
        "last_season": args.last_season,
        "as_of_from": args.as_of_from,
        "as_of_to": args.as_of_to,
        "lambda_offdef": args.lambda_offdef,
        "lambda_pace": args.lambda_pace,
        "half_life_days": args.half_life_days,
        "rows": len(ratings),
        "rating_dates": ratings.as_of_date.nunique(),
        "fit_max_game_date": str(ratings.fit_max_game_date.max().date()),
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_tmp.write_text(json.dumps(metadata, indent=2) + "\n")
    metadata_tmp.replace(metadata_path)
    print(f"Wrote {len(ratings)} rows to {args.output}")


if __name__ == "__main__":
    main()
