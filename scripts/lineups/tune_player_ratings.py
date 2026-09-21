"""Tune lineup rating ridge penalties on explicit walk-forward date ranges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.lineups.rating_cv import tune_rating_lambdas

from scripts.lineups.stint_source import add_source_arguments, load_stints


def _grid(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_arguments(parser)
    parser.add_argument("--first-season", type=int, default=2018)
    parser.add_argument("--last-season", type=int, required=True)
    parser.add_argument("--validation-from", required=True)
    parser.add_argument("--validation-to", required=True)
    parser.add_argument("--offdef-grid", default="10,30,100,300,1000")
    parser.add_argument("--pace-grid", default="100,300,1000,3000,10000")
    parser.add_argument("--half-life-days", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    seasons = list(range(args.first_season, args.last_season + 1))
    stints = load_stints(args.source, seasons, args.local_root)
    start, end = pd.Timestamp(args.validation_from), pd.Timestamp(args.validation_to)
    dates = sorted(
        date
        for date in pd.to_datetime(stints.game_date).dt.normalize().unique()
        if start <= date <= end
    )
    results, best = tune_rating_lambdas(
        stints,
        dates,
        offdef_candidates=_grid(args.offdef_grid),
        pace_candidates=_grid(args.pace_grid),
        half_life_days=args.half_life_days,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "rating_lambda_cv.csv", index=False)
    (args.output_dir / "rating_lambda_best.json").write_text(
        json.dumps(best, indent=2) + "\n"
    )
    print(best)


if __name__ == "__main__":
    main()
