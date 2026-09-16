#!/usr/bin/env python3
"""
Create training dataset up to 2026-01-10 (no date-to-predict / scheduled games).

This script calls `create_df_to_predict` without providing a prediction date
or scheduled-game data. It saves the resulting DataFrame to
`data/train_data/training_data_<schema_version>_YYYYMMDD.csv`
(see nba_ou.config.dataset_versions for the current schema).
"""

from pathlib import Path

import pandas as pd
from nba_ou.config.dataset_versions import TRAINING_DATA_SCHEMA_VERSION
from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Schema reproduced by ``--no-injury-report-features``.
CONTROL_SCHEMA_VERSION = "2_3"


def main(
    limit_date_to_train: str = "2026-01-10",
    n_seasons_to_include: int = None,
    output: Path | None = None,
    normalize_total_lines: bool = True,
    normalize_spread_lines: bool = True,
    null_extreme_spread_prices: bool = True,
    injury_report_features: bool = True,
    status_top_n: dict[str, int] | None = None,
) -> None:
    """Create training data up to `limit_date_to_train`.

    Args:
        limit_date_to_train: Date string YYYY-MM-DD (default: 2026-01-10)
        n_seasons_to_include: Number of seasons to include (default: None, uses all from 2017-18)
    """

    # Call create_df_to_predict without a scheduled date (no todays prediction)
    df_train = create_df_to_predict(
        todays_prediction=False,
        recent_limit_to_include=limit_date_to_train,
        older_season_limit=n_seasons_to_include,
        normalize_total_lines=normalize_total_lines,
        normalize_spread_lines=normalize_spread_lines,
        null_extreme_spread_prices=null_extreme_spread_prices,
        injury_report_features=injury_report_features,
        status_top_n=status_top_n,
    )

    if output is None:
        output_path = PROJECT_ROOT / "data" / "train_data"
        output_path.mkdir(parents=True, exist_ok=True)
        # Schema version in the name, never overwritten in place: spread and
        # moneyline additions, then spread-normalization semantics, must land beside
        # older files that pinned checksums still refer to.
        # Without the injury report the build is the schema 2_3 control, so it
        # is named as a 2_3 rebuild rather than as a 2_4 variant.
        schema, variant = (
            (TRAINING_DATA_SCHEMA_VERSION, "")
            if injury_report_features
            else (CONTROL_SCHEMA_VERSION, "_rebuild")
        )
        output = (
            output_path
            / f"training_data_{schema}_"
            f"{pd.to_datetime(limit_date_to_train).strftime('%Y%m%d')}{variant}.csv"
        )
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df_train.to_csv(output, index=False)
    print(f"Training data saved to {output}")

    from training_pipeline.data import compute_file_checksum

    print(f'expected_checksum: "{compute_file_checksum(output)}"')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create training dataset up to a given date (default: 2026-01-10)"
    )
    parser.add_argument(
        "--limit",
        "-l",
        dest="limit",
        default="2026-07-04",
        help="Limit date to train (YYYY-MM-DD). Defaults to 2026-07-04",
    )
    parser.add_argument(
        "--n-seasons",
        "-n",
        dest="n_seasons",
        type=int,
        default=None,
        help="Number of seasons to include. Defaults to None (all from 2017-18)",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--no-normalize-total-lines",
        action="store_true",
        help="Keep original asymmetrically priced total lines.",
    )
    parser.add_argument(
        "--no-normalize-spread-lines",
        action="store_true",
        help="Keep original asymmetrically priced spread lines.",
    )
    parser.add_argument(
        "--keep-extreme-spread-prices",
        action="store_true",
        help="Keep extreme spread price cells instead of setting them to NaN.",
    )

    parser.add_argument(
        "--no-injury-report-features",
        action="store_true",
        help=(
            "Schema 2_3 control: out set from the inactive list only, no injury "
            "report or listed-status columns. Written as training_data_2_3_*_rebuild.csv."
        ),
    )
    for status, default in (("questionable", 2), ("probable", 1), ("doubtful", 1)):
        parser.add_argument(
            f"--n-top-{status}",
            type=int,
            default=default,
            help=f"{status.title()} players per side with per-player columns (default {default}).",
        )

    args = parser.parse_args()
    main(
        args.limit,
        args.n_seasons,
        output=args.output,
        normalize_total_lines=not args.no_normalize_total_lines,
        normalize_spread_lines=not args.no_normalize_spread_lines,
        null_extreme_spread_prices=not args.keep_extreme_spread_prices,
        injury_report_features=not args.no_injury_report_features,
        status_top_n={
            "questionable": args.n_top_questionable,
            "probable": args.n_top_probable,
            "doubtful": args.n_top_doubtful,
        },
    )
