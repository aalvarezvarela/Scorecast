#!/usr/bin/env python3
"""
Create the closing-line training dataset up to a limit date (no scheduled games).

This script calls `create_df_to_predict` without providing a prediction date
or scheduled-game data. That builds the base schema (2_5), saved to
`data/train_data/closing_line_data_2_5_<limit YYYYMMDD>.parquet`
(nba_ou.config.dataset_versions.training_dataset_filename). With a newer
`--schema-version` (the default is the newest), the newer file is then derived
from that one by adding only its columns (`training_pipeline.layered_dataset`),
so one run writes both versions. Each file gets a `.manifest.json` saying what
it is; see nba_ou.config.dataset_versions.
"""

from pathlib import Path

import pandas as pd
from nba_ou.config.dataset_versions import (
    BASE_SCHEMA_VERSION,
    TRAINING_DATA_SCHEMA_VERSION,
    training_dataset_filename,
)
from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict
from nba_ou.data_processing.referees.referee_tendencies import (
    DEFAULT_REFEREE_HISTORY_SEASONS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main(
    limit_date_to_train: str = "2026-01-10",
    n_seasons_to_include: int = None,
    output: Path | None = None,
    normalize_total_lines: bool = True,
    normalize_spread_lines: bool = True,
    null_extreme_spread_prices: bool = True,
    injury_report_features: bool = True,
    status_top_n: dict[str, int] | None = None,
    referee_history_seasons: int = DEFAULT_REFEREE_HISTORY_SEASONS,
    include_same_season_referee_variants: bool = False,
    schema_version: str = TRAINING_DATA_SCHEMA_VERSION,
) -> None:
    """Create training data up to `limit_date_to_train`.

    Args:
        limit_date_to_train: Date string YYYY-MM-DD (default: 2026-01-10)
        n_seasons_to_include: Number of seasons to include (default: None, uses all from 2017-18)
        schema_version: Newest version to write. The base version is always
            written; every version above it is layered on from that file.
    """
    from nba_ou.create_training_data.schema_layers import CLOSING_LINE, check_version

    check_version(schema_version)
    if output is not None and schema_version != BASE_SCHEMA_VERSION:
        from training_pipeline.layered_dataset import check_layerable_output

        check_layerable_output(output, base_version=BASE_SCHEMA_VERSION)
    build_args = {
        "limit_date_to_train": limit_date_to_train,
        "n_seasons_to_include": n_seasons_to_include,
        "normalize_total_lines": normalize_total_lines,
        "normalize_spread_lines": normalize_spread_lines,
        "null_extreme_spread_prices": null_extreme_spread_prices,
        "injury_report_features": injury_report_features,
        "status_top_n": status_top_n,
        "referee_history_seasons": referee_history_seasons,
        "include_same_season_referee_variants": include_same_season_referee_variants,
    }

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
        referee_history_seasons=referee_history_seasons,
        include_same_season_referee_variants=include_same_season_referee_variants,
    )

    if output is None:
        output_path = PROJECT_ROOT / "data" / "train_data"
        output_path.mkdir(parents=True, exist_ok=True)
        # The suffix distinguishes a build without report-derived availability
        # from the default build; layered versions keep it. The builder
        # produces the base schema; newer versions are layered on below and
        # named by swapping the version segment.
        variant = "" if injury_report_features else "without_injury_reports"
        output = output_path / training_dataset_filename(
            "closing",
            limit_date_to_train,
            variant=variant,
            schema_version=BASE_SCHEMA_VERSION,
        )
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    from training_pipeline.layered_dataset import (
        build_schema_version_file,
        write_base_manifest,
    )
    from training_pipeline.parquet_dataset import write_training_dataset

    write_training_dataset(df_train, output)
    n_rows, n_columns = df_train.shape
    # The layers read the file back; free the build first.
    del df_train
    write_base_manifest(
        output,
        schema_version=BASE_SCHEMA_VERSION,
        dataset_type=CLOSING_LINE,
        n_rows=n_rows,
        n_columns=n_columns,
        build_args=build_args,
    )
    if schema_version != BASE_SCHEMA_VERSION:
        build_schema_version_file(
            output, to_version=schema_version, build_args=build_args
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create training dataset up to a given date (default: 2026-01-10)"
    )
    parser.add_argument(
        "--limit",
        "-l",
        dest="limit",
        default=pd.Timestamp.today().strftime("%Y-%m-%d"),
        help=(
            "Last game date to include (YYYY-MM-DD), also stamped in the output "
            "name. Defaults to today."
        ),
    )
    parser.add_argument(
        "--n-seasons",
        "-n",
        dest="n_seasons",
        type=int,
        default=None,
        help="Number of seasons to include. Defaults to None (all from 2017-18)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Parquet path to write.",
    )
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
        "--referee-history-seasons",
        type=int,
        default=DEFAULT_REFEREE_HISTORY_SEASONS,
        help="Seasons of officiating history behind the REF_CREW_* features.",
    )
    parser.add_argument(
        "--referee-same-season-variants",
        action="store_true",
        help="Also emit REF_CREW_SS_* same-season-only tendencies (history ablation).",
    )

    parser.add_argument(
        "--no-injury-report-features",
        action="store_true",
        help=(
            "Use inactive-list availability without report-derived columns. "
            "The current schema version is retained with a distinct filename suffix."
        ),
    )
    parser.add_argument(
        "--schema-version",
        default=TRAINING_DATA_SCHEMA_VERSION,
        help=(
            f"Newest schema version to write (default {TRAINING_DATA_SCHEMA_VERSION}). "
            f"The base {BASE_SCHEMA_VERSION} file is always written; newer versions "
            "are layered on from it."
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
        referee_history_seasons=args.referee_history_seasons,
        include_same_season_referee_variants=args.referee_same_season_variants,
        schema_version=args.schema_version,
    )
