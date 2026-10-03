#!/usr/bin/env python3
"""Upgrade an existing dataset file to a newer schema version.

Computes only the columns the newer version adds and appends them to a copy of
the file, so a 2_5 build becomes a 2_6 build without being rebuilt:

    poetry run python scripts/create_train_data/build_schema_version.py \\
        data/train_data/closing_line_data_2_5_20261003.parquet --to 2_6

    poetry run python scripts/create_train_data/build_schema_version.py \\
        data/train_data/intermediate_line_data_2_5_20261003.parquet

The dataset type and starting version come from the filename (and its manifest,
when there is one). The output lands beside the input with the version segment
swapped -- ``closing_line_data_2_6_20261003.parquet`` -- plus a
``.manifest.json`` recording the parent file's checksum. A Parquet input is
streamed one row group at a time; an older CSV input is loaded and written once.
The output is always Parquet. An intermediate file keeps using its parent's
``_scoring`` sidecar: the rows are the same.

The printed ``expected_checksum`` goes into a campaign config as usual.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nba_ou.config.dataset_versions import TRAINING_DATA_SCHEMA_VERSION

from training_pipeline.data import compute_file_checksum
from training_pipeline.layered_dataset import build_schema_version_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("parent", type=Path, help="Dataset file of an older version.")
    parser.add_argument(
        "--to",
        default=TRAINING_DATA_SCHEMA_VERSION,
        help=f"Target schema version (default {TRAINING_DATA_SCHEMA_VERSION}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where to write. Default: beside the parent.",
    )
    args = parser.parse_args()

    out = build_schema_version_file(
        args.parent,
        to_version=args.to,
        out_dir=args.out_dir,
        build_args={"parent": str(args.parent), "to": args.to},
    )
    print(f"data.csv_path: {out}")
    print(f'expected_checksum: "{compute_file_checksum(out)}"')


if __name__ == "__main__":
    main()
