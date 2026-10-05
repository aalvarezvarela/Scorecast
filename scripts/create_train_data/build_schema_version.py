#!/usr/bin/env python3
"""Upgrade an existing dataset file to a newer schema version.

Computes only the columns the newer version adds and appends them to a copy of
the file, so a 2_5 build becomes a 2_6 build without being rebuilt:

    poetry run python scripts/create_train_data/build_schema_version.py \\
        data/train_data/closing_line_data_2_5_20261003.parquet --to 2_6

    poetry run python scripts/create_train_data/build_schema_version.py \\
        data/train_data/intermediate_line_data_2_5_20261003.parquet --to 2_6

``--to`` is required: the target is always stated, never inferred from the
checkout's current default version. The dataset type and starting version come
from the filename (and its manifest, when there is one). The output lands beside
the input with the version segment swapped --
``closing_line_data_2_6_20261003.parquet`` -- plus a ``.manifest.json``
recording the parent file's checksum. A Parquet input is
streamed one row group at a time; an older CSV input is loaded and written once.
The output is always Parquet. An intermediate file keeps using its parent's
``_scoring`` sidecar -- the rows are the same -- and a copy is written beside
the output so its own upgrade finds it. Without a sidecar, a layer that needs
snapshot times fails unless ``--allow-schedule-tipoffs`` lets it read them from
the live line-history schedule; the manifest records which source was used.

The printed ``expected_checksum`` goes into a campaign config as usual.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from training_pipeline.data import compute_file_checksum
from training_pipeline.layered_dataset import build_schema_version_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("parent", type=Path, help="Dataset file of an older version.")
    parser.add_argument(
        "--to",
        required=True,
        help="Target schema version, e.g. 2_6. Required: there is no default.",
    )
    parser.add_argument(
        "--scoring-path",
        type=Path,
        default=None,
        help=(
            "Intermediate scoring sidecar with snapshot UTC timestamps. "
            "Default: the parent's _scoring sidecar."
        ),
    )
    parser.add_argument(
        "--allow-schedule-tipoffs",
        action="store_true",
        help=(
            "Without a scoring sidecar, read snapshot tipoffs from the live "
            "line-history schedule instead of failing."
        ),
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
        scoring_path=args.scoring_path,
        allow_schedule_tipoffs=args.allow_schedule_tipoffs,
    )
    print(f"data.csv_path: {out}")
    print(f'expected_checksum: "{compute_file_checksum(out)}"')


if __name__ == "__main__":
    main()
