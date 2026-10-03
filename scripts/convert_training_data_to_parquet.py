#!/usr/bin/env python3
"""Convert an existing training CSV build to Parquet, verified row group by row group.

    poetry run python scripts/convert_training_data_to_parquet.py \\
        data/train_data/intermediate_line_data_2_5_20260613.csv

For datasets built while the builders still wrote CSV. The Parquet file is a
new dataset file with its own checksum, printed at the end: point the configs
that should read it at the ``.parquet`` path and pin that checksum. Configs left
on the CSV keep reading the CSV. How the write stays exact is documented in
``training_pipeline.parquet_dataset``.

**Memory:** the whole CSV is loaded once, ~14GB for the 4.6GB intermediate
file. Run it with nothing else loading a dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training_pipeline.parquet_dataset import (  # noqa: E402
    DEFAULT_COMPRESSION,
    DEFAULT_ROW_GROUP_SIZE,
    convert_csv_to_parquet,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", type=Path, help="Training CSV to convert.")
    parser.add_argument(
        "--output", type=Path, help="Default: the CSV's path with a .parquet suffix."
    )
    parser.add_argument("--row-group-size", type=int, default=DEFAULT_ROW_GROUP_SIZE)
    parser.add_argument("--compression", default=DEFAULT_COMPRESSION)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_path = args.output or args.csv.with_suffix(".parquet")
    if out_path.exists() and not args.overwrite:
        print(f"{out_path} exists; pass --overwrite to replace it.", file=sys.stderr)
        return 1
    convert_csv_to_parquet(
        args.csv,
        out_path,
        row_group_size=args.row_group_size,
        compression=args.compression,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
