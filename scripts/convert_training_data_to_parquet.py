#!/usr/bin/env python3
"""Convert an existing training CSV to Parquet, verified row group by row group.

    poetry run python scripts/convert_training_data_to_parquet.py \\
        data/train_data/intermediate_line_data_2_5_20260613.csv

For datasets built before the builders learned ``--format parquet``. Configs
that pin the CSV read the copy from then on (data.resolve_dataset_source), so
none need editing. The CSV is
kept: its pinned checksum is what older runs reproduce against, so archive it
(or rely on its S3 backup) before deleting it. How the conversion stays exact is
documented in ``training_pipeline.parquet_dataset``.

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

from training_pipeline.data import (  # noqa: E402
    compute_file_checksum,
    parquet_source_checksum,
)
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
    written = convert_csv_to_parquet(
        args.csv,
        out_path,
        row_group_size=args.row_group_size,
        compression=args.compression,
    )
    print(
        f"Configs pinning {args.csv.name} ({parquet_source_checksum(written)}) now read "
        "this copy automatically, and keep working if the CSV is archived away."
    )
    print(f'The copy\'s own checksum, to point a config at it directly: '
          f'"{compute_file_checksum(written)}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
