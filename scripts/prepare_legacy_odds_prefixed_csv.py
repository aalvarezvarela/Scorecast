#!/usr/bin/env python3
"""Make a legacy training CSV consumable by the current odds-column contract.

Only the header is rewritten. Data rows are copied byte-for-byte, so this does
not regenerate features or otherwise turn the historical snapshot into a
modern dataset. The rename is delegated to ``apply_odds_prefix`` -- the same
canonical mapping used by the current feature builders.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nba_ou.config.odds_columns import (  # noqa: E402
    apply_odds_prefix,
    assert_odds_columns_prefixed,
)
from training_pipeline.data import compute_file_checksum  # noqa: E402


DEFAULT_INPUT = REPO_ROOT / "data/train_data/training_data_2_0_20260704.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts/derived_data/training_data_2_0_20260704_odds_prefixed.csv"
)
EXPECTED_INPUT_CHECKSUM = "sha256:2fc9ed86d2f42a78"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--expected-input-checksum",
        default=EXPECTED_INPUT_CHECKSUM,
        help="Refuse to adapt a different historical snapshot.",
    )
    return parser.parse_args()


def _renamed_header(header_bytes: bytes) -> tuple[bytes, int]:
    text = header_bytes.decode("utf-8-sig")
    columns = next(csv.reader([text]))
    renamed = apply_odds_prefix(pd.DataFrame(columns=columns)).columns.tolist()
    assert_odds_columns_prefixed(renamed, context="legacy compatibility CSV")
    if len(renamed) != len(set(renamed)):
        raise ValueError("Odds-prefix normalization created duplicate columns.")

    stream = io.StringIO(newline="")
    csv.writer(stream, lineterminator="\n").writerow(renamed)
    n_changed = sum(old != new for old, new in zip(columns, renamed, strict=True))
    return stream.getvalue().encode("utf-8"), n_changed


def main() -> int:
    args = _parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    source_checksum = compute_file_checksum(source)
    if args.expected_input_checksum and source_checksum != args.expected_input_checksum:
        raise ValueError(
            f"Historical input checksum mismatch: expected "
            f"{args.expected_input_checksum}, got {source_checksum}."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as input_handle:
        header, n_changed = _renamed_header(input_handle.readline())
        fd, temporary_name = tempfile.mkstemp(
            dir=output.parent, prefix=f".{output.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as output_handle:
                output_handle.write(header)
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            os.replace(temporary_name, output)
        except BaseException:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    print(f"input={source}")
    print(f"input_checksum={source_checksum}")
    print(f"renamed_columns={n_changed}")
    print(f"output={output}")
    print(f"output_checksum={compute_file_checksum(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
