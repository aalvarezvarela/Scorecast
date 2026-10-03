#!/usr/bin/env python3
"""Write a single-snapshot slice of the intermediate-line training CSV.

    poetry run python scripts/create_train_data/slice_intermediate_snapshot.py \
        --snapshot 720

The intermediate-line dataset has one row per (game, pre-game snapshot). Keeping
one ``TIME_TO_MATCH_MIN`` gives a file with **one row per game**, structurally
identical to the closing-line training CSV -- so every row-counted window in
``experiments/_base.yaml`` (``train_games`` and friends) means games again, with
no rescaling, and ``evaluate_betting`` needs no grouping.

This exists for the CONTROL run. The model you actually bet with is trained on
every snapshot at once so it can condition on time-to-tip; a single-snapshot
model cannot serve a bet placed at an hour it never saw. The control's only job
is to answer "is pooling earning its complexity?" -- if the pooled model cannot
beat a 12h-only model at 12h, the pooling is costing more than it returns.

Purely a file transform: no feature is recomputed, so the slice cannot disagree
with the pooled dataset about any value. ``TIME_TO_MATCH_MIN`` is constant in
the output and is dropped by the pipeline's own constant-column cleaning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nba_ou.config.identity_columns import identifier_dtypes  # noqa: E402

DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "train_data" / "intermediate_line_data_20260412.csv"
)
SNAPSHOT_COLUMN = "TIME_TO_MATCH_MIN"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--snapshot",
        type=int,
        required=True,
        help="Minutes before tip to keep, e.g. 720 for the 12-hour slice.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    # Identifier columns as str so GAME_ID keeps its leading zeros -- the
    # pipeline resolves season type from its 3-character prefix, and 22100001
    # does not map where 0022100001 does.
    header = pd.read_csv(args.input, nrows=0)
    df = pd.read_csv(
        args.input, dtype=identifier_dtypes(header.columns), low_memory=False
    )

    if SNAPSHOT_COLUMN not in df.columns:
        raise SystemExit(f"{args.input} has no {SNAPSHOT_COLUMN} column.")

    available = sorted(df[SNAPSHOT_COLUMN].dropna().unique())
    if args.snapshot not in available:
        raise SystemExit(
            f"Snapshot {args.snapshot} not present. Available: {available}"
        )

    sliced = df[df[SNAPSHOT_COLUMN] == args.snapshot].copy()

    games = sliced["GAME_ID"].nunique() if "GAME_ID" in sliced.columns else None
    if games is not None and games != len(sliced):
        raise SystemExit(
            f"Slice has {len(sliced)} rows for {games} games; a single snapshot "
            "must be one row per game. Refusing to write a file that would "
            "silently reintroduce the duplicate-game problem."
        )

    output = args.output or args.input.with_name(
        f"{args.input.stem}_t{args.snapshot}.csv"
    )
    sliced.to_csv(output, index=False)

    print(f"Snapshot {args.snapshot}: {len(sliced):,} rows, {games:,} games")
    print(f"Wrote {output}")

    from training_pipeline.data import compute_file_checksum

    print(f'expected_checksum: "{compute_file_checksum(output)}"')
    print(
        "\nRow-counted windows in experiments/_base.yaml need NO rescaling for "
        "this file:\nit is one row per game, exactly what those defaults assume."
    )


if __name__ == "__main__":
    main()
