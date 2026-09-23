"""Derive the campaign's control CSV: the treatment file without the LU_* columns.

Works on the text, not through pandas, so every remaining cell is
byte-identical to the treatment file -- a pandas round-trip would re-infer
``GAME_ID`` as an integer and drop its leading zeros. Then verifies that, and
prints the checksums to pin in the configs.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.lineups.features import LINEUP_FEATURE_COLUMNS

from training_pipeline.data import compute_file_checksum

ROOT = Path(__file__).resolve().parents[2] / "data" / "train_data"
TREATMENT = ROOT / "training_data_2_6_20260704_with_lineup_features.csv"
CONTROL = ROOT / "training_data_2_6_20260704_lineup_control.csv"


def main() -> None:
    csv.field_size_limit(sys.maxsize)
    with TREATMENT.open(newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        missing = set(LINEUP_FEATURE_COLUMNS) - set(header)
        if missing:
            raise SystemExit(f"Treatment file lacks {sorted(missing)}")
        keep = [
            i for i, name in enumerate(header) if name not in LINEUP_FEATURE_COLUMNS
        ]
        temporary = CONTROL.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow([header[i] for i in keep])
            for row in reader:
                writer.writerow([row[i] for i in keep])
        temporary.replace(CONTROL)

    treatment = pd.read_csv(TREATMENT, dtype=str, keep_default_na=False)
    control = pd.read_csv(CONTROL, dtype=str, keep_default_na=False)
    expected = treatment.drop(columns=list(LINEUP_FEATURE_COLUMNS))
    pd.testing.assert_frame_equal(control, expected)
    print(f"control: {len(control):,} rows x {control.shape[1]:,} columns, identical")
    print(f'  treatment expected_checksum: "{compute_file_checksum(TREATMENT)}"')
    print(f'  control   expected_checksum: "{compute_file_checksum(CONTROL)}"')


if __name__ == "__main__":
    main()
