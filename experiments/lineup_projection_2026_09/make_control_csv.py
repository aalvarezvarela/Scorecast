"""Derive the campaign's control CSV: the 2_6 file without the LU_* columns.

The control keeps 2_6's starter-history columns, so the pair differs in the
lineup family alone -- the 2_5 file would differ in starter history as well.

Reads the treatment the way training reads it, drops the ``LU_*`` columns,
writes the rest through ``write_training_dataset`` and verifies that the control
loads as exactly the treatment minus those columns. Prints the checksums to pin
in the configs. (The name is historical: both files are Parquet now.)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from nba_ou.data_processing.lineups.features import LINEUP_FEATURE_COLUMNS

from training_pipeline.data import compute_file_checksum, load_raw_training_csv
from training_pipeline.parquet_dataset import write_training_dataset

ROOT = Path(__file__).resolve().parents[2] / "data" / "train_data"
TREATMENT = ROOT / "closing_line_data_2_6_20260704.parquet"
CONTROL = ROOT / "closing_line_data_2_6_20260704_lineup_control.parquet"


def main() -> None:
    treatment = load_raw_training_csv(TREATMENT)
    missing = set(LINEUP_FEATURE_COLUMNS) - set(treatment.columns)
    if missing:
        raise SystemExit(f"Treatment file lacks {sorted(missing)}")
    expected = treatment.drop(columns=list(LINEUP_FEATURE_COLUMNS))
    write_training_dataset(expected, CONTROL)

    control = load_raw_training_csv(CONTROL)
    pd.testing.assert_frame_equal(control, expected, check_exact=True)
    print(f"control: {len(control):,} rows x {control.shape[1]:,} columns, identical")
    print(f'  treatment expected_checksum: "{compute_file_checksum(TREATMENT)}"')
    print(f'  control   expected_checksum: "{compute_file_checksum(CONTROL)}"')


if __name__ == "__main__":
    main()
