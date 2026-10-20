#!/usr/bin/env bash
# Build the seven-snapshot pooled dataset and a matched single-snapshot
# control slice for EVERY point on the grid (30/60/120/240/360/480/720),
# not just the 6h/4h/12h subset built by prepare_line_error_7snapshot_6h_4h.sh.
#
# Standalone: this does not require the 6h/4h/12h script to have run first.
# If that script already built the pooled CSV and the T=240/360/720 slices,
# this reuses those bytes (same output paths) and only adds T=30/60/120/480.
#
# Run from anywhere:
#   bash experiments/intermediate_line_2026_08/prepare_line_error_7snapshot_all_controls.sh
#
# Existing outputs are protected. To deliberately rebuild them in place:
#   OVERWRITE=1 bash experiments/intermediate_line_2026_08/prepare_line_error_7snapshot_all_controls.sh
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

POOLED_DATASET="data/train_data/intermediate_line_data_20260412_7snap.csv"
GRID="30,60,120,240,360,480,720"

declare -A SNAPSHOT_DATASETS=(
  [30]="data/train_data/intermediate_line_data_20260412_7snap_t30.csv"
  [60]="data/train_data/intermediate_line_data_20260412_7snap_t60.csv"
  [120]="data/train_data/intermediate_line_data_20260412_7snap_t120.csv"
  [240]="data/train_data/intermediate_line_data_20260412_7snap_t240.csv"
  [360]="data/train_data/intermediate_line_data_20260412_7snap_t360.csv"
  [480]="data/train_data/intermediate_line_data_20260412_7snap_t480.csv"
  [720]="data/train_data/intermediate_line_data_20260412_7snap_t720.csv"
)
SNAPSHOTS=(30 60 120 240 360 480 720)

OUTPUTS=("$POOLED_DATASET")
for snap in "${SNAPSHOTS[@]}"; do
  OUTPUTS+=("${SNAPSHOT_DATASETS[$snap]}")
done

EXISTING=()
for path in "${OUTPUTS[@]}"; do
  if [[ -e "$path" ]]; then
    EXISTING+=("$path")
  fi
done

if (( ${#EXISTING[@]} > 0 )) && [[ "${OVERWRITE:-0}" != "1" ]]; then
  echo "Refusing to overwrite existing campaign data:"
  printf '  %s\n' "${EXISTING[@]}"
  echo
  echo "Use OVERWRITE=1 only when you intentionally want to replace all outputs."
  exit 1
fi

PY=(poetry run python -u)

echo "Building pooled snapshots: ${GRID}"
"${PY[@]}" scripts/create_train_data/create_intermediate_line_train_data.py \
  --seasons 2021,2022,2023,2024,2025 \
  --recent-limit 2026-04-12 \
  --snapshot-grid "$GRID" \
  --anchor-book bet365 \
  --output "$POOLED_DATASET"

for snap in "${SNAPSHOTS[@]}"; do
  out="${SNAPSHOT_DATASETS[$snap]}"
  echo
  echo "Building the independent T=${snap} control from the same pooled bytes..."
  "${PY[@]}" scripts/create_train_data/slice_intermediate_snapshot.py \
    --input "$POOLED_DATASET" \
    --snapshot "$snap" \
    --output "$out"
done

echo
echo "Validating the materialized snapshot grains..."
"${PY[@]}" -c '
import sys
from pathlib import Path

import pandas as pd

pooled_path = Path(sys.argv[1])
snapshots = [int(s) for s in sys.argv[2].split(",")]
slice_paths = list(map(Path, sys.argv[3:]))
expected = set(snapshots)

def load_keys(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["GAME_ID", "TIME_TO_MATCH_MIN"], dtype={"GAME_ID": str})

pooled = load_keys(pooled_path)
actual = set(pooled["TIME_TO_MATCH_MIN"].dropna().astype(int).unique())
if actual != expected:
    raise SystemExit(f"{pooled_path}: expected snapshots {sorted(expected)}, got {sorted(actual)}")
if pooled.duplicated(["GAME_ID", "TIME_TO_MATCH_MIN"]).any():
    raise SystemExit(f"{pooled_path}: duplicate (GAME_ID, TIME_TO_MATCH_MIN) rows")

for path, horizon in zip(slice_paths, snapshots):
    frame = load_keys(path)
    horizons = set(frame["TIME_TO_MATCH_MIN"].dropna().astype(int).unique())
    if horizons != {horizon}:
        raise SystemExit(f"{path}: expected only T={horizon}, got {sorted(horizons)}")
    if frame["GAME_ID"].duplicated().any():
        raise SystemExit(f"{path}: more than one row for a game")
    print(f"  {path}: {len(frame):,} games at T={horizon}")

counts = pooled.groupby("TIME_TO_MATCH_MIN")["GAME_ID"].nunique().sort_index()
print(f"  {pooled_path}: {len(pooled):,} rows")
print(counts.to_string())
' "$POOLED_DATASET" "$GRID" "${SNAPSHOT_DATASETS[30]}" "${SNAPSHOT_DATASETS[60]}" \
  "${SNAPSHOT_DATASETS[120]}" "${SNAPSHOT_DATASETS[240]}" "${SNAPSHOT_DATASETS[360]}" \
  "${SNAPSHOT_DATASETS[480]}" "${SNAPSHOT_DATASETS[720]}"

echo
echo "Data preparation complete. Copy the eight printed expected_checksum values"
echo "into the matching YAML files before running the campaign if you want strict"
echo "byte-for-byte dataset pinning."
echo
echo "Next:"
echo "  bash experiments/intermediate_line_2026_08/run_line_error_7snapshot_all_controls.sh"
