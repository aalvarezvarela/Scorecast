#!/usr/bin/env bash
# Build the seven-snapshot pooled dataset and its matched 6h/4h controls.
#
# Run from anywhere:
#   bash experiments/intermediate_line_2026_08/prepare_line_error_7snapshot_6h_4h.sh
#
# Existing outputs are protected. To deliberately rebuild them in place:
#   OVERWRITE=1 bash experiments/intermediate_line_2026_08/prepare_line_error_7snapshot_6h_4h.sh
set -euo pipefail

cd "$(dirname "$0")/../.." || exit 1

POOLED_DATASET="data/train_data/intermediate_line_data_20260412_7snap.csv"
SCORING_DATASET="data/train_data/intermediate_line_data_20260412_7snap_scoring.csv"
T360_DATASET="data/train_data/intermediate_line_data_20260412_7snap_t360.csv"
T240_DATASET="data/train_data/intermediate_line_data_20260412_7snap_t240.csv"
# The 12h control is rebuilt from THIS grid too. The previous t720 slice came
# from the six-snapshot build, whose feature set predates the fanatics-column
# removal, so comparing across those bytes would confound the campaign.
T720_DATASET="data/train_data/intermediate_line_data_20260412_7snap_t720.csv"
GRID="30,60,120,240,360,480,720"

OUTPUTS=(
  "$POOLED_DATASET"
  "$SCORING_DATASET"
  "$T360_DATASET"
  "$T240_DATASET"
  "$T720_DATASET"
)

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

echo
echo "Building the independent T=360 control from the same pooled bytes..."
"${PY[@]}" scripts/create_train_data/slice_intermediate_snapshot.py \
  --input "$POOLED_DATASET" \
  --snapshot 360 \
  --output "$T360_DATASET"

echo
echo "Building the independent T=240 control from the same pooled bytes..."
"${PY[@]}" scripts/create_train_data/slice_intermediate_snapshot.py \
  --input "$POOLED_DATASET" \
  --snapshot 240 \
  --output "$T240_DATASET"

echo
echo "Building the independent T=720 control from the same pooled bytes..."
"${PY[@]}" scripts/create_train_data/slice_intermediate_snapshot.py \
  --input "$POOLED_DATASET" \
  --snapshot 720 \
  --output "$T720_DATASET"

echo
echo "Validating the materialized snapshot grains..."
"${PY[@]}" -c '
import sys
from pathlib import Path

import pandas as pd

pooled_path, t360_path, t240_path, t720_path = map(Path, sys.argv[1:])
expected = {30, 60, 120, 240, 360, 480, 720}

def load_keys(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=["GAME_ID", "TIME_TO_MATCH_MIN"], dtype={"GAME_ID": str})

pooled = load_keys(pooled_path)
actual = set(pooled["TIME_TO_MATCH_MIN"].dropna().astype(int).unique())
if actual != expected:
    raise SystemExit(f"{pooled_path}: expected snapshots {sorted(expected)}, got {sorted(actual)}")
if pooled.duplicated(["GAME_ID", "TIME_TO_MATCH_MIN"]).any():
    raise SystemExit(f"{pooled_path}: duplicate (GAME_ID, TIME_TO_MATCH_MIN) rows")

for path, horizon in ((t360_path, 360), (t240_path, 240), (t720_path, 720)):
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
' "$POOLED_DATASET" "$T360_DATASET" "$T240_DATASET" "$T720_DATASET"

echo
echo "Data preparation complete. Copy the three printed expected_checksum values"
echo "into the matching YAML files before running the campaign if you want strict"
echo "byte-for-byte dataset pinning."
echo
echo "Next:"
echo "  bash experiments/intermediate_line_2026_08/run_line_error_7snapshot_6h_4h.sh"
