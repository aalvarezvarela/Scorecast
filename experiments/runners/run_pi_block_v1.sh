#!/usr/bin/env bash
# Phase 5 paired walk-forward (pi_block_v1_closing_2026_10): arm A's ten seeds,
# its seed band, arm B's ten seeds, then the pre-registered report.
#
#   bash experiments/runners/run_pi_block_v1.sh <cpu|cuda>
#
# The device is chosen once with scripts/player_graph/benchmark_pi_fits.py and
# must be the same for every run of the comparison. Interrupted? Run the same
# command again: finished months of every seed are skipped.
set -uo pipefail
cd "$(dirname "$0")/../.."

DEVICE="${1:?usage: run_pi_block_v1.sh <cpu|cuda>}"
CAMPAIGN=experiments/pi_block_v1_closing_2026_10
OUT=artifacts/experiments/pi_block_v1
SEEDS="101 202 303 404 505 606 707 808 909 1010"
RESEARCH_FILE=$(sed -n 's/^  csv_path: //p' "$CAMPAIGN/a_baseline_2_5.yaml")
mkdir -p artifacts/logs "$OUT"
LOG="artifacts/logs/pi_block_v1_$(date +%Y%m%d_%H%M%S).log"
PY="${PYTHON:-poetry run python}"

echo "device $DEVICE, research file $RESEARCH_FILE, log $LOG" | tee -a "$LOG"
$PY scripts/player_graph/run_pi_walk_forward.py --config "$CAMPAIGN/a_baseline_2_5.yaml" \
    --arm a_baseline --expect-pi none --device "$DEVICE" --out "$OUT" --seeds $SEEDS \
    2>&1 | tee -a "$LOG"
$PY scripts/player_graph/report_pi_walk_forward.py --out "$OUT" --seed-band-only \
    --research-file "$RESEARCH_FILE" 2>&1 | tee -a "$LOG"
$PY scripts/player_graph/run_pi_walk_forward.py --config "$CAMPAIGN/b_pi_primary.yaml" \
    --arm b_pi_primary --expect-pi primary --device "$DEVICE" --out "$OUT" --seeds $SEEDS \
    2>&1 | tee -a "$LOG"
$PY scripts/player_graph/report_pi_walk_forward.py --out "$OUT" \
    --research-file "$RESEARCH_FILE" 2>&1 | tee -a "$LOG"
