#!/usr/bin/env bash
# Phase 6 GNN ladder (development experiment), declared 2026-10-15; NOT run yet
# (about 1-2 h on this machine's CPU; benchmark_pretraining.py gives the GPU
# figure). All rungs are interaction-only residuals over the daily as-of RAPM
# for points and pace, with the additive absorber (trained, dropped at
# prediction); the four auxiliary targets keep their additive heads.
#   A  daily RAPM alone (oof/rapm_daily)
#   B  gnn_control   intercept / home only (no interaction readout)
#   C  gnn_node      capacity-matched node-only neural model (no message passing)
#   D  gnn_no_guard  teammate / opponent message passing
#   E  gnn_full      + guard edges (expected shares and their attributes)
#   C, D, E have identical parameters, inputs, heads and training budget.
#   training   width 32, two layers, delta penalty 300, weight penalty 1e-4,
#              3 epochs, batch 1,024, lr 3e-3, the 25 monthly folds
#              2016-12 -> 2019-06
#   seeds      0 1 2, paired; calibration as calibrate_residuals.py
#   report     fm_ladder_report.py --ladder gnn
# Nothing is tuned on these results. Interrupted? Run it again.
#   bash experiments/runners/run_gnn_ladder.sh [cpu|cuda]
set -uo pipefail
cd "$(dirname "$0")/../.."
DEVICE="${1:-cpu}"
mkdir -p artifacts/logs
LOG="artifacts/logs/gnn_ladder_$(date +%Y%m%d_%H%M%S).log"
PY="${PYTHON:-poetry run python}"
for variant in gnn_control gnn_node gnn_no_guard gnn_full; do
    for seed in 0 1 2; do
        $PY scripts/player_graph/fm_oof.py --variant "$variant" --seed "$seed" \
            --name "fm_${variant}_s${seed}" --until 2019-07-01 --device "$DEVICE" \
            2>&1 | tee -a "$LOG"
    done
done
$PY scripts/player_graph/calibrate_residuals.py \
    --variants gnn_control gnn_node gnn_no_guard gnn_full --seeds 0 1 2 \
    2>&1 | tee -a "$LOG"
$PY scripts/player_graph/fm_ladder_report.py --ladder gnn 2>&1 | tee -a "$LOG"
