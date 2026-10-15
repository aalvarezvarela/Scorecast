#!/usr/bin/env bash
# Phase 6 controlled FM comparison (development experiment, CPU), declared
# 2026-10-15 before any run; the last FM design step before the GNN.
#   base        daily as-of RAPM (oof/rapm_daily, already stored)
#   stint       interaction-only residuals for points and pace (no per-player
#               additive term; interaction vectors centered on their running
#               mean): int_control (intercept / home only), int_chemistry_style,
#               int_matchup; the other four targets keep their additive heads
#   calibration walk-forward shrinkage in [0, 1] of each run's points / pace
#               correction, fitted on its own earlier out-of-fold rows
#               (calibrate_residuals.py); raw and calibrated both reported;
#               step 4's res_additive also calibrated as a reference
#   edges       edge-only capacity control: edge_capacity (attacker x learned
#               defender-independent vector), + defenders
#               (edge_capacity_defender), + matchup (edge_capacity_matchup);
#               edge_attacker / edge_matchup from step 4 reused
#   training    FMConfig defaults (dim 16, delta penalty 300), 3 epochs, batch
#               1,024, lr 3e-3, the 25 monthly folds 2016-12 -> 2019-06
#   seeds       0 1 2, paired; primary = mean over seeds
#   report      fm_ladder_report.py --ladder interaction
# Nothing is tuned on these results. Interrupted? Run it again.
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p artifacts/logs
LOG="artifacts/logs/fm_interaction_ladder_$(date +%Y%m%d_%H%M%S).log"
PY="${PYTHON:-poetry run python}"
for variant in int_control int_chemistry_style int_matchup \
    edge_capacity edge_capacity_defender edge_capacity_matchup; do
    for seed in 0 1 2; do
        $PY scripts/player_graph/fm_oof.py --variant "$variant" --seed "$seed" \
            --name "fm_${variant}_s${seed}" --until 2019-07-01 2>&1 | tee -a "$LOG"
    done
done
$PY scripts/player_graph/calibrate_residuals.py \
    --variants int_control int_chemistry_style int_matchup res_additive \
    --seeds 0 1 2 2>&1 | tee -a "$LOG"
$PY scripts/player_graph/fm_ladder_report.py --ladder interaction 2>&1 | tee -a "$LOG"
