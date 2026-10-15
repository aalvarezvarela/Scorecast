#!/usr/bin/env bash
# Phase 6 FM residual ladder and edge-head ablation (development experiment,
# CPU), declared 2026-10-15 before any run:
#   base      daily as-of RAPM (pretraining_daily_rapm.py): each stint's points
#             and pace from the frozen phase 3 ratings fitted on stints before
#             its own date, for training residuals and evaluation alike
#   ladder    1 RAPM alone; 2 res_additive; 3 res_no_matchup (chemistry/style);
#             4 res_matchup; 5 res_matchup_edges (edge loss weight 1.0,
#             provisional, not optimized). Residual over the base for points
#             and pace; the other four targets keep their own heads
#   ablation  edge-only models: edge_attacker, edge_defender (+ defenders'
#             terms), edge_matchup (+ matchup product)
#   training  FMConfig defaults (dim 16, delta penalty 300), 3 epochs, batch
#             1,024, lr 3e-3, the 25 monthly folds 2016-12 -> 2019-06
#   seeds     0 1 2, paired across variants; primary = mean over seeds
#   report    fm_ladder_report.py --ladder residual (development folds only)
# Nothing is tuned on these results. Interrupted? Run it again.
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p artifacts/logs
LOG="artifacts/logs/fm_residual_ladder_$(date +%Y%m%d_%H%M%S).log"
PY="${PYTHON:-poetry run python}"
$PY scripts/player_graph/pretraining_daily_rapm.py 2>&1 | tee -a "$LOG"
for variant in res_additive res_no_matchup res_matchup res_matchup_edges \
    edge_attacker edge_defender edge_matchup; do
    for seed in 0 1 2; do
        $PY scripts/player_graph/fm_oof.py --variant "$variant" --seed "$seed" \
            --name "fm_${variant}_s${seed}" --until 2019-07-01 2>&1 | tee -a "$LOG"
    done
done
$PY scripts/player_graph/fm_ladder_report.py --ladder residual 2>&1 | tee -a "$LOG"
