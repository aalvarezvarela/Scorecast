#!/usr/bin/env bash
# Phase 6 FM ladder (development experiment, CPU), declared 2026-10-14 before
# any run:
#   variants  additive, no_matchup, matchup, matchup_edges (FMConfig defaults:
#             dim 16, delta penalty 300, edge task weight 1.0 = provisional,
#             not optimized); 3 epochs, batch 1,024, lr 3e-3; no subsampling
#   folds     the 25 monthly folds 2016-12 -> 2019-06 (same for every variant;
#             each trains on every stint before its checkpoint)
#   seeds     0 1 2 per variant; the primary comparison uses each variant's
#             mean prediction over seeds, the spread across seeds is the
#             training-noise band
#   report    scripts/player_graph/fm_ladder_report.py (development folds only)
# Nothing is tuned on these results. Interrupted? Run it again: stored folds
# are skipped.
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p artifacts/logs
LOG="artifacts/logs/fm_ladder_$(date +%Y%m%d_%H%M%S).log"
PY="${PYTHON:-poetry run python}"
for variant in additive no_matchup matchup matchup_edges; do
    for seed in 0 1 2; do
        $PY scripts/player_graph/fm_oof.py --variant "$variant" --seed "$seed" \
            --name "fm_${variant}_s${seed}" --until 2019-07-01 2>&1 | tee -a "$LOG"
    done
done
$PY scripts/player_graph/fm_ladder_report.py 2>&1 | tee -a "$LOG"
