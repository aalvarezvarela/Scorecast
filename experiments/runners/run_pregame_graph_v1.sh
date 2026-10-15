#!/usr/bin/env bash
# Pregame graph campaign v1 (phase 7 -> 8), prepared 2026-10-15; NOT run yet.
#   bash experiments/runners/run_pregame_graph_v1.sh <stage> [cpu|cuda]
# Stages, in order; each is resumable (finished folds, seasons and months are
# skipped on a rerun):
#   encoders  train the two generation encoders on all 88 monthly folds
#             (gnn_node and gnn_full, seed 0, exact-mean centering), each fold
#             on stints before its month: about 40 + 55 min on this CPU
#   store     the pregame graph store, seasons 2018-2025 (needs the database)
#   outputs   out-of-fold encoder outputs for every closing game 2018-2025
#   prepare   the research file + the five arm configs (prints the checksum)
#   campaign  arms A-E x ten seeds, daily walk-forward (phase 5's runner and
#             protocol; about 2.5x phase 5's 44 h on this CPU)
#   report    the pre-registered comparisons
set -uo pipefail
cd "$(dirname "$0")/../.."
STAGE="${1:?usage: run_pregame_graph_v1.sh <stage> [cpu|cuda]}"
DEVICE="${2:-cpu}"
PY="${PYTHON:-poetry run python}"
CAMPAIGN=experiments/pregame_graph_v1_closing
OUT=artifacts/experiments/pregame_graph_v1
SEEDS="101 202 303 404 505 606 707 808 909 1010"
mkdir -p artifacts/logs
LOG="artifacts/logs/pregame_graph_v1_${STAGE}_$(date +%Y%m%d_%H%M%S).log"
case "$STAGE" in
encoders)
    for variant in gnn_node gnn_full; do
        $PY scripts/player_graph/fm_oof.py --variant "$variant" --seed 0 \
            --name "fm_${variant}_gen_s0" --until 2026-05-01 \
            --allow-evaluation-seasons --device "$DEVICE" 2>&1 | tee -a "$LOG"
    done ;;
store)
    $PY scripts/player_graph/build_pregame_graph_store.py --seasons 2018-2025 \
        2>&1 | tee -a "$LOG" ;;
outputs)
    $PY scripts/player_graph/generate_pregame_outputs.py \
        --encoder-run fm_gnn_node_gen_s0 --name gnn_node_gen --seasons 2018-2025 \
        --outputs style embedding --embedding-prefix GE_NODE_EMB \
        --allow-evaluation-seasons 2>&1 | tee -a "$LOG"
    $PY scripts/player_graph/generate_pregame_outputs.py \
        --encoder-run fm_gnn_full_gen_s0 --name gnn_full_gen --seasons 2018-2025 \
        --outputs style embedding --embedding-prefix GE_GRAPH_EMB \
        --allow-evaluation-seasons 2>&1 | tee -a "$LOG" ;;
prepare)
    $PY scripts/player_graph/prepare_pregame_campaign.py --style-run gnn_node_gen \
        --node-run gnn_node_gen --graph-run gnn_full_gen 2>&1 | tee -a "$LOG" ;;
campaign)
    for arm in a_baseline b_pi_primary c_style d_node_emb e_graph_emb; do
        expect=primary; [ "$arm" = a_baseline ] && expect=none
        $PY scripts/player_graph/run_pi_walk_forward.py --config "$CAMPAIGN/$arm.yaml" \
            --arm "$arm" --expect-pi "$expect" --device "$DEVICE" --out "$OUT" \
            --seeds $SEEDS 2>&1 | tee -a "$LOG"
    done ;;
report)
    $PY scripts/player_graph/report_pregame_walk_forward.py --out "$OUT" \
        2>&1 | tee -a "$LOG" ;;
*) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
