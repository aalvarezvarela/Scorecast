#!/usr/bin/env bash
# Rolling-origin CV + tuned rounds + tuned training window -- 2 runs, sequential.
#
#   bash experiments/runners/run_rolling_origin_campaign.sh
#
# Detached (survives closing the terminal):
#   nohup bash experiments/runners/run_rolling_origin_campaign.sh > /dev/null 2>&1 &
#
# Follow along:
#   tail -f artifacts/logs/rolling_origin_*.log
#
# Survey afterwards in experiments/notebooks/survey_experiments.ipynb with
#   SOURCES = ["experiments/rolling_origin_2026_08",
#              "experiments/window_sweep_2026_08",
#              "experiments/strategy_window_2026_08"]
# and RESCORE_EDGE_THRESHOLD = 0.1 -- these runs span a config change, so each
# one froze its own threshold into betting_metrics.json and the columns would
# otherwise be measuring different things.
set -uo pipefail          # NOT -e: one failed run must not cancel the rest

cd "$(dirname "$0")/../.." || exit 1

LOG_DIR="artifacts/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/rolling_origin_$(date +%Y%m%d_%H%M%S).log"

CONFIG_DIR="experiments/rolling_origin_2026_08"
# Strictly ordered by how many things each cell changes. The single-change cell
# runs first, so a campaign that dies overnight still leaves the one result that
# can be read on its own.
CONFIGS=(
  # The protocol itself comes from experiments/_base.yaml, so these two differ
  # only in what they predict. The window curve does not need its own cells any
  # more: train_games is a tuned parameter, so every trial records the window it
  # used next to its score in optuna_trials.csv.
  "$CONFIG_DIR/line_error.yaml"
  # Parity -- both tracks share splits.py, tuning.py and scorers.py.
  "$CONFIG_DIR/total_points.yaml"
)

DATASETS=(
  "data/train_data/all_odds_training_data_until_20260318.csv"
)

# -u keeps output unbuffered so `tail -f` shows progress live.
PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)

log() { echo "$@" | tee -a "$LOG"; }

log "=========================================================="
log "Rolling-origin campaign -- started $(date)"
log "  ${#CONFIGS[@]} runs, single seed (random_state 16), GPU"
log "  Optuna chooses the training window AND the model complexity"
log "  measured layout at eval_span_games=850: 30 folds, 117 game-days,"
log "  855 validation games, window ceiling 4117"
log "=========================================================="

for dataset in "${DATASETS[@]}"; do
  if [[ ! -s "$dataset" ]]; then
    log "ABORT: required dataset missing or empty: $dataset"
    exit 1
  fi
  log "Dataset ready: $dataset ($(du -h "$dataset" | cut -f1))"
done

# --- pre-flight gates the campaign -----------------------------------------
# Under rolling_origin this reports the REALISED fold count and validation
# volume (eval_span_games is a target, not a promise) and checks every
# train_games choice against the earliest fold's history. A choice that does not
# fit would make that fold train short for some trials and not others, so the
# window axis would be measuring the shortfall.
log ""
if [[ "${SKIP_PREFLIGHT:-0}" == "1" ]]; then
  # The slow part is cleaning the 394MB CSV (~8 min), which the campaign is
  # about to do anyway. Skipping is reasonable when an identical data + fold
  # configuration has already passed -- and unreasonable otherwise, because the
  # check it removes is the one that catches a training window silently
  # shrinking rather than erroring.
  log "SKIP_PREFLIGHT=1: skipping the pre-flight window check."
  log "  Only safe if the same dataset and walk_forward settings already passed."
else
  log "Running pre-flight (this cleans the dataset once, so it takes ~8 min)..."
  log "  Skip with: SKIP_PREFLIGHT=1 bash $0"
  if ! "${PY[@]}" scripts/preflight_campaign.py "$CONFIG_DIR" 2>&1 | tee -a "$LOG"; then
    log "ABORT: pre-flight failed. Fix the reported problems before running."
    exit 1
  fi
fi

# --- the runs --------------------------------------------------------------
FAILED=0
for cfg in "${CONFIGS[@]}"; do
  log ""
  log "----------------------------------------------------------"
  log "START $(date +%H:%M:%S)  $(basename "$cfg")"
  log "----------------------------------------------------------"

  START=$SECONDS
  # --no-save-model: this is an evaluation campaign, not a deployment. Promote
  # a winner afterwards with `python -m training_pipeline.promote <run_dir>`.
  if "${CLI[@]}" "$cfg" --no-save-model 2>&1 | tee -a "$LOG"; then
    STATUS="OK"
  else
    STATUS="FAILED"
    FAILED=$((FAILED + 1))
  fi
  ELAPSED=$((SECONDS - START))

  printf 'END   %s  %s  (%dh %02dm)  %s\n' \
    "$(date +%H:%M:%S)" "$STATUS" $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60)) \
    "$(basename "$cfg")" | tee -a "$LOG"
done

log ""
log "=========================================================="
log "Finished $(date) -- $FAILED of ${#CONFIGS[@]} runs failed"
log "Read the seed spread FIRST: nothing smaller than one config's own range"
log "across seeds is a result. Then check whether the selected train_games is"
log "STABLE across seeds -- a window that changes with the seed was chosen by"
log "noise, whatever its ROI says. The window curve itself is in"
log "optuna_trials.csv: group by train_games and read the objective."
log "=========================================================="

exit $FAILED
