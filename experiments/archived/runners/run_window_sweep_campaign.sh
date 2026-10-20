#!/usr/bin/env bash
# Window sweep + sample weighting + fold-count precision -- 11 runs, sequential.
#
#   bash experiments/archived/runners/run_window_sweep_campaign.sh
#
# Detached (survives closing the terminal):
#   nohup bash experiments/archived/runners/run_window_sweep_campaign.sh > /dev/null 2>&1 &
#
# Follow along:
#   tail -f artifacts/logs/window_sweep_*.log
#
# Survey afterwards in experiments/notebooks/survey_experiments.ipynb with
#   SOURCES = ["experiments/archived/window_sweep_2026_08",
#              "experiments/archived/strategy_window_2026_08"]
# so the new 3000/3850 points read against the existing 2500/3750 ones.
set -uo pipefail          # NOT -e: one failed run must not cancel the rest

cd "$(dirname "$0")/../../.." || exit 1

LOG_DIR="artifacts/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/window_sweep_$(date +%Y%m%d_%H%M%S).log"

CONFIG_DIR="experiments/archived/window_sweep_2026_08"
# Ordered so the questions most likely to change what gets trained run first.
# A campaign that dies overnight then still leaves the useful half done.
CONFIGS=(
  "$CONFIG_DIR/line_error_weighted.yaml"      # does recency weighting pay?
  "$CONFIG_DIR/line_error_3750_anchor.yaml"   # the 5-seed reference it needs
  "$CONFIG_DIR/line_error_3000.yaml"          # window curve
  "$CONFIG_DIR/line_error_3850.yaml"
  "$CONFIG_DIR/total_points_weighted.yaml"    # weighting on the repaired path
  "$CONFIG_DIR/line_error_16folds.yaml"       # precision, the real bottleneck
  "$CONFIG_DIR/total_points_3000.yaml"
  "$CONFIG_DIR/total_points_3850.yaml"
  "$CONFIG_DIR/classifier_3000.yaml"
  "$CONFIG_DIR/classifier_3850.yaml"
  "$CONFIG_DIR/line_error_old_data.yaml"      # third and final attempt
)

DATASETS=(
  "data/train_data/training_data_2_0_20260704.csv"
  "data/train_data/old_training_data_until_20260704.csv"
)

# -u keeps output unbuffered so `tail -f` shows progress live.
PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)

log() { echo "$@" | tee -a "$LOG"; }

log "=========================================================="
log "Window sweep campaign -- started $(date)"
log "  ${#CONFIGS[@]} runs, 5 seeds each, GPU"
log "  axes: training window / sample weighting / fold count / dataset"
log "=========================================================="

for dataset in "${DATASETS[@]}"; do
  if [[ ! -s "$dataset" ]]; then
    log "ABORT: required dataset missing or empty: $dataset"
    exit 1
  fi
  log "Dataset ready: $dataset ($(du -h "$dataset" | cut -f1))"
done

# --- pre-flight gates the campaign -----------------------------------------
# This is the check that matters: a training window larger than the data
# supports does NOT raise, it silently shrinks the folds. Six of eleven runs in
# the previous campaign trained short before this gate existed.
log ""
log "Running pre-flight (this cleans each dataset once, so it takes a while)..."
if ! "${PY[@]}" scripts/preflight_campaign.py "$CONFIG_DIR" 2>&1 | tee -a "$LOG"; then
  log "ABORT: pre-flight failed. Fix the reported problems before running."
  exit 1
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
log "Read seed_roi_std FIRST (comparable across seed counts, unlike the range)."
log "With ${#CONFIGS[@]} cells at this noise floor, expect about one to look"
log "good by luck alone."
log "=========================================================="

exit $FAILED
