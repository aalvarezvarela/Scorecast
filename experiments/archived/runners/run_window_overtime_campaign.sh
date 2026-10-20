#!/usr/bin/env bash
# Window / dataset / overtime campaign -- 8 runs, sequentially.
#
#   bash experiments/archived/runners/run_window_overtime_campaign.sh
#
# Detached (survives closing the terminal):
#   nohup bash experiments/archived/runners/run_window_overtime_campaign.sh > /dev/null 2>&1 &
#
# Follow along:
#   tail -f artifacts/logs/window_overtime_*.log
#
# Then survey it together with the campaign it follows up on, by setting
# SOURCES in experiments/notebooks/survey_experiments.ipynb to:
#   ["experiments/archived/window_overtime_2026_08", "experiments/archived/strategy_window_2026_08"]
set -uo pipefail          # NOT -e: one failed run must not cancel the rest

cd "$(dirname "$0")/../../.." || exit 1

LOG_DIR="artifacts/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/window_overtime_$(date +%Y%m%d_%H%M%S).log"

CONFIG_DIR="experiments/archived/window_overtime_2026_08"
# Ordered cheapest-signal-first: the two single-change line_error cells answer
# the campaign's main questions, so a run that dies overnight still leaves the
# comparison interpretable.
CONFIGS=(
  "$CONFIG_DIR/line_error_4500.yaml"                        # window 3750 -> 4500
  "$CONFIG_DIR/line_error_3750_no_ot.yaml"                  # overtime out of training
  "$CONFIG_DIR/line_error_4500_maxna_200.yaml"              # recover ~386 dropped rows
  "$CONFIG_DIR/line_error_4500_with_playoffs.yaml"          # playoffs kept (read cv_roi)
  "$CONFIG_DIR/line_error_4500_maxna_120.yaml"              # milder row recovery
  "$CONFIG_DIR/line_error_3750_old_data.yaml"               # old 3240-col feature build
  "$CONFIG_DIR/line_error_4500_maxna_200_no_consensus.yaml" # rows + drop consensus block
  "$CONFIG_DIR/line_error_4500_no_ot.yaml"                  # window + overtime together
  "$CONFIG_DIR/total_points_4500.yaml"                      # window control
  "$CONFIG_DIR/total_points_4500_no_ot.yaml"                # overtime control
  "$CONFIG_DIR/classifier_4500.yaml"                        # window, third strategy
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
log "Window / dataset / overtime campaign -- started $(date)"
log "  ${#CONFIGS[@]} runs, 3 seeds each, GPU"
log "  axes: window / overtime / dataset / playoffs / missing-data"
log "=========================================================="

# --- both datasets must exist before committing to the campaign ------------
for dataset in "${DATASETS[@]}"; do
  if [[ ! -s "$dataset" ]]; then
    log "ABORT: required dataset missing or empty: $dataset"
    exit 1
  fi
  log "Dataset ready: $dataset ($(du -h "$dataset" | cut -f1))"
done

# --- validate every config first: a typo should cost seconds, not hours -----
for cfg in "${CONFIGS[@]}"; do
  if ! "${CLI[@]}" "$cfg" --dry-run > /dev/null 2>>"$LOG"; then
    log "ABORT: $cfg failed validation. See $LOG"
    exit 1
  fi
done
log "validated all ${#CONFIGS[@]} configs"

# --- the runs --------------------------------------------------------------
FAILED=0
for cfg in "${CONFIGS[@]}"; do
  log ""
  log "----------------------------------------------------------"
  log "START $(date +%H:%M:%S)  $(basename "$cfg")"
  log "----------------------------------------------------------"

  START=$SECONDS
  # --no-save-model: this is an evaluation campaign, not a deployment. Promote
  # the winner afterwards with `python -m training_pipeline.promote <run_dir>`.
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
log "Survey with, in experiments/notebooks/survey_experiments.ipynb:"
log "  SOURCES = [\"experiments/archived/window_overtime_2026_08\","
log "             \"experiments/archived/strategy_window_2026_08\"]"
log "Read seed_roi_range FIRST: with ${#CONFIGS[@]} new cells, about one will look good"
log "by luck alone."
log "=========================================================="

exit $FAILED
