#!/usr/bin/env bash
# Overnight TOTAL_POINTS dataset A/B.
#
# Both aligned datasets must already exist. The script checks them before
# validating and running the experiments; it does not generate training data.
#
#   bash experiments/archived/runners/run_total_points_dataset_ab.sh
#
# Detached (survives closing the terminal):
#   nohup bash experiments/archived/runners/run_total_points_dataset_ab.sh > /dev/null 2>&1 &
#
# Follow along:
#   tail -f artifacts/logs/overnight_*.log
set -uo pipefail          # NOT -e: run 2 should still start if run 1 fails

cd "$(dirname "$0")/../../.." || exit 1

LOG_DIR="artifacts/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/overnight_${STAMP}.log"

DATASET_A="data/train_data/old_training_data_until_20260704.csv"
DATASET_B="data/train_data/training_data_2_0_20260704.csv"

CONFIGS=(
  "experiments/archived/total_points/2500_games_20260704.yaml"            # 3.5 h tuning
  "experiments/archived/total_points/2500_games_train_2_0_20260704.yaml"  # 3.0 h tuning
)

# -u keeps output unbuffered so `tail -f` shows progress live.
PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)

log() { echo "$@" | tee -a "$LOG"; }

log "=========================================================="
log "Overnight run started $(date)"
log "=========================================================="

# --- require both aligned datasets before starting -------------------------
for dataset in "$DATASET_A" "$DATASET_B"; do
  if [[ ! -s "$dataset" ]]; then
    log "ABORT: required dataset is missing or empty: $dataset"
    exit 1
  fi
  log "Dataset ready: $dataset ($(du -h "$dataset" | cut -f1))"
  log "  checksum: $("${PY[@]}" -c "from training_pipeline.data import compute_file_checksum as c; print(c('$dataset'))" 2>/dev/null)"
done

# --- validate every config before committing the night to them -------------
for cfg in "${CONFIGS[@]}"; do
  if ! "${CLI[@]}" "$cfg" --dry-run > /dev/null 2>>"$LOG"; then
    log "ABORT: $cfg failed validation. See $LOG"
    exit 1
  fi
  log "validated: $cfg"
done

# --- the experiments, sequentially -----------------------------------------
# Sequential on purpose: XGBoost already uses every core (n_jobs=-1), so running
# them in parallel would only make both slower.
FAILED=0
for cfg in "${CONFIGS[@]}"; do
  log ""
  log "----------------------------------------------------------"
  log "START $(date +%H:%M:%S)  $cfg"
  log "----------------------------------------------------------"

  START=$SECONDS
  # --no-save-model: this is an evaluation, not a deployment. Promote the
  # winner afterwards with `python -m training_pipeline.promote <run_dir>`.
  if "${CLI[@]}" "$cfg" --no-save-model 2>&1 | tee -a "$LOG"; then
    STATUS="OK"
  else
    STATUS="FAILED"
    FAILED=$((FAILED + 1))
  fi
  ELAPSED=$((SECONDS - START))

  printf 'END   %s  %s  (%dh %02dm)  %s\n' \
    "$(date +%H:%M:%S)" "$STATUS" $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60)) "$cfg" \
    | tee -a "$LOG"
done

log ""
log "=========================================================="
log "Finished $(date) -- $FAILED of ${#CONFIGS[@]} experiments failed"
log "Runs saved under artifacts/experiments/"
log "Compare them in experiments/compare_experiments.ipynb with"
log "  TARGET_FAMILY    = \"total_points\""
log "  COMPARISON_GROUP = \"total_points_dataset_ab_2026_08\""
log "=========================================================="

exit $FAILED
