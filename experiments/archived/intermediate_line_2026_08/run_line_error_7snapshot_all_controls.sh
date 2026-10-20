#!/usr/bin/env bash
# Run the unweighted LINE_ERROR seven-snapshot pooled model and a matched
# single-snapshot control for EVERY point on the grid (30/60/120/240/360/480/
# 720), so every elapsed time the pooled model reports has a control to read
# it against -- not just the 6h/4h/12h subset in run_line_error_7snapshot_6h_4h.sh.
#
# Foreground:
#   bash experiments/intermediate_line_2026_08/run_line_error_7snapshot_all_controls.sh
#
# Detached:
#   nohup bash experiments/intermediate_line_2026_08/run_line_error_7snapshot_all_controls.sh \
#       > /dev/null 2>&1 &
#
# Existing artifacts with the exact experiment name are skipped by default.
# Force all runs to rerun with SKIP_EXISTING=0. The targeted full-data
# preflight is mandatory unless explicitly bypassed with SKIP_PREFLIGHT=1.
#
# NOT `set -e`: one failed training cell must not discard the remaining cells.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="intermediate_line_7snapshot_all_controls_2026_08"
CONFIG_DIR="experiments/intermediate_line_2026_08"
POOLED_DATASET="data/train_data/intermediate_line_data_20260412_7snap.csv"

declare -A SNAPSHOT_DATASETS=(
  [30]="data/train_data/intermediate_line_data_20260412_7snap_t30.csv"
  [60]="data/train_data/intermediate_line_data_20260412_7snap_t60.csv"
  [120]="data/train_data/intermediate_line_data_20260412_7snap_t120.csv"
  [240]="data/train_data/intermediate_line_data_20260412_7snap_t240.csv"
  [360]="data/train_data/intermediate_line_data_20260412_7snap_t360.csv"
  [480]="data/train_data/intermediate_line_data_20260412_7snap_t480.csv"
  [720]="data/train_data/intermediate_line_data_20260412_7snap_t720.csv"
)
SNAPSHOTS=(30 60 120 240 360 480 720)

NAMES=("intermediate_pooled_7snapshot_line_error_no_time_decay")
CONFIGS=("${CONFIG_DIR}/pooled_7snapshot_line_error_no_time_decay.yaml")
RUNNERS=("snapshot")
DATASETS=("$POOLED_DATASET")

for snap in "${SNAPSHOTS[@]}"; do
  NAMES+=("intermediate_t${snap}_control_line_error_no_time_decay")
  CONFIGS+=("${CONFIG_DIR}/t${snap}_control_line_error_no_time_decay.yaml")
  RUNNERS+=("cli")
  DATASETS+=("${SNAPSHOT_DATASETS[$snap]}")
done

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="artifacts/logs/${CAMPAIGN}_${STAMP}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"

PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
SNAPSHOT=("${PY[@]}" scripts/run_intermediate_snapshot_experiment.py)

log() { echo "$@" | tee -a "$LOG"; }

has_complete_artifact() {
  local name="$1"
  local runner="$2"
  local run_dir
  local run_dirs=()

  # A run directory is created before every terminal report is written. Only
  # skip an experiment when its required end-of-run artifacts are present;
  # otherwise a failed snapshot-report step would be skipped forever.
  shopt -s nullglob
  run_dirs=(
    artifacts/experiments/"${name}"_20[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]
  )
  shopt -u nullglob

  for run_dir in "${run_dirs[@]}"; do
    if [[ "$runner" == "snapshot" ]]; then
      if [[ -s "$run_dir/snapshot_cv_metrics.csv" \
            && -s "$run_dir/snapshot_holdout_metrics.csv" ]]; then
        return 0
      fi
    elif [[ -s "$run_dir/final_test_metrics.json" \
            && -s "$run_dir/cv_betting_summary.json" ]]; then
      return 0
    fi
  done
  return 1
}

log "=========================================================="
log " Campaign : ${CAMPAIGN}"
log " Started  : $(date)"
log " Runs     : ${#CONFIGS[@]} (1 pooled, 7 single-snapshot controls: 30/60/120/240/360/480/720 min)"
log " Target   : LINE_ERROR only"
log " Decay    : disabled in every config"
log " Skip old : ${SKIP_EXISTING:-1}"
log " Logs     : ${LOG_DIR}"
log "=========================================================="

log ""
log "Datasets"
for i in "${!DATASETS[@]}"; do
  dataset="${DATASETS[$i]}"
  if [[ ! -s "$dataset" ]]; then
    log "ABORT: missing or empty dataset: $dataset"
    log "Build all eight with:"
    log "  bash ${CONFIG_DIR}/prepare_line_error_7snapshot_all_controls.sh"
    exit 1
  fi
  checksum="$("${PY[@]}" -c \
    "from training_pipeline.data import compute_file_checksum as c; print(c('$dataset'))" \
    2>/dev/null)"
  log "  $dataset ($(du -h "$dataset" | cut -f1), $checksum)"
done

# Validate each config with the same entry point that will ultimately execute
# it. This catches schema mistakes without starting Optuna.
log ""
log "Validating configs..."
for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  runner="${RUNNERS[$i]}"
  if [[ ! -f "$cfg" ]]; then
    log "ABORT: missing config $cfg"
    exit 1
  fi
  if [[ "$runner" == "snapshot" ]]; then
    if ! "${SNAPSHOT[@]}" "$cfg" --dry-run > /dev/null 2>>"$LOG"; then
      log "ABORT: $cfg failed snapshot-wrapper validation. See $LOG"
      exit 1
    fi
  elif ! "${CLI[@]}" "$cfg" --dry-run > /dev/null 2>>"$LOG"; then
    log "ABORT: $cfg failed CLI validation. See $LOG"
    exit 1
  fi
  log "  ok  ${NAMES[$i]}"
done

# Unlike a directory-wide preflight, passing the eight YAML paths checks only
# this matched comparison group even though the configs live beside the older
# six-snapshot and 6h/4h-only campaign definitions.
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  log ""
  log "Running the targeted data/window preflight..."
  if ! "${PY[@]}" scripts/preflight_campaign.py "${CONFIGS[@]}" 2>&1 \
      | tee -a "$LOG"; then
    log "ABORT: preflight failed. No experiment was started."
    exit 1
  fi
else
  log ""
  log "WARNING: SKIP_PREFLIGHT=1; actual fold sizes were not verified."
fi

FAILED=0
COMPLETED=0
SKIPPED=0
CAMPAIGN_START=$SECONDS

# Sequential on purpose: each XGBoost run already owns the GPU budget.
for i in "${!CONFIGS[@]}"; do
  name="${NAMES[$i]}"
  cfg="${CONFIGS[$i]}"
  runner="${RUNNERS[$i]}"
  run_log="${LOG_DIR}/${name}.log"

  if [[ "${SKIP_EXISTING:-1}" == "1" ]] \
     && has_complete_artifact "$name" "$runner"; then
    log ""
    log "SKIP  ${name}  (artifacts already exist)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  log ""
  log "----------------------------------------------------------"
  log "START $(date +%H:%M:%S)  ${name} [$((COMPLETED + FAILED + SKIPPED + 1))/${#CONFIGS[@]}]"
  log "  config: ${cfg}"
  log "  runner: ${runner}"
  log "  detail: ${run_log}"
  log "----------------------------------------------------------"

  START=$SECONDS
  if [[ "$runner" == "snapshot" ]]; then
    if "${SNAPSHOT[@]}" "$cfg" --no-save-model > "$run_log" 2>&1; then
      STATUS="OK"
      COMPLETED=$((COMPLETED + 1))
    else
      STATUS="FAILED"
      FAILED=$((FAILED + 1))
      log "  !! failed -- last 15 lines:"
      tail -15 "$run_log" | sed 's/^/     /' | tee -a "$LOG"
    fi
  elif "${CLI[@]}" "$cfg" --no-save-model > "$run_log" 2>&1; then
    STATUS="OK"
    COMPLETED=$((COMPLETED + 1))
  else
    STATUS="FAILED"
    FAILED=$((FAILED + 1))
    log "  !! failed -- last 15 lines:"
    tail -15 "$run_log" | sed 's/^/     /' | tee -a "$LOG"
  fi

  ELAPSED=$((SECONDS - START))
  printf 'END   %s  %-7s (%dh %02dm)  %s\n' \
    "$(date +%H:%M:%S)" "$STATUS" $((ELAPSED / 3600)) \
    $(((ELAPSED % 3600) / 60)) "$name" | tee -a "$LOG"
done

TOTAL=$((SECONDS - CAMPAIGN_START))
log ""
log "=========================================================="
log " Finished $(date)"
log " ${COMPLETED} ok, ${FAILED} failed, ${SKIPPED} skipped, in $((TOTAL / 3600))h $(((TOTAL % 3600) / 60))m"
log "=========================================================="
log ""
log "Read the pooled run from snapshot_cv_metrics.csv and"
log "snapshot_holdout_metrics.csv, never from its pooled ALL row."
log ""
log "Planned comparisons (check fold dates in the preflight/report):"
log "  1. pooled T=30  (30m) vs t30 control"
log "  2. pooled T=60  (1h)  vs t60 control"
log "  3. pooled T=120 (2h)  vs t120 control"
log "  4. pooled T=240 (4h)  vs t240 control"
log "  5. pooled T=360 (6h)  vs t360 control"
log "  6. pooled T=480 (8h)  vs t480 control"
log "  7. pooled T=720 (12h) vs t720 control -- the pair the earlier"
log "     campaign disagreed on, now on matched bytes and 7 seeds"
log "  8. holdout ROI only as a smaller-sample sanity check"

exit "$FAILED"
