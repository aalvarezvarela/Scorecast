#!/usr/bin/env bash
# Intermediate-line campaign: pooled snapshot models plus controls.
#
#   bash experiments/archived/runners/run_intermediate_line_campaign.sh
#
# Detached, so closing the terminal does not kill it:
#   nohup bash experiments/archived/runners/run_intermediate_line_campaign.sh > /dev/null 2>&1 &
#
# Follow along:
#   tail -f artifacts/logs/campaign_*/campaign.log      # the summary
#   tail -f artifacts/logs/campaign_*/<experiment>.log  # one run in detail
#
# Interrupted or extending the campaign? Existing experiment artifacts are
# skipped by default. Set SKIP_EXISTING=0 to force a full rerun:
#   SKIP_EXISTING=0 bash experiments/archived/runners/run_intermediate_line_campaign.sh
#
# The pooled configs MUST use scripts/run_intermediate_snapshot_experiment.py so
# their artifacts include per-snapshot betting reports. The single-snapshot
# controls are one row per game, so the ordinary CLI is the correct runner for
# those.
#
# NOT `set -e`: one failing run must not cancel the remaining campaign cells.
set -uo pipefail

cd "$(dirname "$0")/../../.." || exit 1

CAMPAIGN="intermediate_line_2026_08"
CONFIG_DIR="experiments/archived/${CAMPAIGN}"
POOLED_DATASET="data/train_data/intermediate_line_data_20260412.csv"
T720_DATASET="data/train_data/intermediate_line_data_20260412_t720.csv"
T480_DATASET="data/train_data/intermediate_line_data_20260412_t480.csv"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="artifacts/logs/campaign_${STAMP}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"

# Primary deliverables first. If the run is cut short, the pooled target
# comparison is still interpretable; controls only answer whether pooling is
# earning its complexity at the 12h horizon.
NAMES=(
  "intermediate_pooled_line_error"
  "intermediate_pooled_total_points"
  "intermediate_t720_control_line_error"
  "intermediate_t720_control_total_points"
  "intermediate_pooled_line_error_no_time_decay"
  "intermediate_t720_control_line_error_no_time_decay"
  "intermediate_t480_control_line_error"
)
CONFIGS=(
  "${CONFIG_DIR}/pooled_line_error.yaml"
  "${CONFIG_DIR}/pooled_total_points.yaml"
  "${CONFIG_DIR}/t720_control_line_error.yaml"
  "${CONFIG_DIR}/t720_control_total_points.yaml"
  "${CONFIG_DIR}/pooled_line_error_no_time_decay.yaml"
  "${CONFIG_DIR}/t720_control_line_error_no_time_decay.yaml"
  "${CONFIG_DIR}/t480_control_line_error.yaml"
)
RUNNERS=(
  "snapshot"
  "snapshot"
  "cli"
  "cli"
  "snapshot"
  "cli"
  "cli"
)
DATASETS=(
  "$POOLED_DATASET"
  "$POOLED_DATASET"
  "$T720_DATASET"
  "$T720_DATASET"
  "$POOLED_DATASET"
  "$T720_DATASET"
  "$T480_DATASET"
)

# -u keeps stdout unbuffered so `tail -f` shows progress live rather than in
# buffered bursts.
PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
SNAPSHOT=("${PY[@]}" scripts/run_intermediate_snapshot_experiment.py)

log() { echo "$@" | tee -a "$LOG"; }

log "=========================================================="
log " Campaign : ${CAMPAIGN}"
log " Started  : $(date)"
log " Runs     : ${#CONFIGS[@]}   (3 pooled snapshot runs, 4 controls)"
log " Skip old : ${SKIP_EXISTING:-1}"
log " Logs     : ${LOG_DIR}"
log "=========================================================="

log ""
if [[ -s "$POOLED_DATASET" ]]; then
  log "Dataset : $POOLED_DATASET ($(du -h "$POOLED_DATASET" | cut -f1))"
  log "Checksum: $("${PY[@]}" -c \
    "from training_pipeline.data import compute_file_checksum as c; print(c('$POOLED_DATASET'))" \
    2>/dev/null)"
  if ! head -1 "$POOLED_DATASET" | tr ',' '\n' | grep -qx "TIME_TO_MATCH_MIN"; then
    log "Dataset : $POOLED_DATASET is missing TIME_TO_MATCH_MIN"
  fi
else
  log "Dataset : $POOLED_DATASET (missing; pooled runs need it)"
fi

if [[ -s "$T720_DATASET" ]]; then
  log "Dataset : $T720_DATASET ($(du -h "$T720_DATASET" | cut -f1))"
  log "Checksum: $("${PY[@]}" -c \
    "from training_pipeline.data import compute_file_checksum as c; print(c('$T720_DATASET'))" \
    2>/dev/null)"
else
  log "Dataset : $T720_DATASET (missing; t720 controls need it)"
fi

if [[ -s "$T480_DATASET" ]]; then
  log "Dataset : $T480_DATASET ($(du -h "$T480_DATASET" | cut -f1))"
  log "Checksum: $("${PY[@]}" -c \
    "from training_pipeline.data import compute_file_checksum as c; print(c('$T480_DATASET'))" \
    2>/dev/null)"
else
  log "Dataset : $T480_DATASET (missing; only the t480 control needs it)"
fi
log "  ^ paste these into data.expected_checksum in the configs to be told"
log "    loudly if a CSV is ever rebuilt underneath this campaign."

# --- validate every config BEFORE committing the campaign -------------------
log ""
log "Validating configs..."
for i in "${!CONFIGS[@]}"; do
  name="${NAMES[$i]}"
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
  else
    if ! "${CLI[@]}" "$cfg" --dry-run > /dev/null 2>>"$LOG"; then
      log "ABORT: $cfg failed CLI validation. See $LOG"
      exit 1
    fi
  fi

  log "  ok  ${name}"
done

# --- run them, one after another -------------------------------------------
# Sequential on purpose: XGBoost already saturates the machine/GPU budget, so
# running two at once would only make both slower and muddy timings.
FAILED=0
COMPLETED=0
SKIPPED=0
CAMPAIGN_START=$SECONDS

for i in "${!CONFIGS[@]}"; do
  name="${NAMES[$i]}"
  cfg="${CONFIGS[$i]}"
  runner="${RUNNERS[$i]}"
  dataset="${DATASETS[$i]}"
  run_log="${LOG_DIR}/${name}.log"

  # Match ONLY the exact run-dir suffix `_YYYYMMDD_HHMMSS`. A loose prefix
  # check can skip a different experiment whose name merely starts the same way.
  if [[ "${SKIP_EXISTING:-1}" == "1" ]] \
     && compgen -G "artifacts/experiments/${name}_20[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]" > /dev/null; then
    log ""
    log "SKIP  ${name}  (artifacts already exist)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  if [[ ! -s "$dataset" ]]; then
    log ""
    log "FAIL  ${name}  (dataset missing or empty: ${dataset})"
    if [[ "$dataset" == "$T480_DATASET" ]]; then
      log "  Build it with:"
      log "    poetry run python scripts/create_train_data/slice_intermediate_snapshot.py --snapshot 480"
    fi
    FAILED=$((FAILED + 1))
    continue
  fi

  log ""
  log "----------------------------------------------------------"
  log "START $(date +%H:%M:%S)  ${name}   [$((COMPLETED + FAILED + SKIPPED + 1))/${#CONFIGS[@]}]"
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
  else
    if "${CLI[@]}" "$cfg" --no-save-model > "$run_log" 2>&1; then
      STATUS="OK"
      COMPLETED=$((COMPLETED + 1))
    else
      STATUS="FAILED"
      FAILED=$((FAILED + 1))
      log "  !! failed -- last 15 lines:"
      tail -15 "$run_log" | sed 's/^/     /' | tee -a "$LOG"
    fi
  fi
  ELAPSED=$((SECONDS - START))

  printf 'END   %s  %-7s (%dh %02dm)  %s\n' \
    "$(date +%H:%M:%S)" "$STATUS" $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60)) "$name" \
    | tee -a "$LOG"
done

TOTAL=$((SECONDS - CAMPAIGN_START))
log ""
log "=========================================================="
log " Finished $(date)"
log " ${COMPLETED} ok, ${FAILED} failed, ${SKIPPED} skipped, in $((TOTAL / 3600))h $(((TOTAL % 3600) / 60))m"
log "=========================================================="
log ""
log "Read the pooled runs from their snapshot_cv_metrics.csv and"
log "snapshot_holdout_metrics.csv files, not from the pooled betting row."
log ""
log "Survey with, in experiments/notebooks/survey_experiments.ipynb:"
log "  SOURCES = [\"experiments/archived/intermediate_line_2026_08\"]"
log ""
log "Primary read order:"
log "  1. pooled_line_error vs pooled_total_points by per-snapshot cv_roi"
log "  2. pooled_line_error at snapshot 720 vs t720_control_line_error cv_roi"
log "  3. pooled_line_error_no_time_decay vs pooled_line_error for decay"
log "  4. pooled_line_error at snapshot 480 vs t480_control_line_error cv_roi"
log "  5. holdout ROI only as a sanity check, because it is a smaller slice"

exit $FAILED
