#!/usr/bin/env bash
# Report text, last game: does each team's previous-game report embedding add
# anything to line_error? Fifteen cells: close / T-480 / T-720 x five arms
# (control, target PCA 64, target residual PCA 16, raw PCA 64, raw residual
# PCA 16). See experiments/report_text_last_game_2026_11/README.md.
#
# No Optuna: every cell uses the production spec's hyperparameters, so arms of a
# horizon differ only in their text columns. Five seeds per cell. Controls run
# first, then each arm at the three horizons.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="report_text_last_game_2026_11"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_$(date +%Y%m%d_%H%M%S)${RUN_TAG:+_$RUN_TAG}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"
CONFIGS=(
  "$CONFIG_DIR/close_a_control.yaml"
  "$CONFIG_DIR/t480_a_control.yaml"
  "$CONFIG_DIR/t720_a_control.yaml"
  "$CONFIG_DIR/close_b_tgt64.yaml"
  "$CONFIG_DIR/t480_b_tgt64.yaml"
  "$CONFIG_DIR/t720_b_tgt64.yaml"
  "$CONFIG_DIR/close_c_tgt16r.yaml"
  "$CONFIG_DIR/t480_c_tgt16r.yaml"
  "$CONFIG_DIR/t720_c_tgt16r.yaml"
  "$CONFIG_DIR/close_d_raw64.yaml"
  "$CONFIG_DIR/t480_d_raw64.yaml"
  "$CONFIG_DIR/t720_d_raw64.yaml"
  "$CONFIG_DIR/close_e_raw16r.yaml"
  "$CONFIG_DIR/t480_e_raw16r.yaml"
  "$CONFIG_DIR/t720_e_raw16r.yaml"
)
# Optional subset, to run two runners side by side on disjoint cells:
#   RUN_TAG=b bash run_report_text_last_game_2026_11.sh close_b_tgt64 t480_b_tgt64 ...
# WAIT_PID: start only after that process has exited.
if (( $# > 0 )); then
  CONFIGS=()
  for name in "$@"; do CONFIGS+=("$CONFIG_DIR/${name}.yaml"); done
fi
PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
log() { echo "$@" | tee -a "$LOG"; }

# Start a cell only with >= MIN_AVAIL_GB free and no other training_pipeline
# process younger than 5 min.
MIN_AVAIL_GB="${MIN_AVAIL_GB:-6}"
wait_for_memory_gap() {
  while true; do
    local avail_gb young
    avail_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    young=$(ps -eo etimes=,args= | awk '/training_pipeline\.cli/ && !/awk/ && $1 < 300' | wc -l)
    if (( avail_gb >= MIN_AVAIL_GB && young == 0 )); then
      return
    fi
    log "  waiting: ${avail_gb}GB available, ${young} young training process(es)"
    sleep 60
  done
}

if [[ -n "${WAIT_PID:-}" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
log "$CAMPAIGN started $(date)"
log "Production hyperparameters, no tuning; seeds 16/17 + 101/202/303/404."
log "Logs: $LOG_DIR"

FAILED=0
for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  wait_for_memory_gap
  run_log="$LOG_DIR/${name}.log"
  log "START $(date +%H:%M:%S) $name"
  START=$SECONDS
  nice -n 5 "${CLI[@]}" "$cfg" --no-save-model > "$run_log" 2>&1
  RUN_STATUS=$?
  if (( RUN_STATUS == 0 )); then
    STATUS="OK"
  else
    STATUS="FAILED(exit $RUN_STATUS)"
    FAILED=$((FAILED + 1))
    if (( RUN_STATUS > 128 )); then
      log "  killed by signal $((RUN_STATUS - 128))$( ((RUN_STATUS == 137)) && echo ' -- OOM killer' )"
    fi
    tail -30 "$run_log" | tee -a "$LOG"
  fi
  ELAPSED=$((SECONDS - START))
  log "END $(date +%H:%M:%S) $STATUS ($((ELAPSED / 60))m) $name"
done
log "Finished $(date): $FAILED failed"
exit "$FAILED"
