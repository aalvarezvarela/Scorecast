#!/usr/bin/env bash
# Report text with Optuna: does each team's last-game report embedding plus its
# mean over the last 5 reports add anything to line_error once both arms are
# tuned? Six cells: T-480 / T-720 x (control, control replicate, text).
# See experiments/report_text_optuna_2026_11/README.md.
#
# Every cell re-tunes the production recipe with the same fixed number of
# Optuna trials (not a timeout, so arms that share the machine with the
# predictor get the same budget). The cells run niced and with a raised OOM
# score, so under memory pressure the kernel kills a cell, not a predictor run.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="report_text_optuna_2026_11"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_$(date +%Y%m%d_%H%M%S)${RUN_TAG:+_$RUN_TAG}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"
CONFIGS=(
  "$CONFIG_DIR/t720_a_control.yaml"
  "$CONFIG_DIR/t720_t_text.yaml"
  "$CONFIG_DIR/t720_a2_control.yaml"
  "$CONFIG_DIR/t480_a_control.yaml"
  "$CONFIG_DIR/t480_t_text.yaml"
  "$CONFIG_DIR/t480_a2_control.yaml"
)
# Optional subset, to run several runners side by side on disjoint cells:
#   RUN_TAG=a bash run_report_text_optuna_2026_11.sh t720_a_control t480_a_control
if (( $# > 0 )); then
  CONFIGS=()
  for name in "$@"; do CONFIGS+=("$CONFIG_DIR/${name}.yaml"); done
fi
# PYTHON: an interpreter whose environment has the repo's dependencies. The
# repository root and src/ go first on the path, so this checkout's code runs
# even when the environment's editable install points at another checkout.
PY=("${PYTHON:-poetry run python}" -u)
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
CLI=(${PY[@]} -m training_pipeline.cli)
log() { echo "$@" | tee -a "$LOG"; }

echo "${OOM_SCORE_ADJ:-800}" > /proc/self/oom_score_adj 2>/dev/null || true

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

log "$CAMPAIGN started $(date)"
log "Optuna re-tuning, same n_trials per cell; seeds 16 (a, t) / 17 (a2) + 101/202."
log "Python: ${PY[*]}; code: $PWD ($(git rev-parse --short HEAD))"
log "Logs: $LOG_DIR"

FAILED=0
for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  wait_for_memory_gap
  run_log="$LOG_DIR/${name}.log"
  log "START $(date +%H:%M:%S) $name"
  START=$SECONDS
  nice -n "${NICE:-10}" "${CLI[@]}" "$cfg" --no-save-model > "$run_log" 2>&1
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
