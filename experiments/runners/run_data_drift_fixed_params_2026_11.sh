#!/usr/bin/env bash
# Data drift at fixed hyperparameters: do the 2_5 rebuilds since 20261016
# (#64, #86, #107, #112) change the production line_error models? Six cells:
# close / T-480 / T-720 x two arms (old = the 20261016 files, new = the S3
# season partitions of 2026-11-05 restricted to the same rows). See
# experiments/data_drift_fixed_params_2026_11/README.md.
#
# No Optuna: every cell uses the production spec's hyperparameters, so the two
# arms of a horizon differ only in their data file. Five paired seeds per cell,
# then seed-replicate cells so CV has per-seed values too.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="data_drift_fixed_params_2026_11"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_$(date +%Y%m%d_%H%M%S)${RUN_TAG:+_$RUN_TAG}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"
CONFIGS=(
  "$CONFIG_DIR/close_old.yaml"
  "$CONFIG_DIR/close_new.yaml"
  "$CONFIG_DIR/t480_old.yaml"
  "$CONFIG_DIR/t480_new.yaml"
  "$CONFIG_DIR/t720_old.yaml"
  "$CONFIG_DIR/t720_new.yaml"
)
# Seed replicates (random_state 101..404, no evaluation seeds): CV is scored
# with the primary seed only, so these give CV its per-seed noise.
for cell in close t480 t720; do
  for seed in 101 202 303 404; do
    for arm in old new; do CONFIGS+=("$CONFIG_DIR/${cell}_${arm}_s${seed}.yaml"); done
  done
done
# Optional subset, to run two runners side by side on disjoint cells:
#   RUN_TAG=b bash run_data_drift_fixed_params_2026_11.sh t720_old t720_new
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
log "Production hyperparameters, no tuning; seeds 16/17 + 101/202/303/404, paired across arms."
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
