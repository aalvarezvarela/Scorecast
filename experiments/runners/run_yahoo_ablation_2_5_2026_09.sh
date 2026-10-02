#!/usr/bin/env bash
# Schema-2.5 Yahoo ablation: does line_error need the Yahoo public-betting
# features (pct_bets / pct_money)? Four cells, two pairs, one difference each:
#
#   a_close_with / a_close_without   closing line, hyperparameters of close6x100_25_ref
#   b_t240_with  / b_t240_without    T-240,        hyperparameters of promo25_t240_line_error
#
# No Optuna: every cell uses optuna.fixed_params from its reference run, so the
# pair differs only in the feature set (145 closing / 129 T-240 columns). Five
# seeds per cell (random_state 16 + evaluation seeds 101..404).
#
# Measured 2026-09-25 (validation folds and the 610 holdout games identical within
# each pair): dropping the columns lets 66 (closing) / 14 (T-240) extra 2019-20
# games through max_na_per_row. They only reach the oldest end of CV folds 1-5.
#
# The T-240 cells read a pre-sliced CSV (T-240 rows only, 286MB) so they can run
# beside the promote campaign instead of adding a 14GB load to the box.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="yahoo_ablation_2_5_2026_09"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"
CONFIGS=(
  "$CONFIG_DIR/a_close_with.yaml"
  "$CONFIG_DIR/a_close_without.yaml"
  "$CONFIG_DIR/b_t240_with.yaml"
  "$CONFIG_DIR/b_t240_without.yaml"
)
PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
log() { echo "$@" | tee -a "$LOG"; }

# Wait out a 4.6GB intermediate CSV load by another campaign: start only with
# >= MIN_AVAIL_GB free and no other training_pipeline process younger than 5 min.
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
log "Fixed hyperparameters, no tuning; seeds 16 + 101/202/303/404."
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
