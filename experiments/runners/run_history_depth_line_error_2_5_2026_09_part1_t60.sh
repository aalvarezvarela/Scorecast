#!/usr/bin/env bash
# Schema-2.5 history-depth A/B for line error at T-60: three (season floor, NaN budget) arms.
# Arm d (2021/800) is NOT run at this horizon. Measured 2026-09-22, it cleans to
# the same 6,165 games as arm a (2021/500), because max_na_per_row is applied
# after the season-gated columns are dropped and so binds on nothing at the 2021
# floor. Same rows, same pruning, same seed: the cell would reproduce arm a
# exactly. It survives at T-720 (part 2), where it admits 13 more games.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="history_depth_line_error_2_5_2026_09"
PART="part1_t60"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_${PART}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"
CONFIGS=(
  "$CONFIG_DIR/l_t60_a_floor2021_na500.yaml"
  "$CONFIG_DIR/l_t60_b_floor2020_na700.yaml"
  "$CONFIG_DIR/l_t60_c_floor2019_na800.yaml"
)
PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
log() { echo "$@" | tee -a "$LOG"; }

log "$CAMPAIGN $PART started $(date)"
log "Sequential CUDA runs; 150 trials each; seed 16 plus evaluation seeds 101, 202."
log "Baseline arm (a) runs first so an interrupted part still leaves the reference behind it."
log "Logs: $LOG_DIR"

CUDA_CHECK="$("${PY[@]}" -c "
import warnings
import numpy as np
import xgboost
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    xgboost.train(
        {'device': 'cuda'},
        xgboost.DMatrix(np.random.rand(50, 3), label=np.random.rand(50)),
        num_boost_round=2,
    )
    print('CPU_FALLBACK' if any('not compiled with CUDA' in str(w.message)
                                for w in caught) else 'CUDA_OK')
" 2>/dev/null)"
if [[ "$CUDA_CHECK" != "CUDA_OK" ]]; then
  log "ABORT: XGBoost CUDA check failed (${CUDA_CHECK:-no output})."
  exit 1
fi
log "CUDA verified."

# One raw load of the 4.6GB intermediate CSV peaks near 14GB of RSS, so the two
# parts of this campaign cannot share this 15GB box. On 2026-09-22 part 2's
# pre-flight was SIGKILLed by the OOM killer while part 1 held the memory, and
# the log said only "Killed". This lock serialises the parts rather than letting
# the kernel pick which one dies: launching both at once is still fine, the
# second simply waits.
LOCK_FILE="artifacts/logs/.${CAMPAIGN}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  log "Another part of $CAMPAIGN holds the memory lock; waiting for it to finish."
  flock 9
fi
log "Memory lock held (parts run one at a time)."

# Full preflight, not --skip-data: the per-arm window ceiling is the whole point.
"${PY[@]}" scripts/preflight_campaign.py "${CONFIGS[@]}" 2>&1 | tee -a "$LOG"
PREFLIGHT_STATUS=${PIPESTATUS[0]}
if (( PREFLIGHT_STATUS != 0 )); then
  if (( PREFLIGHT_STATUS > 128 )); then
    log "ABORT: pre-flight killed by signal $((PREFLIGHT_STATUS - 128)) (exit $PREFLIGHT_STATUS)."
    log "       Signal 9 is the OOM killer -- cleaning this CSV peaks near 14GB and"
    log "       nothing else on this box may hold memory while it runs."
  else
    log "ABORT: campaign preflight failed (exit $PREFLIGHT_STATUS)."
  fi
  exit 1
fi

FAILED=0
SKIPPED=0
for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  experiment_name="$("${PY[@]}" -c \
    "from training_pipeline.cli import load_config; print(load_config('$cfg').experiment_name)" \
    2>/dev/null)"
  if [[ "${SKIP_EXISTING:-0}" == "1" ]] \
     && compgen -G "artifacts/experiments/${CAMPAIGN}/${experiment_name}_20*" > /dev/null; then
    log "SKIP $name (artifact already exists)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi
  run_log="$LOG_DIR/${name}.log"
  log "START $(date +%H:%M:%S) $name"
  START=$SECONDS
  "${CLI[@]}" "$cfg" --no-save-model > "$run_log" 2>&1
  RUN_STATUS=$?
  if (( RUN_STATUS == 0 )); then
    STATUS="OK"
  else
    STATUS="FAILED(exit $RUN_STATUS)"
    FAILED=$((FAILED + 1))
    # A signalled run leaves no traceback in the log, so name the signal here:
    # 9 is the OOM killer, which is what a second memory-hungry process causes.
    if (( RUN_STATUS > 128 )); then
      log "  killed by signal $((RUN_STATUS - 128))$( ((RUN_STATUS == 137)) && echo ' -- OOM killer' )"
    fi
    tail -30 "$run_log" | tee -a "$LOG"
  fi
  ELAPSED=$((SECONDS - START))
  log "END $(date +%H:%M:%S) $STATUS ($((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m) $name"
done
log "Finished $(date): $FAILED failed, $SKIPPED skipped"
exit "$FAILED"
