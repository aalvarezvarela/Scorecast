#!/usr/bin/env bash
# Schema-2.5 intermediate line-error promotion sweep, part1 t30 t420.
#
# One promotion candidate per pre-game horizon, T-30 .. T-1080 (T-0 is covered by
# the closing-line slot). Every cell shares one recipe -- 2019 floor, max_na 800,
# the longest window every fold supports at that horizon, 6x100 CV, 150 trials,
# seed 16, evaluation seeds 101/202 -- so the cells differ only in the horizon.
# See experiments/promote_intermediate_line_error_2_5_2026_09/README.md.
#
# The other half is run_promote_intermediate_line_error_2_5_2026_09_part2_t480_t1080.sh; the controls are run_promote_intermediate_line_error_2_5_2026_09_part3_controls.sh.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="promote_intermediate_line_error_2_5_2026_09"
PART="part1_t30_t420"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_${PART}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"
CONFIGS=(
  "$CONFIG_DIR/p_t30_line_error.yaml"
  "$CONFIG_DIR/p_t60_line_error.yaml"
  "$CONFIG_DIR/p_t120_line_error.yaml"
  "$CONFIG_DIR/p_t180_line_error.yaml"
  "$CONFIG_DIR/p_t240_line_error.yaml"
  "$CONFIG_DIR/p_t300_line_error.yaml"
  "$CONFIG_DIR/p_t360_line_error.yaml"
  "$CONFIG_DIR/p_t420_line_error.yaml"
)
PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
log() { echo "$@" | tee -a "$LOG"; }

log "$CAMPAIGN $PART started $(date)"
log "Sequential CUDA runs; 150 trials each; train_games FIXED at each horizon's maximum."
log "Seed 16; evaluation seeds 101, 202 throughout."
log "Logs: $LOG_DIR"

# DELIBERATELY UNLOCKED: the parts run concurrently and compete for memory
# instead of queueing. One raw load of the 4.6GB intermediate CSV peaks near
# 14GB of RSS on this 15GB box, so if both parts load a CSV in the same ~2
# minutes, one may be OOM-killed -- the exit-status reporting below names signal
# 9 when that happens, and SKIP_EXISTING=1 resumes the part that lost. To
# serialise the parts instead, add to BOTH, here, before the pre-flight:
#
#   LOCK_FILE="artifacts/logs/.${CAMPAIGN}.lock"
#   exec 9>"$LOCK_FILE"
#   flock 9

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
