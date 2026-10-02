#!/usr/bin/env bash
# Schema-2.5 closing-line line_error, always 6x100 test-anchored CV: seven cells,
# each changing ONE thing from the reference.
#
#   c_ref            NaN 300, window 6,200, ODDS_ corr 0.99, seed 16
#   c_ref_rep        identical, seed 17 -- the noise floor; read it first
#   c_tg4000         window 4,000         (also the control for the NaN cells)
#   c_tg2500         window 2,500
#   c_na150_tg4000   NaN budget 150 at window 4,000  -> read against c_tg4000
#   c_na80_tg4000    NaN budget 80  at window 4,000  -> read against c_tg4000
#   c_corr0995_tg4000 ODDS_ corr 0.995 at window 4,000  -> read against c_tg4000
#                    (changes features AND, via the NaN budget, drops 380 games;
#                    its ceiling is ~5,836, so it cannot run at 6,200)
#
# No lock. The closing CSV is 222MB, so cells peak at a few GB and can run
# alongside an intermediate campaign.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="closing_line_error_6x100_2_5_2026_09"
PART="closing"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"
# Reference and replicate first: an interrupted night still leaves the pair
# every other number is read against.
CONFIGS=(
  "$CONFIG_DIR/c_ref.yaml"
  "$CONFIG_DIR/c_ref_rep.yaml"
  "$CONFIG_DIR/c_tg4000.yaml"
  "$CONFIG_DIR/c_na150_tg4000.yaml"
  "$CONFIG_DIR/c_na80_tg4000.yaml"
  "$CONFIG_DIR/c_tg2500.yaml"
  "$CONFIG_DIR/c_corr0995_tg4000.yaml"
)
PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
log() { echo "$@" | tee -a "$LOG"; }

log "$CAMPAIGN started $(date)"
log "Sequential CUDA runs; 150 trials each; 6x100 folds; train_games fixed per cell."
log "Seed 16 except the replicate (17); evaluation seeds 101, 202 throughout."
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

# Full preflight, not --skip-data: the per-arm window ceiling is the whole point.
"${PY[@]}" scripts/preflight_campaign.py "${CONFIGS[@]}" 2>&1 | tee -a "$LOG"
PREFLIGHT_STATUS=${PIPESTATUS[0]}
if (( PREFLIGHT_STATUS != 0 )); then
  if (( PREFLIGHT_STATUS > 128 )); then
    log "ABORT: pre-flight killed by signal $((PREFLIGHT_STATUS - 128)) (exit $PREFLIGHT_STATUS)."
    log "       Signal 9 is the OOM killer -- most likely an intermediate campaign"
    log "       holding ~14GB alongside this one."
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
