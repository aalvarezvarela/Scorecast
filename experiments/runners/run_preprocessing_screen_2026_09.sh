#!/usr/bin/env bash
# Six sequential fixed-hyperparameter CUDA cells for NaN/correlation screening.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="preprocessing_screen_2026_09"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"

CONFIGS=(
  "$CONFIG_DIR/a_na300_corr095.yaml"
  "$CONFIG_DIR/b_na80_corr095.yaml"
  "$CONFIG_DIR/c_na150_corr095.yaml"
  "$CONFIG_DIR/d_na300_corr099.yaml"
  "$CONFIG_DIR/e_na300_corr0995.yaml"
  "$CONFIG_DIR/f_na80_corr0995.yaml"
)

PY=(poetry run python -u)
log() { echo "$@" | tee -a "$LOG"; }

log "Preprocessing screen started $(date)"
log "Runs: ${#CONFIGS[@]} sequential fixed-hyperparameter CUDA cells"
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

if ! "${PY[@]}" scripts/preflight_campaign.py "${CONFIGS[@]}" 2>&1 | tee -a "$LOG"; then
  log "ABORT: campaign preflight failed."
  exit 1
fi

OUTPUT_DIR="artifacts/probes/${CAMPAIGN}"
RUN_LOG="$LOG_DIR/fixed_cv_screen.log"
log "START $(date +%H:%M:%S) fixed CV-only screen (holdout will not be scored)"
START=$SECONDS
if "${PY[@]}" scripts/probe_preprocessing_fixed_cv.py \
    "${CONFIGS[@]}" --output-dir "$OUTPUT_DIR" > "$RUN_LOG" 2>&1; then
  ELAPSED=$((SECONDS - START))
  log "END $(date +%H:%M:%S) OK ($((ELAPSED / 60))m $((ELAPSED % 60))s)"
  tail -30 "$RUN_LOG" | tee -a "$LOG"
  exit 0
fi

ELAPSED=$((SECONDS - START))
log "END $(date +%H:%M:%S) FAILED ($((ELAPSED / 60))m $((ELAPSED % 60))s)"
tail -50 "$RUN_LOG" | tee -a "$LOG"
exit 1
