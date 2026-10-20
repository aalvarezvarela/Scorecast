#!/usr/bin/env bash
# Schema-2.5 early totals+spread windows, part2 (spread). Run both parts concurrently.
# Set SKIP_EXISTING=1 to resume completed configs.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="early_totals_spread_window_2_5_2026_09"
PART="part2_spread"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_${PART}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"
CONFIGS=(
  "$CONFIG_DIR/s_t0_spread.yaml"
  "$CONFIG_DIR/s_t60_spread.yaml"
  "$CONFIG_DIR/s_t180_spread.yaml"
  "$CONFIG_DIR/s_t300_spread.yaml"
  "$CONFIG_DIR/s_t420_spread.yaml"
  "$CONFIG_DIR/s_t540_spread.yaml"
  "$CONFIG_DIR/s_t660_spread.yaml"
  "$CONFIG_DIR/s_t840_spread.yaml"
  "$CONFIG_DIR/s_t1080_spread.yaml"
)
PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
log() { echo "$@" | tee -a "$LOG"; }

log "$CAMPAIGN $PART started $(date)"
log "${#CONFIGS[@]} sequential CUDA runs in this runner; the two runners may run concurrently."
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

if ! "${PY[@]}" scripts/preflight_campaign.py "${CONFIGS[@]}" --skip-data 2>&1 | tee -a "$LOG"; then
  log "ABORT: config/checksum preflight failed."
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
  if "${CLI[@]}" "$cfg" --no-save-model > "$run_log" 2>&1; then
    STATUS="OK"
  else
    STATUS="FAILED"
    FAILED=$((FAILED + 1))
    tail -30 "$run_log" | tee -a "$LOG"
  fi
  ELAPSED=$((SECONDS - START))
  log "END $(date +%H:%M:%S) $STATUS ($((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m) $name"
done
log "Finished $(date): $FAILED failed, $SKIPPED skipped"
exit "$FAILED"
