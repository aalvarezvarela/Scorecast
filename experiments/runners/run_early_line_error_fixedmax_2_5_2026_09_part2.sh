#!/usr/bin/env bash
# Schema-2.5 line-error, top-6 horizons with the training window FIXED at each
# horizon's maximum, with and without tuned time decay. part2.
# Run both parts concurrently. Set SKIP_EXISTING=1 to resume completed configs.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="early_line_error_fixedmax_2_5_2026_09"
PART="part2"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_${PART}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"
# Each horizon's no-decay baseline runs immediately before its decay arm, so a
# part that is interrupted still leaves complete contrasts behind it.
CONFIGS=(
  "$CONFIG_DIR/l_t540_line_error_fixedmax.yaml"
  "$CONFIG_DIR/l_t540_line_error_fixedmax_decay.yaml"
  "$CONFIG_DIR/l_t240_line_error_fixedmax.yaml"
  "$CONFIG_DIR/l_t240_line_error_fixedmax_decay.yaml"
  "$CONFIG_DIR/l_t660_line_error_fixedmax.yaml"
  "$CONFIG_DIR/l_t660_line_error_fixedmax_decay.yaml"
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

# --skip-data only: the window question this campaign exists to ask is already
# settled. train_games is fixed at the value the tuned campaign measured as the
# maximum, and the fixed path accepts a fold on exactly the same condition the
# tuned path does (len(pool) >= min_train_games), so the folds are the ones the
# tuned runs used. Set PREFLIGHT_FULL=1 to re-verify that with the slow check.
PREFLIGHT_ARGS=(--skip-data)
if [[ "${PREFLIGHT_FULL:-0}" == "1" ]]; then
  PREFLIGHT_ARGS=()
fi
if ! "${PY[@]}" scripts/preflight_campaign.py "${CONFIGS[@]}" "${PREFLIGHT_ARGS[@]}" 2>&1 | tee -a "$LOG"; then
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
