#!/usr/bin/env bash
# Three of the seven no-decay experiments. Start part 1 and part 2 together.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="time_decay_2026_09"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_part2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"

CONFIGS=(
  "$CONFIG_DIR/b_closing_total_points.yaml"
  "$CONFIG_DIR/j_t720_spread_error.yaml"
  "$CONFIG_DIR/m_closing_spread_error_2_1.yaml"
)

PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
log() { echo "$@" | tee -a "$LOG"; }

DATASETS=(
  "data/train_data/training_data_2_2_20260901.csv"
  "data/train_data/training_data_2_1_20260828.csv"
  "data/train_data/intermediate_line_data_2_2_20260901.csv"
  "data/train_data/intermediate_line_data_2_2_20260901_scoring.csv"
)

log "No-decay campaign part 2 started $(date)"
log "Runs: ${#CONFIGS[@]} sequential CUDA experiments, 150 trials each"
log "Logs: $LOG_DIR"

for dataset in "${DATASETS[@]}"; do
  if [[ ! -s "$dataset" ]]; then
    log "ABORT: required dataset missing or empty: $dataset"
    exit 1
  fi
done

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
    tail -20 "$run_log" | tee -a "$LOG"
  fi
  ELAPSED=$((SECONDS - START))
  log "END $(date +%H:%M:%S) $STATUS ($((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m) $name"
done

log "Finished $(date): $FAILED failed, $SKIPPED skipped"
exit "$FAILED"
