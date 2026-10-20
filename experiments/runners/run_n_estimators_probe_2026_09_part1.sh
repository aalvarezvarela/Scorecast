#!/usr/bin/env bash
# Part 1: line-error controls on modern 2.0/2.2 and the historical July CSV,
# plus the modern-2.2 closing SPREAD cell (i), which is the first spread run
# measured after the rotation-depth leak was dropped in cleaning. Cell i runs
# last so a failure there cannot cost the four line-error controls.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="n_estimators_probe_2026_09"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_part1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"

CONFIGS=(
  "$CONFIG_DIR/a_modern20_line_early_stop.yaml"
  "$CONFIG_DIR/b_modern22_line_early_stop.yaml"
  "$CONFIG_DIR/c_legacy_line_early_stop.yaml"
  "$CONFIG_DIR/d_legacy_line_tuned_rounds.yaml"
  "$CONFIG_DIR/i_modern22_spread_early_stop.yaml"
)

PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
log() { echo "$@" | tee -a "$LOG"; }

log "n_estimators probe part 1 started $(date)"
log "Runs: ${#CONFIGS[@]} sequential CUDA experiments (4 line-error, 1 spread-error)"
log "Logs: $LOG_DIR"

LEGACY_DERIVED="artifacts/derived_data/training_data_2_0_20260704_odds_prefixed.csv"
if [[ ! -s "$LEGACY_DERIVED" ]]; then
  log "Preparing header-normalized historical CSV"
  if ! "${PY[@]}" scripts/prepare_legacy_odds_prefixed_csv.py 2>&1 | tee -a "$LOG"; then
    log "ABORT: historical CSV preparation failed."
    exit 1
  fi
fi

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

if ! "${PY[@]}" scripts/preflight_campaign.py "${CONFIGS[@]}" \
    --allow-short-training-windows 2>&1 | tee -a "$LOG"; then
  log "ABORT: campaign preflight failed."
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
