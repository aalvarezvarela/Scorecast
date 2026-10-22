#!/usr/bin/env bash
# The three promoted closing configurations (line_error, total_points,
# spread_error; schema 2_5), each fitted twice with its selected hyperparameters
# fixed -- no Optuna -- differing ONLY in device: CUDA versus CPU. Same seeds in
# both cells (random_state + evaluation_seeds 101, 202).
#
# Question: the daily refit and promote's production fit now run on CPU. CUDA
# and CPU XGBoost build different models from one seed, and these
# configurations were only ever evaluated on CUDA. Does the CPU fit score the
# same?
#
#   bash experiments/runners/run_cpu_vs_cuda_closing_2_5_2026_10.sh
#
# Detached, surviving the terminal or session that started it:
#   setsid nohup bash experiments/runners/run_cpu_vs_cuda_closing_2_5_2026_10.sh > /dev/null 2>&1 &
#
# Resume after an interruption without repeating a completed cell:
#   SKIP_EXISTING=1 SKIP_PREFLIGHT=1 setsid nohup bash experiments/runners/run_cpu_vs_cuda_closing_2_5_2026_10.sh > /dev/null 2>&1 &
#
# Pairs run back to back (CUDA then CPU) so an interrupted campaign still leaves
# complete pairs on disk.
set -uo pipefail  # One failed cell must not cancel the remaining experiments.

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="cpu_vs_cuda_closing_2_5_2026_10"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"

CONFIGS=(
  "$CONFIG_DIR/le_cuda.yaml"
  "$CONFIG_DIR/le_cpu.yaml"
  "$CONFIG_DIR/tp_cuda.yaml"
  "$CONFIG_DIR/tp_cpu.yaml"
  "$CONFIG_DIR/sp_cuda.yaml"
  "$CONFIG_DIR/sp_cpu.yaml"
)

DATASETS=(
  "data/train_data/closing_line_data_2_5_20261016.parquet"
)

PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
log() { echo "$@" | tee -a "$LOG"; }

log "=========================================================="
log "Campaign : ${CAMPAIGN}"
log "Started  : $(date)"
log "PID      : $$"
log "Runs     : ${#CONFIGS[@]} sequential runs, fixed hyperparameters (no Optuna)"
log "Variable : device cuda -> cpu, everything else identical"
log "Logs     : ${LOG_DIR}"
log "=========================================================="

for dataset in "${DATASETS[@]}"; do
  if [[ ! -s "$dataset" ]]; then
    log "ABORT: required dataset missing or empty: $dataset"
    exit 1
  fi
  log "Dataset ready: $dataset ($(du -h "$dataset" | cut -f1))"
done

# XGBoost can warn and silently fall back to CPU when CUDA support is absent,
# which would turn every CUDA cell into a second CPU cell.
log ""
log "Verifying CUDA support..."
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

log ""
if [[ "${SKIP_PREFLIGHT:-0}" == "1" ]]; then
  log "SKIP_PREFLIGHT=1: skipping configuration/data preflight."
else
  log "Running campaign preflight (checksums, cleaning, per-fold windows)..."
  if ! "${PY[@]}" scripts/preflight_campaign.py "$CONFIG_DIR" 2>&1 | tee -a "$LOG"; then
    log "ABORT: preflight failed. No experiment was started."
    exit 1
  fi
fi

FAILED=0
COMPLETED=0
SKIPPED=0
for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  experiment_name="$("${PY[@]}" -c \
    "from training_pipeline.cli import load_config; print(load_config('$cfg').experiment_name)" \
    2>/dev/null)"

  if [[ "${SKIP_EXISTING:-0}" == "1" ]] \
     && compgen -G "artifacts/experiments/${CAMPAIGN}/${experiment_name}_20[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]" > /dev/null; then
    log ""
    log "SKIP  ${name} (an artifact already exists)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  run_log="$LOG_DIR/${name}.log"
  log ""
  log "----------------------------------------------------------"
  log "START $(date +%H:%M:%S)  ${name}"
  log "Detail: ${run_log}"
  log "----------------------------------------------------------"
  START=$SECONDS
  if "${CLI[@]}" "$cfg" --no-save-model > "$run_log" 2>&1; then
    STATUS="OK"
    COMPLETED=$((COMPLETED + 1))
  else
    STATUS="FAILED"
    FAILED=$((FAILED + 1))
    log "Last 20 lines from the failed run:"
    tail -20 "$run_log" | sed 's/^/  /' | tee -a "$LOG"
  fi
  ELAPSED=$((SECONDS - START))
  printf 'END   %s  %-7s (%dh %02dm)  %s\n' \
    "$(date +%H:%M:%S)" "$STATUS" $((ELAPSED / 3600)) \
    $(((ELAPSED % 3600) / 60)) "$name" | tee -a "$LOG"
done

log ""
log "=========================================================="
log "Finished $(date): ${COMPLETED} ok, ${FAILED} failed, ${SKIPPED} skipped"
log ""
log "READ IN THIS ORDER. Stop at the first failure."
log "  1. NO-OP CHECK. A _cpu cell byte-identical to its _cuda cell means the"
log "     device setting never reached the fits; the comparison is void."
log "  2. CONTROL. Each _cuda cell's primary-seed holdout should reproduce its"
log "     source run (artifacts/experiments/retrain_*_2_5_2026_10). If not, the"
log "     fixed-params path differs from the tuned one: compare _cpu with _cuda"
log "     only, never with the source run."
log "  3. NOISE. Each cell has three seeds (random_state, 101, 202). CPU counts"
log "     as 'the same' when its CV and holdout win rate / ROI fall inside the"
log "     CUDA cell's seed range. Read CV first: ~600 games, the tighter number."
log "  4. VOLUME. The holdout is ~610 games; a Wilson interval on it cannot"
log "     separate two configurations a couple of points apart."
log "=========================================================="

exit "$FAILED"
