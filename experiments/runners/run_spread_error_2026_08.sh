#!/usr/bin/env bash
# Closing, pooled, and six single-horizon spread-error experiments.
#
#   bash experiments/runners/run_spread_error_2026_08.sh
#   SKIP_EXISTING=1 bash experiments/runners/run_spread_error_2026_08.sh
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="spread_error_2026_08"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"

CONFIGS=(
  "$CONFIG_DIR/a_closing_reference.yaml"
  "$CONFIG_DIR/h_intermediate_t0.yaml"
  "$CONFIG_DIR/d_intermediate_t360.yaml"
  "$CONFIG_DIR/c_intermediate_t720.yaml"
  "$CONFIG_DIR/e_intermediate_t120.yaml"
  "$CONFIG_DIR/f_intermediate_t60.yaml"
  "$CONFIG_DIR/g_intermediate_t30.yaml"
  "$CONFIG_DIR/b_intermediate_pooled.yaml"
)
DATASETS=(
  "data/train_data/training_data_2_1_20260828.csv"
  "data/train_data/intermediate_line_data_2_1_20260828.csv"
  "data/train_data/intermediate_line_data_2_1_20260828_scoring.csv"
)

PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
log() { echo "$@" | tee -a "$LOG"; }

for dataset in "${DATASETS[@]}"; do
  if [[ ! -s "$dataset" ]]; then
    log "ABORT: required dataset missing or empty: $dataset"
    exit 1
  fi
done

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  if ! "${PY[@]}" scripts/preflight_campaign.py "$CONFIG_DIR" 2>&1 | tee -a "$LOG"; then
    log "ABORT: preflight failed."
    exit 1
  fi
fi

FAILED=0
for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  experiment_name="$("${PY[@]}" -c \
    "from training_pipeline.cli import load_config; print(load_config('$cfg').experiment_name)" \
    2>/dev/null)"
  if [[ "${SKIP_EXISTING:-0}" == "1" ]] \
     && compgen -G "artifacts/experiments/${CAMPAIGN}/${experiment_name}_20*" >/dev/null; then
    log "SKIP $name"
    continue
  fi
  log "START $name"
  if "${CLI[@]}" "$cfg" --no-save-model > "$LOG_DIR/${name}.log" 2>&1; then
    log "OK    $name"
  else
    log "FAIL  $name"
    tail -20 "$LOG_DIR/${name}.log" | tee -a "$LOG"
    FAILED=$((FAILED + 1))
  fi
done

log "Finished with ${FAILED} failed cell(s)."
exit "$FAILED"
