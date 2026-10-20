#!/usr/bin/env bash
# Evaluation only: closing plus 16 intermediate horizons. No model promotion.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
TARGET="spread_error"
CAMPAIGN="retrain_${TARGET}_2_5_2026_10"
CONFIG_DIR="experiments/$CAMPAIGN"
ARTIFACT_ROOT="artifacts/experiments/$CAMPAIGN"
LOG_DIR="artifacts/logs/${CAMPAIGN}_$(date +%Y%m%d_%H%M%S)"
PREFLIGHT_ONLY=0
SKIP_EXISTING=${SKIP_EXISTING:-0}
for arg in "$@"; do
  case "$arg" in
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    --skip-existing) SKIP_EXISTING=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY=("$PYTHON_BIN" -u)
elif [[ -x .venv/bin/python ]]; then
  PY=(.venv/bin/python -u)
else
  PY=(poetry run python -u)
fi
# Optional CPU affinity on the GPU host; e.g. CPUSET=0-7 bash <this script>.
if [[ -n "${CPUSET:-}" ]]; then PY=(taskset -c "$CPUSET" "${PY[@]}"); fi
CONFIGS=("$CONFIG_DIR/closing_${TARGET}.yaml")
for horizon in 240 720 60 480 960 30 360 840 120 600 1080 180 540 300 660 420; do
  CONFIGS+=("$CONFIG_DIR/intermediate_${TARGET}_t${horizon}.yaml")
done
if [[ ! -f "$CONFIG_DIR/datasets.sha256" ]]; then
  echo "NOT READY: pin both rebuilt datasets and scoring; see $CONFIG_DIR/README.md" >&2
  exit 1
fi
sha256sum --check "$CONFIG_DIR/datasets.sha256" || exit 1
mkdir -p "$LOG_DIR" "$ARTIFACT_ROOT/.completed"
exec 9>"$ARTIFACT_ROOT/.campaign.lock"
flock -n 9 || { echo "$CAMPAIGN is already running" >&2; exit 1; }
LOG="$LOG_DIR/campaign.log"
log() { echo "$*" | tee -a "$LOG"; }
log "$CAMPAIGN started $(date); 17 models, 150 trials/model, evaluation only"
# Full data-dependent preflight for EVERY cell before spending GPU time.
"${PY[@]}" scripts/preflight_campaign.py "${CONFIGS[@]}" 2>&1 | tee "$LOG_DIR/preflight.log"
PREFLIGHT_STATUS=${PIPESTATUS[0]}
if (( PREFLIGHT_STATUS != 0 )); then
  log "ABORT: preflight failed (exit $PREFLIGHT_STATUS). No training started."
  exit 1
fi
if (( PREFLIGHT_ONLY == 1 )); then log "Full preflight passed."; exit 0; fi
"${PY[@]}" - <<'PY' > "$LOG_DIR/cuda_check.log" 2>&1
import json
import numpy as np
import xgboost as xgb
model = xgb.train(
    {"device": "cuda", "tree_method": "hist", "nthread": 1},
    xgb.DMatrix(np.arange(60, dtype=float).reshape(20, 3), label=np.arange(20)),
    num_boost_round=1,
)
device = json.loads(model.save_config())["learner"]["generic_param"]["device"]
if not device.startswith("cuda"):
    raise RuntimeError(f"XGBoost fell back to {device}; a working CUDA GPU is required")
print(f"CUDA verified: {device}")
PY
if (( $? != 0 )); then
  log "ABORT: CUDA check failed; see $LOG_DIR/cuda_check.log"
  exit 1
fi
FAILED=0
for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  receipt="$ARTIFACT_ROOT/.completed/$name.done"
  signature=$(sha256sum "$cfg" experiments/_base.yaml "$CONFIG_DIR/datasets.sha256" | sha256sum | cut -d' ' -f1)
  saved_signature=""; saved_run=""
  if [[ -f "$receipt" ]]; then
    { read -r saved_signature; read -r saved_run; } < "$receipt"
  fi
  if [[ "$SKIP_EXISTING" == 1 && "$saved_signature" == "$signature" \
        && -f "$saved_run/seed_stability.csv" && -f "$saved_run/cv_betting_summary.json" \
        && -f "$saved_run/final_test_metrics.json" && -f "$saved_run/config.json" \
        && -f "$saved_run/optuna_selected_trial.json" ]]; then
    log "SKIP $name (matching successful evaluation)"
    continue
  fi
  experiment_name=$("${PY[@]}" -c 'import sys; from training_pipeline.cli import load_config; print(load_config(sys.argv[1]).experiment_name)' "$cfg")
  if [[ -z "$experiment_name" ]]; then log "FAILED: cannot load $cfg"; FAILED=$((FAILED + 1)); continue; fi
  log "START $name; log=$LOG_DIR/$name.log"
  "${PY[@]}" -m training_pipeline.cli "$cfg" --no-save-model > "$LOG_DIR/$name.log" 2>&1
  RUN_STATUS=$?
  run_dir=$(ls -dt "$ARTIFACT_ROOT/${experiment_name}_"20* 2>/dev/null | head -1)
  if (( RUN_STATUS == 0 )) && [[ -f "$run_dir/seed_stability.csv" \
       && -f "$run_dir/final_test_metrics.json" && -f "$run_dir/cv_betting_summary.json" \
       && -f "$run_dir/config.json" && -f "$run_dir/optuna_selected_trial.json" ]]; then
    printf '%s\n%s\n' "$signature" "$run_dir" > "$receipt"
    log "OK $name: $run_dir"
  else
    FAILED=$((FAILED + 1))
    log "FAILED $name (exit $RUN_STATUS or incomplete artifacts); see $LOG_DIR/$name.log"
    tail -20 "$LOG_DIR/$name.log" | tee -a "$LOG"
  fi
done
log "Finished $CAMPAIGN: $FAILED failed. Logs: $LOG_DIR"
(( FAILED == 0 ))
