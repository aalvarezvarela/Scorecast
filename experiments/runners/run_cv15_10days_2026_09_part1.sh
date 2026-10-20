#!/usr/bin/env bash
# Part 1: closing, schema 2.2. Experiments run sequentially; this runner never saves a production model.
set -euo pipefail
cd "$(dirname "$0")/../.."

CAMPAIGN="cv15_10days_2026_09"
PART="part1"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_${PART}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"
CONFIGS=(
  "$CONFIG_DIR/a_closing_spread_2_2.yaml"
  "$CONFIG_DIR/b_closing_total_points_2_2.yaml"
  "$CONFIG_DIR/c_closing_line_error_2_2.yaml"
)
PY=(poetry run python -u)
log() { echo "$@" | tee -a "$LOG"; }

log "$CAMPAIGN $PART started $(date)"
log "Three targets; 80 trials each; seed 16; CUDA."
log "CV: latest 15 folds, nominally 10 game-days; daily holdout: 90 calendar days."
log "Selector: pooled MAE quantile band (floor .001, cap .04), then pooled directional accuracy."
log "Pruning disabled effectively: warmup 16 exceeds the 15 folds."
log "Closing history admitted from season 2019."
log "Logs: $LOG_DIR"

# Check the actual booster device: XGBoost may silently fall back to CPU.
if ! "${PY[@]}" - <<'PY' 2>&1 | tee -a "$LOG"
import json
import numpy as np
import xgboost as xgb

rng = np.random.default_rng(16)
booster = xgb.train(
    {"device": "cuda", "tree_method": "hist", "seed": 16},
    xgb.DMatrix(rng.normal(size=(50, 3)), label=rng.normal(size=50)),
    num_boost_round=2,
)
device = json.loads(booster.save_config())["learner"]["generic_param"]["device"]
if not device.startswith("cuda"):
    raise SystemExit(f"CUDA required; effective device is {device}")
print(f"CUDA verified: {device}; XGBoost {xgb.__version__}")
PY
then
  log "ABORT: CUDA check failed."
  exit 1
fi

# Includes real cleaning/splits and checks that every requested X fits.
if ! "${PY[@]}" scripts/preflight_campaign.py "${CONFIGS[@]}" 2>&1 | tee -a "$LOG"; then
  log "ABORT: campaign preflight failed."
  exit 1
fi

FAILED=0
for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  run_log="$LOG_DIR/${name}.log"
  log "START $(date +%H:%M:%S) $name"
  START=$SECONDS
  if "${PY[@]}" -m training_pipeline.cli "$cfg" --no-save-model > "$run_log" 2>&1; then
    STATUS="OK"
  else
    STATUS="FAILED"
    FAILED=$((FAILED + 1))
    tail -n 20 "$run_log" | tee -a "$LOG"
  fi
  ELAPSED=$((SECONDS - START))
  log "END $(date +%H:%M:%S) $STATUS ($((ELAPSED / 60)) min) $name"
done
log "Finished $(date): $FAILED failed"
exit "$FAILED"

