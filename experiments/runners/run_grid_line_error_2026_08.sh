#!/usr/bin/env bash
# Eight experiments, sequential on one CUDA machine: six line-error horizon
# cells plus two of the campaign's four classifier cells.
#
# TWO RUNNERS, STARTED TOGETHER, EIGHT CELLS EACH. Only two fit on this GPU at
# once, and the two streams are deliberately the same size so they finish
# together: six line-error horizon cells plus two of the four classifier cells.
# That split is a GPU-slot arrangement, NOT a comparison group.
# reporting/factors.py matches on config and never on directory, so read any
# classifier cell against the other three across BOTH folders, and each
# regressor cell against the same horizon in the other folder.
#
# THE CLASSIFIER SCAN IS COARSE ON PURPOSE: four horizons (0/120/360/720), not
# ten. It answers "does the horizon matter at all", not "which horizon is
# best", and no single one of the four should be named a winner. T-360 is a
# replicate of the horizon campaign's cell at the same snapshot and is the only
# whole-pipeline noise estimate this campaign produces, seeds being off.
#
# BOTH REGRESSOR FAMILIES RUN, total points included, even though it lost to
# line error on holdout MAE in 9 of 9 paired cells last campaign and its deficit
# was worst on exactly this dataset. Six new horizons per family is the test of
# whether that was the formulation or those four snapshots.
#
#   bash experiments/runners/run_grid_line_error_2026_08.sh
#
# Detached:
#   nohup bash experiments/runners/run_grid_line_error_2026_08.sh > /dev/null 2>&1 &
#
# Resume after an interruption without repeating completed cells:
#   SKIP_EXISTING=1 bash experiments/runners/run_grid_line_error_2026_08.sh
#
# ORDER. Both classifier cells first -- they are ~4h each against a regressor's
# ~1-1.5h, so front-loading them means an interrupted weekend loses cheap cells
# rather than expensive ones -- then the six regressor cells, which complete one
# whole horizon curve.
#
# NO SEEDS, ANYWHERE. evaluation_seeds is [] in every cell, so no run in this
# campaign has an error bar of its own. What replaces it is the SHAPE of a
# curve -- six regressor points per family, four classifier points -- whose
# cells differ only in snapshot_minutes, plus the T-360 classifier replicate
# of the horizon campaign. The horizon
# campaign measured a single-seed ROI spread of 0.9-11.4 points on one fixed
# config, so a curve wandering less than that is flat. Do not read one cell.
#
# 150 TRIALS, down from 300. Justified on the horizon campaign's own 20 runs:
# the best objective reachable within the first 150 trials matched the 300-trial
# best in 13 of 20 cells, median gap 0.000 MAE and worst case 0.026 -- inside
# that campaign's seed MAE spread of 0.017-0.110.
set -uo pipefail  # One failed cell must not cancel the remaining experiments.

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="grid_line_error_2026_08"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"

CONFIGS=(
  "$CONFIG_DIR/a_classifier_intermediate_t0.yaml"
  "$CONFIG_DIR/b_classifier_intermediate_t360.yaml"
  "$CONFIG_DIR/c_line_error_intermediate_t0.yaml"
  "$CONFIG_DIR/d_line_error_intermediate_t30.yaml"
  "$CONFIG_DIR/e_line_error_intermediate_t60.yaml"
  "$CONFIG_DIR/f_line_error_intermediate_t180.yaml"
  "$CONFIG_DIR/g_line_error_intermediate_t300.yaml"
  "$CONFIG_DIR/h_line_error_intermediate_t480.yaml"
)

DATASETS=(
  "data/train_data/intermediate_line_data_10snap.csv"
  "data/train_data/intermediate_line_data_10snap_scoring.csv"
)

PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
log() { echo "$@" | tee -a "$LOG"; }

log "=========================================================="
log "Campaign : ${CAMPAIGN}"
log "Started  : $(date)"
log "Runs     : ${#CONFIGS[@]} sequential runs, 150 trials each, 1 seed each, CUDA"
log "Estimate : ~15h. Start the other grid runner at the same time."
log "Logs     : ${LOG_DIR}"
log "=========================================================="

for dataset in "${DATASETS[@]}"; do
  if [[ ! -s "$dataset" ]]; then
    log "ABORT: required dataset missing or empty: $dataset"
    exit 1
  fi
  log "Dataset ready: $dataset ($(du -h "$dataset" | cut -f1))"
done

# XGBoost can warn and silently fall back to CPU when CUDA support is absent.
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
  log "Running campaign preflight. The 1.8GB snapshot dataset makes this slow,"
  log "but it verifies checksums, cleaning and every requested game window --"
  log "and every cell here reads a DIFFERENT snapshot of it."
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
log "READ IN THIS ORDER, and stop at the first question that fails:"
log "  1. THE CURVE, NOT A CELL. Plot classifier holdout win rate against"
log "     snapshot_minutes across all FOUR cells from BOTH folders. No cell here"
log "     has a seed error bar, so the curve's own scatter is the only noise"
log "     estimate you have. Flat within that scatter = no horizon effect, and"
log "     that is a perfectly good answer."
log "  2. Volume, before any win rate. At -110 break-even is 52.38%, and the"
log "     classifier bets only 33-44% of games -- 134-182 bets on this holdout,"
log "     where a Wilson interval spans roughly +/-8 points. Compute it first."
log "  3. Classifier against the regressor at the SAME horizon, remembering the"
log "     cohort differs: pushes have no classifier label and are dropped."
log "  4. line_error against total_points at the same horizon. On the horizon"
log "     campaign line error won 9 of 9 on holdout MAE and the gap widened as"
log "     the line got harder to reproduce. If that does not reproduce across"
log "     these six new horizons, the earlier result was the four snapshots."
log "  5. Only then, ROI -- and only as the next campaign's hypothesis."
log ""
log "16 cells across the two streams. Against this noise floor expect one or two"
log "to look good by luck, and with seeds off you cannot tell which."
log "=========================================================="

exit "$FAILED"
