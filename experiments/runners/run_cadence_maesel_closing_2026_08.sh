#!/usr/bin/env bash
# LANE 2 of retrain_cadence_maesel_2026_08: pairs 2 and 3, both on the CLOSING
# dataset. Pair 2 is line error, pair 3 is total points; each is the 4 game-day
# / 25-game control against the daily origin with a 5-game floor.
#
# Lane 1 (experiments/runners/run_cadence_maesel_intermediate_2026_08.sh) holds
# pair 1, on the intermediate T-360 dataset. The two lanes touch DIFFERENT
# datasets and different runs, so they can be started together:
#
#   nohup bash experiments/runners/run_cadence_maesel_intermediate_2026_08.sh > /dev/null 2>&1 &
#   nohup bash experiments/runners/run_cadence_maesel_closing_2026_08.sh      > /dev/null 2>&1 &
#
# WHAT RUNNING THEM CONCURRENTLY DOES AND DOES NOT CHANGE. Results: nothing.
# Every fit is seeded from random_state, the data and splits are fixed, and
# neither lane reads the other's artifacts. Wall clock: both lanes slow down,
# because they share one GPU. Host RAM is the real risk -- lane 1 loads a 1.8 GB
# CSV -- so on a tight box run the lanes one after the other. The estimates
# below are single-lane.
#
# ORDER: both CONTROLS first, then both daily cells. The controls are ~4x
# cheaper, so an interrupted lane still leaves a reference cell for each pair on
# disk rather than one complete pair and one empty one.
#
# Estimated single-lane total: ~13h (controls ~1h30m each, daily cells ~5h
# each, at 150 trials on the closing dataset -- ~34 s/trial measured on
# horizon_*_2026_08 -- plus 2 extra holdout walk-forwards per cell).
#
# Resume after an interruption without repeating a completed cell:
#   SKIP_EXISTING=1 bash experiments/runners/run_cadence_maesel_closing_2026_08.sh
set -uo pipefail  # One failed cell must not cancel the remaining experiments.

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="retrain_cadence_maesel_2026_08"
LANE="closing"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_${LANE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"

CONFIGS=(
  "$CONFIG_DIR/c_close_line_error_4d.yaml"
  "$CONFIG_DIR/e_close_total_points_4d.yaml"
  "$CONFIG_DIR/d_close_line_error_1d.yaml"
  "$CONFIG_DIR/f_close_total_points_1d.yaml"
)

DATASETS=(
  "data/train_data/training_data_2_0_20260819.csv"
)

PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
log() { echo "$@" | tee -a "$LOG"; }

log "=========================================================="
log "Campaign : ${CAMPAIGN}  (lane: ${LANE})"
log "Started  : $(date)"
log "Runs     : ${#CONFIGS[@]} sequential runs, 150 trials each, CUDA"
log "Variable : walk_forward.retrain_every_days 4 -> 1 (fold floor 25 -> 5)"
log "Targets  : line_error (pair 2) and total_points (pair 3), same dataset"
log "Fixed    : lowest-CV-MAE trial selection, evaluation_seeds [101, 202]"
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
  log "Running preflight on THIS LANE's configs only -- lane 1 checks its own,"
  log "so neither pays for the other's dataset. It verifies the checksum, the"
  log "cleaning and every requested game window, and prints the realised FOLD"
  log "COUNT and validation-game count per cell. cv_n_validation_games must"
  log "match WITHIN each pair or that pair is unreadable; the 4500-game window"
  log "must fit the earliest fold or the run refuses to start."
  if ! "${PY[@]}" scripts/preflight_campaign.py "${CONFIGS[@]}" 2>&1 | tee -a "$LOG"; then
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
log "Lane ${LANE} finished $(date): ${COMPLETED} ok, ${FAILED} failed, ${SKIPPED} skipped"
log ""
log "READ IN THIS ORDER, and only once BOTH lanes are done. Stop at the first"
log "failure -- later questions are meaningless once an earlier one fails."
log "  1. VALIDITY, per pair. metadata.json: cv_n_validation_games must match"
log "     within the pair, and tie_n_completed must be comparable. On"
log "     2026-08-25 the daily cell completed 49 of 150 trials against the"
log "     control's 66, because MedianPruner reads a running metric whose steps"
log "     are 5-15-game folds. If that repeats here, it IS the finding about"
log "     daily folds and the pair is not an equal-search cadence test."
log "  2. NOISE. Read seed_stability.csv first, not the headline. Compare the"
log "     3-seed MEANS; the measured single-seed holdout ROI spread on ONE"
log "     FIXED config is 4.9-12.0 points."
log "  3. VOLUME. ~416 holdout games per cell on this dataset. A Wilson"
log "     interval on that cannot clear the 52.38% break-even at -110."
log "  4. AGREEMENT, across pairs. Three pairs is three replicates of ONE"
log "     question. A result is a cadence gap with the SAME SIGN in all three,"
log "     surviving the seed range. Do NOT compare a cell in pair 2 against a"
log "     cell in pair 3 directly -- different targets, and MAE is not even on"
log "     the same scale. Only the cadence GAP travels between pairs."
log ""
log "Survey both lanes together:"
log "  SOURCES = ['experiments/${CAMPAIGN}']  in"
log "  experiments/notebooks/survey_experiments.ipynb"
log "=========================================================="

exit "$FAILED"
