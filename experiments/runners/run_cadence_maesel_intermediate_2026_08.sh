#!/usr/bin/env bash
# LANE 1 of retrain_cadence_maesel_2026_08: pair 1, line error at the six-hour
# intermediate snapshot. The 4 game-day / 25-game control against the daily
# origin with a 5-game floor.
#
# Lane 2 (experiments/runners/run_cadence_maesel_closing_2026_08.sh) holds the
# other two pairs, both on the closing dataset. The two lanes touch DIFFERENT
# datasets and different runs, so they can be started together:
#
#   nohup bash experiments/runners/run_cadence_maesel_intermediate_2026_08.sh > /dev/null 2>&1 &
#   nohup bash experiments/runners/run_cadence_maesel_closing_2026_08.sh      > /dev/null 2>&1 &
#
# WHAT RUNNING THEM CONCURRENTLY DOES AND DOES NOT CHANGE. Results: nothing.
# Every fit is seeded from random_state, the data and splits are fixed, and
# neither lane reads the other's artifacts. Wall clock: both lanes slow down,
# because they share one GPU -- expect noticeably more than the single-lane
# estimates below. Host RAM is the real risk: this lane alone loads a 1.8 GB
# CSV and cleans it, so if the box is tight, run the lanes one after the other
# instead. The per-cell hours below are single-lane measurements from
# retrain_cadence_2026_08 on the same hardware.
#
# Estimated single-lane total: ~10.5h (a ~2h15m, b ~7h45m, plus ~2 extra
# holdout walk-forwards per cell for evaluation_seeds).
#
# Resume after an interruption without repeating a completed cell:
#   SKIP_EXISTING=1 bash experiments/runners/run_cadence_maesel_intermediate_2026_08.sh
#
# The control runs FIRST and is much the cheaper of the two, so an interrupted
# lane still leaves the reference cell on disk.
set -uo pipefail  # One failed cell must not cancel the remaining experiments.

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="retrain_cadence_maesel_2026_08"
LANE="intermediate"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_${LANE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"

CONFIGS=(
  "$CONFIG_DIR/a_t360_line_error_4d.yaml"
  "$CONFIG_DIR/b_t360_line_error_1d.yaml"
)

DATASETS=(
  "data/train_data/intermediate_line_data_10snap.csv"
  "data/train_data/intermediate_line_data_10snap_scoring.csv"
)

PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
log() { echo "$@" | tee -a "$LOG"; }

log "=========================================================="
log "Campaign : ${CAMPAIGN}  (lane: ${LANE})"
log "Started  : $(date)"
log "Runs     : ${#CONFIGS[@]} sequential runs, 150 trials each, CUDA"
log "Variable : walk_forward.retrain_every_days 4 -> 1 (fold floor 25 -> 5)"
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
  log "Running preflight on THIS LANE's configs only -- the other lane checks"
  log "its own, so neither pays for the other's dataset. The 1.8GB pooled file"
  log "makes this slow, but it verifies the checksum, the cleaning and every"
  log "requested game window, and it prints the realised FOLD COUNT and"
  log "validation-game count for both cadences. Those two numbers decide"
  log "whether the pair is readable at all: cv_n_validation_games must match."
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
log "     are 5-15-game folds. If that repeats, it IS the finding about daily"
log "     folds and the pair is not an equal-search cadence test."
log "  2. NOISE. Read seed_stability.csv first, not the headline. The measured"
log "     single-seed holdout ROI spread on ONE FIXED config is 4.9-12.0"
log "     points; compare the 3-seed MEANS and treat anything inside the range"
log "     as nothing."
log "  3. VOLUME. ~410 holdout games per cell. A Wilson interval on that cannot"
log "     clear the 52.38% break-even at -110 whatever it shows."
log "  4. AGREEMENT, across pairs. Three pairs is three replicates of ONE"
log "     question. A result is a cadence gap with the SAME SIGN in all three,"
log "     surviving the seed range. One pair out of three looking good is the"
log "     expected yield of luck at this noise floor."
log ""
log "Survey both lanes together:"
log "  SOURCES = ['experiments/${CAMPAIGN}']  in"
log "  experiments/notebooks/survey_experiments.ipynb"
log "=========================================================="

exit "$FAILED"
