#!/usr/bin/env bash
# Two line-error cells at the six-hour intermediate snapshot, differing ONLY in
# the rolling-origin retrain cadence: the default 4 game-days / 25-game fold
# floor versus a daily origin with a 5-game floor.
#
#   bash experiments/runners/run_retrain_cadence_2026_08.sh
#
# Detached:
#   nohup bash experiments/runners/run_retrain_cadence_2026_08.sh > /dev/null 2>&1 &
#
# Resume after an interruption without repeating a completed cell:
#   SKIP_EXISTING=1 bash experiments/runners/run_retrain_cadence_2026_08.sh
#
# The control runs FIRST and is much the cheaper of the two, so an interrupted
# campaign still leaves the reference cell on disk.
set -uo pipefail  # One failed cell must not cancel the remaining experiments.

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="retrain_cadence_2026_08"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"

CONFIGS=(
  "$CONFIG_DIR/a_cadence_4d_25g.yaml"
  "$CONFIG_DIR/b_cadence_1d_5g.yaml"
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
log "Runs     : ${#CONFIGS[@]} sequential runs, 150 trials each, CUDA"
log "Variable : walk_forward.retrain_every_days 4 -> 1 (fold floor 25 -> 5)"
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
  log "Running campaign preflight. The 1.8GB pooled dataset makes this slow,"
  log "but it verifies checksums, cleaning and every requested game window --"
  log "and it prints the realised FOLD COUNT and validation-game count for both"
  log "cadences, which is the number that decides whether this pair is readable."
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
log "  1. VALIDITY. metadata.json in both runs: cv_n_validation_games must"
log "     match, and the completed-trial counts must be comparable. If cell B"
log "     completed far fewer trials, the pruner killed it early on tiny folds"
log "     and the pair received unequal search -- the comparison is void."
log "  2. NOISE. evaluation_seeds is OFF, so NEITHER cell has an error bar and"
log "     every holdout number is one seed. The measured single-seed"
log "     holdout ROI spread on ONE FIXED config is 4.9-12.0 points, so treat a"
log "     gap below ~10 points as nothing. Read CV first: both cells score the"
log "     same 850 validation games, which makes it the tighter comparison."
log "  3. VOLUME. ~410 holdout games, ~166 bets. A Wilson interval on that"
log "     cannot clear the 52.38% break-even at -110 whatever it shows."
log "  4. AGREEMENT. CV and holdout must move the same way. If they disagree,"
log "     the answer is 'no measurable effect', not the prettier number."
log ""
log "If the pair DOES separate and you intend to act on it, set"
log "evaluation_seeds: [101, 202] and rerun. That reuses the Optuna study (seeds"
log "are outside the fingerprint) and pays only for the re-evaluations."
log ""
log "Only then: does the daily origin justify ~4x the fits per trial? The"
log "expected answer is no, and a null result closes the question."
log "=========================================================="

exit "$FAILED"
