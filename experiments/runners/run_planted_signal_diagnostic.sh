#!/usr/bin/env bash
# Planted-signal diagnostic for line_error -- 4 runs, sequential.
#
# DIAGNOSTIC CAMPAIGN. Every run here carries a synthetic feature derived from
# the target. The numbers measure whether this pipeline can find a signal it is
# handed; they say nothing about live performance and no run here can be
# promoted (training_pipeline.promote refuses them by name).
#
#   bash experiments/runners/run_planted_signal_diagnostic.sh
#
# Detached (survives closing the terminal):
#   nohup bash experiments/runners/run_planted_signal_diagnostic.sh > /dev/null 2>&1 &
#
# Follow along:
#   tail -f artifacts/logs/planted_signal_*.log
#
# Compare afterwards:
#   poetry run python scripts/compare_planted_signal.py
#
# Budget: 100 trials x 30 folds, reduced from the real campaign's 200 to cut
# runtime. That is ~12,000 fits across the four cells before pruning, plus 3
# holdout evaluations each. Answers a slightly different question than "can
# our protocol find it at full budget" -- say so when reporting results.
#
# To sanity-check the machinery before committing that, run the single most
# discriminating cell on its own -- it is the one whose failure would already
# answer the question:
#   poetry run python -m training_pipeline.cli \
#     experiments/diagnostics_planted_signal_2026_08/signal_020.yaml --no-save-model
# Lowering optuna.n_trials in the YAMLs also works, but answers a reduced
# question and must be said out loud when reporting the result.
set -uo pipefail          # NOT -e: one failed run must not cancel the rest

cd "$(dirname "$0")/../.." || exit 1

LOG_DIR="artifacts/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/planted_signal_$(date +%Y%m%d_%H%M%S).log"

CONFIG_DIR="experiments/diagnostics_planted_signal_2026_08"
# Ordered strongest-first. If the campaign dies overnight, the cell that
# actually discriminates -- 2% planted, the easy positive control -- is the one
# already finished, and a failure there indicts the protocol on its own.
CONFIGS=(
  "$CONFIG_DIR/signal_020.yaml"     # easy positive control
  "$CONFIG_DIR/control_000.yaml"    # the control it must be read against
  "$CONFIG_DIR/signal_010.yaml"     # the main test
  "$CONFIG_DIR/signal_005.yaml"     # weak
)

DATASETS=(
  "data/train_data/training_data_2_0_20260704.csv"
)

# -u keeps output unbuffered so `tail -f` shows progress live.
PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)

log() { echo "$@" | tee -a "$LOG"; }

log "=========================================================="
log "PLANTED-SIGNAL DIAGNOSTIC -- started $(date)"
log "  ${#CONFIGS[@]} runs. Synthetic target-derived feature: PLANTED_SIGNAL."
log "  These runs measure the PIPELINE, not the market. Not promotable."
log "  planted variance: 0.0 / 0.005 / 0.01 / 0.02 of LINE_ERROR variance"
log "=========================================================="

for dataset in "${DATASETS[@]}"; do
  if [[ ! -s "$dataset" ]]; then
    log "ABORT: required dataset missing or empty: $dataset"
    exit 1
  fi
  log "Dataset ready: $dataset ($(du -h "$dataset" | cut -f1))"
done

# --- pre-flight gates the campaign -----------------------------------------
log ""
if [[ "${SKIP_PREFLIGHT:-0}" == "1" ]]; then
  # The slow part is cleaning the 394MB CSV (~8 min), which the campaign is
  # about to do anyway. Skipping is reasonable when an identical data + fold
  # configuration has already passed -- and unreasonable otherwise, because the
  # check it removes is the one that catches a training window silently
  # shrinking rather than erroring.
  log "SKIP_PREFLIGHT=1: skipping the pre-flight window check."
  log "  Only safe if the same dataset and walk_forward settings already passed."
else
  log "Running pre-flight (this cleans the dataset once, so it takes ~8 min)..."
  log "  Skip with: SKIP_PREFLIGHT=1 bash $0"
  if ! "${PY[@]}" scripts/preflight_campaign.py "$CONFIG_DIR" 2>&1 | tee -a "$LOG"; then
    log "ABORT: pre-flight failed. Fix the reported problems before running."
    exit 1
  fi
fi

# --- the runs --------------------------------------------------------------
FAILED=0
for cfg in "${CONFIGS[@]}"; do
  log ""
  log "----------------------------------------------------------"
  log "START $(date +%H:%M:%S)  $(basename "$cfg")"
  log "----------------------------------------------------------"

  START=$SECONDS
  # --no-save-model is belt AND braces: the config already refuses to train a
  # production model on a diagnostic run, and run_experiment refuses the
  # save_model override too.
  if "${CLI[@]}" "$cfg" --no-save-model 2>&1 | tee -a "$LOG"; then
    STATUS="OK"
  else
    STATUS="FAILED"
    FAILED=$((FAILED + 1))
  fi
  ELAPSED=$((SECONDS - START))

  printf 'END   %s  %s  (%dh %02dm)  %s\n' \
    "$(date +%H:%M:%S)" "$STATUS" $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60)) \
    "$(basename "$cfg")" | tee -a "$LOG"
done

log ""
log "=========================================================="
log "Finished $(date) -- $FAILED of ${#CONFIGS[@]} runs failed"
log ""
log "Compare the cells:"
log "  poetry run python scripts/compare_planted_signal.py"
log ""
log "Read fold_use_rate first. If it is 0.00 at 2% planted variance, the tree"
log "builder never once split on a feature explaining 2% of the target across"
log "30 folds -- and the search space, not the market, is the constraint."
log "=========================================================="

exit $FAILED
