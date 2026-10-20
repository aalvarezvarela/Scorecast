#!/usr/bin/env bash
# Public betting columns vs two extra seasons -- 3 runs, sequential.
#
#   bash experiments/runners/run_public_betting_tradeoff.sh
#
# Detached (survives closing the terminal):
#   nohup bash experiments/runners/run_public_betting_tradeoff.sh > /dev/null 2>&1 &
#
# Follow along:
#   tail -f artifacts/logs/public_betting_tradeoff_*.log
#
# Survey afterwards in experiments/notebooks/survey_experiments.ipynb with
#   SOURCES = ["experiments/public_betting_tradeoff_2026_08"]
# and RESCORE_EDGE_THRESHOLD = 0.1, so all three cells are scored at one common
# threshold rather than each at whichever one it froze into betting_metrics.json.
set -uo pipefail          # NOT -e: one failed run must not cancel the rest

cd "$(dirname "$0")/../.." || exit 1

LOG_DIR="artifacts/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/public_betting_tradeoff_$(date +%Y%m%d_%H%M%S).log"

CONFIG_DIR="experiments/public_betting_tradeoff_2026_08"
# Ordered so the two single-change contrasts land first. If the campaign dies
# overnight after two runs, A and B still answer "do public betting percentages
# earn their place?" on their own; C alone would answer nothing.
CONFIGS=(
  "$CONFIG_DIR/a_keep_columns.yaml"
  "$CONFIG_DIR/b_drop_columns.yaml"
  "$CONFIG_DIR/c_drop_and_extend.yaml"
)

DATASETS=(
  "data/train_data/training_data_2_0_20260819.csv"
)

# -u keeps output unbuffered so `tail -f` shows progress live.
PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)

log() { echo "$@" | tee -a "$LOG"; }

log "=========================================================="
log "Public betting tradeoff campaign -- started $(date)"
log "  3 runs, 3 evaluations each (primary + 2 seeds), 120 trials"
log "  A keep columns 2021+ | B drop columns 2021+ | C drop columns 2019+"
log "  measured layout: 30 folds, 117 game-days, 855 validation games,"
log "  416-game holdout -- identical in all three cells"
log "=========================================================="

for dataset in "${DATASETS[@]}"; do
  if [[ ! -s "$dataset" ]]; then
    log "ABORT: required dataset missing or empty: $dataset"
    exit 1
  fi
  log "Dataset ready: $dataset ($(du -h "$dataset" | cut -f1))"
done

# --- pre-flight gates the campaign -----------------------------------------
# The check that earns its keep here is cell C's window ceiling. C is the only
# cell allowed 5000- and 6000-game windows, and those are the entire mechanism
# by which its two extra seasons reach the model -- if they did not fit, that
# cell would quietly become a replication of cell B and the campaign would
# conclude "extra history does not help" having never used any.
log ""
if [[ "${SKIP_PREFLIGHT:-0}" == "1" ]]; then
  log "SKIP_PREFLIGHT=1: skipping the pre-flight window check."
  log "  Only safe if the same dataset and walk_forward settings already passed."
else
  log "Running pre-flight (cleans each distinct dataset once)..."
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
  # --no-save-model: this is an evaluation campaign, not a deployment. Promote
  # a winner afterwards with `python -m training_pipeline.promote <run_dir>`.
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
log "Judge on holdout MAE, not ROI. Runs are single-seed, so no cell carries"
log "an error bar. The floor measured 4.9-12.0 ROI points for one fixed"
log "config, so treat any smaller ROI gap as indistinguishable."
log "Then, in order:"
log "  A vs B  -- do public betting percentages earn their place?"
log "  B vs C  -- does bubble/COVID history help once they are gone?"
log "Only if C wins BOTH is the production switch worth flipping. Also check"
log "whether C's selected train_games landed above 4000: if it did not, the"
log "extra seasons were available and the tuner declined them, which is itself"
log "the answer."
log "=========================================================="

exit $FAILED
