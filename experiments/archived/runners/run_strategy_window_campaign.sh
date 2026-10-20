#!/usr/bin/env bash
# Overnight campaign: 3 prediction strategies x 2 training windows, 6 runs.
#
#   bash experiments/archived/runners/run_strategy_window_campaign.sh
#
# Detached, so closing the terminal does not kill it:
#   nohup bash experiments/archived/runners/run_strategy_window_campaign.sh > /dev/null 2>&1 &
#
# Follow along:
#   tail -f artifacts/logs/campaign_*/campaign.log      # the summary
#   tail -f artifacts/logs/campaign_*/<experiment>.log  # one run in detail
#
# Interrupted? Re-run with SKIP_EXISTING=1 to continue where it stopped:
#   SKIP_EXISTING=1 bash experiments/archived/runners/run_strategy_window_campaign.sh
#
# NOT `set -e`: one failing run must not cancel the remaining five. An overnight
# window is too expensive to lose to a single bad config.
set -uo pipefail

cd "$(dirname "$0")/../../.." || exit 1

CAMPAIGN="strategy_window_2026_08"
CONFIG_DIR="experiments/archived/${CAMPAIGN}"
DATASET="data/train_data/training_data_2_0_20260704.csv"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="artifacts/logs/campaign_${STAMP}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"

# Cheapest and most informative first. If the night is cut short you still have
# the strategy comparison at the reference window, which is the primary question;
# the 3750 runs only refine it.
CONFIGS=(
  "line_error_2500"      # pre-registered favourite
  "total_points_2500"    # the incumbent it has to beat
  "classifier_2500"      # the new strategy
  "line_error_3750"
  "total_points_3750"
  "classifier_3750"
)

# -u keeps stdout unbuffered so `tail -f` shows progress live rather than in
# 8KB bursts.
PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)

log() { echo "$@" | tee -a "$LOG"; }

log "=========================================================="
log " Campaign : ${CAMPAIGN}"
log " Started  : $(date)"
log " Runs     : ${#CONFIGS[@]}   (~1.2-1.8 h each on 4 CPUs)"
log " Logs     : ${LOG_DIR}"
log "=========================================================="

# --- the dataset must exist, and its identity gets recorded ----------------
if [[ ! -s "$DATASET" ]]; then
  log "ABORT: dataset missing or empty: $DATASET"
  log "  Regenerate it with:"
  log "    poetry run python scripts/create_train_data/create_train_data.py --limit 2026-07-04"
  exit 1
fi

CHECKSUM="$("${PY[@]}" -c \
  "from training_pipeline.data import compute_file_checksum as c; print(c('$DATASET'))" \
  2>/dev/null)"
log ""
log "Dataset : $DATASET ($(du -h "$DATASET" | cut -f1))"
log "Checksum: ${CHECKSUM}"
log "  ^ paste into data.expected_checksum in the six configs to be told loudly"
log "    if the CSV is ever rebuilt underneath this campaign."

# The overtime filter needs the column these configs' CSV must now carry. It is
# off for this campaign, but a missing column means the CSV predates the change
# and the follow-up overtime campaign would fail after hours of work.
if ! head -1 "$DATASET" | tr ',' '\n' | grep -qx "IS_OVERTIME"; then
  log ""
  log "WARNING: IS_OVERTIME is not in this CSV, so it predates the pipeline change."
  log "  This campaign runs fine (it does not filter overtime), but the follow-up"
  log "  overtime comparison will need the CSV regenerated first."
fi

# --- validate every config BEFORE committing the night to them --------------
log ""
log "Validating configs..."
for name in "${CONFIGS[@]}"; do
  cfg="${CONFIG_DIR}/${name}.yaml"
  if [[ ! -f "$cfg" ]]; then
    log "ABORT: missing config $cfg"
    exit 1
  fi
  if ! "${CLI[@]}" "$cfg" --dry-run > /dev/null 2>>"$LOG"; then
    log "ABORT: $cfg failed validation. See $LOG"
    exit 1
  fi
  log "  ok  $name"
done

# --- run them, one after another -------------------------------------------
# Sequential on purpose: XGBoost already saturates every core (n_jobs=-1), so
# running two at once would only make both slower and muddy the timings.
FAILED=0
COMPLETED=0
SKIPPED=0
CAMPAIGN_START=$SECONDS

for name in "${CONFIGS[@]}"; do
  cfg="${CONFIG_DIR}/${name}.yaml"
  run_log="${LOG_DIR}/${name}.log"

  # Match ONLY the exact run-dir suffix `_YYYYMMDD_HHMMSS`. A loose `${name}_*`
  # also matches longer experiment names that merely start the same way --
  # `total_points_2500_old_games_...` from July would make this skip
  # `total_points_2500` and silently drop it from the campaign.
  if [[ "${SKIP_EXISTING:-0}" == "1" ]] \
     && compgen -G "artifacts/experiments/${name}_20[0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]" > /dev/null; then
    log ""
    log "SKIP  ${name}  (artifacts already exist)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi

  log ""
  log "----------------------------------------------------------"
  log "START $(date +%H:%M:%S)  ${name}   [$((COMPLETED + FAILED + SKIPPED + 1))/${#CONFIGS[@]}]"
  log "  detail: ${run_log}"
  log "----------------------------------------------------------"

  START=$SECONDS
  # --no-save-model: this is an evaluation campaign, not a deployment. Promote a
  # winner afterwards with `python -m training_pipeline.promote <run_dir>`.
  if "${CLI[@]}" "$cfg" --no-save-model > "$run_log" 2>&1; then
    STATUS="OK"
    COMPLETED=$((COMPLETED + 1))
  else
    STATUS="FAILED"
    FAILED=$((FAILED + 1))
    log "  !! failed -- last 15 lines:"
    tail -15 "$run_log" | sed 's/^/     /' | tee -a "$LOG"
  fi
  ELAPSED=$((SECONDS - START))

  printf 'END   %s  %-7s (%dh %02dm)  %s\n' \
    "$(date +%H:%M:%S)" "$STATUS" $((ELAPSED / 3600)) $(((ELAPSED % 3600) / 60)) "$name" \
    | tee -a "$LOG"
done

TOTAL=$((SECONDS - CAMPAIGN_START))
log ""
log "=========================================================="
log " Finished $(date)"
log " ${COMPLETED} ok, ${FAILED} failed, ${SKIPPED} skipped, in $((TOTAL / 3600))h $(((TOTAL % 3600) / 60))m"
log "=========================================================="
log ""
log "Compare them in experiments/compare_experiments.ipynb with:"
log "    EXPERIMENT_IDS   = []"
log "    TARGET_FAMILY    = None"
log "    COMPARISON_GROUP = \"${CAMPAIGN}\""
log ""
log "Read in this order:"
log "  1. Section 6  seed_roi_range -- a gap under ~4.9pp is not a result"
log "  2. Section 5  cv_roi         -- the ranking metric (~600 games)"
log "  3. Section 3  roi            -- out-of-sample sanity check (~166 bets)"
log "  4. Section 13 cv_log_loss_improvement -- judge the classifier here FIRST;"
log "                if it is negative the probabilities carry no information"
log "                and its ROI is noise."

exit $FAILED
