#!/usr/bin/env bash
# Schema-2.5 training-window sweep for line error at T-60: does older data help?
#
# Six cells. Five fix train_games at 1,500 / 2,500 / 3,500 / 4,500 / 6,250 on one
# identical dataset; the sixth repeats the 3,500 arm under a different seed to
# measure how much of any gap is noise.
#
# This replaces history_depth_line_error_2_5_2026_09, which could not answer the
# question for two reasons, both fixed here:
#
#   1. It varied season_year_floor across arms. Cleaning runs per arm, so a
#      different floor meant a different correlation structure and a different
#      surviving feature set -- 1,380 features in one arm against 2,407 in
#      another. "More history" and "more features" moved together. Here the
#      floor is 2019 in EVERY arm, cleaning is therefore identical, and the only
#      thing that differs is how far back the window reaches.
#   2. It tuned train_games with Optuna, so the answer was an argmax. Arms a and
#      d at T-720 differed by 13 rows in 6,017 and picked windows 325 games
#      apart. Here the window is fixed per arm and the comparison is between
#      runs, with the replicate arm supplying the error bar.
#
# Read the results in this order: the 3,500-vs-3,500-replicate gap first -- that
# is the noise floor -- then any window difference larger than it. Nothing
# smaller counts.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="train_window_line_error_2_5_2026_09"
PART="part1_t60"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_${PART}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"
# Anchor and its replicate first: an interrupted part still leaves behind the
# pair that every other number has to be read against.
CONFIGS=(
  "$CONFIG_DIR/w_t60_g3500.yaml"
  "$CONFIG_DIR/w_t60_g3500_rep.yaml"
  "$CONFIG_DIR/w_t60_g1500.yaml"
  "$CONFIG_DIR/w_t60_g2500.yaml"
  "$CONFIG_DIR/w_t60_g4500.yaml"
  "$CONFIG_DIR/w_t60_g6250.yaml"
)
PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
log() { echo "$@" | tee -a "$LOG"; }

log "$CAMPAIGN $PART started $(date)"
log "Sequential CUDA runs; 150 trials each; train_games FIXED per arm (not tuned)."
log "Seed 16 except the replicate arm (17); evaluation seeds 101, 202 throughout."
log "Logs: $LOG_DIR"

# DELIBERATELY UNLOCKED, like part 2. Neither part takes a lock, so the two run
# concurrently and compete for memory instead of queueing. That is a choice, not
# an oversight: one raw load of the 4.6GB intermediate CSV peaks near 14GB of RSS
# against 15GB of physical memory plus 31GB of swap, so two at once either
# thrashes through swap or ends with the OOM killer taking one of them. The
# predecessor campaign died exactly that way on 2026-09-22, leaving only "Killed"
# in the log; the exit-status reporting below names the signal when it happens,
# and SKIP_EXISTING=1 resumes whichever part lost. The peaks only overlap while a
# cell loads its CSV, which is the first ~2 minutes of each cell, so a collision
# is possible rather than certain. To serialise the parts instead, add:
#
#   LOCK_FILE="artifacts/logs/.${CAMPAIGN}.lock"
#   exec 9>"$LOCK_FILE"
#   flock 9
#
# to BOTH parts, here, before the pre-flight.

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

# Full preflight, not --skip-data: the per-arm window ceiling is the whole point.
"${PY[@]}" scripts/preflight_campaign.py "${CONFIGS[@]}" 2>&1 | tee -a "$LOG"
PREFLIGHT_STATUS=${PIPESTATUS[0]}
if (( PREFLIGHT_STATUS != 0 )); then
  if (( PREFLIGHT_STATUS > 128 )); then
    log "ABORT: pre-flight killed by signal $((PREFLIGHT_STATUS - 128)) (exit $PREFLIGHT_STATUS)."
    log "       Signal 9 is the OOM killer -- cleaning this CSV peaks near 14GB and"
    log "       nothing else on this box may hold memory while it runs."
  else
    log "ABORT: campaign preflight failed (exit $PREFLIGHT_STATUS)."
  fi
  exit 1
fi

FAILED=0
SKIPPED=0
for cfg in "${CONFIGS[@]}"; do
  name="$(basename "$cfg" .yaml)"
  experiment_name="$("${PY[@]}" -c \
    "from training_pipeline.cli import load_config; print(load_config('$cfg').experiment_name)" \
    2>/dev/null)"
  if [[ "${SKIP_EXISTING:-0}" == "1" ]] \
     && compgen -G "artifacts/experiments/${CAMPAIGN}/${experiment_name}_20*" > /dev/null; then
    log "SKIP $name (artifact already exists)"
    SKIPPED=$((SKIPPED + 1))
    continue
  fi
  run_log="$LOG_DIR/${name}.log"
  log "START $(date +%H:%M:%S) $name"
  START=$SECONDS
  "${CLI[@]}" "$cfg" --no-save-model > "$run_log" 2>&1
  RUN_STATUS=$?
  if (( RUN_STATUS == 0 )); then
    STATUS="OK"
  else
    STATUS="FAILED(exit $RUN_STATUS)"
    FAILED=$((FAILED + 1))
    # A signalled run leaves no traceback in the log, so name the signal here:
    # 9 is the OOM killer, which is what a second memory-hungry process causes.
    if (( RUN_STATUS > 128 )); then
      log "  killed by signal $((RUN_STATUS - 128))$( ((RUN_STATUS == 137)) && echo ' -- OOM killer' )"
    fi
    tail -30 "$run_log" | tee -a "$LOG"
  fi
  ELAPSED=$((SECONDS - START))
  log "END $(date +%H:%M:%S) $STATUS ($((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m) $name"
done
log "Finished $(date): $FAILED failed, $SKIPPED skipped"
exit "$FAILED"
