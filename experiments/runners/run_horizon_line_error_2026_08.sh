#!/usr/bin/env bash
# Sixteen experiments, sequential on one CUDA machine: fifteen line-error cells
# plus one closing-line classifier cell.
#
# The classifier cell is a GUEST here. There are only two runners because only
# two fit on this GPU at once, so the classifier has no stream of its own and one
# of its cells lives in each. Read it against the matching cell in BOTH streams,
# never against the rest of this folder -- the folder is a runner grouping, not a
# comparison group. reporting/factors.py matches on config, never on directory,
# so the cross-stream comparison is the one the tooling will make anyway.
#
# Cells are ordered so a weekend that runs short still leaves the comparisons
# that matter: the baseline, the noise floor, the correlation contrast and the
# classifier first, then the horizon curve, then the two expensive cells. The
# pooled cell runs LAST because it costs as much as three others and its controls
# have to exist before it can be read.
#
# Two of these runners, started together. That is what the estimate assumes, and
# the measured per-cell times it is built from were themselves recorded under
# two-way contention.
#
#   bash experiments/runners/run_horizon_line_error_2026_08.sh
#
# Detached:
#   nohup bash experiments/runners/run_horizon_line_error_2026_08.sh > /dev/null 2>&1 &
#
# Resume after an interruption without repeating completed cells:
#   SKIP_EXISTING=1 bash experiments/runners/run_horizon_line_error_2026_08.sh
set -uo pipefail  # One failed cell must not cancel the remaining experiments.

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="horizon_line_error_2026_08"
CONFIG_DIR="experiments/${CAMPAIGN}"
LOG_DIR="artifacts/logs/${CAMPAIGN}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/campaign.log"

CONFIGS=(
  "$CONFIG_DIR/a_closing_reference.yaml"
  "$CONFIG_DIR/b_closing_corr099_control.yaml"
  "$CONFIG_DIR/h_closing_replicate.yaml"
  "$CONFIG_DIR/i_classifier_closing_reference.yaml"
  "$CONFIG_DIR/f_intermediate_t360.yaml"
  "$CONFIG_DIR/l_intermediate_t300.yaml"
  "$CONFIG_DIR/g_intermediate_t240.yaml"
  "$CONFIG_DIR/m_intermediate_t180.yaml"
  "$CONFIG_DIR/j_intermediate_t120.yaml"
  "$CONFIG_DIR/n_intermediate_t60.yaml"
  "$CONFIG_DIR/o_intermediate_t30.yaml"
  "$CONFIG_DIR/p_intermediate_t0.yaml"
  "$CONFIG_DIR/e_intermediate_t720.yaml"
  "$CONFIG_DIR/k_intermediate_t480.yaml"
  "$CONFIG_DIR/c_closing_2020_extended.yaml"
  "$CONFIG_DIR/d_intermediate_pooled.yaml"
)

DATASETS=(
  "data/train_data/training_data_2_0_20260819.csv"
  "data/train_data/intermediate_line_data_10snap.csv"
  "data/train_data/intermediate_line_data_10snap_scoring.csv"
)

PY=(poetry run python -u)
CLI=("${PY[@]}" -m training_pipeline.cli)
log() { echo "$@" | tee -a "$LOG"; }

log "=========================================================="
log "Campaign : ${CAMPAIGN}"
log "Started  : $(date)"
log "Runs     : ${#CONFIGS[@]} sequential runs, 300 trials each, 3 seeds each, CUDA"
log "Estimate : ~58h full, ~22h gap-only with SKIP_EXISTING=1."
log "           Start the other horizon runner at the same time."
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
  log "but it verifies checksums, cleaning and every requested game window."
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
log "  1. seed_stability.csv, and a_closing_reference against h_closing_replicate."
log "     That pair is the whole-pipeline noise floor. Nothing smaller is a result."
log "  2. Volume. At -110 break-even is 52.38% and a Wilson interval around a"
log "     genuine 55% does not clear it at 400 bets. Check the interval first."
log "  3. a_closing_reference minus b_closing_corr099_control: the ONLY effect of"
log "     the 0.995 odds-correlation change. It is a closing-line question --"
log "     measured, the same change moves 7-8 features on the intermediate"
log "     dataset against 59 on the closing one, so do not look for it there."
log "  4. Pooled versus dedicated AT THE SAME HORIZON (T-240/T-360/T-720). The"
log "     pooled cell'"'"'s headline ROI counts one game once per horizon: read"
log "     snapshot_holdout_metrics.csv, never the pooled row."
log "  5. For the 2020 cell, verify selected train_games exceeds 4500 before"
log "     claiming the extra season helped: otherwise Optuna had access and"
log "     declined it, which is the answer."
log ""
log "  6. The classifier cell is read against the SAME cell in the other stream"
log "     and in this one, target formulation being the only difference. Its"
log "     holdout is a slightly different cohort: pushes have no label for a"
log "     classifier and are dropped, 6,075 games against 6,148."
log ""
log "32 cells across the two streams. Against this noise floor expect about two"
log "to look good by luck. Treat anything that survives as the next campaign'"'"'s"
log "hypothesis, not as a finding."
log "=========================================================="

exit "$FAILED"
