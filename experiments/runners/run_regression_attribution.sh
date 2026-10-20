#!/usr/bin/env bash
#
# Why did line_error stop beating the bookmaker?
#
# Every archived line_error run beat the line on the 416-game holdout (model MAE
# 14.3332-14.4934 against a line MAE of 14.5036). Cells A and C of the
# public-betting campaign do not: 14.5186 and 14.5100, i.e. the model is now
# WORSE than doing nothing. The holdout cohort is byte-identical across the two
# eras -- same 60-day window, same 416 games, same line MAE to four decimals --
# so this is not a cohort artefact, and MAE is far less noisy than ROI, so it is
# not the seed spread either.
#
# Four things changed. Two are testable:
#
#   d_corr995_control      cleaning.corr_threshold 0.95 -> 0.995, ODDS_ override
#                          removed. Returns 150 of the 300 lost features.
#   e_test_anchored_control  rolling_origin/30/pooled -> test_anchored/12/mean.
#                          Explains the CV MAE sign flip if the folds, not the
#                          model, are what changed.
#
# Two are not. The 20260704 CSV predates the ODDS_ prefix unification and has
# zero ODDS_ columns, so it cannot be loaded by today's code at all -- and 93 of
# the 146 features it had that the new one lacks are DIFF_FROM_LINE_caesars_*,
# removed by the Caesars book merge and not coming back. n_trials (50 -> 120)
# cannot remove a feature; test it by re-reading the first 50 trials out of the
# persistent study rather than by spending another cell.
#
# READ AGAINST pubbet_a_keep_columns, which is already in
# artifacts/experiments/public_betting_tradeoff_2026_08/. The decisive number is
# holdout MAE against 14.5036, not ROI: at this bet volume ROI moves 4.5-9.3
# points on seed alone, and the archived runs' own ROI median was only +0.60%.
# Two cells against one baseline is two comparisons -- one marginal win is not
# a result. What would be: one cell moving MAE back under the line while the
# other does not.
#
#   screen -dmS attrib bash experiments/runners/run_regression_attribution.sh
#   screen -r attrib          # attach;  Ctrl-A then D to detach
#   tail -f artifacts/logs/regression_attribution_*.log
#
# set -u and pipefail, but deliberately NOT -e: one failed cell must not cancel
# the other.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="experiments/public_betting_tradeoff_2026_08"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="artifacts/logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/regression_attribution_${STAMP}.log"

exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== $(date -Is) starting regression-attribution cells ==="

# ---------------------------------------------------------------------------
# GPU gate. XGBoost 2.1.4 does NOT error when asked for cuda in a build without
# CUDA -- it warns and silently runs on CPU, so a run can look GPU-configured
# and take CPU time. Both cells declare device: cuda, and cell A (the baseline
# they are read against) ran on GPU, so a CPU fallback here would add a hardware
# confound to a comparison whose whole point is a single factor.
# ---------------------------------------------------------------------------
echo "--- verifying CUDA is genuine ---"
CUDA_CHECK="$(poetry run python -c "
import warnings, numpy as np, xgboost
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    xgboost.train({'device': 'cuda'},
                  xgboost.DMatrix(np.random.rand(64, 4), label=np.random.rand(64)),
                  num_boost_round=2)
    print('CPU_FALLBACK' if any('not compiled with CUDA' in str(w.message)
                               for w in caught) else 'CUDA_OK')
" 2>/dev/null | tail -1)"

if [ "$CUDA_CHECK" != "CUDA_OK" ]; then
  echo "ABORT: XGBoost would fall back to CPU despite device: cuda ($CUDA_CHECK)."
  echo "       Run this on the GPU machine. Do NOT switch these cells to cpu:"
  echo "       the baseline they are read against ran on GPU."
  exit 1
fi
echo "CUDA verified."

# ---------------------------------------------------------------------------
# Pre-flight. Parses both configs, verifies the CSV checksum, and -- the reason
# it exists -- builds the real splits and reports the ACTUAL per-fold training
# size. Cell E is the one that needs it: test_anchored folds reach further back
# than rolling_origin's for the same fold count, so its ceiling is lower than
# cell A's 4877 and a window past it would train short without erroring.
# ---------------------------------------------------------------------------
echo "--- pre-flight ---"
if ! poetry run python scripts/preflight_campaign.py \
      "$CAMPAIGN/d_corr995_control.yaml" \
      "$CAMPAIGN/e_test_anchored_control.yaml"; then
  echo "ABORT: pre-flight failed. Nothing was run."
  exit 1
fi

# D first: it is the single-factor cell and the one with measured mechanism
# behind it (+108 features recovered, overlap with the archived model 1158 ->
# 1261, verified without training). If only one finishes, that is the one to
# have.
CELLS=("d_corr995_control" "e_test_anchored_control")

for cell in "${CELLS[@]}"; do
  echo ""
  echo "=== $(date -Is) $cell ==="
  # --no-save-model: this is an evaluation campaign, not a deployment. Without
  # it the CLI passes save_model=True, which OVERRIDES the config's
  # refit.train_production_model: false -- and cell D then collides with the
  # bundle cell A already wrote at models/line_error/tuned_window/, because
  # both tune the window and so share the "tuned_window" label.
  poetry run python -m training_pipeline.cli "$CAMPAIGN/$cell.yaml" --no-save-model
  status=$?
  if [ $status -ne 0 ]; then
    echo "!!! $cell FAILED (exit $status) -- continuing with the rest"
  else
    echo "--- $(date -Is) $cell done ---"
  fi
done

echo ""
echo "=== $(date -Is) attribution finished ==="
echo "Runs are under artifacts/experiments/public_betting_tradeoff_2026_08/,"
echo "alongside cells A and C. Read them with"
echo "experiments/notebooks/survey_experiments.ipynb with SOURCES pointing at"
echo "that folder and RESCORE_EDGE_THRESHOLD = 0.05, so all four cells are"
echo "scored at one common threshold. Compare final_test_mae against the"
echo "line's 14.5036 FIRST -- it is far less noisy than ROI, and these runs"
echo "are single-seed so no cell has an error bar of its own. Treat any ROI"
echo "gap under ~5 points as indistinguishable."
