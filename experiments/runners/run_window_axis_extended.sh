#!/usr/bin/env bash
#
# The training-window axis on the extended-history data, measured at three
# FIXED windows instead of sampled by Optuna.
#
#   d1_window_3000   control -- the window cell C actually selected
#   d2_window_4500   the largest window the old data regime could also reach
#   d3_window_6500   treatment -- reachable only via the 2019/2020 seasons
#
# The data regime is identical in all three (extend_history_dropping_season_gated
# _columns: true, 8,279 rows / 1,232 columns). The window is the only difference,
# which is the whole point: in public_betting_tradeoff_2026_08 the window was a
# tuned hyperparameter, the pruner killed 94 of 95 trials at the first fold it
# was allowed to act on, and the large windows ended up with 5 completed trials
# between them. See d1_window_3000.yaml for the full post-mortem.
#
#   nohup bash experiments/runners/run_window_axis_extended.sh > /dev/null 2>&1 &
#   tail -f artifacts/logs/window_axis_extended_*.log
#
# Budget ~3h per cell (~9h total). Tuning is the minority of that: measured on
# cell C, 120 trials took ~32 min and the rest went to the daily walk-forward
# holdout. That budget was measured when the holdout was evaluated under three
# seeds; at one seed the holdout stage costs roughly a third of that. Raising
# pruner_warmup_fraction to 0.5 roughly doubles the cost of a pruned trial, so
# expect tuning nearer 60 min per cell.
#
# set -u and pipefail, but deliberately NOT -e: one failed cell must not cancel
# the others.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="experiments/window_axis_extended_2026_08"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="artifacts/logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/window_axis_extended_${STAMP}.log"

exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== $(date -Is) starting window-axis campaign ==="

# ---------------------------------------------------------------------------
# GPU gate. XGBoost 2.1.4 does NOT error when asked for cuda in a build without
# CUDA -- it warns and silently runs on CPU, so a run can look GPU-configured
# and take CPU time. All three cells declare device: cuda, so check it here
# once, loudly, rather than discovering it from the wall clock in the morning.
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
  echo "       Run this on the GPU machine, or set device: cpu on ALL THREE cells"
  echo "       (identically -- a device split between cells is not comparable)."
  exit 1
fi
echo "CUDA verified."

# ---------------------------------------------------------------------------
# Pre-flight. The check that earns its keep here is the window ceiling on
# d3: 6500 must fit the EARLIEST fold's history. tail(n) returns what it has
# rather than raising, so a window past the ceiling does not fail -- the early
# folds quietly train short and the cell stops being the comparison it was
# designed to be, which is precisely the failure this campaign is correcting.
# ---------------------------------------------------------------------------
echo "--- pre-flight ---"
if ! poetry run python scripts/preflight_campaign.py "$CAMPAIGN"; then
  echo "ABORT: pre-flight failed. Nothing was run."
  exit 1
fi

# Control first, then ascending window: if the campaign dies partway, what
# survives is still an interpretable prefix of the axis.
CELLS=("d1_window_3000" "d2_window_4500" "d3_window_6500")

for cell in "${CELLS[@]}"; do
  echo ""
  echo "=== $(date -Is) $cell ==="
  # --no-save-model: this is an evaluation campaign, not a deployment. Without
  # it every cell writes a production bundle under models/, and since the bundle
  # name is <window_label>_xgb_<target>_<DD_MM_YY> the cells would collide on
  # the date and every cell after the first would die on the overwrite guard.
  # Promote a winner afterwards with:
  #   poetry run python -m training_pipeline.promote <run_dir>
  poetry run python -m training_pipeline.cli "$CAMPAIGN/$cell.yaml" --no-save-model
  status=$?
  if [ $status -ne 0 ]; then
    echo "!!! $cell FAILED (exit $status) -- continuing with the rest"
  else
    echo "--- $(date -Is) $cell done ---"
  fi
done

echo ""
echo "=== $(date -Is) campaign finished ==="
echo "Runs are under artifacts/experiments/window_axis_extended_2026_08/"
echo ""
echo "Judge on holdout MAE, not ROI. These runs are single-seed, so no cell"
echo "carries an error bar of its own. The noise floor has not gone away --"
echo "it was measured at 4.9-12.0 ROI points for ONE fixed config re-fitted"
echo "under other seeds -- it is just no longer measured per run. Treat any"
echo "ROI gap under ~5 points as indistinguishable, and at three cells expect"
echo "roughly one to look good by luck."
echo ""
echo "Then, in order:"
echo "  D1 vs D2  -- does more history help, using ordinary seasons only?"
echo "  D2 vs D3  -- does the bubble/COVID history add anything on top?"
echo "Only if D3 wins BOTH is the production switch worth flipping."
echo ""
echo "Survey with experiments/notebooks/survey_experiments.ipynb, pointing"
echo "SOURCES at $CAMPAIGN and setting RESCORE_EDGE_THRESHOLD so all three"
echo "cells are scored at one common threshold rather than each at whichever"
echo "one it froze into betting_metrics.json."
