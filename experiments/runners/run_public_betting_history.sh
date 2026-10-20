#!/usr/bin/env bash
#
# Overnight: line_error on the closing-line dataset, WITH and WITHOUT seasons
# 2019-20.
#
#   a_keep_columns     floor 2021, every column        6,148 rows / 1,362 features
#   c_drop_and_extend  floor 2019, season-gated gone   8,279 rows / 1,227 features
#
# Read this as the PRODUCTION-DECISION contrast, not a single-factor one. The two
# arms differ in history AND in 213 columns, and that is unavoidable rather than
# sloppy: admitting 2019-20 without dropping the columns absent in them leaves
# the missingness pattern as a season indicator, and dropping them without
# admitting the seasons discards features for nothing. One switch, one decision.
# b_drop_columns is the cell that separates the two halves -- it is NOT run here;
# add it when you want the decomposition rather than the verdict.
#
# Cell C also reaches further back in the window (up to 6500 games vs 4500).
# That is the mechanism, not a confound: under rolling_origin each origin trains
# on the last train_games games before it, so at any window both arms would pull
# the SAME games and C's extra 2,131 rows would sit behind the window, never
# read. Without the larger windows this comparison would conclude "extra history
# does not help" having never used any.
#
# What the extra seasons ARE matters for reading the result: 2019-20 is the
# bubble and 2020-21 the 72-game compressed season. This trades distribution
# match for volume, so a win is weaker evidence than the same win from ordinary
# seasons, and a loss is not evidence that history per se is unhelpful.
#
#   nohup bash experiments/runners/run_public_betting_history.sh > /dev/null 2>&1 &
#   tail -f artifacts/logs/public_betting_history_*.log
#
# set -u and pipefail, but deliberately NOT -e: one failed cell must not cancel
# the other.
set -uo pipefail

cd "$(dirname "$0")/../.." || exit 1

CAMPAIGN="experiments/public_betting_tradeoff_2026_08"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="artifacts/logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/public_betting_history_${STAMP}.log"

exec > >(tee -a "$MAIN_LOG") 2>&1

echo "=== $(date -Is) starting public-betting history campaign ==="

# ---------------------------------------------------------------------------
# GPU gate. XGBoost 2.1.4 does NOT error when asked for cuda in a build without
# CUDA -- it warns and silently runs on CPU, so a run can look GPU-configured
# and take CPU time. Both cells declare device: cuda, so check it here once,
# loudly, rather than discovering it from the wall clock in the morning.
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
  echo "       Run this on the GPU machine, or set device: cpu on BOTH cells"
  echo "       (identically -- a device split between cells is not comparable)."
  exit 1
fi
echo "CUDA verified."

# ---------------------------------------------------------------------------
# Pre-flight. Parses every config, verifies the CSV checksum, and -- the reason
# it exists -- builds the real splits and reports the ACTUAL per-fold training
# size, catching a window that would otherwise silently shrink. Not optional
# before an overnight run; --skip-data would skip exactly that check.
# ---------------------------------------------------------------------------
echo "--- pre-flight ---"
if ! poetry run python scripts/preflight_campaign.py "$CAMPAIGN"; then
  echo "ABORT: pre-flight failed. Nothing was run."
  exit 1
fi

# Baseline first, so a crash in the treatment arm still leaves something to read.
CELLS=("a_keep_columns" "c_drop_and_extend")

for cell in "${CELLS[@]}"; do
  echo ""
  echo "=== $(date -Is) $cell ==="
  # --no-save-model: this is an evaluation campaign, not a deployment. Without
  # it every cell writes a production bundle named
  # <window_name_label>_xgb_<target>_<DD_MM_YY> -- and since no cell sets
  # window_name_label (both tune train_games, so both resolve to
  # "tuned_window") and the date comes from the END of the dev split, which the
  # extra history does not move, EVERY cell resolves to the same path. The
  # first cell writes it and the rest die on the overwrite guard before tuning.
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
echo "Runs are under artifacts/experiments/public_betting_tradeoff_2026_08/"
echo "Read them with experiments/notebooks/survey_experiments.ipynb, pointing"
echo "SOURCES at $CAMPAIGN and setting RESCORE_EDGE_THRESHOLD so both cells are"
echo "scored at one common threshold."
