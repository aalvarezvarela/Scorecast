#!/usr/bin/env bash
# Build, and self-check, the 2_5 speed-up references from a pinned worktree.
#
#   scripts/perf/run_reference_suite.sh <code-root worktree> <out dir> [fast|full|all]
#
# fast: closing 2 seasons + intermediate 2025-26, each recorded, replayed and
#       compared; closing also rebuilt live a second time (determinism).
# Re-running resumes: finished steps leave <name>.done and are skipped. Delete a
# crashed build's output directory before re-running it.
# full: closing all seasons + intermediate all store seasons to 2026-10-03,
#       recorded, compared with the stored 2_5 files, then replayed.
set -uo pipefail
CODE_ROOT=$(realpath "$1"); OUT=$(realpath -m "$2"); TIER=${3:-all}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY="$REPO/.venv/bin/python"
BUILD="$PY $REPO/scripts/perf/reference_build.py --code-root $CODE_ROOT"
COMPARE="$PY $REPO/scripts/perf/compare_builds.py"
export TQDM_DISABLE=1 PYTHONWARNINGS=ignore
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"

step() {  # step <name> <command...>; a step that finished before is skipped
  local name=$1; shift
  if [[ -f "$OUT/$name.done" ]]; then
    echo "[$(date '+%F %T')] SKIP  $name (done)" | tee -a "$SUMMARY"; return
  fi
  echo "[$(date '+%F %T')] START $name" | tee -a "$SUMMARY"
  "$@" > "$OUT/$name.log" 2>&1
  local status=$?
  local result; result=$(grep -h '^RESULT:' "$OUT/$name.log" | tail -1)
  # A build is done when it exits 0; a comparison is done once it has a verdict.
  if [[ $status -eq 0 || -n $result ]]; then touch "$OUT/$name.done"; fi
  echo "[$(date '+%F %T')] END   $name exit=$status $result" | tee -a "$SUMMARY"
}

if [[ $TIER == fast || $TIER == all ]]; then
  step closing_2s_record   $BUILD closing --limit 2026-04-10 --n-seasons 2 --mode record --out "$OUT/closing_2s"
  step closing_2s_replay   $BUILD closing --limit 2026-04-10 --n-seasons 2 --mode replay --tape "$OUT/closing_2s/tape" --out "$OUT/closing_2s_replay"
  step closing_2s_cmp_replay $COMPARE "$OUT/closing_2s/dataset.parquet" "$OUT/closing_2s_replay/dataset.parquet" --exact
  step closing_2s_live     $BUILD closing --limit 2026-04-10 --n-seasons 2 --mode live --out "$OUT/closing_2s_live"
  step closing_2s_cmp_live $COMPARE "$OUT/closing_2s/dataset.parquet" "$OUT/closing_2s_live/dataset.parquet" --exact
  step inter_2025_record   $BUILD intermediate --limit 2026-04-10 --seasons 2025 --mode record --out "$OUT/inter_2025"
  step inter_2025_replay   $BUILD intermediate --limit 2026-04-10 --seasons 2025 --mode replay --tape "$OUT/inter_2025/tape" --out "$OUT/inter_2025_replay"
  step inter_2025_cmp_replay $COMPARE "$OUT/inter_2025/dataset.parquet" "$OUT/inter_2025_replay/dataset.parquet" --exact
  step inter_2025_cmp_replay_scoring $COMPARE "$OUT/inter_2025/scoring.parquet" "$OUT/inter_2025_replay/scoring.parquet" --exact
fi

if [[ $TIER == full || $TIER == all ]]; then
  step closing_full_record $BUILD closing --limit 2026-10-03 --mode record --out "$OUT/closing_full"
  step closing_full_cmp_stored $COMPARE "$REPO/data/train_data/closing_line_data_2_5_20261003.parquet" "$OUT/closing_full/dataset.parquet" --exact
  step inter_full_record   $BUILD intermediate --limit 2026-10-03 --mode record --out "$OUT/inter_full"
  step inter_full_cmp_stored $COMPARE "$REPO/data/train_data/intermediate_line_data_2_5_20261003.parquet" "$OUT/inter_full/dataset.parquet" --exact
  step inter_full_cmp_stored_scoring $COMPARE "$REPO/data/train_data/intermediate_line_data_2_5_20261003_scoring.parquet" "$OUT/inter_full/scoring.parquet" --exact
  step closing_full_replay $BUILD closing --limit 2026-10-03 --mode replay --tape "$OUT/closing_full/tape" --out "$OUT/closing_full_replay"
  step closing_full_cmp_replay $COMPARE "$OUT/closing_full/dataset.parquet" "$OUT/closing_full_replay/dataset.parquet" --exact
  step inter_full_replay   $BUILD intermediate --limit 2026-10-03 --mode replay --tape "$OUT/inter_full/tape" --out "$OUT/inter_full_replay"
  step inter_full_cmp_replay $COMPARE "$OUT/inter_full/dataset.parquet" "$OUT/inter_full_replay/dataset.parquet" --exact
fi
echo "[$(date '+%F %T')] SUITE DONE ($TIER)" | tee -a "$SUMMARY"
