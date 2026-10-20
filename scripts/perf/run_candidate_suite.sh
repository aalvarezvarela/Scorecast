#!/usr/bin/env bash
# Accept a speed-up: replay every reference tape with the candidate code and
# require each output to equal its reference bit for bit.
#
#   scripts/perf/run_candidate_suite.sh <candidate worktree> <reference dir> <out dir> [fast|full|all]
#
# <reference dir> is a run_reference_suite.sh output (e.g. artifacts/perf/ref_7bd96a9).
# fast: closing 2 seasons + intermediate 2025-26; full: the all-season builds.
# Run the candidate from a worktree pinned to its commit, so edits in the main
# checkout cannot leak into a running build. Resumable like the reference suite.
set -uo pipefail
CODE_ROOT=$(realpath "$1"); REF=$(realpath "$2"); OUT=$(realpath -m "$3"); TIER=${4:-all}
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
  if [[ $status -eq 0 || -n $result ]]; then touch "$OUT/$name.done"; fi
  echo "[$(date '+%F %T')] END   $name exit=$status $result" | tee -a "$SUMMARY"
}

seconds() { "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['build_seconds'])" "$1"; }

replay_and_compare() {  # replay_and_compare <name> <kind> <build args...>
  local name=$1 kind=$2; shift 2
  step "${name}_replay" $BUILD "$kind" "$@" --mode replay --tape "$REF/$name/tape" --out "$OUT/$name"
  step "${name}_cmp" $COMPARE "$REF/$name/dataset.parquet" "$OUT/$name/dataset.parquet" --exact
  if [[ -f "$REF/$name/scoring.parquet" ]]; then
    step "${name}_cmp_scoring" $COMPARE "$REF/$name/scoring.parquet" "$OUT/$name/scoring.parquet" --exact
  fi
  if [[ -f "$OUT/$name/manifest.json" ]]; then
    echo "[$(date '+%F %T')] TIME  $name reference_record=$(seconds "$REF/$name/manifest.json")s" \
      "reference_replay=$(seconds "$REF/${name}_replay/manifest.json" 2>/dev/null || echo ?)s" \
      "candidate_replay=$(seconds "$OUT/$name/manifest.json")s" | tee -a "$SUMMARY"
  fi
}

if [[ $TIER == fast || $TIER == all ]]; then
  replay_and_compare closing_2s closing --limit 2026-04-10 --n-seasons 2
  replay_and_compare inter_2025 intermediate --limit 2026-04-10 --seasons 2025
fi
if [[ $TIER == full || $TIER == all ]]; then
  replay_and_compare closing_full closing --limit 2026-10-03
  replay_and_compare inter_full intermediate --limit 2026-10-03
fi
echo "[$(date '+%F %T')] SUITE DONE ($TIER)" | tee -a "$SUMMARY"
