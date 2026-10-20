#!/usr/bin/env bash
# Acceptance of the tonight-only builds at one pinned commit.
#
#   scripts/perf/run_tonight_acceptance.sh <worktree at the commit> <out dir> [old closing ref]
#
# Every build is live (the databases are not written during the offseason) and
# every comparison exact, except the first: the full closing build against an
# older reference, at rtol 1e-12, whose exact differences must be the
# point-in-time fixes' columns only (REF_*). Steps that finish leave <name>.done
# and are skipped when re-run; a build that hit a network error is retried.
set -uo pipefail
CODE_ROOT=$(realpath "$1"); OUT=$(realpath -m "$2"); OLD_REF=${3:-}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY="$REPO/.venv/bin/python"
BUILD="taskset -c 0-10 $PY $REPO/scripts/perf/reference_build.py --code-root $CODE_ROOT"
PIT="taskset -c 0-10 $PY $REPO/scripts/perf/point_in_time_parity.py"
COMPARE="$PY $REPO/scripts/perf/compare_builds.py"
NET='Temporary failure in name resolution|OperationalError|Error loading'
export TQDM_DISABLE=1 PYTHONWARNINGS=ignore PYTHONHASHSEED=0
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"

step() {  # step <name> <command...>
  local name=$1; shift
  if [[ -f "$OUT/$name.done" ]]; then echo "[$(date '+%F %T')] SKIP  $name" | tee -a "$SUMMARY"; return; fi
  for attempt in 1 2 3; do
    echo "[$(date '+%F %T')] START $name (attempt $attempt)" | tee -a "$SUMMARY"
    "$@" > "$OUT/$name.log" 2>&1
    local status=$?
    if grep -Eq "$NET" "$OUT/$name.log" && ! grep -q '^RESULT: EQUAL' "$OUT/$name.log"; then
      echo "  network error, retrying in 120 s" | tee -a "$SUMMARY"; sleep 120; continue
    fi
    local result; result=$(grep -h '^RESULT:\|columns with differing' "$OUT/$name.log" | tail -2 | tr '\n' ' ')
    [[ $status -eq 0 || -n $result ]] && touch "$OUT/$name.done"
    echo "[$(date '+%F %T')] END   $name exit=$status $result" | tee -a "$SUMMARY"
    return
  done
}
fresh() { [[ -f "$OUT/$1.done" ]] || rm -rf "$OUT/$1"; }

CLOSING_DATES="2019-10-22 2020-12-22 2021-05-18 2022-04-16 2023-02-15 2024-04-16 2025-10-22 2026-01-15 2026-04-10"
INTER_DATES="2025-10-22 2026-01-15 2026-03-01 2026-04-10"

# ---- closing ---------------------------------------------------------------
fresh closing_full
step closing_full env NBA_OU_HISTORY_MEMO=0 $BUILD closing --limit 2026-10-03 --mode live --out "$OUT/closing_full"
if [[ -n $OLD_REF ]]; then
  step closing_vs_old_rtol $COMPARE "$OLD_REF" "$OUT/closing_full/dataset.parquet" --rtol 1e-12
  step closing_vs_old_exact $COMPARE "$OLD_REF" "$OUT/closing_full/dataset.parquet" --exact --show 60
fi
step closing_pit $PIT "$OUT/closing_full" "$OUT/closing_pit" --code-root "$CODE_ROOT" --dates $CLOSING_DATES
step closing_parity $PIT "$OUT/closing_full" "$OUT/closing_parity" --limit 2026-10-03 --code-root "$CODE_ROOT" --dates $CLOSING_DATES
step closing_prediction env NBA_OU_CODE_ROOT="$CODE_ROOT" taskset -c 0-10 $PY $REPO/scripts/perf/closing_prediction_parity.py --date 2026-04-10 --as-of 2026-04-10T22:30:00Z --out "$OUT/closing_prediction"

# ---- intermediate, 2025-26 ---------------------------------------------------
fresh inter_2025
step inter_2025 env NBA_OU_HISTORY_MEMO=0 $BUILD intermediate --limit 2026-04-10 --seasons 2025 --mode live --out "$OUT/inter_2025"
step inter_2025_parity $PIT "$OUT/inter_2025" "$OUT/inter_2025_parity" --kind intermediate --seasons 2025 --targets --limit 2026-04-10 --code-root "$CODE_ROOT" --dates $INTER_DATES
step inter_2025_pit $PIT "$OUT/inter_2025" "$OUT/inter_2025_pit" --kind intermediate --seasons 2025 --targets --code-root "$CODE_ROOT" --dates $INTER_DATES
step live_replay env NBA_OU_CODE_ROOT="$CODE_ROOT" taskset -c 0-10 $PY $REPO/scripts/perf/live_replay_parity.py --date 2026-03-01 --as-of 2026-03-01T22:00:00Z --seasons 2025 --out "$OUT/live_replay"

# ---- intermediate, every line-history season ----------------------------------
fresh inter_full
step inter_full env NBA_OU_HISTORY_MEMO=0 $BUILD intermediate --limit 2026-04-10 --mode live --out "$OUT/inter_full"
step inter_full_parity $PIT "$OUT/inter_full" "$OUT/inter_full_parity" --kind intermediate --targets --limit 2026-04-10 --code-root "$CODE_ROOT" --dates 2024-04-16 2026-04-10

# ---- tests -------------------------------------------------------------------
step pytest env PYTHONPATH="$CODE_ROOT/src" $PY -m pytest -q -p no:cacheprovider "$CODE_ROOT/tests"
echo "[$(date '+%F %T')] DONE" | tee -a "$SUMMARY"
