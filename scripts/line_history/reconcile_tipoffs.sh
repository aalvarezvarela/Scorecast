#!/usr/bin/env bash
# Rerun reconcile_tipoffs.py until it succeeds.
#
# stats.nba.com stops answering a process after ~300 requests, so the script
# raises after --max-requests (default 290) with dates still uncached. Every
# fetched date is cached, so each fresh run continues where the last stopped.
# All arguments are passed through:
#
#   scripts/line_history/reconcile_tipoffs.sh --season-year 2023 --dry-run
#   scripts/line_history/reconcile_tipoffs.sh --all-seasons --write-corrections
#
# Environment: PAUSE_SECONDS between runs (default 30), MAX_RUNS (default 20).
set -uo pipefail
cd "$(dirname "$0")/../.."

PAUSE_SECONDS="${PAUSE_SECONDS:-30}"
MAX_RUNS="${MAX_RUNS:-20}"

for run in $(seq 1 "$MAX_RUNS"); do
    echo "=== reconcile_tipoffs run ${run}/${MAX_RUNS} ==="
    if python scripts/line_history/reconcile_tipoffs.py "$@"; then
        exit 0
    fi
    echo "Run ${run} failed; restarting in ${PAUSE_SECONDS}s."
    sleep "$PAUSE_SECONDS"
done

echo "Still failing after ${MAX_RUNS} runs." >&2
exit 1
