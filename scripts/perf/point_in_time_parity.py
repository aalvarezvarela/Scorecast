#!/usr/bin/env python3
"""A build that stops at date D gives D's games the full build's rows.

    python scripts/perf/point_in_time_parity.py REFERENCE_DIR OUT_DIR \
        [--dates 2019-10-22 2026-04-10 ...] [--code-root WORKTREE]

For each date: ``reference_build.py closing --limit D --targets <D's games>
--mode live`` -- the history a prediction for D sees, nothing after it -- and
``compare_builds.py --exact`` against the reference build's rows for D. The
reference is a full closing build with a later limit (live, so both read the
same database; run it the same day, when nothing is being written).

This is the property the history memo and every prediction rely on: a game's
row depends only on what came before it. ``target_parity.py`` cannot check it,
because its target builds load the reference's whole span.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
NETWORK_ERRORS = re.compile(
    r"Temporary failure in name resolution|OperationalError|Error loading"
)
#: Not the first loaded date: a build that stops there has no bet365 lines yet,
#: so its column set differs from the full build's. A prediction always loads
#: the whole history up to yesterday.
DEFAULT_DATES = (
    "2019-10-22",
    "2021-05-18",
    "2024-04-16",
    "2025-10-22",
    "2026-04-10",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reference", type=Path, help="reference build directory")
    parser.add_argument("out", type=Path)
    parser.add_argument("--dates", nargs="*", default=list(DEFAULT_DATES))
    parser.add_argument("--code-root", type=Path, default=ROOT)
    args = parser.parse_args()

    reference = pd.read_parquet(args.reference / "dataset.parquet")
    dates = pd.to_datetime(reference["GAME_DATE"]).dt.normalize()
    args.out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for date in args.dates:
        rows = reference[dates.eq(pd.Timestamp(date))]
        if rows.empty:
            print(f"{date}: no reference rows, skipped")
            continue
        day = args.out / date
        day.mkdir(exist_ok=True)
        (day / "targets.txt").write_text("\n".join(rows["GAME_ID"].astype(str)) + "\n")
        rows.to_parquet(day / "reference.parquet")
        build = day / "build"
        log = day / "build.log"
        if log.exists() and NETWORK_ERRORS.search(log.read_text(errors="replace")):
            # A loader that could not reach its database returns nothing and
            # the build goes on without it: never compare such a build.
            shutil.rmtree(build, ignore_errors=True)
        if not (build / "dataset.parquet").exists():
            subprocess.run(
                [
                    PY,
                    str(ROOT / "scripts/perf/reference_build.py"),
                    "closing",
                    "--limit",
                    date,
                    "--mode",
                    "live",
                    "--targets",
                    str(day / "targets.txt"),
                    "--out",
                    str(build),
                    "--code-root",
                    str(args.code_root),
                ],
                stdout=open(log, "w"),
                stderr=subprocess.STDOUT,
                check=False,
            )
        if NETWORK_ERRORS.search(log.read_text(errors="replace")):
            failures += 1
            print(f"{date}: BUILD HIT A NETWORK ERROR, rerun it", flush=True)
            continue
        compare = subprocess.run(
            [
                PY,
                str(ROOT / "scripts/perf/compare_builds.py"),
                str(day / "reference.parquet"),
                str(build / "dataset.parquet"),
                "--exact",
                # A build that stops early creates some columns (Yahoo's
                # percentages, absent before 2020) at another point; the
                # models select columns by name.
                "--ignore-column-order",
            ],
            capture_output=True,
            text=True,
        )
        (day / "compare.log").write_text(compare.stdout + compare.stderr)
        verdict = [
            line for line in compare.stdout.splitlines() if line.startswith("RESULT:")
        ]
        verdict = verdict[-1] if verdict else "RESULT: no verdict"
        failures += verdict != "RESULT: EQUAL"
        print(f"{date}: {len(rows)} games, {verdict}", flush=True)
    print("RESULT:", "EQUAL" if not failures else f"DIFFERENT ({failures} dates)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
