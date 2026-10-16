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
    parser.add_argument(
        "--kind", choices=["closing", "intermediate"], default="closing"
    )
    parser.add_argument(
        "--seasons", default=None, help="intermediate: comma-separated season years"
    )
    parser.add_argument(
        "--targets",
        action="store_true",
        help="intermediate: build D's games alone (target_game_ids); closing "
        "builds always do",
    )
    parser.add_argument(
        "--limit",
        default=None,
        help="build every date with this limit instead of the date itself: "
        "target parity against a reference with the same limit",
    )
    args = parser.parse_args()
    use_targets = args.kind == "closing" or args.targets

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
        game_ids = list(dict.fromkeys(rows["GAME_ID"].astype(str)))
        (day / "targets.txt").write_text("\n".join(game_ids) + "\n")
        rows.to_parquet(day / "reference.parquet")
        build = day / "build"
        log = day / "build.log"
        if log.exists() and NETWORK_ERRORS.search(log.read_text(errors="replace")):
            # A loader that could not reach its database returns nothing and
            # the build goes on without it: never compare such a build.
            shutil.rmtree(build, ignore_errors=True)
        if not (build / "dataset.parquet").exists():
            command = [
                PY,
                str(ROOT / "scripts/perf/reference_build.py"),
                args.kind,
                "--limit",
                args.limit or date,
                "--mode",
                "live",
                "--out",
                str(build),
                "--code-root",
                str(args.code_root),
            ]
            if args.seasons:
                command += ["--seasons", args.seasons]
            if use_targets:
                command += ["--targets", str(day / "targets.txt")]
            subprocess.run(
                command,
                stdout=open(log, "w"),
                stderr=subprocess.STDOUT,
                check=False,
            )
        if NETWORK_ERRORS.search(log.read_text(errors="replace")):
            failures += 1
            print(f"{date}: BUILD HIT A NETWORK ERROR, rerun it", flush=True)
            continue
        verdicts = []
        for name in ("dataset", "scoring"):
            built = build / f"{name}.parquet"
            if name == "scoring" and not (args.reference / "scoring.parquet").exists():
                continue
            if name == "scoring":
                scoring = pd.read_parquet(args.reference / "scoring.parquet")
                scoring[scoring["GAME_ID"].astype(str).isin(set(game_ids))].to_parquet(
                    day / "reference_scoring.parquet"
                )
            if built.exists() and not use_targets:
                # A whole build: keep D's rows, in the build's own order.
                frame = pd.read_parquet(built)
                frame = frame[frame["GAME_ID"].astype(str).isin(set(game_ids))]
                built = day / f"build_{name}_rows.parquet"
                frame.to_parquet(built)
            compare = subprocess.run(
                [
                    PY,
                    str(ROOT / "scripts/perf/compare_builds.py"),
                    str(
                        day
                        / (
                            "reference.parquet"
                            if name == "dataset"
                            else "reference_scoring.parquet"
                        )
                    ),
                    str(built),
                    "--exact",
                    # A build that stops early creates some columns (Yahoo's
                    # percentages, absent before 2020) at another point; the
                    # models select columns by name.
                    "--ignore-column-order",
                ],
                capture_output=True,
                text=True,
            )
            (day / f"compare_{name}.log").write_text(compare.stdout + compare.stderr)
            verdict = [
                line
                for line in compare.stdout.splitlines()
                if line.startswith("RESULT:")
            ]
            verdicts.append(
                f"{name} " + (verdict[-1] if verdict else "RESULT: no verdict")
            )
        failures += any(not v.endswith("RESULT: EQUAL") for v in verdicts)
        print(f"{date}: {len(set(game_ids))} games, {'; '.join(verdicts)}", flush=True)
    print("RESULT:", "EQUAL" if not failures else f"DIFFERENT ({failures} dates)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
