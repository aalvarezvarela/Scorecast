#!/usr/bin/env python3
"""Time one build stage in isolation, and check it against the reference exactly.

A full replayed intermediate build takes 20 minutes for one season. Most
speed-ups touch a single stage, so this captures that stage's inputs and
output once, from a replayed reference build, and then reruns only the
stage:

    # Capture, with the reference code: inputs + output of the stage
    python scripts/perf/stage_harness.py capture snapshot_injuries \
        --code-root ../NBA_ou_ref_7bd96a9 \
        --tape artifacts/perf/ref_7bd96a9/inter_2025/tape \
        --limit 2026-04-10 --seasons 2025 --out artifacts/perf/stage/inj_2025

    # Rerun the stage with the code under test; exit 1 unless bit-identical
    python scripts/perf/stage_harness.py run snapshot_injuries \
        --tape artifacts/perf/ref_7bd96a9/inter_2025/tape \
        --inputs artifacts/perf/stage/inj_2025

The stage's own queries are answered from the same tape, so ``run`` needs no
database either. This is a development check; a change is accepted only by a
full replayed build compared with ``compare_builds.py --exact``.

Pins ``PYTHONHASHSEED=0`` like ``reference_build.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _code_root() -> Path:
    if "--code-root" in sys.argv:
        return Path(sys.argv[sys.argv.index("--code-root") + 1]).resolve()
    return PROJECT_ROOT


CODE_ROOT = _code_root()
sys.path[:0] = [str(CODE_ROOT / "src"), str(CODE_ROOT), str(Path(__file__).parent)]

#: stage name -> (module whose global the builder calls, function name)
STAGES = {
    "snapshot_injuries": (
        "nba_ou.create_training_data.create_intermediate_line_df",
        "nba_ou.create_training_data.intermediate_injuries",
        "add_snapshot_injury_features",
    ),
}


class _Captured(BaseException):
    """Ends the capture build once the stage has run."""


def _ensure_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


def _check_origin() -> None:
    import nba_ou

    if not Path(nba_ou.__file__).resolve().is_relative_to(CODE_ROOT):
        raise SystemExit(f"nba_ou imported from {nba_ou.__file__}, not {CODE_ROOT}")


def capture(args: argparse.Namespace) -> int:
    import importlib

    import db_tape

    caller_name, _, function_name = STAGES[args.stage]
    caller = importlib.import_module(caller_name)
    original = getattr(caller, function_name)
    args.out.mkdir(parents=True, exist_ok=False)

    def wrapper(*call_args, **call_kwargs):
        with open(args.out / "inputs.pkl", "wb") as handle:
            pickle.dump((call_args, call_kwargs), handle, protocol=5)
        started = time.perf_counter()
        result = original(*call_args, **call_kwargs)
        seconds = time.perf_counter() - started
        result.to_pickle(args.out / "output.pkl")
        (args.out / "manifest.json").write_text(
            json.dumps(
                {
                    "stage": args.stage,
                    "code_root": str(CODE_ROOT),
                    "tape": str(args.tape),
                    "limit": args.limit,
                    "seasons": args.seasons,
                    "seconds": round(seconds, 1),
                    "shape": list(result.shape),
                },
                indent=1,
            )
        )
        raise _Captured

    setattr(caller, function_name, wrapper)
    from nba_ou.create_training_data.create_intermediate_line_df import (
        create_intermediate_line_df,
    )

    try:
        with db_tape.replay(args.tape):
            create_intermediate_line_df(
                recent_limit_to_include=args.limit,
                season_years=[int(s) for s in args.seasons.split(",")],
                return_scoring=True,
                verbose=True,
            )
    except _Captured:
        print(f"captured {args.stage} into {args.out}")
        return 0
    raise SystemExit("the build finished without reaching the stage")


def run(args: argparse.Namespace) -> int:
    import importlib

    import db_tape
    import pandas as pd
    from compare_builds import compare

    _, module_name, function_name = STAGES[args.stage]
    function = getattr(importlib.import_module(module_name), function_name)
    with open(args.inputs / "inputs.pkl", "rb") as handle:
        call_args, call_kwargs = pickle.load(handle)
    expected = pd.read_pickle(args.inputs / "output.pkl")

    started = time.perf_counter()
    with db_tape.replay(args.tape):
        result = function(*call_args, **call_kwargs)
    seconds = time.perf_counter() - started
    reference_seconds = json.loads((args.inputs / "manifest.json").read_text())[
        "seconds"
    ]
    print(f"{args.stage}: {seconds:.1f} s (reference {reference_seconds:.1f} s)")
    if args.save:
        result.to_pickle(args.save)
    return 0 if compare(expected, result, exact=True) else 1


def main() -> int:
    _ensure_hash_seed()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "run"):
        command = sub.add_parser(name)
        command.add_argument("stage", choices=sorted(STAGES))
        command.add_argument("--tape", type=Path, required=True)
        command.add_argument("--code-root", type=Path, default=PROJECT_ROOT)
    sub.choices["capture"].add_argument("--limit", required=True)
    sub.choices["capture"].add_argument("--seasons", required=True)
    sub.choices["capture"].add_argument("--out", type=Path, required=True)
    sub.choices["run"].add_argument("--inputs", type=Path, required=True)
    sub.choices["run"].add_argument("--save", type=Path, default=None)
    args = parser.parse_args()
    _check_origin()
    return capture(args) if args.command == "capture" else run(args)


if __name__ == "__main__":
    sys.exit(main())
