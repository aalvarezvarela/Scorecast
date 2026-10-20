#!/usr/bin/env python3
"""Build a 2_5 dataset as a speed-up reference, recording or replaying its inputs.

    # Record: live databases, every query result saved to <out>/tape
    python scripts/perf/reference_build.py closing --limit 2026-04-10 --n-seasons 2 \
        --mode record --out artifacts/perf/ref/closing_2s

    # Replay: the same build with every query answered from that tape
    python scripts/perf/reference_build.py closing --limit 2026-04-10 --n-seasons 2 \
        --mode replay --tape artifacts/perf/ref/closing_2s/tape \
        --out artifacts/perf/cand/closing_2s

Then compare the two outputs with ``compare_builds.py --exact``.

Output in ``--out``: ``dataset.parquet`` (plus ``scoring.parquet`` for
intermediate), written with the pipeline's own ``write_training_dataset``, and
``manifest.json`` with the commit, the build arguments, the timings, the tape
summary and the frame digest.

Pins ``PYTHONHASHSEED=0`` by re-executing itself: roster continuity's last bits
depend on set iteration order, and a reference that is not reproducible
cannot gate anything.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _code_root() -> Path:
    """``--code-root`` from argv, before argparse runs, so imports resolve there.

    Reference builds run from a worktree pinned to one commit, so the checkout
    switching branch mid-build cannot mix two versions of the code.
    """
    if "--code-root" in sys.argv:
        return Path(sys.argv[sys.argv.index("--code-root") + 1]).resolve()
    return PROJECT_ROOT


CODE_ROOT = _code_root()
sys.path[:0] = [str(CODE_ROOT / "src"), str(CODE_ROOT), str(Path(__file__).parent)]


def _ensure_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=CODE_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


class _HttpGuard:
    """Log HTTP calls while recording; refuse them on replay."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.urls: list[str] = []

    def install(self) -> None:
        try:
            import requests.sessions
        except ImportError:
            return
        original = requests.sessions.Session.request
        guard = self

        def request(session, method, url, *args, **kwargs):
            guard.urls.append(f"{method} {url}")
            if guard.mode == "replay":
                raise RuntimeError(
                    f"HTTP request during a replayed build: {method} {url}"
                )
            return original(session, method, url, *args, **kwargs)

        requests.sessions.Session.request = request


def _build(args: argparse.Namespace):
    if args.kind == "closing":
        from nba_ou.create_training_data.create_df_to_predict import (
            create_df_to_predict,
        )

        targets = (
            args.targets.read_text().split() if args.targets is not None else None
        )
        built = create_df_to_predict(
            todays_prediction=False,
            recent_limit_to_include=args.limit,
            older_season_limit=args.n_seasons,
            target_game_ids=targets,
            return_layer_inputs=targets is not None,
        )
        if targets is None:
            return built, None
        frame, layer_inputs = built
        # Every built row's layer inputs: a full build's own key columns.
        layer_inputs.reset_index(drop=True).to_parquet(args.out / "layer_inputs.parquet")
        return frame, None
    from nba_ou.create_training_data.create_intermediate_line_df import (
        create_intermediate_line_df,
    )

    seasons = [int(s) for s in args.seasons.split(",")] if args.seasons else None
    return create_intermediate_line_df(
        recent_limit_to_include=args.limit,
        season_years=seasons,
        return_scoring=True,
        target_game_ids=(
            args.targets.read_text().split() if args.targets is not None else None
        ),
        verbose=True,
    )


def main() -> int:
    _ensure_hash_seed()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("kind", choices=["closing", "intermediate"])
    parser.add_argument("--limit", required=True, help="recent_limit_to_include")
    parser.add_argument(
        "--n-seasons", type=int, default=None, help="closing: older_season_limit"
    )
    parser.add_argument(
        "--seasons", default=None, help="intermediate: comma-separated season years"
    )
    parser.add_argument("--mode", choices=["record", "replay", "live"], required=True)
    parser.add_argument(
        "--tape", type=Path, default=None, help="replay: tape directory"
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help="replay: serve each result in another order its ORDER BY permits",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=None,
        help="file of GAME_IDs to build alone (target_game_ids)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--code-root",
        type=Path,
        default=PROJECT_ROOT,
        help="checkout whose src/ is built (default: this one)",
    )
    args = parser.parse_args()

    if args.mode == "replay" and args.tape is None:
        parser.error("--mode replay needs --tape")
    args.out.mkdir(parents=True, exist_ok=False)

    import db_tape
    import nba_ou
    import training_pipeline
    from compare_builds import frame_digest

    for module in (nba_ou, training_pipeline):
        if not Path(module.__file__).resolve().is_relative_to(CODE_ROOT):
            raise SystemExit(
                f"{module.__name__} imported from {module.__file__}, not {CODE_ROOT}"
            )

    guard = _HttpGuard(args.mode)
    guard.install()
    if args.mode in ("record", "replay"):
        # A tape holds the build's own queries. The input cache would add
        # fingerprint queries and skip fetches, so taped builds bypass it.
        os.environ["NBA_OU_INPUT_CACHE"] = "0"
        os.environ["NBA_OU_S3_SEASONS"] = "0"
        # The history memo only from a local directory, never from S3.
        if not os.environ.get("NBA_OU_HISTORY_MEMO_DIR"):
            os.environ["NBA_OU_HISTORY_MEMO"] = "0"
    if args.mode == "record":
        context = db_tape.record(args.out / "tape")
    elif args.mode == "replay":
        perturb = None
        if args.shuffle_seed is not None:
            import random

            from row_order import shuffle_within_order

            rng = random.Random(args.shuffle_seed)
            perturb = (
                lambda query, description, rows: shuffle_within_order(  # noqa: E731
                    query, description, rows, rng
                )
            )
        context = db_tape.replay(args.tape, perturb=perturb)
    else:
        import contextlib

        context = contextlib.nullcontext()

    started = time.perf_counter()
    with context:
        frame, scoring = _build(args)
    build_seconds = time.perf_counter() - started

    from training_pipeline.parquet_dataset import write_training_dataset

    dataset_sha = write_training_dataset(frame, args.out / "dataset.parquet")
    scoring_sha = (
        write_training_dataset(scoring, args.out / "scoring.parquet")
        if scoring is not None
        else None
    )
    import pandas as pd

    written = pd.read_parquet(args.out / "dataset.parquet")
    manifest = {
        "kind": args.kind,
        "limit": args.limit,
        "n_seasons": args.n_seasons,
        "seasons": args.seasons,
        "mode": args.mode,
        "tape": str(args.tape or (args.out / "tape" if args.mode == "record" else "")),
        "code_root": str(CODE_ROOT),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": _git(
            "status", "--porcelain", "--untracked-files=no"
        ).splitlines(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "shuffle_seed": args.shuffle_seed,
        "targets": str(args.targets) if args.targets else None,
        "input_cache": os.environ.get("NBA_OU_INPUT_CACHE", "1") != "0",
        "s3_seasons": os.environ.get("NBA_OU_S3_SEASONS", "1") != "0",
        "input_cache_dir": os.environ.get("NBA_OU_INPUT_CACHE_DIR"),
        "history_memo": os.environ.get("NBA_OU_HISTORY_MEMO", "1") != "0",
        "history_memo_dir": os.environ.get("NBA_OU_HISTORY_MEMO_DIR"),
        "build_seconds": round(build_seconds, 1),
        "shape": list(written.shape),
        "dataset_file_sha256": dataset_sha,
        "scoring_file_sha256": scoring_sha,
        "frame_digest": frame_digest(written),
        "http_requests": guard.urls,
        "created": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    if args.mode == "record":
        manifest["tape_summary"] = db_tape.tape_summary(args.out / "tape")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        json.dumps({k: v for k, v in manifest.items() if k != "dirty_files"}, indent=1)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
