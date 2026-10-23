#!/usr/bin/env python3
"""One predictor run for a replayed moment: its frames and slot predictions.

    # Reference: the code before a change, from a worktree pinned to it. Records
    # the inputs, every database read (the tape) and each slot's build.
    python scripts/perf/prediction_runtime_parity.py run --code-root <worktree> \
        --date 2026-04-10 --as-of 2026-04-10T22:00:00Z --record \
        --out artifacts/perf/runtime/ref_20260410_pm

    # Candidate: the changed code, on the reference's inputs, tape and builds.
    python scripts/perf/prediction_runtime_parity.py run \
        --date 2026-04-10 --as-of 2026-04-10T22:00:00Z \
        --replay artifacts/perf/runtime/ref_20260410_pm \
        --out artifacts/perf/runtime/cand_20260410_pm

    python scripts/perf/prediction_runtime_parity.py compare \
        artifacts/perf/runtime/ref_20260410_pm artifacts/perf/runtime/cand_20260410_pm

``run`` does what ``predict_nba_games.py`` does from step 3 on: tonight's
closing frame (schema layers included), tonight's intermediate frame and every
enabled slot's predictions. The run's inputs are made identical on both sides:

* tonight's scheduled data and live game day come from storage
  (``live_replay``), are pickled by the reference and loaded by the candidate;
* every database read is recorded to a tape (``db_tape``) and replayed. The
  input cache and the S3 season store are off unless ``--s3-cache``, as in
  ``reference_build.py``;
* each slot predicts with the build the reference resolved (``builds.json``),
  so a promotion in between cannot move a prediction;
* ``PYTHONHASHSEED=0``, and predictions are stamped with ``as_of``.

Nothing shared is written: predictions are not stored, the history memo reads
S3 but writes to ``--memo-dir`` only, and the season store, roster, crew and
injury-report writes are refused. ``timings.json`` holds each stage's seconds
and, per slot, model resolution apart from prediction (step 0's profile).

``compare`` checks both frames with ``compare_builds.compare(exact=True)`` and
the predictions exactly, but for their wall-clock stamp. ``--latest-only`` compares the intermediate frames on
their ``LATEST_INPUT`` rows only (a candidate that builds only those).
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _code_root() -> Path:
    """``--code-root`` from argv, before argparse runs, so imports resolve there."""
    if "--code-root" in sys.argv:
        return Path(sys.argv[sys.argv.index("--code-root") + 1]).resolve()
    return PROJECT_ROOT


CODE_ROOT = _code_root()
sys.path[:0] = [str(CODE_ROOT / "src"), str(CODE_ROOT), str(Path(__file__).parent)]

INPUTS = "inputs.pkl"
BUILDS = "builds.json"
FRAMES = ("closing_line", "intermediate_line")
PREDICTIONS = "predictions.parquet"
SLOT_COLUMN = "SLOT"
#: Stamped with the wall clock when a slot predicts (``prediction.py``), so
#: they differ between any two runs; left out of the comparison.
WALL_CLOCK_COLUMNS = ("PREDICTED_AT_UTC",)


def _ensure_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") != "0":
        env = {**os.environ, "PYTHONHASHSEED": "0"}
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=CODE_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


# -- no shared writes ----------------------------------------------------------


class _Refused:
    """Every refused write, by name."""

    calls: list[str] = []

    @classmethod
    def stub(cls, name: str, result=None):
        def refuse(*args, **kwargs):
            cls.calls.append(name)
            return result

        return refuse


class _MemoOverlay:
    """The production memo read from S3; new entries written locally only."""

    def __init__(self, local, remote) -> None:
        self.local, self.remote = local, remote

    def get(self, key):
        data = self.local.get(key)
        if data is None and self.remote is not None:
            data = self.remote.get(key)
        return data

    def put(self, key, data) -> None:
        self.local.put(key, data)


def _refuse_shared_writes(memo_dir: Path, memo_mode: str) -> None:
    from nba_ou.create_training_data import history_memo
    from nba_ou.fetch_data.live_rosters import RosterStore
    from nba_ou.fetch_data.referees.assignment_store import AssignmentStore
    from nba_ou.postgre_db.season_store import SeasonStore

    memo_dir.mkdir(parents=True, exist_ok=True)
    local = history_memo._LocalBackend(memo_dir)
    if memo_mode == "local":
        history_memo.default_backend = lambda: local
    else:
        from nba_ou.postgre_db.season_store import default_store

        store = default_store()
        remote = history_memo._S3Backend(store) if store is not None else None
        overlay = _MemoOverlay(local, remote)
        history_memo.default_backend = lambda: overlay
    os.environ.pop("NBA_OU_HISTORY_MEMO_DIR", None)

    RosterStore.save = _Refused.stub("RosterStore.save")
    AssignmentStore.save = _Refused.stub("AssignmentStore.save", False)
    SeasonStore.write = _Refused.stub("SeasonStore.write")
    SeasonStore.touch = _Refused.stub("SeasonStore.touch")
    SeasonStore._put = _Refused.stub("SeasonStore._put")

    import nba_ou.create_training_data.live_game_day as live_game_day

    live_game_day.synchronize_injury_reports = _Refused.stub(
        "synchronize_injury_reports"
    )


class _HttpGuard:
    """Log HTTP calls while recording; refuse them on replay."""

    def __init__(self, replaying: bool) -> None:
        self.replaying = replaying
        self.urls: list[str] = []

    def install(self) -> None:
        import requests.sessions

        original = requests.sessions.Session.request
        guard = self

        def request(session, method, url, *args, **kwargs):
            guard.urls.append(f"{method} {url}")
            if guard.replaying:
                raise RuntimeError(
                    f"HTTP request during a replayed run: {method} {url}"
                )
            return original(session, method, url, *args, **kwargs)

        requests.sessions.Session.request = request


# -- inputs and builds ---------------------------------------------------------


def _replayed_inputs(date: str, as_of):
    """Tonight's scheduled data and live game day, from storage."""
    from nba_ou.create_training_data.live_replay import (
        replay_live_game_day,
        replay_scheduled_data,
    )

    scheduled_data = replay_scheduled_data(date, as_of)
    live = replay_live_game_day(date, as_of)
    # What fetch_live_game_day hands the builder in a real run.
    live = dataclasses.replace(
        live,
        scheduled_games=scheduled_data["scheduled_games"],
        df_referees_scheduled=scheduled_data["df_referees_scheduled"],
        roster_snapshots=scheduled_data["roster_snapshots"],
    )
    return scheduled_data, live


@contextlib.contextmanager
def _pinned_builds(builds: dict[str, str], record: bool, slot_seconds: dict):
    """``resolve_channel`` timed per slot; on replay, the recorded build."""
    from nba_ou.modeling import registry_store

    original = registry_store.resolve_channel

    def resolve(*, s3_client, bucket, slot, channel=None):
        started = time.perf_counter()
        label = slot.describe()
        if record:
            kwargs = {} if channel is None else {"channel": channel}
            resolved = original(s3_client=s3_client, bucket=bucket, slot=slot, **kwargs)
            builds[label] = resolved.fit.fit_id
        else:
            resolved = registry_store.resolve_build(
                s3_client=s3_client, bucket=bucket, slot=slot, fit_id=builds[label]
            )
        slot_seconds[label] = {"resolve": round(time.perf_counter() - started, 2)}
        return resolved

    registry_store.resolve_channel = resolve
    try:
        yield
    finally:
        registry_store.resolve_channel = original


# -- run -------------------------------------------------------------------------


def _parameters(function) -> set[str]:
    import inspect

    return set(inspect.signature(function).parameters)


def _closing_frame(scheduled_data, date: str, configured_slots):
    """``predict_nba_games``' step 3: tonight's closing rows, layers included."""
    import pandas as pd
    from nba_ou.config.dataset_versions import BASE_SCHEMA_VERSION
    from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict
    from nba_ou.create_training_data.schema_layers import (
        CLOSING_LINE,
        compute_new_columns,
        is_layered,
        newest_version,
    )

    tonight_ids = scheduled_data["scheduled_games"]["GAME_ID"].astype(str).tolist()
    frame, layer_inputs = create_df_to_predict(
        todays_prediction=True,
        scheduled_data=scheduled_data,
        strict_mode=30,
        target_game_ids=tonight_ids,
        return_layer_inputs=True,
    )
    version = newest_version(slot.schema_version for slot in configured_slots)
    if is_layered(version):
        added = compute_new_columns(
            layer_inputs,
            from_version=BASE_SCHEMA_VERSION,
            to_version=version,
            dataset_type=CLOSING_LINE,
            existing_columns=frame.columns,
        )
        frame = pd.concat([frame, added.loc[frame.index]], axis=1)
    return frame[frame["GAME_DATE"] == date].copy(), version


def run(args: argparse.Namespace) -> int:
    import pandas as pd

    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out if args.record else args.replay
    if args.s3_cache:
        os.environ["NBA_OU_INPUT_CACHE_DIR"] = str(args.out / "input_cache")
    else:
        # A tape holds the run's own queries. The input cache would add
        # fingerprint queries and skip fetches (see reference_build.py).
        os.environ["NBA_OU_INPUT_CACHE"] = "0"
        os.environ["NBA_OU_S3_SEASONS"] = "0"
    _refuse_shared_writes(args.memo_dir or args.out / "memo", args.memo)
    http = _HttpGuard(replaying=not args.record)

    import db_tape
    from nba_ou.config.dataset_versions import BASE_SCHEMA_VERSION
    from nba_ou.config.settings import SETTINGS
    from nba_ou.prediction.intermediate_frame import (
        PASSTHROUGH_COLUMNS,
        tonight_intermediate_frame,
    )
    from nba_ou.prediction.prediction import load_registry_model_and_predict
    from nba_ou.utils.s3_models import make_s3_client

    from scripts.predict_nba_games import latest_input_slots

    as_of = pd.Timestamp(args.as_of)
    timings: dict = {}
    started = time.perf_counter()
    if args.record:
        scheduled_data, live = _replayed_inputs(args.date, as_of)
        with open(args.out / INPUTS, "wb") as handle:
            pickle.dump({"scheduled_data": scheduled_data, "live": live}, handle)
        builds: dict[str, str] = {}
    else:
        with open(source / INPUTS, "rb") as handle:
            inputs = pickle.load(handle)
        scheduled_data, live = inputs["scheduled_data"], inputs["live"]
        builds = json.loads((source / BUILDS).read_text())
    timings["inputs"] = round(time.perf_counter() - started, 1)
    http.install()

    import nba_ou.prediction.intermediate_frame as intermediate_frame

    # The replayed game day stands in for the live scrape.
    intermediate_frame.fetch_live_game_day = lambda *a, **k: live

    configured_slots = SETTINGS.prediction_model_slots
    latest_slots = latest_input_slots(configured_slots, args.intermediate_input)
    tape = (
        db_tape.record(args.out / "tape")
        if args.record
        else db_tape.replay(source / "tape", subset_cutoffs=True)
    )
    slot_seconds: dict = {}
    failures: list[str] = []
    with tape:
        started = time.perf_counter()
        closing, frame_version = _closing_frame(
            scheduled_data, args.date, configured_slots
        )
        timings["closing_rows"] = round(time.perf_counter() - started, 1)

        started = time.perf_counter()
        # As predict_nba_games, whose code may predate latest_only.
        latest_only = bool(latest_slots) and all(
            slot in latest_slots for slot in configured_slots if slot.horizon_minutes
        )
        intermediate = tonight_intermediate_frame(
            scheduled_data,
            game_date=args.date,
            as_of=as_of,
            latest_input=bool(latest_slots),
            **(
                {"latest_only": True}
                if latest_only
                and "latest_only" in _parameters(tonight_intermediate_frame)
                else {}
            ),
        )
        timings["intermediate_rows"] = round(time.perf_counter() - started, 1)

        frames = {"closing_line": closing, "intermediate_line": intermediate}
        frame_versions = {
            "closing_line": frame_version,
            "intermediate_line": BASE_SCHEMA_VERSION,
        }
        s3 = make_s3_client(
            profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region
        )
        prediction_time = as_of.tz_convert("Europe/Madrid").to_pydatetime()
        predictions = []
        started = time.perf_counter()
        with _pinned_builds(builds, args.record, slot_seconds):
            for slot in configured_slots:
                label = slot.describe()
                slot_started = time.perf_counter()
                try:
                    predicted = load_registry_model_and_predict(
                        s3_client=s3,
                        bucket=SETTINGS.s3_bucket,
                        slot=slot,
                        df=closing,
                        prediction_datetime=prediction_time,
                        frames=frames,
                        frame_versions=frame_versions,
                        passthrough_columns=PASSTHROUGH_COLUMNS,
                        persist=False,
                        latest_input=slot in latest_slots,
                    )
                except Exception as exc:  # noqa: BLE001 -- as the predictor does
                    failures.append(f"{label}: {exc}")
                    continue
                total = time.perf_counter() - slot_started
                entry = slot_seconds.setdefault(label, {})
                entry["total"] = round(total, 2)
                entry["predict"] = round(total - entry.get("resolve", 0.0), 2)
                if predicted is not None and not predicted.empty:
                    predictions.append(predicted.assign(**{SLOT_COLUMN: label}))
        timings["slots"] = round(time.perf_counter() - started, 1)

    for name, frame in frames.items():
        frame.reset_index(drop=True).to_parquet(args.out / f"{name}.parquet")
    if predictions:
        pd.concat(predictions, ignore_index=True).to_parquet(args.out / PREDICTIONS)
    if args.record:
        (args.out / BUILDS).write_text(json.dumps(builds, indent=1, sort_keys=True))
    manifest = {
        "date": args.date,
        "as_of": as_of.isoformat(),
        "mode": "record" if args.record else f"replay {source}",
        "code_root": str(CODE_ROOT),
        "commit": _git("rev-parse", "HEAD"),
        "dirty_files": _git("status", "--porcelain", "--untracked-files=no").split(
            "\n"
        ),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "s3_cache": args.s3_cache,
        "memo": args.memo,
        "latest_slots": len(latest_slots),
        "rows": {name: len(frame) for name, frame in frames.items()},
        "predictions": sum(len(p) for p in predictions),
        "failures": failures,
        "refused_writes": sorted(set(_Refused.calls)),
        "http_requests": len(http.urls),
        "timings": timings,
        "slot_seconds": slot_seconds,
    }
    (args.out / "timings.json").write_text(json.dumps(manifest, indent=1))
    print(
        json.dumps({k: v for k, v in manifest.items() if k != "slot_seconds"}, indent=1)
    )
    return 1 if failures else 0


# -- compare -----------------------------------------------------------------------


def compare_runs(args: argparse.Namespace) -> int:
    import pandas as pd
    from compare_builds import compare
    from nba_ou.prediction.input_limits import LATEST_INPUT_COLUMN

    ok = True
    for name in FRAMES:
        reference = pd.read_parquet(args.reference / f"{name}.parquet")
        candidate = pd.read_parquet(args.candidate / f"{name}.parquet")
        if name == "intermediate_line" and args.latest_only:
            reference = reference[reference[LATEST_INPUT_COLUMN]].reset_index(drop=True)
            candidate = candidate[candidate[LATEST_INPUT_COLUMN]].reset_index(drop=True)
        print(f"--- {name}")
        ok &= compare(reference, candidate, exact=True)

    print("--- predictions")
    paths = [args.reference / PREDICTIONS, args.candidate / PREDICTIONS]
    if not all(path.exists() for path in paths):
        print(f"predictions missing: {[str(p) for p in paths if not p.exists()]}")
        ok = False
    else:
        # One row per slot and game (and snapshot): the slot joins the key.
        reference, candidate = (
            pd.read_parquet(path)
            .drop(columns=list(WALL_CLOCK_COLUMNS))
            .assign(GAME_ID=lambda p: p[SLOT_COLUMN] + "|" + p["GAME_ID"].astype(str))
            for path in paths
        )
        ok &= compare(reference, candidate, exact=True)
    print("OVERALL:", "EQUAL" if ok else "DIFFERENT")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="build and predict one moment")
    run_parser.add_argument("--date", required=True, help="the game date replayed")
    run_parser.add_argument("--as-of", required=True, help="the moment, UTC")
    run_parser.add_argument("--out", type=Path, required=True)
    source = run_parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--record", action="store_true", help="record inputs, tape and builds"
    )
    source.add_argument(
        "--replay", type=Path, help="a reference run's directory to replay"
    )
    run_parser.add_argument(
        "--code-root", type=Path, help="the checkout to run (default: this one)"
    )
    run_parser.add_argument(
        "--memo",
        choices=("s3", "local"),
        default="s3",
        help="history memo: S3 read with local writes (default), or local only",
    )
    run_parser.add_argument(
        "--memo-dir", type=Path, help="local memo directory (default: <out>/memo)"
    )
    run_parser.add_argument(
        "--s3-cache",
        action="store_true",
        help="keep the input cache and S3 season store on (to test them)",
    )
    run_parser.add_argument(
        "--intermediate-input",
        choices=("latest", "own"),
        help="override INTERMEDIATE_INPUT_MODE, as the predictor's flag",
    )
    run_parser.set_defaults(func=run)

    compare_parser = commands.add_parser("compare", help="compare two runs exactly")
    compare_parser.add_argument("reference", type=Path)
    compare_parser.add_argument("candidate", type=Path)
    compare_parser.add_argument(
        "--latest-only",
        action="store_true",
        help="compare the intermediate frames on their LATEST_INPUT rows only",
    )
    compare_parser.set_defaults(func=compare_runs)

    args = parser.parse_args()
    if args.command == "run":
        _ensure_hash_seed()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
