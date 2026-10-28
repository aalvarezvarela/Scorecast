"""One build in a side process, beside the one this process runs.

The predictor builds tonight's closing rows here while it builds the
intermediate rows itself: the two are independent until the predictions, and
on a warm run each is mostly one core's work. Both hold the warm gate
(``nba_ou.utils.warm_gate``), so a build that turns cold waits for the other
and finishes alone: the side by side part is only ever two warm builds.

The side process is spawned, not forked: a fork would share this process's
open S3 and database connections with it. Its output goes to a log file,
printed here in one block when it ends, so the two builds' lines do not mix.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from nba_ou.create_training_data.history_memo import (
    default_backend,
    history_memo_rebuild,
)
from nba_ou.utils import warm_gate
from nba_ou.utils.phase_timer import duration, peak_rss

LOCK_FILE = "warm.lock"
LOG_FILE = "side.log"
RESULT_FILE = "result.pkl"


class SideBuildFailed(RuntimeError):
    """The side build raised; the message carries its traceback."""


def side_by_side_reason() -> str | None:
    """Why the builds cannot run side by side here, or None when they can.

    Without a usable history memo every season is computed: the run is cold
    from the start, and the gate would never see a miss to hold it at.
    """
    if "spawn" not in multiprocessing.get_all_start_methods():
        return "no spawn start method"
    if history_memo_rebuild():
        return "the history memo is being rebuilt"
    if default_backend() is None:
        return "no history memo"
    return None


def _run(fn, args, kwargs, workdir: str, current_season: int, label: str) -> None:
    """The side process: runs ``fn`` under the gate and leaves its result."""
    log = open(Path(workdir) / LOG_FILE, "w", buffering=1)  # noqa: SIM115
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = sys.stderr = log
    gate = warm_gate.WarmGate(
        Path(workdir) / LOCK_FILE, current_season=current_season, label=label
    )
    warm_gate.install(gate)
    started = time.perf_counter()
    try:
        outcome = ("ok", fn(*args, **kwargs))
    except BaseException:  # noqa: BLE001 -- handed to the parent to raise
        traceback.print_exc()
        outcome = ("error", traceback.format_exc())
    seconds = time.perf_counter() - started
    gate.release()
    print(f"[time] {label} (side process): {duration(seconds)}; peak RSS {peak_rss()}")
    with open(Path(workdir) / RESULT_FILE, "wb") as out:
        pickle.dump(
            (*outcome, seconds, gate.cold_cause, gate.waited),
            out,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


@dataclass
class SideResult:
    value: object
    seconds: float
    cold_cause: str | None
    waited: float


class SideBuild:
    """``fn(*args, **kwargs)`` running in a side process.

    ``start`` also gives this process its own hold on the gate: the build it
    runs meanwhile is gated too until ``result`` (or ``release_gate``).
    """

    def __init__(self, fn, args, kwargs, *, current_season: int, label: str):
        self.label = label
        self.workdir = tempfile.mkdtemp(prefix="nba_ou_side_build_")
        self.gate = warm_gate.WarmGate(
            Path(self.workdir) / LOCK_FILE,
            current_season=current_season,
            label="this build",
        )
        warm_gate.install(self.gate)
        context = multiprocessing.get_context("spawn")
        self.process = context.Process(
            target=_run,
            args=(fn, args, kwargs, self.workdir, current_season, label),
            name=label,
        )
        self.process.start()

    @classmethod
    def start(cls, fn, *args, current_season: int, label: str, **kwargs):
        return cls(fn, args, kwargs, current_season=current_season, label=label)

    def release_gate(self) -> None:
        """This process's build is done: the side build may go on alone."""
        self.gate.release()
        warm_gate.install(None)

    def result(self) -> SideResult | None:
        """Waits for the side build; None when it died without a result
        (killed, e.g. out of memory), so the caller can build it itself.

        Raises ``SideBuildFailed`` when it raised.
        """
        self.release_gate()
        self.process.join()
        workdir = Path(self.workdir)
        try:
            log = workdir / LOG_FILE
            print(f"----- {self.label} (side process) -----", flush=True)
            if log.exists():
                print(log.read_text(errors="replace"), end="", flush=True)
            print(f"----- end of {self.label} -----", flush=True)
            path = workdir / RESULT_FILE
            if not path.exists():
                print(
                    f"{self.label}: the side process ended without a result "
                    f"(exit code {self.process.exitcode})",
                    flush=True,
                )
                return None
            with open(path, "rb") as source:
                status, value, seconds, cold_cause, waited = pickle.load(source)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        if status == "error":
            raise SideBuildFailed(f"{self.label} failed in the side process:\n{value}")
        return SideResult(value, seconds, cold_cause, waited)
