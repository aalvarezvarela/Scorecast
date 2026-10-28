"""One build in a side process, beside the one this process runs.

The predictor builds tonight's closing rows here while it builds the
intermediate rows itself: the two are independent until the predictions, and
on a warm run each is mostly one core's work. Both hold the warm gate
(``nba_ou.utils.warm_gate``), so a build that turns cold waits for the other
and finishes alone: the side by side part is only ever two warm builds.

The side process is spawned, not forked: a fork would share this process's
open S3 and database connections with it. Its output goes to a log file, so
the two builds' lines do not mix: ``result`` prints it once this process's
build is done, what it holds so far and then live, each line marked with the
side build's label, and a line every few minutes when the build is quiet (a
cold build computes for long stretches without printing).
"""

from __future__ import annotations

import codecs
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
#: A quiet side build says it is still running this often while waited for.
HEARTBEAT_SECONDS = 120.0
#: How often the side build's log is read while it is followed.
POLL_SECONDS = 0.5


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
    with open(Path(workdir) / RESULT_FILE, "wb") as out:
        pickle.dump(
            (*outcome, seconds, gate.cold_cause, gate.waited, peak_rss()),
            out,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


@dataclass
class SideResult:
    value: object
    seconds: float
    cold_cause: str | None
    waited: float
    #: The side process's memory peak (``phase_timer.peak_rss``).
    peak: str | None = None


class SideBuild:
    """``fn(*args, **kwargs)`` running in a side process.

    ``start`` also gives this process its own hold on the gate: the build it
    runs meanwhile is gated too until ``result`` (or ``release_gate``).
    """

    def __init__(
        self,
        fn,
        args,
        kwargs,
        *,
        current_season: int,
        label: str,
        own_label: str = "this build",
    ):
        self.label = label
        self.started = time.perf_counter()
        self.workdir = tempfile.mkdtemp(prefix="nba_ou_side_build_")
        self.gate = warm_gate.WarmGate(
            Path(self.workdir) / LOCK_FILE,
            current_season=current_season,
            label=own_label,
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
    def start(
        cls,
        fn,
        *args,
        current_season: int,
        label: str,
        own_label: str = "this build",
        **kwargs,
    ):
        """``own_label`` names this process's build in the gate's messages."""
        return cls(
            fn,
            args,
            kwargs,
            current_season=current_season,
            label=label,
            own_label=own_label,
        )

    def release_gate(self) -> None:
        """This process's build is done: the side build may go on alone."""
        self.gate.release()
        warm_gate.install(None)

    def result(self, *, heartbeat_seconds: float = HEARTBEAT_SECONDS):
        """Waits for the side build, printing its log; None when it died
        without a result (killed, e.g. out of memory), so the caller can
        build it itself.

        Raises ``SideBuildFailed`` when it raised.
        """
        self.release_gate()
        workdir = Path(self.workdir)
        try:
            state = "running" if self.process.is_alive() else "done"
            print(
                f"{self.label} (side process, {state} after "
                f"{duration(time.perf_counter() - self.started)}): its log, "
                f"lines marked [{self.label}]",
                flush=True,
            )
            self._follow(workdir / LOG_FILE, heartbeat_seconds)
            self.process.join()
            path = workdir / RESULT_FILE
            if not path.exists():
                print(
                    f"{self.label}: the side process ended without a result "
                    f"(exit code {self.process.exitcode})",
                    flush=True,
                )
                return None
            with open(path, "rb") as source:
                status, value, seconds, cold_cause, waited, peak = pickle.load(source)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        if status == "error":
            raise SideBuildFailed(f"{self.label} failed in the side process:\n{value}")
        return SideResult(value, seconds, cold_cause, waited, peak)

    def _follow(self, log: Path, heartbeat_seconds: float) -> None:
        """Prints ``log`` as the side process writes it, until it ends."""
        mark = f"[{self.label}]"
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        offset = 0
        quiet_since = time.perf_counter()
        while True:
            # Read after the check: what an ended process wrote is all there.
            alive = self.process.is_alive()
            chunk = ""
            if log.exists():
                with open(log, "rb") as source:
                    source.seek(offset)
                    data = source.read()
                offset += len(data)
                chunk = decoder.decode(data, final=not alive)
            if chunk:
                *lines, pending = (pending + chunk).split("\n")
                for line in lines:
                    print(f"{mark} {line}".rstrip(), flush=True)
                quiet_since = time.perf_counter()
            if not alive:
                break
            if time.perf_counter() - quiet_since >= heartbeat_seconds:
                print(
                    f"{mark} ... still running "
                    f"({duration(time.perf_counter() - self.started)})",
                    flush=True,
                )
                quiet_since = time.perf_counter()
            time.sleep(POLL_SECONDS)
        if pending:
            print(f"{mark} {pending}".rstrip(), flush=True)
