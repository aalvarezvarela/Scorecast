"""Start, progress, wall time and memory peak of a job's phases.

A ``[time]`` line per phase lets a job's log say where its minutes went
(``docs/serving/runner_timings.md``); the memory peak says whether two
phases could run side by side on the runner. In Actions, a heartbeat every
minute identifies the phase still running. ``report`` keeps timing summaries
available when a command fails or returns early.
"""

from __future__ import annotations

import resource
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from nba_ou.utils.run_log import RunLog, in_github_actions


def duration(seconds: float) -> str:
    minutes, seconds = divmod(int(round(seconds)), 60)
    return f"{minutes}m {seconds:02d}s"


def peak_rss() -> str:
    """This process's memory peak so far, and the largest finished child's
    (the intermediate builder forks its horizons). Linux reports KiB."""
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
    child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2
    return f"{own:.1f} GB" + (f" (largest child {child:.1f} GB)" if child else "")


class PhaseTimer:
    """Times phases in turn; ``summary`` lists them with the job's total."""

    def __init__(self, *, heartbeat_seconds: float | None = None) -> None:
        self.started = time.perf_counter()
        self.phases: list[tuple[str, float, bool]] = []
        # Local runs stay quiet between boundaries; Actions gets a heartbeat
        # when a scraper, feature build or hosted fit takes several minutes.
        self.heartbeat_seconds = (
            (60.0 if in_github_actions() else 0.0)
            if heartbeat_seconds is None
            else heartbeat_seconds
        )

    def _heartbeat(self, label: str, started: float, stopped: threading.Event) -> None:
        while not stopped.wait(self.heartbeat_seconds):
            print(
                f"[progress] {label}: still running, "
                f"{duration(time.perf_counter() - started)} elapsed",
                flush=True,
            )

    @contextmanager
    def phase(self, label: str) -> Iterator[None]:
        started = time.perf_counter()
        print(f"[start] {label}", flush=True)
        stopped = threading.Event()
        heartbeat = None
        if self.heartbeat_seconds > 0:
            heartbeat = threading.Thread(
                target=self._heartbeat,
                args=(label, started, stopped),
                daemon=True,
            )
            heartbeat.start()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            stopped.set()
            if heartbeat is not None:
                heartbeat.join()
            self.record(label, time.perf_counter() - started, failed=failed)

    @contextmanager
    def report(self, title: str) -> Iterator[None]:
        """Print and publish timings even after an exception or early return.

        This cannot report a forcibly killed process. Nested phases overlap;
        the total is elapsed wall time, rather than the sum of the rows.
        """
        interrupted = False
        try:
            yield
        except BaseException:
            interrupted = True
            raise
        finally:
            print(self.summary(), flush=True)
            RunLog().summary(
                f"### {title}{' (interrupted)' if interrupted else ''}\n\n"
                + self.markdown()
            )

    def record(
        self,
        label: str,
        seconds: float,
        *,
        failed: bool = False,
        peak: str | None = None,
    ) -> None:
        """A phase timed by the caller, as ``phase`` records its own.

        ``peak`` is the memory peak to print in place of this process's (a
        phase run in another process).
        """
        self.phases.append((label, seconds, failed))
        print(
            f"[time] {label}: {duration(seconds)}"
            f"{' (did not finish)' if failed else ''}; peak RSS {peak or peak_rss()}",
            flush=True,
        )

    def summary(self) -> str:
        width = max((len(label) for label, _, _ in self.phases), default=0)
        lines = ["Timings (wall clock):"]
        for label, seconds, failed in self.phases:
            lines.append(
                f"  {label:<{width}}  {duration(seconds)}"
                + ("  (did not finish)" if failed else "")
            )
        lines.append(
            f"  {'total':<{width}}  {duration(time.perf_counter() - self.started)}"
            f"; peak RSS {peak_rss()}"
        )
        return "\n".join(lines)

    def markdown(self) -> str:
        """``summary`` as a Markdown table (a GitHub Actions job summary)."""
        lines = ["| Phase | Wall clock |", "| --- | ---: |"]
        for label, seconds, failed in self.phases:
            lines.append(
                f"| {label}{' (did not finish)' if failed else ''} "
                f"| {duration(seconds)} |"
            )
        lines.append(
            f"| **total** (peak RSS {peak_rss()}) "
            f"| **{duration(time.perf_counter() - self.started)}** |"
        )
        return "\n".join(lines)
