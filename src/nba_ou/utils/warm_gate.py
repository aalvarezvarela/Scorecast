"""Lets two builds share the machine only while both stay warm.

Builds that run side by side each hold a file lock, shared. A warm build reads
every past season from the history memo and fits next to the other one; a
cold one computes past seasons, the slow and memory-heavy part. So when a
build starts computing one (``HistoryMemo.table`` reports it) it takes the
lock exclusive: it waits there until the other build has finished and
released it, and goes on alone. The current season is part of every warm run
(its digest moves with each game and report) and passes.

A process without a gate (``install`` never called) is never held.
"""

from __future__ import annotations

import fcntl
import os
import threading
import time

_GATE: WarmGate | None = None


class WarmGate:
    """One build's hold on the shared lock at ``path``.

    Opened per process, never inherited: a forked copy of the descriptor
    would share this process's lock instead of contending for it.
    """

    def __init__(self, path: str | os.PathLike, *, current_season: int, label: str):
        self.current_season = int(current_season)
        self.label = label
        #: What made this build cold, once it is (``"<memo> <season>"``).
        self.cold_cause: str | None = None
        #: Seconds this build waited for the other one to finish.
        self.waited = 0.0
        self._lock = threading.Lock()
        self._fd: int | None = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(self._fd, fcntl.LOCK_SH)

    def cold(self, what: str, season: int) -> None:
        """A memo miss of ``what`` for ``season``; waits when it is a past one."""
        if int(season) >= self.current_season:
            return
        with self._lock:
            if self.cold_cause is not None or self._fd is None:
                return
            self.cold_cause = f"{what} {season}"
            print(
                f"Warm gate: {self.label} is cold ({self.cold_cause}); "
                "it waits for the build beside it to finish",
                flush=True,
            )
            started = time.perf_counter()
            # Shared to exclusive is not atomic on Linux: the shared hold is
            # dropped first, so two builds turning cold at once cannot deadlock.
            fcntl.flock(self._fd, fcntl.LOCK_EX)
            self.waited = time.perf_counter() - started
            print(
                f"Warm gate: {self.label} goes on alone after waiting "
                f"{self.waited:.0f} s",
                flush=True,
            )

    def release(self) -> None:
        """Lets the other build go on alone; this build takes no more turns."""
        if self._fd is None:
            return
        fcntl.flock(self._fd, fcntl.LOCK_UN)
        os.close(self._fd)
        self._fd = None


def install(gate: WarmGate | None) -> None:
    """Makes ``gate`` the one this process's memo misses report to."""
    global _GATE
    _GATE = gate


def cold(what: str, season) -> None:
    """Reports a memo miss to the installed gate, if any (see ``WarmGate.cold``)."""
    if _GATE is None:
        return
    try:
        season = int(season)
    except (TypeError, ValueError):
        return
    _GATE.cold(what, season)
