"""Prevent two processes from writing the same raw manifest concurrently."""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path


class BackfillAlreadyRunning(RuntimeError):
    pass


@contextmanager
def lineup_run_lock(local_root: Path):
    path = local_root / "nba_api_raw" / "backfill.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Append mode does not truncate the current owner's diagnostic text before
    # a competing process discovers that it cannot acquire the lock.
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BackfillAlreadyRunning(
                f"Another lineup fetch/build process holds {path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
