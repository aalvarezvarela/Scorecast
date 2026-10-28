"""A prediction job's log as numbered steps, easy to follow in GitHub Actions.

Each step prints its number, title and the clock. In GitHub Actions a step is
also a collapsible group, so a run's log reads as its list of steps and the
closing summary, printed outside every group, is what shows at the end.
Warnings and failures become annotations on the run's page as well as lines
of the log, and ``summary`` adds a table to the job's summary page
(``$GITHUB_STEP_SUMMARY``).
"""

from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

WIDTH = 70


def in_github_actions() -> bool:
    return os.getenv("GITHUB_ACTIONS", "").lower() == "true"


def _escaped(message: str) -> str:
    """``message`` as a workflow command's data (one line)."""
    return message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


class RunLog:
    """The steps of one run, numbered in the order they start."""

    def __init__(self, *, github: bool | None = None) -> None:
        self.github = in_github_actions() if github is None else github
        self.steps = 0
        self._grouped = False

    def step(self, title: str, *, group: bool = True) -> None:
        """Starts the next step; ``group=False`` keeps it outside any group."""
        self.end_group()
        self.steps += 1
        clock = datetime.now(ZoneInfo("UTC")).strftime("%H:%M:%S")
        header = f"STEP {self.steps} — {title} [{clock} UTC]"
        if self.github and group:
            print(f"::group::{header}", flush=True)
            self._grouped = True
            return
        print("-" * WIDTH)
        print(f" {header} ".center(WIDTH))
        print("-" * WIDTH, flush=True)

    def end_group(self) -> None:
        """Closes the step's group, so what follows (a traceback) shows."""
        if self._grouped:
            print("::endgroup::", flush=True)
            self._grouped = False

    def status(self, message: str, ok: bool = True) -> None:
        print(f"  {'✓' if ok else '✖'} {message}", flush=True)

    def warning(self, message: str) -> None:
        """A line of the log, and in GitHub Actions a warning on the run's page."""
        print(f"  ⚠ {message}", flush=True)
        if self.github:
            print(f"::warning::{_escaped(message)}", flush=True)

    def error(self, message: str) -> None:
        """A line of the log, and in GitHub Actions an error on the run's page."""
        print(f"  ✖ {message}", flush=True)
        if self.github:
            print(f"::error::{_escaped(message)}", flush=True)

    def summary(self, markdown: str) -> None:
        """Adds ``markdown`` to the job's summary page, in GitHub Actions."""
        path = os.getenv("GITHUB_STEP_SUMMARY")
        if not path:
            return
        try:
            with open(path, "a", encoding="utf-8") as out:
                out.write(markdown.rstrip() + "\n\n")
        except OSError as exc:
            print(f"  (job summary not written: {exc})")


def progress_disabled(label: str | None) -> bool | None:
    """tqdm's ``disable`` for a bar shown only when given a ``label``.

    None, not False, when there is one: tqdm then hides the bar on a log
    that is not a terminal (a GitHub Actions job) and keeps it in a terminal.
    An explicit False would override ``TQDM_DISABLE``.
    """
    return True if label is None else None
