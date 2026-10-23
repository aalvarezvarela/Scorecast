"""Messages that stand out in a GitHub Actions run, plain text anywhere else."""

from __future__ import annotations

import os


def annotate(level: str, message: str) -> None:
    """Print ``message``; under GitHub Actions also as a run annotation.

    ``level`` is ``"warning"`` or ``"error"``. An annotation shows on the run's
    summary page, so a degraded-but-green run is still noticed.
    """
    if os.getenv("GITHUB_ACTIONS", "").lower() == "true":
        # Annotations are one line; newlines would cut the message short.
        print(f"::{level}::{message.replace(chr(10), ' ')}")
    else:
        print(f"{level.upper()}: {message}")
