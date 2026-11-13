"""Shared paths and yt-dlp options for the caption-vs-Whisper probe."""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "coach_pressers" / "caption_vs_whisper_probe"


def youtube_options(root: Path, js_runtime: str) -> dict:
    """Quiet yt-dlp options; ``js_runtime`` is a name on PATH or a binary path."""
    path = Path(js_runtime)
    name = path.stem if path.exists() else js_runtime
    if not path.exists() and not shutil.which(js_runtime):
        raise RuntimeError(f"JavaScript runtime not found: {js_runtime}")
    runtime = {"path": str(path)} if path.exists() else {}
    return {
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": {name: runtime},
        "cachedir": str(root / ".yt-dlp-cache"),
    }
