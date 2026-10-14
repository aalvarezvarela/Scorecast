"""A local cache of database reads, validated against the server on every use.

Training builds read the same finished seasons from the databases every time,
and most of a closing build's wall time is spent waiting on them. This keeps
a loader's result per season on disk and reuses it only while the season's
data on the server is unchanged.

"Unchanged" is checked, not assumed. Each cached season stores a fingerprint:
a read-only query the loader supplies, computed on the server (row counts and
a hash over every column the loader reads). Before reusing an entry, the
fingerprint is computed again; any difference -- a repair script rewriting
closes, a tipoff retime, a backfilled book -- refetches that season. The
fingerprint is taken *before* a fetch, so data changing mid-fetch leaves an
entry with the older fingerprint, which the next build refetches.

An entry is also keyed on the loader's arguments, on the source code of the
functions that produce it (a changed query or decoding invalidates every
entry), and on the pandas and numpy versions. Frames are pickled, so a cached
frame is the fetched frame exactly: dtypes, index and values.

Set ``NBA_OU_INPUT_CACHE=0`` to bypass the cache; ``NBA_OU_INPUT_CACHE_DIR``
moves it (default ``data/cache/inputs`` in the checkout).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import pickle
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache" / "inputs"


def input_cache_enabled() -> bool:
    return os.environ.get("NBA_OU_INPUT_CACHE", "1") != "0"


def cache_dir() -> Path:
    return Path(os.environ.get("NBA_OU_INPUT_CACHE_DIR", DEFAULT_CACHE_DIR))


def code_key(*functions: Callable, **arguments) -> str:
    """A key for one loader configuration: its code, arguments and libraries."""
    digest = hashlib.sha256()
    for function in functions:
        digest.update(inspect.getsource(function).encode())
    digest.update(json.dumps(arguments, sort_keys=True, default=repr).encode())
    digest.update(f"pandas={pd.__version__} numpy={np.__version__}".encode())
    return digest.hexdigest()[:20]


class SeasonCache:
    """Per-season frames of one loader configuration, under ``<dir>/<name>/<key>/``."""

    def __init__(self, name: str, key: str, root: Path | None = None) -> None:
        self.directory = (root or cache_dir()) / name / key
        self.hits = 0
        self.misses = 0

    def _paths(self, season: int) -> tuple[Path, Path]:
        stem = self.directory / f"season={int(season)}"
        return stem.with_suffix(".pkl"), stem.with_suffix(".json")

    def load(self, season: int, fingerprint: list) -> pd.DataFrame | None:
        """The cached frame, if one was stored under this exact fingerprint."""
        frame_path, meta_path = self._paths(season)
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            return None
        if meta.get("fingerprint") != fingerprint:
            return None
        try:
            payload = frame_path.read_bytes()
        except OSError:
            return None
        if hashlib.sha256(payload).hexdigest() != meta.get("sha256"):
            return None
        return pickle.loads(payload)

    def store(self, season: int, fingerprint: list, frame: pd.DataFrame) -> None:
        frame_path, meta_path = self._paths(season)
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = pickle.dumps(frame, protocol=5)
        meta = {
            "fingerprint": fingerprint,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "rows": len(frame),
            "created": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        # Frame first, then the manifest that vouches for it, each replaced
        # atomically: a reader never pairs a fingerprint with a partial frame.
        for path, data in (
            (frame_path, payload),
            (meta_path, json.dumps(meta).encode()),
        ):
            fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".tmp-")
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp, path)

    def get(
        self,
        season: int,
        fingerprint: Callable[[int], list],
        fetch: Callable[[int], pd.DataFrame],
    ) -> pd.DataFrame:
        """The season's frame: cached if its fingerprint still matches, else fetched."""
        current = _jsonable(fingerprint(season))
        frame = self.load(season, current)
        if frame is not None:
            self.hits += 1
            return frame
        self.misses += 1
        frame = fetch(season)
        self.store(season, current, frame)
        return frame


def _jsonable(values) -> list:
    """A fingerprint as JSON round-trips it, so stored and fresh ones compare."""
    return json.loads(json.dumps(list(values), default=str))
