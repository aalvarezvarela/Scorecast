"""Per-season results of history-only work, reused across target builds.

A target build (``create_df_to_predict(target_game_ids=...)``) still does some
work for every earlier game, because target rows read it: the player step
records each history row's availability and top-player ids, and the
availability effects compute each history row's raw effects for the
shrinkage fit. For a finished season those results are the same every day,
so this keeps them per season and reuses them.

Reuse is never assumed. Each season's entry is keyed on

* the code: every source file of the ``nba_ou`` package, and the pandas and
  numpy versions (any change refuses every entry); and
* a digest of the season's inputs as this build loaded them -- games, player
  box scores, odds after the closing-line repair, injuries and injury
  reports -- chained over every earlier season too. A history row of season
  ``S`` reads nothing from later seasons, so the chain up to ``S`` covers
  everything it can depend on, and a repaired or backfilled season (or any
  earlier one) refuses its entry.

Only seasons before the targets' first season are kept: the current season
gains games every day, so its digest never repeats. The parity test
(``scripts/perf/target_parity.py``) checks a build that reads every entry
against the full build, bit for bit.

Entries live in S3 next to the input cache (``.../memo/<name>/<code>/
season=S.pkl``) or, with ``NBA_OU_HISTORY_MEMO_DIR``, in a local directory.
``NBA_OU_HISTORY_MEMO=0`` turns it off.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections.abc import Iterable
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

MEMO_PREFIX = "train_data/input_cache/v1/memo"


def history_memo_enabled() -> bool:
    return os.environ.get("NBA_OU_HISTORY_MEMO", "1") != "0"


@cache
def package_code_key() -> str:
    """Every ``nba_ou`` source file and the pandas/numpy versions."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    digest.update(f"pandas={pd.__version__} numpy={np.__version__}".encode())
    return digest.hexdigest()[:20]


def season_of_game_ids(game_ids: pd.Series) -> np.ndarray:
    """Season start year from NBA game ids (``00SYYNNNNN``); -1 if unparseable."""
    ids = game_ids.astype(str)
    text = ids.str.zfill(10).str[3:5].where(ids.str.isdigit())
    years = pd.to_numeric(text, errors="coerce")
    return np.where(years.notna(), 2000 + years.fillna(0), -1).astype("int64")


def season_of_game_id(game_id) -> int | None:
    """``season_of_game_ids`` for one id."""
    text = str(game_id)
    return 2000 + int(text.zfill(10)[3:5]) if text.isdigit() else None


def _frame_digest(frame: pd.DataFrame) -> bytes:
    digest = hashlib.sha256()
    for column in frame.columns:
        digest.update(str(column).encode())
        digest.update(str(frame[column].dtype).encode())
        try:
            hashed = pd.util.hash_pandas_object(frame[column], index=False)
        except TypeError:  # unhashable values: lists, dicts
            hashed = pd.util.hash_pandas_object(frame[column].map(repr), index=False)
        digest.update(hashed.to_numpy().tobytes())
    return digest.digest()


def season_input_digests(
    inputs: dict[str, tuple[pd.DataFrame, str]], seasons: Iterable[int]
) -> dict[int, str]:
    """Chained digest per season of every input frame's rows up to it.

    ``inputs`` maps a name to ``(frame, game_id_column)``; a row belongs to
    the season its game id names. Rows keep the frame's order.
    """
    per_frame = {}
    for name, (frame, game_id_column) in sorted(inputs.items()):
        if frame is None or frame.empty:
            per_frame[name] = (frame, np.array([], dtype="int64"))
            continue
        per_frame[name] = (frame, season_of_game_ids(frame[game_id_column]))
    out = {}
    chain = hashlib.sha256(package_code_key().encode()).digest()
    for season in sorted(set(seasons)):
        digest = hashlib.sha256(chain)
        digest.update(str(season).encode())
        for name, (frame, frame_seasons) in sorted(per_frame.items()):
            digest.update(name.encode())
            if frame is not None and len(frame_seasons):
                digest.update(_frame_digest(frame[frame_seasons == season]))
        chain = digest.digest()
        out[season] = digest.hexdigest()
    return out


class _LocalBackend:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def get(self, key: str) -> bytes | None:
        path = self.root / key
        return path.read_bytes() if path.exists() else None

    def put(self, key: str, data: bytes) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)


class _S3Backend:
    def __init__(self, store) -> None:
        self.store = store

    def get(self, key: str) -> bytes | None:
        return self.store._get(f"{MEMO_PREFIX}/{key}")

    def put(self, key: str, data: bytes) -> None:
        self.store._put(f"{MEMO_PREFIX}/{key}", data)


def default_backend():
    """Local directory if ``NBA_OU_HISTORY_MEMO_DIR`` is set, else S3, else None."""
    if not history_memo_enabled():
        return None
    local = os.environ.get("NBA_OU_HISTORY_MEMO_DIR")
    if local:
        return _LocalBackend(Path(local))
    from nba_ou.postgre_db.season_store import default_store

    store = default_store()
    return _S3Backend(store) if store is not None else None


class HistoryMemo:
    """Season entries of named results, valid for the given input digests.

    ``max_season`` is the last season whose entries may be read or written.
    Any failure to read or write is reported and treated as a miss: the
    caller computes the season itself, so a broken store costs time, never
    a value.
    """

    def __init__(self, backend, digests: dict[int, str], max_season: int) -> None:
        self.backend = backend
        self.digests = digests
        self.max_season = max_season
        self.code = package_code_key()
        self.report: list[str] = []
        self._tables: dict[tuple[str, int], dict] = {}
        self._sizes: dict[tuple[str, int], int] = {}

    def table(self, name: str, season) -> dict | None:
        """The season's entries as a dict to read and add to; None if unusable.

        Loaded once per build and shared by every caller; ``flush`` writes the
        tables that gained entries.
        """
        if not self.usable(season):
            return None
        key = (name, int(season))
        if key not in self._tables:
            loaded = self.load(name, int(season))
            self._tables[key] = loaded if isinstance(loaded, dict) else {}
            self._sizes[key] = len(self._tables[key])
        return self._tables[key]

    def flush(self) -> None:
        for (name, season), table in self._tables.items():
            if len(table) > self._sizes[(name, season)]:
                self.save(name, season, table)
                self._sizes[(name, season)] = len(table)
        loaded = sum(self._sizes.values())
        print(
            f"History memo: {len(self._tables)} season tables, {loaded:,} entries"
            + (f"; {'; '.join(self.report)}" if self.report else "")
        )

    def usable(self, season) -> bool:
        try:
            season = int(season)
        except (TypeError, ValueError):
            return False
        return season <= self.max_season and season in self.digests

    def _key(self, name: str, season: int, suffix: str) -> str:
        return f"{name}/{self.code}/season={season}.{suffix}"

    def load(self, name: str, season: int):
        if not self.usable(season):
            return None
        try:
            raw = self.backend.get(self._key(name, season, "json"))
            if raw is None:
                return None
            meta = json.loads(raw)
            if meta.get("digest") != self.digests[season]:
                self.report.append(f"{name} {season}: inputs changed")
                return None
            payload = self.backend.get(self._key(name, season, "pkl"))
            if payload is None or hashlib.sha256(payload).hexdigest() != meta["sha256"]:
                self.report.append(f"{name} {season}: payload missing or corrupt")
                return None
            return pickle.loads(payload)
        except Exception as exc:  # noqa: BLE001 -- a broken store is a miss
            self.report.append(f"{name} {season}: read failed ({exc})")
            return None

    def save(self, name: str, season: int, value) -> None:
        if not self.usable(season):
            return
        try:
            payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            meta = {
                "digest": self.digests[season],
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            self.backend.put(self._key(name, season, "pkl"), payload)
            self.backend.put(self._key(name, season, "json"), json.dumps(meta).encode())
        except Exception as exc:  # noqa: BLE001
            self.report.append(f"{name} {season}: write failed ({exc})")
