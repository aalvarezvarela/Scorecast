"""Record every database read of a build, and replay it later without the server.

A reference build is only worth comparing against if the inputs stay the same.
The databases do not: games, injury reports and line history are appended
daily and corrected by the repair scripts. So a reference build records the
result of every query it runs (the "tape"), and later builds replay that tape.
Same inputs, so any difference in the output is a difference in the code.

The patch sits on ``psycopg.Cursor``. Every builder reads through plain
psycopg cursors, directly or through ``pd.read_sql_query``, so whatever path a
query takes, it is caught. Nothing in ``src/`` knows about this.

* ``record(tape_dir)``: queries run against the live server as usual. Each
  result (column description and rows, exactly as psycopg returned them) is
  pickled to the tape, and the caller gets the same rows.
* ``replay(tape_dir)``: connections still open, because psycopg's ``sql``
  composition needs a real connection to render the query text. But no query
  reaches the server: each one is answered from the tape. A query that is not
  on the tape raises :class:`TapeMiss`. Replay is strict on purpose, so it
  never falls back to live data silently.

A query is keyed by its rendered text plus ``repr(params)``. A key recorded
more than once is replayed in the same order. Statements that return no rows
(``SET search_path``) are recorded as such and skipped on replay.

The live-DB check (``verify_tape.py``) uses ``content_sha256``. It is
order-insensitive (a sha256 over the sorted row reprs) because queries without
``ORDER BY`` may legitimately return rows in another order.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import pickle
import threading
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

INDEX_NAME = "index.json"


class TapeMiss(BaseException):
    """A replayed build ran a query the reference build never ran.

    A ``BaseException`` on purpose: several loaders wrap their query in
    ``except Exception`` and return ``None``, which would turn a miss into a
    silently different build. Misses are also counted on the tape, and
    ``replay`` raises at exit if there were any.
    """


def _query_text(cursor: psycopg.Cursor, query: Any) -> str:
    if isinstance(query, sql.Composable):
        return query.as_string(cursor.connection)
    if isinstance(query, bytes):
        return query.decode()
    return str(query)


def tape_key(query_text: str, params: Any) -> str:
    return f"{query_text}\x00{params!r}"


def content_sha256(rows: list) -> str:
    """Order-insensitive digest of a result set."""
    digest = hashlib.sha256()
    for text in sorted(repr(row) for row in rows):
        digest.update(text.encode())
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass
class _Served:
    """Rows a cursor hands back after an intercepted execute."""

    description: list | None
    rows: list
    position: int = 0


@dataclass
class _Tape:
    directory: Path
    mode: str  # "record" | "replay"
    entries: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))
    replay_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    lock: threading.Lock = field(default_factory=threading.Lock)
    n_files: int = 0
    misses: list[str] = field(default_factory=list)

    def save_index(self) -> None:
        index = {
            "mode": "record",
            "entries": [
                {"key": key, **entry}
                for key, recorded in self.entries.items()
                for entry in recorded
            ],
        }
        (self.directory / INDEX_NAME).write_text(json.dumps(index, indent=1))

    @classmethod
    def load(cls, directory: Path) -> _Tape:
        tape = cls(directory=directory, mode="replay")
        index = json.loads((directory / INDEX_NAME).read_text())
        for entry in index["entries"]:
            key = entry.pop("key")
            tape.entries[key].append(entry)
        return tape


_state: dict[str, _Tape | None] = {"tape": None}
_served: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_original = {
    name: getattr(psycopg.Cursor, name)
    for name in ("execute", "fetchall", "fetchone", "fetchmany", "__iter__")
}
_original_description = psycopg.Cursor.description


def _execute(self, query, params=None, *, prepare=None, binary=None):
    tape = _state["tape"]
    if tape is None:
        return _original["execute"](self, query, params, prepare=prepare, binary=binary)
    text = _query_text(self, query)
    key = tape_key(text, params)

    if tape.mode == "replay":
        with tape.lock:
            recorded = tape.entries.get(key)
            if not recorded:
                tape.misses.append(key)
                raise TapeMiss(
                    "Query not on the tape (replay never reads live data):\n"
                    f"{text[:2000]}\nparams={params!r}"
                )
            position = min(tape.replay_counts[key], len(recorded) - 1)
            tape.replay_counts[key] += 1
            entry = recorded[position]
        if entry["file"] is None:
            _served[self] = _Served(description=None, rows=[])
            return self
        with open(tape.directory / entry["file"], "rb") as handle:
            description, rows = pickle.load(handle)
        _served[self] = _Served(description=description, rows=rows)
        return self

    started = time.perf_counter()
    _original["execute"](self, query, params, prepare=prepare, binary=binary)
    raw_description = _original_description.fget(self)
    if raw_description is None:
        description, rows, file_name = None, [], None
    else:
        description = [
            (column.name, column.type_code, None, None, None, None, None)
            for column in raw_description
        ]
        rows = _original["fetchall"](self)
    with tape.lock:
        if raw_description is not None:
            file_name = f"{tape.n_files:05d}.pkl"
            tape.n_files += 1
        tape.entries[key].append(
            {
                "file": file_name,
                "n_rows": len(rows),
                "content_sha256": content_sha256(rows) if file_name else None,
                "seconds": round(time.perf_counter() - started, 3),
                "query": text[:4000],
                "params": repr(params)[:4000],
            }
        )
    if file_name is not None:
        with open(tape.directory / file_name, "wb") as handle:
            pickle.dump((description, rows), handle, protocol=pickle.HIGHEST_PROTOCOL)
    _served[self] = _Served(description=description, rows=rows)
    return self


def _description(self):
    served = _served.get(self)
    if served is not None:
        return served.description
    return _original_description.fget(self)


def _fetchall(self):
    served = _served.get(self)
    if served is None:
        return _original["fetchall"](self)
    out = served.rows[served.position :]
    served.position = len(served.rows)
    return out


def _fetchone(self):
    served = _served.get(self)
    if served is None:
        return _original["fetchone"](self)
    if served.position >= len(served.rows):
        return None
    row = served.rows[served.position]
    served.position += 1
    return row


def _fetchmany(self, size: int = 0):
    served = _served.get(self)
    if served is None:
        return _original["fetchmany"](self, size)
    size = size or self.arraysize
    out = served.rows[served.position : served.position + size]
    served.position += len(out)
    return out


def _iter(self):
    served = _served.get(self)
    if served is None:
        return _original["__iter__"](self)
    return iter(_fetchall(self))


@contextlib.contextmanager
def _patched(tape: _Tape):
    if _state["tape"] is not None:
        raise RuntimeError("A tape is already active in this process.")
    _state["tape"] = tape
    psycopg.Cursor.execute = _execute
    psycopg.Cursor.fetchall = _fetchall
    psycopg.Cursor.fetchone = _fetchone
    psycopg.Cursor.fetchmany = _fetchmany
    psycopg.Cursor.__iter__ = _iter
    psycopg.Cursor.description = property(_description)
    try:
        yield tape
    finally:
        for name, function in _original.items():
            setattr(psycopg.Cursor, name, function)
        psycopg.Cursor.description = _original_description
        _state["tape"] = None


@contextlib.contextmanager
def record(directory: str | Path):
    """Run the block against the live databases, recording every result."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tape = _Tape(directory=directory, mode="record")
    try:
        with _patched(tape):
            yield tape
    finally:
        tape.save_index()


@contextlib.contextmanager
def replay(directory: str | Path):
    """Run the block with every query answered from the tape."""
    tape = _Tape.load(Path(directory))
    with _patched(tape):
        yield tape
    if tape.misses:
        raise TapeMiss(f"{len(tape.misses)} query(ies) were not on the tape.")
    unused = [key for key, recorded in tape.entries.items() if key not in tape.replay_counts]
    if unused:
        print(f"[db_tape] {len(unused)} recorded query key(s) were not replayed.")


def tape_summary(directory: str | Path) -> dict:
    index = json.loads((Path(directory) / INDEX_NAME).read_text())
    entries = index["entries"]
    return {
        "n_queries": len(entries),
        "n_result_sets": sum(1 for e in entries if e["file"]),
        "n_rows": sum(e["n_rows"] for e in entries),
        "query_seconds": round(sum(e["seconds"] for e in entries), 1),
    }
