"""Loads of stored history, shared by the builds of one run in the process.

A predictor run carrying TabPFN jobs builds tonight's rows more than once, and
one intermediate build already reads the closing odds twice (base features and
referees). The odds, games, players and injury-report tables they read are
stored history: while nothing writes to the databases, every read returns the
same frame. Inside ``history_cache()`` a function marked ``@cached`` loads once
per set of arguments and every call gets its own copy, so a caller may change
what it gets. The steps computed from those frames alone (the team rolling
statistics, the prepared player and All-Star lookups) are marked too: a frame
argument keys on its content, so the next build with the same inputs reuses
the result.

Outside the scope every call reads as before. A run enters it once its
database update and live fetches are done, and calls ``clear`` before
anything it reads may have changed (a refresh of the live data).
"""

from __future__ import annotations

import copy
import functools
import hashlib
import inspect
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import pandas as pd

_STORE: dict | None = None
_LOCK = threading.Lock()
_KEY_LOCKS: dict = {}


@contextmanager
def history_cache() -> Iterator[None]:
    """Shares the ``@cached`` loads until the block ends; nests as one scope."""
    global _STORE
    outer = _STORE is not None
    if not outer:
        _STORE = {}
    try:
        yield
    finally:
        if not outer:
            with _LOCK:
                _STORE = None
                _KEY_LOCKS.clear()


def clear() -> None:
    """Forgets every load, so the next call reads again."""
    with _LOCK:
        if _STORE is not None:
            _STORE.clear()
        _KEY_LOCKS.clear()


def active() -> bool:
    return _STORE is not None


def _frozen(value):
    """``value`` as part of a key; TypeError for what has no stable one."""
    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return tuple(_frozen(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted(_frozen(item) for item in value)))
    if isinstance(value, dict):
        return ("dict", tuple(sorted((k, _frozen(v)) for k, v in value.items())))
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return ("frame", _fingerprint(value))
    if isinstance(value, (np.ndarray, pd.Index)):
        return tuple(_frozen(item) for item in np.asarray(value).tolist())
    raise TypeError(f"no cache key for {type(value).__name__}")


def _fingerprint(frame: pd.DataFrame | pd.Series) -> str:
    """A digest of ``frame``'s content: index, columns, dtypes, values, attrs."""
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=("series", frame.name))
    digest = hashlib.sha256(repr(_frozen(frame.attrs)).encode())
    try:
        digest.update(pd.util.hash_pandas_object(frame.index).to_numpy().tobytes())
        for position, (name, column) in enumerate(frame.items()):
            digest.update(repr((position, name, str(column.dtype))).encode())
            values = pd.util.hash_pandas_object(column, index=False)
            digest.update(values.to_numpy().tobytes())
    except (TypeError, ValueError) as exc:  # unhashable cells (lists, dicts)
        raise TypeError(f"no cache key for this frame: {exc}") from None
    return digest.hexdigest()


def _copied(value):
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.copy()
    if isinstance(value, tuple):
        return tuple(_copied(item) for item in value)
    if isinstance(value, list):
        return [_copied(item) for item in value]
    if isinstance(value, dict):
        return {key: _copied(item) for key, item in value.items()}
    return copy.deepcopy(value)


def cached(*, ignore: tuple[str, ...] = (), shared: bool = False) -> Callable:
    """Marks a load of stored history; ``ignore`` names arguments that do not
    change what it returns (a connection, verbosity). ``shared=True`` hands
    every caller the one result, for prepared lookups its callers only read."""

    def decorate(fn: Callable) -> Callable:
        signature = inspect.signature(fn)
        name = f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if _STORE is None:
                return fn(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            try:
                key = (name,) + _frozen(
                    [(k, v) for k, v in bound.arguments.items() if k not in ignore]
                )
            except TypeError:
                return fn(*args, **kwargs)
            with _LOCK:
                store = _STORE
                if store is None:
                    return fn(*args, **kwargs)
                key_lock = _KEY_LOCKS.setdefault(key, threading.Lock())
            # One load per key; builds running in threads wait for it.
            with key_lock:
                if key not in store:
                    result = fn(*args, **kwargs)
                    if result is None:
                        # A loader's failed read: the next call tries again.
                        return None
                    store[key] = result
                return store[key] if shared else _copied(store[key])

        return wrapper

    return decorate
