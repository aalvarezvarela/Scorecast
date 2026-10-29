"""Database connections kept open for the rest of a run, one per thread and
database.

Every read of a build opens its own connection, and each open repeats the TLS
and authentication handshake with Supabase or Aiven. Inside
``kept_connections()`` closing a connection, or leaving ``with conn:``, ends
its transaction and keeps it for the next caller in the same thread, who gets
it back with the session settings reset (``RESET ALL``) and its own
``search_path``. Only a closed connection is kept: a caller opening one while
another is open gets a new one, and of the two the one closed last is closed
for real; one never closed is freed as before. A kept connection that fails
its reset is dropped and replaced. Leaving the scope closes them all; outside
it every call opens a new connection.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import psycopg
from psycopg import sql
from psycopg.pq import TransactionStatus

_KEPT: dict | None = None
_LOCK = threading.Lock()


@contextmanager
def kept_connections() -> Iterator[None]:
    """Keeps the connections closed until the block ends; nests as one scope."""
    global _KEPT
    with _LOCK:
        outer = _KEPT is not None
        if not outer:
            _KEPT = {}
    try:
        yield
    finally:
        if not outer:
            with _LOCK:
                kept, _KEPT = _KEPT, None
            for conn in kept.values():
                _close(conn)


def active() -> bool:
    return _KEPT is not None


def connection(
    env: str, schema: str | None, open_connection: Callable[[], psycopg.Connection]
) -> psycopg.Connection:
    """A connection to ``env`` with ``search_path`` set to ``schema`` when one
    is given: the thread's kept one inside the scope, else a new one."""
    key = (threading.get_ident(), env)
    with _LOCK:
        kept = _KEPT
        conn = None if kept is None else kept.pop(key, None)
    if conn is not None:
        try:
            _reset(conn, schema)
        except psycopg.Error:
            # Dropped by the server while kept (idle timeout, restart).
            _close(conn)
            conn = None
    if conn is None:
        conn = open_connection()
        if schema is not None:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SET search_path TO {}, public;").format(
                        sql.Identifier(schema)
                    )
                )
            conn.commit()
        if kept is not None:
            # A weak reference: a connection its caller drops unclosed is freed.
            ref = weakref.ref(conn)
            conn.close = lambda: _release(kept, key, ref())
    return conn


def _reset(conn: psycopg.Connection, schema: str | None) -> None:
    statement = sql.SQL("RESET ALL;")
    if schema is not None:
        statement += sql.SQL(" SET search_path TO {}, public;").format(
            sql.Identifier(schema)
        )
    # In autocommit: one round trip, no transaction left open.
    conn.autocommit = True
    try:
        conn.execute(statement)
    finally:
        conn.autocommit = False


def _release(kept: dict, key: tuple, conn: psycopg.Connection) -> None:
    """``close()`` of a connection opened in the scope: ends its transaction
    and keeps it, unless the thread already keeps one."""
    if not conn.closed and not conn.broken:
        try:
            if conn.info.transaction_status != TransactionStatus.IDLE:
                conn.rollback()  # what closing does to an open transaction
            with _LOCK:
                if _KEPT is kept and key not in kept:
                    kept[key] = conn
                    return
        except psycopg.Error:
            pass
    _close(conn)


def _close(conn: psycopg.Connection) -> None:
    conn.__dict__.pop("close", None)
    try:
        conn.close()
    except psycopg.Error:
        pass
