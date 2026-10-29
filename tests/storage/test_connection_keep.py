"""Connections kept for the rest of a run: one per thread and database inside
the scope, reset for each caller, a new one for a caller nesting a second."""

from __future__ import annotations

import threading
import weakref
from types import SimpleNamespace

import psycopg
import pytest
from nba_ou.postgre_db.config import connection_keep
from nba_ou.postgre_db.config.connection_keep import connection, kept_connections
from psycopg.pq import TransactionStatus


class _Conn:
    """What the keep uses of a psycopg connection; ``with`` as psycopg's."""

    def __init__(self, opened):
        opened.append(self)
        self.statements = []
        self.closed = False
        self.broken = False
        self.autocommit = False
        self.info = SimpleNamespace(transaction_status=TransactionStatus.IDLE)
        self.rollbacks = 0
        self.fail_next = False

    def execute(self, statement):
        if self.fail_next:
            raise psycopg.OperationalError("server closed the connection")
        self.statements.append(statement.as_string(None))
        if not self.autocommit:
            self.info.transaction_status = TransactionStatus.INTRANS

    def cursor(self):
        conn = self

        class _Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, statement):
                conn.execute(statement)

        return _Cursor()

    def commit(self):
        self.info.transaction_status = TransactionStatus.IDLE

    def rollback(self):
        self.rollbacks += 1
        self.info.transaction_status = TransactionStatus.IDLE

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        (self.rollback if exc_type else self.commit)()
        self.close()


@pytest.fixture
def opened():
    return []


def _connect(opened, env="supabase", schema=None):
    return connection(env, schema, lambda: _Conn(opened))


def test_outside_the_scope_every_call_opens_and_closes(opened):
    for _ in range(2):
        with _connect(opened):
            pass
    assert len(opened) == 2
    assert all(conn.closed for conn in opened)
    assert not connection_keep.active()


def test_inside_the_scope_one_connection_per_database(opened):
    with kept_connections():
        with _connect(opened, schema="nba_games") as games:
            pass
        with _connect(opened, schema="nba_players") as players:
            pass
        conn = _connect(opened)
        conn.close()
        with _connect(opened, env="aiven") as aiven:
            pass
        assert games is players is conn
        assert aiven is not games
        assert not any(c.closed for c in opened)
    assert len(opened) == 2
    assert all(c.closed for c in opened)
    assert games.statements == [
        'SET search_path TO "nba_games", public;',
        'RESET ALL; SET search_path TO "nba_players", public;',
        "RESET ALL;",
    ]
    assert games.autocommit is False


def test_a_connection_in_use_is_not_handed_out_again(opened):
    with kept_connections():
        with _connect(opened) as outer:
            with _connect(opened) as inner:
                pass
            assert inner is not outer and not inner.closed
        # The thread keeps one: the one closed last is closed for real.
        assert outer.closed
        with _connect(opened) as again:
            pass
    assert again is inner
    assert len(opened) == 2


def test_a_connection_dropped_unclosed_is_freed(opened):
    with kept_connections():
        dropped = weakref.ref(_connect(opened))
        opened.clear()
        # Freed by reference counting, as outside the scope: no cycle keeps it.
        assert dropped() is None
        with _connect(opened):
            pass
    assert len(opened) == 1


def test_closing_rolls_back_an_open_transaction(opened):
    with kept_connections():
        conn = _connect(opened)
        conn.execute(psycopg.sql.SQL("SELECT 1"))
        conn.close()
        assert conn.rollbacks == 1 and not conn.closed
        assert conn.info.transaction_status == TransactionStatus.IDLE


def test_a_connection_failing_its_reset_is_replaced(opened):
    with kept_connections():
        with _connect(opened) as first:
            pass
        first.fail_next = True
        with _connect(opened) as second:
            pass
        with _connect(opened) as third:
            pass
    assert first.closed
    assert second is not first and third is second
    assert len(opened) == 2


def test_a_broken_connection_is_not_kept(opened):
    with kept_connections():
        with _connect(opened) as first:
            first.broken = True
        with _connect(opened) as second:
            pass
    assert first.closed and second is not first


def test_each_thread_keeps_its_own(opened):
    seen = []
    both_running = threading.Barrier(2, timeout=5)

    def read():
        with _connect(opened) as conn:
            seen.append(conn)
        both_running.wait()
        with _connect(opened) as conn:
            seen.append(conn)

    with kept_connections():
        threads = [threading.Thread(target=read) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
    assert len(opened) == 2
    assert seen.count(opened[0]) == 2 and seen.count(opened[1]) == 2
    assert all(c.closed for c in opened)


def test_nested_scopes_keep_until_the_outer_one_ends(opened):
    with kept_connections():
        with kept_connections():
            with _connect(opened) as conn:
                pass
        assert not conn.closed
        with _connect(opened) as again:
            pass
    assert again is conn and conn.closed
