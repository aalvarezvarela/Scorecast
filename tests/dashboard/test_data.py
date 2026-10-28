from types import SimpleNamespace
from unittest.mock import MagicMock

import psycopg
import pytest
from nba_ou.dashboard.repositories import connection, predictions
from nba_ou.dashboard.repositories.predictions import DashboardDataError

from .conftest import NOW


def test_reader_uses_one_read_only_snapshot_and_parameterized_slate(monkeypatch):
    conn = MagicMock()
    conn.__enter__.return_value = conn
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.description = [SimpleNamespace(name="game_id")]
    cursor.fetchall.return_value = [(1,)]
    opener = MagicMock(return_value=conn)
    monkeypatch.setattr(predictions, "open_connection", opener)
    data = predictions.read_today(NOW.date(), "postgresql://example.invalid/dashboard")
    assert data.games["game_id"].tolist() == [1]
    assert conn.read_only is True
    assert conn.isolation_level == psycopg.IsolationLevel.REPEATABLE_READ
    assert cursor.execute.call_count == 2
    assert cursor.execute.call_args_list[0].args[1] == (NOW.date(),)
    assert cursor.execute.call_args_list[1].args[1][0] == NOW.date()
    assert all("%s" in call.args[0] for call in cursor.execute.call_args_list)


def test_connection_explicitly_resolves_only_aiven(monkeypatch):
    dsn = MagicMock(return_value="postgresql://example.invalid/dashboard")
    connect = MagicMock()
    monkeypatch.setattr(connection, "get_db_dsn", dsn)
    monkeypatch.setattr(connection.psycopg, "connect", connect)
    connection.open_connection()
    dsn.assert_called_once_with("aiven")
    assert "default_transaction_read_only=on" in connect.call_args.kwargs["options"]
    assert connect.call_args.kwargs["connect_timeout"] == 8


@pytest.mark.parametrize(
    "error",
    [
        psycopg.OperationalError("sensitive connection URI"),
        psycopg.errors.UndefinedTable("sensitive table detail"),
        psycopg.errors.UndefinedColumn("sensitive schema detail"),
        psycopg.errors.InsufficientPrivilege("sensitive role detail"),
        ValueError("sensitive credentials"),
    ],
)
def test_data_errors_are_actionable_and_do_not_leak_credentials(monkeypatch, error):
    monkeypatch.setattr(predictions, "open_connection", MagicMock(side_effect=error))
    with pytest.raises(DashboardDataError) as caught:
        predictions.read_today(NOW.date())
    assert "sensitive" not in str(caught.value)
    assert "Aiven" in str(caught.value)
