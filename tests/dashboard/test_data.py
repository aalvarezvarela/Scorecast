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
    # The games, the predictions and the injury-report filings of the slate.
    assert cursor.execute.call_count == 3
    assert cursor.execute.call_args_list[0].args[1] == (NOW.date(),)
    assert cursor.execute.call_args_list[1].args[1][0] == NOW.date()
    assert cursor.execute.call_args_list[2].args[1][-1] == NOW.date()
    assert all("%s" in call.args[0] for call in cursor.execute.call_args_list)


def _slate_connection(monkeypatch, results):
    """A connection whose queries return ``results`` in turn: rows with their
    columns, or an exception to raise."""
    conn = MagicMock()
    conn.__enter__.return_value = conn
    cursor = conn.cursor.return_value.__enter__.return_value
    pending = iter(results)

    def execute(query, params):
        result = next(pending)
        if isinstance(result, Exception):
            raise result
        columns, rows = result
        cursor.description = [SimpleNamespace(name=name) for name in columns]
        cursor.fetchall.return_value = rows

    cursor.execute.side_effect = execute
    monkeypatch.setattr(predictions, "open_connection", MagicMock(return_value=conn))
    return conn


def test_todays_games_say_which_injury_reports_are_filed(monkeypatch):
    conn = _slate_connection(
        monkeypatch,
        [
            (["game_id"], [(1,), (2,)]),
            (["game_id"], []),
            (
                predictions.REPORT_COLUMNS[:0]
                + ["game_id", *predictions.REPORT_COLUMNS],
                [(1, True, False, NOW), (2, True, True, NOW)],
            ),
        ],
    )
    games = predictions.read_today(NOW.date()).games
    assert games["home_report_filed"].tolist() == [True, True]
    assert games["away_report_filed"].tolist() == [False, True]
    # In a savepoint, so a refused read leaves the slate's transaction alive.
    conn.transaction.assert_called_once()


def test_an_unreadable_report_store_leaves_the_games_unflagged(monkeypatch):
    _slate_connection(
        monkeypatch,
        [
            (["game_id"], [(1,)]),
            (["game_id"], []),
            psycopg.errors.InsufficientPrivilege("injury_report"),
        ],
    )
    games = predictions.read_today(NOW.date()).games
    assert games["game_id"].tolist() == [1]
    assert "home_report_filed" not in games


def test_yesterday_does_not_read_the_report_filings(monkeypatch):
    conn = _slate_connection(monkeypatch, [(["game_id"], [(1,)]), (["game_id"], [])])
    predictions.read_yesterday(NOW.date())
    conn.transaction.assert_not_called()


def test_connection_explicitly_resolves_only_aiven(monkeypatch):
    dsn = MagicMock(return_value="postgresql://example.invalid/dashboard")
    connect = MagicMock()
    monkeypatch.setattr(connection, "get_db_dsn", dsn)
    monkeypatch.setattr(connection.psycopg, "connect", connect)
    connection.open_connection()
    dsn.assert_called_once_with("aiven")
    assert "default_transaction_read_only=on" in connect.call_args.kwargs["options"]
    assert connect.call_args.kwargs["connect_timeout"] == 8


def test_history_reads_scores_and_post_tip_replays_in_one_read_only_snapshot(
    monkeypatch,
):
    conn = MagicMock()
    conn.__enter__.return_value = conn
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.description = [SimpleNamespace(name="game_id")]
    cursor.fetchall.return_value = [(1,)]
    monkeypatch.setattr(predictions, "open_connection", MagicMock(return_value=conn))
    predictions.read_yesterday(NOW.date())
    assert conn.read_only is True
    assert conn.isolation_level == psycopg.IsolationLevel.REPEATABLE_READ
    assert cursor.execute.call_count == 2
    games_query = cursor.execute.call_args_list[0].args[0]
    history_query, parameters = cursor.execute.call_args_list[1].args
    assert "home_points, away_points" in games_query
    assert "workflow IN (1, 2, 3, 4)" in history_query
    assert "time_to_tipoff_seconds >" not in history_query
    assert parameters[0] == NOW.date()


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
