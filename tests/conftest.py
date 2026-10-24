"""Fail immediately when a test reaches an unsimulated external service."""

import os
import socket

import boto3
import botocore.client
import psycopg
import pytest
import requests
from psycopg.conninfo import conninfo_to_dict


@pytest.fixture(autouse=True)
def isolated_services(monkeypatch, request):
    """Clients must be faked; postgres-marked tests may use their local DSN."""

    def blocked(*args, **kwargs):
        # pytest.fail raises an OutcomeException (outside Exception), so a
        # production fallback cannot swallow this and leave the test green.
        pytest.fail(
            "Unmocked external service access. Supply a fake client/connection; "
            "only postgres-marked tests may connect to NBA_OU_TEST_PG_DSN.",
            pytrace=False,
        )

    real_connect = psycopg.connect

    def test_database(conninfo="", **kwargs):
        allowed_dsn = os.getenv("NBA_OU_TEST_PG_DSN")
        if not request.node.get_closest_marker("postgres") or not allowed_dsn:
            blocked()
        if conninfo != allowed_dsn:
            blocked()
        host = conninfo_to_dict(allowed_dsn).get("host", "")
        if host not in {"localhost", "127.0.0.1", "::1"} and not host.startswith("/"):
            blocked()
        return real_connect(conninfo, **kwargs)

    monkeypatch.setattr(boto3, "client", blocked)
    monkeypatch.setattr(boto3, "resource", blocked)
    monkeypatch.setattr(boto3.session.Session, "client", blocked)
    monkeypatch.setattr(boto3.session.Session, "resource", blocked)
    monkeypatch.setattr(botocore.client.BaseClient, "_make_api_call", blocked)
    monkeypatch.setattr(psycopg, "connect", test_database)
    monkeypatch.setattr(psycopg.Connection, "connect", test_database)
    monkeypatch.setattr(psycopg.AsyncConnection, "connect", blocked)
    if not request.node.get_closest_marker("postgres"):
        # Also catch connect aliases imported before the autouse fixture ran.
        monkeypatch.setattr(psycopg.Connection, "_connect_gen", blocked)
        monkeypatch.setattr(psycopg.AsyncConnection, "_connect_gen", blocked)
    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
