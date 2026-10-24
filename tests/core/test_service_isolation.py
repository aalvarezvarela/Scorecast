"""Accidental service calls must fail even inside production fallbacks."""

import socket

import boto3
import psycopg
import pytest
import requests
from psycopg import connect as imported_connect


@pytest.mark.parametrize(
    "access",
    [
        lambda: boto3.client("s3"),
        lambda: boto3.Session().client("s3"),
        lambda: boto3.Session().resource("s3"),
        lambda: psycopg.connect("host=127.0.0.1 dbname=unmocked"),
        lambda: imported_connect("host=127.0.0.1 dbname=unmocked"),
        lambda: requests.get("https://example.invalid"),
        lambda: socket.create_connection(("example.invalid", 443)),
    ],
    ids=[
        "boto3",
        "boto3-session",
        "boto3-resource",
        "postgres",
        "postgres-alias",
        "http",
        "socket",
    ],
)
def test_unmocked_services_cannot_be_hidden_by_fallbacks(access):
    with pytest.raises(pytest.fail.Exception, match="Unmocked external service"):
        try:
            access()
        except Exception:
            # Applications may recover from outages. They must not conceal a
            # test that forgot to fake its dependency.
            pass
