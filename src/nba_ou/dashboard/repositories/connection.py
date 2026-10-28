"""Resolve Aiven credentials without changing the process-wide environment."""

import psycopg

from nba_ou.postgre_db.config.db_config import get_db_credentials, get_db_dsn


def open_connection(dsn_override: str = "") -> psycopg.Connection:
    """Use Aiven explicitly, even when the predictor defaults to another DB."""
    options = {
        "connect_timeout": 8,
        "application_name": "nba_dashboard",
        "options": "-c default_transaction_read_only=on -c statement_timeout=15000",
    }
    dsn = dsn_override or get_db_dsn("aiven")
    if dsn:
        return psycopg.connect(dsn, **options)
    credentials = get_db_credentials(env="aiven")
    return psycopg.connect(
        dbname=credentials["dbname"],
        user=credentials["user"],
        password=credentials["password"],
        host=credentials["host"],
        port=credentials["port"],
        sslmode=credentials["sslmode"] or "require",
        **options,
    )
