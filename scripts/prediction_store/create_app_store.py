"""Create schema ``app`` on Aiven (dashboard users, manual requests, bets).

Needs the prediction store (``create_prediction_store.py``) first. Idempotent.
``--print`` shows the SQL without connecting. ``--role`` also creates the
dashboard's own login role (``dashboard_app``) when missing, reading its
password from ``DASHBOARD_APP_DB_PASSWORD`` or a prompt, and grants it read
access to ``prediction`` and read/write (no DELETE) to ``app``.

    python scripts/prediction_store/create_app_store.py --print
    python scripts/prediction_store/create_app_store.py
    python scripts/prediction_store/create_app_store.py --role
"""

from __future__ import annotations

import argparse
import getpass
import os

from nba_ou.postgre_db.app_aiven.schema import (
    DASHBOARD_ROLE,
    SCHEMA,
    create_schema,
    ddl_script,
    grant_dashboard_role,
)
from nba_ou.postgre_db.config.db_config import connect_nba_db
from nba_ou.postgre_db.predictions_aiven.schema import (
    create_schema as create_prediction_schema,
)
from psycopg import sql


def _ensure_role(conn, role: str) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
        if cur.fetchone():
            print(f"Role {role!r} exists; grants refreshed.")
            return
        password = os.getenv("DASHBOARD_APP_DB_PASSWORD") or getpass.getpass(
            f"Password for the new role {role!r}: "
        )
        if not password:
            raise SystemExit("No password given; role not created.")
        cur.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(
                sql.Identifier(role), sql.Literal(password)
            )
        )
    conn.commit()
    print(f"Role {role!r} created.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="Print the SQL and exit without connecting",
    )
    parser.add_argument(
        "--role",
        action="store_true",
        help=f"Create the {DASHBOARD_ROLE!r} role when missing and grant it",
    )
    args = parser.parse_args()
    if args.print_only:
        print(ddl_script())
        return
    with connect_nba_db(env="aiven") as conn:
        # Adds prediction.runs.request_id, which the key below references.
        create_prediction_schema(conn)
        create_schema(conn)
        if args.role:
            _ensure_role(conn, DASHBOARD_ROLE)
            grant_dashboard_role(conn, DASHBOARD_ROLE)
    print(f"Schema {SCHEMA!r} is in place on Aiven.")


if __name__ == "__main__":
    main()
