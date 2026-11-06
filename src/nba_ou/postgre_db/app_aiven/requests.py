"""Status of a dashboard prediction request, written by the run that answers it.

The dashboard inserts a request as ``queued`` and dispatches a workflow with
its id; ``predict_nba_games.py`` / ``predict_tabpfn.py`` mark it ``running``
when they start and ``done`` / ``partial`` / ``failed`` when they end. A
workflow step marks it ``failed`` when the job dies before the script can:

    python -m nba_ou.postgre_db.app_aiven.requests --request-id 12 \\
        --status failed --error "workflow failed"

Status writes never stop a run: a failure is printed and the run carries on
(its predictions are stored either way, with ``runs.request_id``).
"""

from __future__ import annotations

import argparse

import psycopg
from psycopg import sql

from .schema import (
    REQUEST_DONE,
    REQUEST_FAILED,
    REQUEST_PARTIAL,
    REQUEST_QUEUED,
    REQUEST_RUNNING,
    REQUEST_STATUS_CODES,
    SCHEMA,
)

#: The longest error text stored; the workflow log has the rest.
MAX_ERROR_CHARS = 2000


def _table() -> sql.Composed:
    return sql.SQL("{}.prediction_requests").format(sql.Identifier(SCHEMA))


def mark_request(
    conn: psycopg.Connection,
    request_id: int,
    status: int,
    *,
    error: str | None = None,
) -> bool:
    """Move a request to ``status``; False when there is no such request.

    ``running`` sets ``started_at``; a final status sets ``finished_at``. A
    finished request is never moved back to ``running`` (a retried job
    reports again only when it ends), and ``failed`` from the workflow step
    does not overwrite a status the script already wrote.
    """
    if status == REQUEST_RUNNING:
        query = sql.SQL(
            "UPDATE {} SET status = %s, started_at = now(), error = NULL "
            "WHERE request_id = %s AND status IN (%s, %s)"
        ).format(_table())
        params: tuple = (status, request_id, REQUEST_QUEUED, REQUEST_RUNNING)
    elif status in (REQUEST_DONE, REQUEST_PARTIAL, REQUEST_FAILED):
        query = sql.SQL(
            "UPDATE {} SET status = %s, finished_at = now(), error = %s, "
            "started_at = coalesce(started_at, now()) "
            "WHERE request_id = %s AND status IN (%s, %s)"
        ).format(_table())
        params = (
            status,
            None if error is None else error[:MAX_ERROR_CHARS],
            request_id,
            REQUEST_QUEUED,
            REQUEST_RUNNING,
        )
    else:
        raise ValueError(f"Unknown request status {status!r}")
    with conn.cursor() as cur:
        cur.execute(query, params)
        updated = cur.rowcount > 0
    conn.commit()
    return updated


def report_request(request_id: int | None, status: int, *, error: str | None = None):
    """``mark_request`` on a new Aiven connection; never raises.

    A no-op without a request (every scheduled run).
    """
    if request_id is None:
        return
    from nba_ou.postgre_db.config.db_config import connect_nba_db

    try:
        with connect_nba_db(env="aiven") as conn:
            if not mark_request(conn, int(request_id), status, error=error):
                print(f"  - Request {request_id}: already finished or not found")
    except Exception as exc:  # noqa: BLE001 -- a status write never stops a run
        print(f"  ✖ Request {request_id}: status not written ({exc})")


def final_status(failures: list[str], predicted: int) -> int:
    """``failed`` when nothing was predicted, ``partial`` when something
    failed, ``done`` otherwise."""
    if not predicted:
        return REQUEST_FAILED
    return REQUEST_PARTIAL if failures else REQUEST_DONE


def main() -> None:
    parser = argparse.ArgumentParser(description="Set a prediction request's status.")
    parser.add_argument("--request-id", type=int, required=True)
    parser.add_argument("--status", choices=sorted(REQUEST_STATUS_CODES), required=True)
    parser.add_argument("--error")
    args = parser.parse_args()
    report_request(args.request_id, REQUEST_STATUS_CODES[args.status], error=args.error)


if __name__ == "__main__":
    main()
