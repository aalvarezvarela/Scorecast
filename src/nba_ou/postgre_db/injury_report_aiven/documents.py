"""Durable receipts for live PDFs. Only completed ingests are skippable."""

from __future__ import annotations

from psycopg.rows import dict_row


def ensure_document_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS injury_report.ir_document (
                source_url TEXT PRIMARY KEY,
                expected_at TIMESTAMPTZ NOT NULL,
                observed_at TIMESTAMPTZ,
                s3_key TEXT NOT NULL,
                archived_at TIMESTAMPTZ,
                ingested_at TIMESTAMPTZ,
                last_error TEXT
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS ir_document_expected_idx
            ON injury_report.ir_document (expected_at)
        """)
    conn.commit()


def register_documents(conn, candidates, *, prefix: str) -> None:
    from nba_ou.fetch_data.injury_reports.archive.urls import s3_key

    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO injury_report.ir_document (source_url, expected_at, s3_key)
            VALUES (%s, %s, %s) ON CONFLICT (source_url) DO NOTHING
        """,
            [
                (
                    c.url,
                    c.report_datetime_utc,
                    s3_key(c.report_datetime_et, prefix=prefix),
                )
                for c in candidates
            ],
        )
    conn.commit()


def pending_documents(conn, start, end) -> list[dict]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT * FROM injury_report.ir_document
            WHERE expected_at >= %s AND expected_at <= %s AND ingested_at IS NULL
            ORDER BY expected_at, source_url
        """,
            (start, end),
        )
        rows = cur.fetchall()
    conn.commit()
    return rows


def mark_archived(conn, url: str, observed_at) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE injury_report.ir_document
            SET observed_at = %s, archived_at = COALESCE(archived_at, now()), last_error = NULL
            WHERE source_url = %s
        """,
            (observed_at, url),
        )
    conn.commit()


def mark_failed(conn, url: str, error: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE injury_report.ir_document SET last_error = %s WHERE source_url = %s",
            (error[:2000], url),
        )
    conn.commit()


def mark_ingested(conn, urls: list[str]) -> None:
    """The caller commits this together with report records and status spans."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE injury_report.ir_document SET ingested_at = now(), last_error = NULL
            WHERE source_url = ANY(%s)
        """,
            (urls,),
        )
