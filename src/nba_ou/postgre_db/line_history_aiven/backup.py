"""Copy the Aiven line-history store to S3 as Parquet.

The mechanics -- relkind-based discovery, per-partition export, manifests --
live in :mod:`nba_ou.postgre_db.backup_core` and are shared with the Supabase
backup, so the two cannot drift apart again. This module is just the
line-history-shaped entry point onto them.

Lands beside the Supabase backups, under the same
``backups/db/<YYYY-MM-DD>/<schema>/`` layout, so a restore reads the same way
regardless of which database a table came from. The schema name keeps the two
apart: everything here goes under ``.../line_history/``.
"""

from __future__ import annotations

from typing import Any

import psycopg

from nba_ou.postgre_db import backup_core
from nba_ou.postgre_db.backup_core import (
    CHUNK_SIZE,
    MANIFEST_NAME,
    S3_BACKUP_PREFIX,
    BackupResult,
    BackupTarget,
    VerificationIssue,
    export_table_to_parquet,
    get_s3_settings,
    read_manifest,
)

from .schema import SCHEMA

__all__ = [
    "CHUNK_SIZE",
    "MANIFEST_NAME",
    "S3_BACKUP_PREFIX",
    "BackupResult",
    "BackupTarget",
    "backup_line_history",
    "discover_backup_targets",
    "export_table_to_parquet",
    "get_s3_settings",
    "list_backup_dates",
    "read_manifest",
    "VerificationIssue",
    "verify_line_history",
]


def discover_backup_targets(
    conn: psycopg.Connection,
    schema: str = SCHEMA,
) -> tuple[list[BackupTarget], list[str]]:
    """Relations worth exporting, defaulting to the line-history schema."""
    return backup_core.discover_backup_targets(conn, schema)


def backup_line_history(
    conn: psycopg.Connection,
    *,
    schema: str = SCHEMA,
    dry_run: bool = False,
    date_tag: str | None = None,
    s3_client: Any = None,
    progress: bool = True,
) -> BackupResult:
    """Export every relation in ``schema`` to ``backups/db/<tag>/<schema>/``."""
    return backup_core.backup_database(
        conn,
        schemas=[schema],
        dry_run=dry_run,
        date_tag=date_tag,
        s3_client=s3_client,
        progress=progress,
    )


def list_backup_dates(
    *,
    schema: str = SCHEMA,
    s3_client: Any = None,
    limit: int = 24,
) -> list[str]:
    """Date tags that already hold a line-history backup, newest first."""
    return backup_core.list_backup_dates(
        schema=schema, s3_client=s3_client, limit=limit
    )


def verify_line_history(
    conn: psycopg.Connection,
    date_tag: str,
    *,
    schema: str = SCHEMA,
    s3_client: Any = None,
) -> list[VerificationIssue]:
    """Check that *date_tag* covers every live line-history relation."""
    return backup_core.verify_backup(
        conn, date_tag, schemas=[schema], s3_client=s3_client
    )
