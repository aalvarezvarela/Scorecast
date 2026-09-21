"""Shared backup machinery for both Postgres instances.

Supabase and Aiven are backed up by different entry points but must behave
identically, because a restore reads them the same way:
``s3://<bucket>/backups/db/<YYYY-MM-DD>/<schema>/<table>.parquet``.

Two rules live here rather than in either caller, because both got them wrong
independently at some point:

**Discovery is relkind-based, never ``information_schema``.** A partitioned
parent and its leaves are both reported as ``BASE TABLE`` by
``information_schema.tables``. Backing up that list writes every row twice --
once through the parent, once across the partitions. ``pg_class.relkind``
tells them apart: ``p`` is a parent (skipped, it stores nothing), ``r`` with a
``pg_inherits`` row is a leaf (exported on its own), ``r`` without one is an
ordinary table.

**Schemas are discovered and excluded, never enumerated.** A hand-maintained
include list silently stops covering new schemas the moment someone adds one,
and keeps naming ones that have been dropped. Both happened here: ``lineups``,
``nba_all_star_voting`` and ``nba_predictions`` went unbacked for months while
the list still named the long-empty ``nba_odds``. Everything that is not
platform infrastructure is in scope by default.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import psycopg

from nba_ou.postgre_db.config.db_config import get_config
from nba_ou.utils.s3_models import make_s3_client

#: Shared by both databases so a restore reads the same way either side.
S3_BACKUP_PREFIX = "backups/db"

#: Rows per SELECT batch. Bounds the driver's buffer, not the final frame.
CHUNK_SIZE = 100_000

#: Filename of the per-schema manifest written beside that schema's Parquet.
MANIFEST_NAME = "_manifest.json"

#: Object tag driving lifecycle. Weekly in-season runs pile up fast, so only
#: the first backup of each month is kept long-term; the rest age out. S3
#: lifecycle cannot express "keep the first of each month" by date, but it can
#: filter on a tag, so the decision is made here, at write time.
RETENTION_TAG_KEY = "retention"
RETENTION_MONTHLY = "monthly"
RETENTION_WEEKLY = "weekly"

#: Managed by the platform, not by this project: Supabase internals, the
#: connection pooler, and Postgres' own catalogs. Nothing here is ours to
#: restore, and ``vault`` holds secrets we deliberately never copy.
EXCLUDED_SCHEMAS = frozenset(
    {
        "auth",
        "cron",
        "extensions",
        "graphql",
        "graphql_public",
        "information_schema",
        "net",
        "pgbouncer",
        "realtime",
        "storage",
        "supabase_functions",
        "supabase_migrations",
        "vault",
    }
)

#: ``pg_catalog``, ``pg_toast``, ``pg_temp_*`` and friends.
EXCLUDED_SCHEMA_PREFIXES = ("pg_",)


@dataclass
class TaggingState:
    """Whether this run may attach retention tags.

    Tagging needs ``s3:PutObjectTagging`` on top of ``s3:PutObject``. A role
    that has the latter but not the former would otherwise fail every upload,
    turning a retention optimisation into a total backup outage. Losing the
    tier is recoverable -- losing the backup is not -- so the first refusal
    disables tagging for the rest of the run and the upload is retried plain.
    """

    enabled: bool = True
    warned: bool = False


def is_tagging_permission_error(exc: Exception) -> bool:
    """True when *exc* is S3 refusing the tagging part of a tagged put."""
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error") or {}
    if error.get("Code") not in {"AccessDenied", "AccessDeniedException"}:
        return False
    return "PutObjectTagging" in str(error.get("Message", ""))


def put_backup_object(
    s3_client: Any,
    *,
    bucket: str,
    key: str,
    body: bytes,
    tagging: str | None,
    state: TaggingState,
    progress: bool = True,
) -> None:
    """Upload *body*, attaching the retention tag when permissions allow."""
    if tagging and state.enabled:
        try:
            s3_client.put_object(Bucket=bucket, Key=key, Body=body, Tagging=tagging)
            return
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is tagging
            if not is_tagging_permission_error(exc):
                raise
            state.enabled = False
            if progress and not state.warned:
                state.warned = True
                print(
                    "\n  ! s3:PutObjectTagging denied -- uploading without "
                    "retention tags.\n"
                    "    Backups are still complete, but lifecycle tiering "
                    "will not apply to them.\n"
                    "    Grant the tagging permission, then run "
                    "scripts/backup/tag_existing_backups.py.\n"
                )
    s3_client.put_object(Bucket=bucket, Key=key, Body=body)


@dataclass(frozen=True)
class BackupTarget:
    """One relation to export."""

    table: str
    is_partition: bool

    @property
    def label(self) -> str:
        return "partition" if self.is_partition else "table"


@dataclass
class SchemaResult:
    """What one schema contributed to a run."""

    schema: str
    uploaded: list[str] = field(default_factory=list)
    rows: int = 0
    bytes_written: int = 0
    skipped_parents: list[str] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BackupResult:
    bucket: str
    prefix: str
    uploaded: list[str] = field(default_factory=list)
    rows: int = 0
    bytes_written: int = 0
    skipped_parents: list[str] = field(default_factory=list)
    schemas: list[SchemaResult] = field(default_factory=list)


def retention_class(date_tag: str) -> str:
    """``monthly`` for the first backup of a calendar month, else ``weekly``.

    Weekly runs are seven days apart, so exactly one of them lands in days 1-7
    of any month. Deciding by day-of-month keeps this a pure function of the
    tag -- no listing of previous backups, no state -- while still yielding one
    long-lived backup per month.
    """
    try:
        day = int(date_tag.split("-")[2])
    except (IndexError, ValueError):
        return RETENTION_MONTHLY  # Unparseable: keep it rather than expire it.
    return RETENTION_MONTHLY if day <= 7 else RETENTION_WEEKLY


def get_s3_settings() -> dict[str, str | None]:
    """Bucket / region / profile from ``[S3]``, overridable by env vars."""
    import os

    config = get_config()
    profile = os.getenv("S3_AWS_PROFILE", config.get("S3", "AWS_PROFILE", fallback=""))
    return {
        "bucket": os.getenv("S3_BACKUP_BUCKET") or config.get("S3", "BUCKET"),
        "region": os.getenv("AWS_REGION") or config.get("S3", "AWS_REGION"),
        # Empty means "use the ambient credentials", which is what CI provides
        # via OIDC; a named profile is only for local runs.
        "profile": (profile.strip() or None) if profile else None,
    }


def is_project_schema(name: str) -> bool:
    """True when *name* is ours to back up rather than platform infrastructure."""
    if name in EXCLUDED_SCHEMAS:
        return False
    return not name.startswith(EXCLUDED_SCHEMA_PREFIXES)


def discover_schemas(conn: psycopg.Connection) -> list[str]:
    """Every project schema holding at least one relation, alphabetically.

    Inverted on purpose: new schemas are covered the day they appear, and a
    dropped one stops being named without anyone editing a list.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT n.nspname,
                   count(c.oid) FILTER (WHERE c.relkind IN ('r', 'p')) AS relations
            FROM pg_namespace n
            LEFT JOIN pg_class c ON c.relnamespace = n.oid
            GROUP BY n.nspname
            ORDER BY n.nspname
            """
        )
        rows = cur.fetchall()

    return [name for name, relations in rows if relations and is_project_schema(name)]


def discover_backup_targets(
    conn: psycopg.Connection,
    schema: str,
) -> tuple[list[BackupTarget], list[str]]:
    """Relations worth exporting, plus the partitioned parents deliberately skipped.

    ``relkind`` distinguishes what ``information_schema`` cannot:

    * ``p`` -- partitioned parent. Holds no rows itself; skipped.
    * ``r`` with a ``pg_inherits`` row -- a partition leaf. Exported.
    * ``r`` with none -- an ordinary table. Exported.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.relname,
                   c.relkind,
                   (i.inhrelid IS NOT NULL) AS is_partition
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            LEFT JOIN pg_inherits i ON i.inhrelid = c.oid
            WHERE n.nspname = %s
              AND c.relkind IN ('r', 'p')
            ORDER BY c.relname
            """,
            (schema,),
        )
        rows = cur.fetchall()

    targets: list[BackupTarget] = []
    skipped: list[str] = []
    for relname, relkind, is_partition in rows:
        if relkind == "p":
            skipped.append(relname)
            continue
        targets.append(BackupTarget(table=relname, is_partition=bool(is_partition)))
    return targets, skipped


def export_table_to_parquet(
    conn: psycopg.Connection,
    schema: str,
    table: str,
) -> tuple[bytes, int]:
    """Export one relation to an in-memory Parquet file."""
    query = f'SELECT * FROM "{schema}"."{table}"'
    chunks = [
        chunk
        for chunk in pd.read_sql(query, conn, chunksize=CHUNK_SIZE)  # type: ignore[call-overload]
    ]
    frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    buffer = io.BytesIO()
    frame.to_parquet(buffer, engine="pyarrow", compression="snappy", index=False)
    buffer.seek(0)
    return buffer.read(), len(frame)


def build_manifest(result: SchemaResult, *, date_tag: str) -> dict[str, Any]:
    """What this schema's backup claims to contain.

    Without it "complete" is unverifiable, which is how two gaps went unnoticed:
    nothing recorded what *should* have been there to compare against.
    """
    return {
        "schema": result.schema,
        "date_tag": date_tag,
        "generated_at": datetime.now(UTC).isoformat(),
        "retention": retention_class(date_tag),
        "table_count": len(result.tables),
        "total_rows": result.rows,
        "total_bytes": result.bytes_written,
        "skipped_parents": result.skipped_parents,
        "tables": result.tables,
    }


def backup_schema(
    conn: psycopg.Connection,
    schema: str,
    *,
    date_tag: str,
    bucket: str,
    s3_client: Any = None,
    dry_run: bool = False,
    progress: bool = True,
    tagging_state: TaggingState | None = None,
) -> SchemaResult:
    """Export every relation in *schema* and write its manifest."""
    prefix = f"{S3_BACKUP_PREFIX}/{date_tag}/{schema}"
    tagging = f"{RETENTION_TAG_KEY}={retention_class(date_tag)}"
    state = tagging_state if tagging_state is not None else TaggingState()
    result = SchemaResult(schema=schema)
    targets, result.skipped_parents = discover_backup_targets(conn, schema)

    if progress:
        mode = "[DRY RUN] would write" if dry_run else "writing"
        print(f"{mode} to s3://{bucket}/{prefix}/")
        if result.skipped_parents:
            print(
                "  partitioned parents skipped (their rows live in the "
                f"partitions): {', '.join(result.skipped_parents)}"
            )

    if not targets:
        if progress:
            print(f"  no relations found in schema '{schema}'.")
        return result

    for target in targets:
        key = f"{prefix}/{target.table}.parquet"
        if dry_run:
            if progress:
                print(f"  → {target.table} ({target.label})  →  s3://{bucket}/{key}")
            result.uploaded.append(key)
            continue

        if progress:
            print(f"  exporting {schema}.{target.table} …", end=" ", flush=True)
        data, row_count = export_table_to_parquet(conn, schema, target.table)
        put_backup_object(
            s3_client,
            bucket=bucket,
            key=key,
            body=data,
            tagging=tagging,
            state=state,
            progress=progress,
        )

        result.uploaded.append(key)
        result.rows += row_count
        result.bytes_written += len(data)
        result.tables.append(
            {
                "table": target.table,
                "is_partition": target.is_partition,
                "rows": row_count,
                "bytes": len(data),
                "key": key,
                # Exports are byte-deterministic for unchanged data, so this
                # doubles as a restore integrity check and as the evidence for
                # whether content-addressed dedup would ever pay off.
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        if progress:
            print(f"{row_count:,} rows, {len(data) / (1024 * 1024):.2f} MB ✓")

    if not dry_run:
        manifest = build_manifest(result, date_tag=date_tag)
        put_backup_object(
            s3_client,
            bucket=bucket,
            key=f"{prefix}/{MANIFEST_NAME}",
            body=json.dumps(manifest, indent=2).encode(),
            tagging=tagging,
            state=state,
            progress=progress,
        )

    return result


def backup_database(
    conn: psycopg.Connection,
    *,
    schemas: list[str] | None = None,
    dry_run: bool = False,
    date_tag: str | None = None,
    s3_client: Any = None,
    progress: bool = True,
) -> BackupResult:
    """Export *schemas* (default: every project schema) under one date tag.

    Each run writes under its own date tag, so runs never overwrite one another
    and a restore is just "pick a date".
    """
    tag = date_tag or datetime.now(UTC).strftime("%Y-%m-%d")
    settings = get_s3_settings()
    bucket = str(settings["bucket"])

    result = BackupResult(bucket=bucket, prefix=f"{S3_BACKUP_PREFIX}/{tag}")

    if schemas is None:
        schemas = discover_schemas(conn)
        if progress:
            print(f"discovered {len(schemas)} schema(s): {', '.join(schemas)}\n")

    if s3_client is None and not dry_run:
        s3_client = make_s3_client(
            profile=settings["profile"], region=str(settings["region"])
        )

    tagging_state = TaggingState()
    for schema in schemas:
        schema_result = backup_schema(
            conn,
            schema,
            date_tag=tag,
            bucket=bucket,
            s3_client=s3_client,
            dry_run=dry_run,
            progress=progress,
            tagging_state=tagging_state,
        )
        result.schemas.append(schema_result)
        result.uploaded.extend(schema_result.uploaded)
        result.rows += schema_result.rows
        result.bytes_written += schema_result.bytes_written
        result.skipped_parents.extend(schema_result.skipped_parents)

    if progress:
        verb = "would upload" if dry_run else "uploaded"
        size_mb = result.bytes_written / (1024 * 1024)
        print(
            f"\nDone. {verb} {len(result.uploaded)} file(s) across "
            f"{len(result.schemas)} schema(s), {result.rows:,} rows, "
            f"{size_mb:.2f} MB"
        )
        print(f"S3 prefix: s3://{bucket}/{result.prefix}/")

    return result


def list_backup_dates(
    *,
    schema: str | None = None,
    s3_client: Any = None,
    limit: int = 24,
) -> list[str]:
    """Date tags holding a backup, newest first.

    With *schema*, only tags that actually contain that schema are returned --
    the two databases share the date namespace, so a tag existing says nothing
    about which of them wrote it.
    """
    settings = get_s3_settings()
    bucket = str(settings["bucket"])
    if s3_client is None:
        s3_client = make_s3_client(
            profile=settings["profile"], region=str(settings["region"])
        )

    paginator = s3_client.get_paginator("list_objects_v2")
    tags: set[str] = set()
    for page in paginator.paginate(
        Bucket=bucket, Prefix=f"{S3_BACKUP_PREFIX}/", Delimiter="/"
    ):
        for entry in page.get("CommonPrefixes") or []:
            tag = entry["Prefix"].removeprefix(f"{S3_BACKUP_PREFIX}/").strip("/")
            if tag:
                tags.add(tag)

    dated = sorted(tags, reverse=True)
    if schema is None:
        return dated[:limit]

    keep: list[str] = []
    for tag in dated:
        response = s3_client.list_objects_v2(
            Bucket=bucket, Prefix=f"{S3_BACKUP_PREFIX}/{tag}/{schema}/", MaxKeys=1
        )
        if response.get("KeyCount"):
            keep.append(tag)
        if len(keep) >= limit:
            break
    return keep


def list_backed_up_schemas(
    date_tag: str,
    *,
    s3_client: Any = None,
    bucket: str | None = None,
) -> set[str]:
    """Schema folders present under *date_tag*."""
    settings = get_s3_settings()
    bucket = bucket or str(settings["bucket"])
    if s3_client is None:
        s3_client = make_s3_client(
            profile=settings["profile"], region=str(settings["region"])
        )

    paginator = s3_client.get_paginator("list_objects_v2")
    found: set[str] = set()
    for page in paginator.paginate(
        Bucket=bucket, Prefix=f"{S3_BACKUP_PREFIX}/{date_tag}/", Delimiter="/"
    ):
        for entry in page.get("CommonPrefixes") or []:
            name = (
                entry["Prefix"]
                .removeprefix(f"{S3_BACKUP_PREFIX}/{date_tag}/")
                .strip("/")
            )
            if name:
                found.add(name)
    return found


def read_manifest(
    date_tag: str,
    schema: str,
    *,
    s3_client: Any = None,
    bucket: str | None = None,
) -> dict[str, Any] | None:
    """The manifest for one schema's backup, or None when it has none."""
    settings = get_s3_settings()
    bucket = bucket or str(settings["bucket"])
    if s3_client is None:
        s3_client = make_s3_client(
            profile=settings["profile"], region=str(settings["region"])
        )

    key = f"{S3_BACKUP_PREFIX}/{date_tag}/{schema}/{MANIFEST_NAME}"
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
    except Exception:
        return None
    return json.loads(response["Body"].read())


@dataclass
class VerificationIssue:
    """One way a backup fails to match the live database."""

    kind: str
    schema: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.schema}: {self.detail}"


def verify_backup(
    conn: psycopg.Connection,
    date_tag: str,
    *,
    schemas: list[str] | None = None,
    s3_client: Any = None,
) -> list[VerificationIssue]:
    """Check that *date_tag* covers every live relation.

    Row counts drift constantly on a live database, so they are not compared.
    What must hold is structural: every project schema has a folder, every
    folder has a manifest, and every live relation has a file. That is exactly
    the check whose absence let the 2019/2020 line partitions and three whole
    schemas go missing unnoticed.
    """
    settings = get_s3_settings()
    bucket = str(settings["bucket"])
    if s3_client is None:
        s3_client = make_s3_client(
            profile=settings["profile"], region=str(settings["region"])
        )

    live_schemas = schemas if schemas is not None else discover_schemas(conn)
    backed_up = list_backed_up_schemas(date_tag, s3_client=s3_client, bucket=bucket)

    issues: list[VerificationIssue] = []
    for schema in live_schemas:
        if schema not in backed_up:
            issues.append(
                VerificationIssue("missing-schema", schema, "no folder in this backup")
            )
            continue

        manifest = read_manifest(date_tag, schema, s3_client=s3_client, bucket=bucket)
        if manifest is None:
            issues.append(
                VerificationIssue(
                    "missing-manifest", schema, f"no {MANIFEST_NAME}; cannot verify"
                )
            )
            continue

        recorded = {entry["table"] for entry in manifest.get("tables", [])}
        targets, _ = discover_backup_targets(conn, schema)
        for target in targets:
            if target.table not in recorded:
                issues.append(
                    VerificationIssue(
                        "missing-table",
                        schema,
                        f"{target.table} ({target.label}) is live but not in the backup",
                    )
                )

    return issues
