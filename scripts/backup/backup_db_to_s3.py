"""Back up the Supabase PostgreSQL schemas to S3 as Parquet.

The counterpart to ``backup_line_history_to_s3.py``, which covers Aiven. Both
write to ``s3://<BUCKET>/backups/db/<YYYY-MM-DD>/<schema>/<table>.parquet`` and
share their machinery via ``nba_ou.postgre_db.backup_core``, so a restore reads
the same way whichever database a table came from.

Schemas are discovered, not listed: everything except platform infrastructure
(Supabase internals, the pooler, ``pg_*``) is in scope. ``--schemas`` narrows a
run for ad hoc use, but the scheduled backup should always take the default.

Usage:
    python scripts/backup/backup_db_to_s3.py                    # back up everything
    python scripts/backup/backup_db_to_s3.py --schemas lineups  # just these
    python scripts/backup/backup_db_to_s3.py --dry-run          # list, upload nothing
    python scripts/backup/backup_db_to_s3.py --verify           # check today's backup
    python scripts/backup/backup_db_to_s3.py --list             # existing backups
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from nba_ou.postgre_db.backup_core import (
    backup_database,
    discover_schemas,
    list_backup_dates,
    verify_backup,
)
from nba_ou.postgre_db.config.db_config import connect_nba_db


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schemas",
        nargs="+",
        default=None,
        help="schemas to back up (default: every project schema, discovered)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be written without uploading",
    )
    parser.add_argument(
        "--date-tag",
        default=None,
        help="override the date folder (default: today, UTC)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list the date tags that already hold a backup, newest first",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check that a backup covers every live relation, then exit",
    )
    args = parser.parse_args()

    if args.list:
        tags = list_backup_dates()
        if not tags:
            print("No backups found.")
        else:
            print(f"{len(tags)} backup(s), newest first:")
            for tag in tags:
                print(f"  {tag}")
        return 0

    tag = args.date_tag or datetime.now(UTC).strftime("%Y-%m-%d")

    if args.verify:
        with connect_nba_db() as conn:
            issues = verify_backup(conn, tag, schemas=args.schemas)
        if issues:
            print(f"Backup {tag} is incomplete -- {len(issues)} issue(s):")
            for issue in issues:
                print(f"  {issue}")
            return 1
        print(f"Backup {tag} covers every live relation. ✓")
        return 0

    with connect_nba_db() as conn:
        if args.schemas is None:
            print(f"{len(discover_schemas(conn))} project schema(s) in scope.\n")
        result = backup_database(
            conn,
            schemas=args.schemas,
            dry_run=args.dry_run,
            date_tag=args.date_tag,
        )

    if not result.uploaded:
        print("Nothing was backed up.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
