"""Back up the Aiven line-history store to S3 as Parquet.

The counterpart to ``backup_db_to_s3.py``, which covers Supabase. Both write to
``s3://<BUCKET>/backups/db/<YYYY-MM-DD>/<schema>/<table>.parquet``, so a restore
reads the same way whichever database a table came from; the ``line_history``
schema name keeps this store's files in their own folder.

``lh_line`` is partitioned by season and is exported one partition per file.
The partitioned parent itself is skipped -- it holds no rows, and including it
would write the whole fact table a second time.

Usage:
    python scripts/backup/backup_line_history_to_s3.py              # back it up
    python scripts/backup/backup_line_history_to_s3.py --dry-run    # list, upload nothing
    python scripts/backup/backup_line_history_to_s3.py --list        # existing backups
    python scripts/backup/backup_line_history_to_s3.py --verify      # check today's backup
    python scripts/backup/backup_line_history_to_s3.py --date-tag 2026-08-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime

from nba_ou.postgre_db.config.db_config import connect_line_history_db
from nba_ou.postgre_db.line_history_aiven import backup as backup_mod


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
        tags = backup_mod.list_backup_dates()
        if not tags:
            print("No line-history backups found.")
        else:
            print(f"{len(tags)} backup(s), newest first:")
            for tag in tags:
                print(f"  {tag}")
        return 0

    if args.verify:
        tag = args.date_tag or datetime.now(UTC).strftime("%Y-%m-%d")
        with connect_line_history_db() as conn:
            issues = backup_mod.verify_line_history(conn, tag)
        if issues:
            print(f"Backup {tag} is incomplete -- {len(issues)} issue(s):")
            for issue in issues:
                print(f"  {issue}")
            return 1
        print(f"Backup {tag} covers every live relation. ✓")
        return 0

    with connect_line_history_db() as conn:
        result = backup_mod.backup_line_history(
            conn, dry_run=args.dry_run, date_tag=args.date_tag
        )

    if not result.uploaded:
        print("Nothing was backed up.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
