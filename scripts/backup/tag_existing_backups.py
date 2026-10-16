"""Tag already-uploaded backups with their retention tier.

Backups written before tiered retention existed carry no ``retention`` tag, so
no lifecycle rule matches them and they are kept forever. This applies the same
rule the backup code uses -- first run of a calendar month is kept long-term,
the rest age out -- to objects already in the bucket.

Idempotent: re-running retags to the same values.

Usage:
    python scripts/backup/tag_existing_backups.py --dry-run
    python scripts/backup/tag_existing_backups.py
    python scripts/backup/tag_existing_backups.py --bucket adrian-nba-backups-eu-west-1
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from nba_ou.postgre_db.backup_core import (
    RETENTION_TAG_KEY,
    S3_BACKUP_PREFIX,
    get_s3_settings,
    retention_class,
)
from nba_ou.utils.s3_models import make_s3_client


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bucket", default=None, help="override the configured backup bucket"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be tagged without changing anything",
    )
    args = parser.parse_args()

    settings = get_s3_settings()
    bucket = args.bucket or str(settings["bucket"])
    s3 = make_s3_client(profile=settings["profile"], region=str(settings["region"]))

    counts: Counter[str] = Counter()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{S3_BACKUP_PREFIX}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            parts = key.split("/")
            if len(parts) < 4:
                continue
            tier = retention_class(parts[2])
            counts[tier] += 1
            if args.dry_run:
                continue
            s3.put_object_tagging(
                Bucket=bucket,
                Key=key,
                Tagging={"TagSet": [{"Key": RETENTION_TAG_KEY, "Value": tier}]},
            )

    verb = "would tag" if args.dry_run else "tagged"
    total = sum(counts.values())
    if not total:
        print(f"No backup objects found under s3://{bucket}/{S3_BACKUP_PREFIX}/")
        return 1
    print(f"{verb} {total} object(s) in s3://{bucket}/{S3_BACKUP_PREFIX}/")
    for tier, count in sorted(counts.items()):
        print(f"  {RETENTION_TAG_KEY}={tier}: {count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
