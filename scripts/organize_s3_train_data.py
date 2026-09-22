"""File the flat S3 train_data/ prefix under schema-version folders.

    python scripts/organize_s3_train_data.py              # dry run
    python scripts/organize_s3_train_data.py --execute

Datasets sitting loose at the root of ``train_data/`` move into
``train_data/1_0/``, keeping their names. They predate
``TRAINING_DATA_SCHEMA_VERSION``, so ``1_0`` is a label for "older than every
schema we can name" rather than a recovered value -- the same reasoning that
kept the retired models unconverted.

Only objects directly at the root are touched. Anything already in a subfolder
stays, so re-running this is a no-op and an unrelated archive under the prefix
is never swept.

Server-side copy then delete, so nothing is downloaded. Every copy is done and
checked before any delete: a half-moved tree with holes in it is worse than one
that is merely duplicated.
"""

import argparse
import sys

from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.train_data_store import (
    LEGACY_SCHEMA_VERSION,
    TRAIN_DATA_PREFIX,
    is_unversioned_key,
)
from nba_ou.utils.s3_models import list_s3_objects, make_s3_client


def plan_moves(
    *, s3_client, bucket: str, root: str, schema_version: str
) -> list[tuple[str, str, int]]:
    """(source, target, size) for every dataset loose at the root."""
    destination_root = f"{root}{schema_version}/"
    moves = []
    for obj in list_s3_objects(s3_client=s3_client, bucket=bucket, prefix=root):
        key = str(obj["Key"])
        if not is_unversioned_key(key, root=root):
            continue
        moves.append(
            (key, f"{destination_root}{key[len(root):]}", int(obj.get("Size", 0)))
        )
    return sorted(moves)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the copies and deletes. Without it, this is a dry run.",
    )
    parser.add_argument(
        "--schema-version",
        default=LEGACY_SCHEMA_VERSION,
        help=f"Destination version segment (default: {LEGACY_SCHEMA_VERSION}).",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Copy without deleting the originals. Doubles the stored bytes.",
    )
    args = parser.parse_args()

    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    bucket = SETTINGS.s3_bucket
    root = TRAIN_DATA_PREFIX

    moves = plan_moves(
        s3_client=s3, bucket=bucket, root=root, schema_version=args.schema_version
    )
    if not moves:
        print(f"Nothing to organize under s3://{bucket}/{root}.")
        return 0

    total_bytes = sum(size for _, _, size in moves)
    print(
        f"{'EXECUTE' if args.execute else 'DRY RUN'}: {len(moves)} object(s), "
        f"{total_bytes:,} bytes"
    )
    print(f"  from s3://{bucket}/{root}")
    print(f"  to   s3://{bucket}/{root}{args.schema_version}/\n")
    for source, target, size in moves:
        print(f"  {size:>14,}  {source}")
        print(f"  {'':>14}  -> {target}")

    if not args.execute:
        print("\nDry run: nothing was copied or deleted. Re-run with --execute.")
        return 0

    copied: list[str] = []
    failures = 0
    for source, target, _ in moves:
        try:
            s3.copy_object(
                Bucket=bucket,
                CopySource={"Bucket": bucket, "Key": source},
                Key=target,
            )
            copied.append(source)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  COPY FAILED {source}: {exc}", file=sys.stderr)

    print(f"\nCopied {len(copied)} object(s), {failures} failure(s).")
    if failures:
        print(
            "Refusing to delete the originals while copies are failing.",
            file=sys.stderr,
        )
        return 1

    if args.keep_source:
        print("--keep-source: originals left in place.")
        return 0

    deleted = 0
    for source in copied:
        try:
            s3.delete_object(Bucket=bucket, Key=source)
            deleted += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  DELETE FAILED {source}: {exc}", file=sys.stderr)

    print(f"Deleted {deleted} original(s).")
    print(
        f"\nDone. s3://{bucket}/{root} now holds only schema-version folders."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
