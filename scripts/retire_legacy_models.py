"""Sweep the pre-registry model tree into models/retired/{YYYYMMDD}/.

    python scripts/retire_legacy_models.py              # dry run
    python scripts/retire_legacy_models.py --execute

Everything that predates the slot layout -- the six production families, the
four dead `.joblib` ones, and all of their archives -- moves under one dated
prefix. Afterwards `models/` holds exactly two kinds of thing: dated `retired/`
snapshots, and live schema-version trees such as `models/2_5/`.

A move, not a conversion. Reshaping those bundles into specs would mean giving
each one a `schema_version`, and that information does not exist: their
metadata carries `training_code_tag: "1.0"` and no schema field, and they
predate `TRAINING_DATA_SCHEMA_VERSION` entirely. Any value written there would
be a guess, in the one field whose whole job is to say whether two models are
comparable. Since nothing reads them, they keep their original shape under a
prefix whose name says they are out of service.

Server-side copy then delete, so nothing is downloaded. Copies are done first
and in full; deletes only happen once every copy for that object has succeeded.
"""

import argparse
import re
import sys
from datetime import UTC, datetime

from nba_ou.config.settings import SETTINGS
from nba_ou.utils.s3_models import list_s3_objects, make_s3_client

RETIRED_SEGMENT = "retired"

#: A live tree is named after a training-data schema version, e.g. "2_5".
_SCHEMA_VERSION_RE = re.compile(r"^\d+_\d+$")


def is_legacy_key(key: str, *, root: str) -> bool:
    """True for objects under the old flat ``models/<family>/`` layout."""
    if not key.startswith(root):
        return False
    remainder = key[len(root) :]
    if not remainder or remainder.endswith("/"):
        # Directory placeholder objects: swept with the rest, but they carry no
        # data and must not be mistaken for a family on their own.
        remainder = remainder.rstrip("/")
        if not remainder:
            return False
    top = remainder.split("/", 1)[0]
    if top == RETIRED_SEGMENT:
        return False
    return not _SCHEMA_VERSION_RE.match(top)


def plan_moves(*, s3_client, bucket: str, root: str, stamp: str) -> list[tuple[str, str]]:
    destination_root = f"{root}{RETIRED_SEGMENT}/{stamp}/"
    objects = list_s3_objects(s3_client=s3_client, bucket=bucket, prefix=root)
    moves = []
    for obj in objects:
        key = str(obj["Key"])
        if not is_legacy_key(key, root=root):
            continue
        moves.append((key, f"{destination_root}{key[len(root):]}"))
    return sorted(moves)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the copies and deletes. Without it, this is a dry run.",
    )
    parser.add_argument(
        "--stamp",
        default=datetime.now(tz=UTC).strftime("%Y%m%d"),
        help="Destination date segment (default: today, UTC).",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Copy without deleting the originals. Doubles the stored bytes.",
    )
    args = parser.parse_args()

    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    bucket = SETTINGS.s3_bucket
    root = SETTINGS.s3_models_prefix or "models/"
    if not root.endswith("/"):
        root = f"{root}/"

    moves = plan_moves(s3_client=s3, bucket=bucket, root=root, stamp=args.stamp)
    if not moves:
        print(f"Nothing to retire under s3://{bucket}/{root}.")
        return 0

    families = sorted({key[len(root):].split("/", 1)[0] for key, _ in moves})
    print(f"{'EXECUTE' if args.execute else 'DRY RUN'}: "
          f"{len(moves)} object(s) in {len(families)} family/families")
    print(f"  from s3://{bucket}/{root}")
    print(f"  to   s3://{bucket}/{root}{RETIRED_SEGMENT}/{args.stamp}/\n")
    for family in families:
        count = sum(1 for key, _ in moves if key[len(root):].startswith(f"{family}/"))
        print(f"  {family:<40} {count:>5} object(s)")

    if not args.execute:
        print("\nDry run: nothing was copied or deleted. Re-run with --execute.")
        return 0

    copied: list[str] = []
    failures = 0
    for source, target in moves:
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
        # Nothing is deleted if any copy failed: a half-swept tree with holes in
        # it is worse than one that is simply duplicated.
        print("Refusing to delete the originals while copies are failing.", file=sys.stderr)
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
        f"\nDone. s3://{bucket}/{root} now holds only retired/ snapshots and "
        "live schema-version trees."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
