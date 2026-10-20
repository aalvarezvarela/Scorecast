"""Move superseded registry builds into models/retired/{YYYYMMDD}/.

    python scripts/archive_registry_builds.py --before 20261005               # dry run
    python scripts/archive_registry_builds.py --before 20261005 --target line_error
    python scripts/archive_registry_builds.py --before 20261005 --include-rollback-target --execute

The registry never rewrites a build, and ``promote_build`` keeps the build it
replaced as the rollback target, so superseded builds pile up inside each slot.
This moves the ones fitted before ``--before`` out of the live tree, keeping
their path under the dated prefix:

    models/2_5/line_error/t0060/main/builds/{fit_id}/...
 -> models/retired/20261005/2_5/line_error/t0060/main/builds/{fit_id}/...

What is never moved:

- a build that ``production`` or ``staging`` points at (``unreferenced_fit_ids``
  already guarantees this);
- production's ``previous_fit_id`` -- the build ``promote_build --rollback``
  falls back to -- unless ``--include-rollback-target`` says that rollback is
  no longer wanted;
- a spec that ``config`` names or that a build left in the slot was fitted
  from;
- ``channels/`` and its history: the record of what served when stays with the
  slot, even once the build it names has moved.

A move, not a delete: server-side copy, then delete only once every copy has
succeeded, the same as ``retire_legacy_models.py``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime

from nba_ou.config.settings import SETTINGS
from nba_ou.modeling.registry_models import BuildPointer, ConfigPointer
from nba_ou.modeling.registry_paths import Channel, ModelSlot, parse_horizon
from nba_ou.modeling.registry_store import (
    list_fit_ids,
    list_spec_ids,
    read_channel,
    read_fit,
    unreferenced_fit_ids,
)
from nba_ou.utils.s3_models import list_s3_objects, make_s3_client

RETIRED_SEGMENT = "retired"


@dataclass
class SlotArchivePlan:
    slot: ModelSlot
    fit_ids: list[str] = field(default_factory=list)
    spec_ids: list[str] = field(default_factory=list)
    #: (source, target) for every object under the moved builds and specs.
    moves: list[tuple[str, str]] = field(default_factory=list)
    #: Builds the cutoff selected but a rule protected, with the reason.
    kept: list[tuple[str, str]] = field(default_factory=list)


def discover_slots(
    *, s3_client, bucket: str, root: str, schema_version: str, targets: set[str] | None
) -> list[ModelSlot]:
    """Every slot under ``{root}{schema_version}/`` that holds a build or spec."""
    prefix = f"{root}{schema_version}/"
    slots: dict[str, ModelSlot] = {}
    for obj in list_s3_objects(s3_client=s3_client, bucket=bucket, prefix=prefix):
        parts = str(obj["Key"])[len(root) :].split("/")
        # schema/target/horizon/variant/{builds,specs}/...
        if len(parts) < 6 or parts[4] not in ("builds", "specs"):
            continue
        _, target, horizon, variant = parts[:4]
        if targets and target not in targets:
            continue
        name = "/".join(parts[:4])
        if name not in slots:
            slots[name] = ModelSlot(
                schema_version=schema_version,
                target=target,
                horizon_minutes=parse_horizon(horizon),
                variant=variant,
                root=root,
            )
    return [slots[name] for name in sorted(slots)]


def retired_key(key: str, *, root: str, stamp: str) -> str:
    return f"{root}{RETIRED_SEGMENT}/{stamp}/{key[len(root):]}"


def plan_slot(
    *,
    s3_client,
    bucket: str,
    slot: ModelSlot,
    before: datetime,
    stamp: str,
    include_rollback_target: bool,
) -> SlotArchivePlan:
    plan = SlotArchivePlan(slot=slot)
    candidates = unreferenced_fit_ids(
        s3_client=s3_client, bucket=bucket, slot=slot, keep_recent=0, keep_after=before
    )

    production = read_channel(
        s3_client=s3_client, bucket=bucket, slot=slot, channel=Channel.PRODUCTION
    )
    rollback = (
        production.previous_fit_id if isinstance(production, BuildPointer) else None
    )
    for fit_id in candidates:
        if fit_id == rollback and not include_rollback_target:
            plan.kept.append((fit_id, "production's rollback target"))
        else:
            plan.fit_ids.append(fit_id)

    archived = set(plan.fit_ids)
    remaining = [f for f in list_fit_ids(s3_client=s3_client, bucket=bucket, slot=slot)
                 if f not in archived]
    specs_in_use = {
        read_fit(s3_client=s3_client, bucket=bucket, slot=slot, fit_id=f).spec_id
        for f in remaining
    }
    config = read_channel(
        s3_client=s3_client, bucket=bucket, slot=slot, channel=Channel.CONFIG
    )
    if isinstance(config, ConfigPointer):
        specs_in_use.add(config.spec_id)
    plan.spec_ids = [
        s for s in list_spec_ids(s3_client=s3_client, bucket=bucket, slot=slot)
        if s not in specs_in_use
    ]

    root = slot.root
    for fit_id in plan.fit_ids:
        for obj in list_s3_objects(
            s3_client=s3_client, bucket=bucket, prefix=slot.build_prefix(fit_id)
        ):
            key = str(obj["Key"])
            plan.moves.append((key, retired_key(key, root=root, stamp=stamp)))
    for spec_id in plan.spec_ids:
        key = slot.spec_key(spec_id)
        plan.moves.append((key, retired_key(key, root=root, stamp=stamp)))
    return plan


def execute_moves(
    *, s3_client, bucket: str, moves: list[tuple[str, str]], keep_source: bool
) -> int:
    """Copy everything, then delete the originals only if no copy failed."""
    copied: list[str] = []
    failures = 0
    for source, target in moves:
        try:
            s3_client.copy_object(
                Bucket=bucket, CopySource={"Bucket": bucket, "Key": source}, Key=target
            )
            copied.append(source)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  COPY FAILED {source}: {exc}", file=sys.stderr)

    print(f"\nCopied {len(copied)} object(s), {failures} failure(s).")
    if failures:
        print("Refusing to delete the originals while copies are failing.", file=sys.stderr)
        return 1
    if keep_source:
        print("--keep-source: originals left in place.")
        return 0

    deleted = 0
    for source in copied:
        try:
            s3_client.delete_object(Bucket=bucket, Key=source)
            deleted += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  DELETE FAILED {source}: {exc}", file=sys.stderr)
    print(f"Deleted {deleted} original(s).")
    return 1 if failures else 0


def _parse_day(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y%m%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYYMMDD, got {value!r}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--before",
        type=_parse_day,
        required=True,
        help="Archive builds fitted before this UTC day (YYYYMMDD). Required: "
        "there is no default cutoff.",
    )
    parser.add_argument("--schema-version", default="2_5")
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="Only slots of this target; repeatable. Default: every target.",
    )
    parser.add_argument(
        "--include-rollback-target",
        action="store_true",
        help="Also archive production's previous_fit_id. promote_build "
        "--rollback then has no default target.",
    )
    parser.add_argument(
        "--stamp",
        default=datetime.now(tz=UTC).strftime("%Y%m%d"),
        help="Destination date segment (default: today, UTC).",
    )
    parser.add_argument(
        "--keep-source",
        action="store_true",
        help="Copy without deleting the originals.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the copies and deletes. Without it, this is a dry run.",
    )
    args = parser.parse_args()

    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    bucket = SETTINGS.s3_bucket
    root = SETTINGS.s3_models_prefix or "models/"
    if not root.endswith("/"):
        root = f"{root}/"

    slots = discover_slots(
        s3_client=s3,
        bucket=bucket,
        root=root,
        schema_version=args.schema_version,
        targets=set(args.target) if args.target else None,
    )
    plans = [
        plan_slot(
            s3_client=s3,
            bucket=bucket,
            slot=slot,
            before=args.before,
            stamp=args.stamp,
            include_rollback_target=args.include_rollback_target,
        )
        for slot in slots
    ]

    print(f"{'EXECUTE' if args.execute else 'DRY RUN'}: builds fitted before "
          f"{args.before:%Y-%m-%d}, {len(slots)} slot(s)")
    print(f"  to s3://{bucket}/{root}{RETIRED_SEGMENT}/{args.stamp}/\n")
    for plan in plans:
        print(f"  {plan.slot.describe():<32} {len(plan.fit_ids)} build(s), "
              f"{len(plan.spec_ids)} spec(s)")
        for fit_id in plan.fit_ids:
            print(f"      move  build {fit_id}")
        for spec_id in plan.spec_ids:
            print(f"      move  spec  {spec_id}")
        for fit_id, reason in plan.kept:
            print(f"      keep  build {fit_id} ({reason})")

    moves = [move for plan in plans for move in plan.moves]
    if not moves:
        print("\nNothing to archive.")
        return 0
    if not args.execute:
        print(f"\nDry run: {len(moves)} object(s) would move. Re-run with --execute.")
        return 0
    return execute_moves(
        s3_client=s3, bucket=bucket, moves=moves, keep_source=args.keep_source
    )


if __name__ == "__main__":
    raise SystemExit(main())
