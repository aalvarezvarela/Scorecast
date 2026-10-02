"""Ship today's staged builds, or roll one back.

    python scripts/promote_build.py                      # dry run, every slot
    python scripts/promote_build.py --execute
    python scripts/promote_build.py --slot 2_5/line_error/t0060/main --execute
    python scripts/promote_build.py --status
    python scripts/promote_build.py --rollback --slot 2_5/line_error/t0060/main --execute

Replaces ``promote_prediction_models.py``. A promotion is now one pointer
write, so this script copies nothing and deletes nothing -- the build it
promotes over stays where it is and remains a valid rollback target.
"""

import argparse
import getpass
import sys

from nba_ou.config.settings import SETTINGS
from nba_ou.modeling.build_promotion import (
    describe_slot,
    promote_build,
    rollback_production,
)
from nba_ou.modeling.registry_paths import ModelSlot, parse_horizon
from nba_ou.utils.s3_models import make_s3_client


def _parse_slot(value: str, *, root: str) -> ModelSlot:
    parts = [part for part in value.split("/") if part]
    if len(parts) == 3:
        parts.append("main")
    if len(parts) != 4:
        raise SystemExit(
            f"--slot must be 'schema_version/target/horizon/variant' "
            f"(variant optional), e.g. '2_5/line_error/t0060/main'. Got {value!r}."
        )
    schema_version, target, horizon, variant = parts
    return ModelSlot(
        schema_version=schema_version,
        target=target,
        horizon_minutes=parse_horizon(horizon),
        variant=variant,
        root=root,
    )


def _current_user() -> str | None:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the pointer writes. Without it, this is a dry run.",
    )
    parser.add_argument(
        "--slot",
        action="append",
        default=None,
        help=(
            "Act on this slot only; repeatable. Default: every slot enabled in "
            "[PredictionModels] ENABLED_MODELS."
        ),
    )
    parser.add_argument(
        "--status", action="store_true", help="Print each slot's channels and exit."
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Point production at an earlier build instead of at staging.",
    )
    parser.add_argument(
        "--to-fit",
        help="Rollback target. Default: the build production replaced.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    root = SETTINGS.s3_models_prefix or "models/"

    if args.slot:
        slots = [_parse_slot(value, root=root) for value in args.slot]
    else:
        slots = SETTINGS.prediction_model_slots

    if not slots:
        # An empty table is a valid state, not an error: the registry has to be
        # runnable before the first configuration is promoted.
        print(
            "No model slots enabled in [PredictionModels] ENABLED_MODELS. "
            "Nothing to promote."
        )
        return 0

    if args.rollback and len(slots) != 1:
        raise SystemExit("--rollback acts on exactly one --slot.")

    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    bucket = SETTINGS.s3_bucket
    actor = _current_user()

    if args.status:
        for slot in slots:
            print(describe_slot(s3_client=s3, bucket=bucket, slot=slot))
        return 0

    mode = "EXECUTE" if args.execute else "DRY RUN"
    action = "rollback" if args.rollback else "promotion"
    print(f"{mode} {action} in {bucket}\n")

    failures = 0
    for slot in slots:
        try:
            if args.rollback:
                outcome = rollback_production(
                    s3_client=s3,
                    bucket=bucket,
                    slot=slot,
                    fit_id=args.to_fit,
                    dry_run=not args.execute,
                    set_by=actor,
                )
            else:
                outcome = promote_build(
                    s3_client=s3,
                    bucket=bucket,
                    slot=slot,
                    dry_run=not args.execute,
                    set_by=actor,
                )
        except Exception as exc:  # noqa: BLE001 - one bad slot must not stop the rest
            failures += 1
            print(f"{slot.describe()}\n  FAILED: {exc}\n", file=sys.stderr)
            continue

        print(slot.describe())
        if outcome.promoted:
            print(
                f"  {outcome.previous_fit_id or '(none)'} -> "
                f"{outcome.promoted_fit_id}  (spec {outcome.spec_id})"
            )
        else:
            print(f"  skipped: {outcome.skipped_reason}")
        print()

    if failures:
        print(f"{failures} slot(s) failed.", file=sys.stderr)
        return 1
    if not args.execute:
        print("Dry run: nothing was written. Re-run with --execute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
