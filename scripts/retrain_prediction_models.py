"""Refit every enabled slot from its spec and stage the result.

    python scripts/retrain_prediction_models.py
    python scripts/retrain_prediction_models.py --slot 2_5/line_error/t0000/main

Each slot's ``channels/config.json`` names the spec to use, and that spec --
not yesterday's model -- supplies the features, the cleaning thresholds and the
hyperparameters. Nothing here can change a configuration; adopting a new one is
``python -m training_pipeline.promote <run_dir> --to-s3``.

The staged build does not serve until ``scripts/promote_build.py --execute``
flips the production pointer, so a bad refit is visible before it reaches
anything.
"""

import argparse
import getpass
import os
import sys
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from nba_ou.config.settings import SETTINGS
from nba_ou.modeling.refit import (
    TrainingFrame,
    TrainingFrameUnavailable,
    model_bytes_of,
    refit_from_spec,
    resolve_training_frame,
)
from nba_ou.modeling.registry_paths import Channel, ModelSlot, parse_horizon
from nba_ou.modeling.registry_store import (
    SlotNotPromotedError,
    resolve_config_spec,
    set_build_channel,
    write_build,
)
from nba_ou.utils.s3_models import make_s3_client

TODAY_TIMEZONE = ZoneInfo("Europe/Madrid")


def configure_tqdm_for_environment() -> None:
    """Keep local tqdm behaviour intact while avoiding noisy CI logs.

    tqdm reads ``TQDM_*`` at bar-creation time, so setting it here disables
    downstream bars without touching their call sites. Set
    ``NBA_OU_GITHUB_ACTIONS_TQDM=keep`` to preserve them in a specific run.
    """
    mode = os.getenv("NBA_OU_GITHUB_ACTIONS_TQDM", "disable").strip().lower()
    if os.getenv("GITHUB_ACTIONS", "").lower() == "true" and mode != "keep":
        os.environ.setdefault("TQDM_DISABLE", "1")


def _today_limit_date() -> str:
    return datetime.now(TODAY_TIMEZONE).strftime("%Y-%m-%d")


def _current_user() -> str | None:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return None


def _parse_slot(value: str, *, root: str) -> ModelSlot:
    parts = [part for part in value.split("/") if part]
    if len(parts) == 3:
        parts.append("main")
    if len(parts) != 4:
        raise SystemExit(
            "--slot must be 'schema_version/target/horizon/variant' "
            f"(variant optional). Got {value!r}."
        )
    schema_version, target, horizon, variant = parts
    return ModelSlot(
        schema_version=schema_version,
        target=target,
        horizon_minutes=parse_horizon(horizon),
        variant=variant,
        root=root,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--slot",
        action="append",
        default=None,
        help="Refit this slot only; repeatable. Default: every enabled slot.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be refitted without fitting or writing.",
    )
    return parser


def main() -> int:
    configure_tqdm_for_environment()
    args = _build_arg_parser().parse_args()

    root = SETTINGS.s3_models_prefix or "models/"
    slots = (
        [_parse_slot(value, root=root) for value in args.slot]
        if args.slot
        else SETTINGS.prediction_model_slots
    )

    if not slots:
        # A valid state, not an error: the registry has to be runnable before
        # the first configuration is promoted.
        print(
            "No model slots enabled in [PredictionModels] ENABLED_MODELS. "
            "Nothing to retrain."
        )
        return 0

    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    bucket = SETTINGS.s3_bucket
    actor = _current_user()
    limit_date = _today_limit_date()

    print(f"Refitting {len(slots)} slot(s) up to {limit_date} in {bucket}\n")

    # One frame per dataset flavour, shared by every slot that wants it: the
    # marginal cost of an extra slot should be one fit, not one dataset build.
    frames: dict[tuple[str, str], TrainingFrame] = {}
    specs = {}
    for slot in slots:
        try:
            specs[slot.describe()] = resolve_config_spec(
                s3_client=s3, bucket=bucket, slot=slot
            )
        except SlotNotPromotedError as exc:
            print(f"{slot.describe()}\n  skipped: {exc}\n")

    by_dataset: dict[tuple[str, str], list[ModelSlot]] = defaultdict(list)
    for slot in slots:
        spec = specs.get(slot.describe())
        if spec is not None:
            by_dataset[(spec.identity.dataset_type, slot.schema_version)].append(slot)

    failures = 0
    staged = 0
    for (dataset_type, schema_version), group in by_dataset.items():
        label = f"{dataset_type} / schema {schema_version}"
        try:
            if (dataset_type, schema_version) not in frames:
                if args.dry_run:
                    print(f"[{label}] would build the training frame")
                else:
                    print(f"[{label}] building training frame...")
                    frames[(dataset_type, schema_version)] = resolve_training_frame(
                        specs[group[0].describe()], limit_date=limit_date
                    )
        except TrainingFrameUnavailable as exc:
            failures += len(group)
            print(
                f"[{label}] unavailable: {exc}\n"
                f"  skipping {len(group)} slot(s): "
                f"{', '.join(slot.describe() for slot in group)}\n",
                file=sys.stderr,
            )
            continue

        for slot in group:
            spec = specs[slot.describe()]
            print(slot.describe())
            print(f"  spec       : {spec.spec_id}")
            if args.dry_run:
                print("  dry run    : not fitted\n")
                continue
            try:
                result = refit_from_spec(
                    spec,
                    slot=slot,
                    frame=frames[(dataset_type, schema_version)],
                    fitted_by=actor,
                )
                write_build(
                    s3_client=s3,
                    bucket=bucket,
                    slot=slot,
                    fit=result.fit,
                    model_bytes=model_bytes_of(result.model),
                    spec=spec,
                )
                set_build_channel(
                    s3_client=s3,
                    bucket=bucket,
                    slot=slot,
                    channel=Channel.STAGING,
                    fit_id=result.fit.fit_id,
                    set_by=actor,
                    reason="daily refit",
                )
            except Exception as exc:  # noqa: BLE001 - one slot must not stop the rest
                failures += 1
                print(f"  FAILED     : {exc}\n", file=sys.stderr)
                continue

            staged += 1
            print(f"  staged     : {result.fit.fit_id}")
            print(f"  model_name : {result.fit.model_name}")
            print(
                f"  trained on : {result.fit.n_train_games} games "
                f"({result.fit.train_date_min.date()} .. "
                f"{result.fit.train_date_max.date()})\n"
            )

    print(f"{staged} build(s) staged, {failures} failure(s).")
    if staged and not args.dry_run:
        print("Promote them with: python scripts/promote_build.py --execute")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
