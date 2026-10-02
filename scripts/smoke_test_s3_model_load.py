"""Check that every enabled slot resolves to a loadable model.

    python scripts/smoke_test_s3_model_load.py
    python scripts/smoke_test_s3_model_load.py --slot 2_5/line_error/t0000/main

Exercises the whole resolution chain -- production pointer, fit, spec, booster
-- and asserts the pieces agree with each other. What it is really guarding is
that the spec a served model is read against is the one its own fit names, so a
configuration change cannot leave production uninterpretable.
"""

import argparse
import os
import sys

from nba_ou.config.settings import SETTINGS
from nba_ou.modeling.registry_paths import Channel, ModelSlot, parse_horizon
from nba_ou.modeling.registry_store import resolve_channel
from nba_ou.utils.s3_models import make_s3_client
from xgboost import XGBRegressor


def _env_or_default(env_name: str, default: str) -> str:
    value = os.getenv(env_name)
    if value is None:
        return default
    return value.strip() or default


def _parse_slot(value: str, *, root: str) -> ModelSlot:
    parts = [part for part in value.split("/") if part]
    if len(parts) == 3:
        parts.append("main")
    if len(parts) != 4:
        raise SystemExit(f"--slot must have 3 or 4 segments. Got {value!r}.")
    schema_version, target, horizon, variant = parts
    return ModelSlot(
        schema_version=schema_version,
        target=target,
        horizon_minutes=parse_horizon(horizon),
        variant=variant,
        root=root,
    )


def check_slot(*, s3_client, bucket: str, slot: ModelSlot, channel: Channel) -> None:
    resolved = resolve_channel(
        s3_client=s3_client, bucket=bucket, slot=slot, channel=channel
    )
    print(f"  {channel.value:<10}: {resolved.fit_id}")
    print(f"  spec      : {resolved.spec_id}")
    print(f"  model_name: {resolved.fit.model_name}")
    print(f"  target    : {resolved.spec.identity.target}")
    print(f"  horizon   : {resolved.spec.identity.horizon_minutes}")
    print(f"  features  : {resolved.spec.features.n_features}")

    model = XGBRegressor()
    model.load_model(bytearray(resolved.model_bytes))
    booster_features = model.get_booster().num_features()
    if booster_features != resolved.spec.features.n_features:
        raise AssertionError(
            f"booster expects {booster_features} features, spec declares "
            f"{resolved.spec.features.n_features}"
        )
    # A fit whose feature count disagrees with its spec would already have been
    # caught by resolve_channel; this re-checks against the BOOSTER, which is
    # the only one of the three that cannot be edited by hand.
    print(f"  loaded    : {booster_features} features, agrees with spec")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slot", action="append", default=None)
    parser.add_argument(
        "--channel",
        default=Channel.PRODUCTION.value,
        choices=[Channel.PRODUCTION.value, Channel.STAGING.value],
    )
    args = parser.parse_args()

    region = _env_or_default("S3_AWS_REGION", SETTINGS.s3_aws_region)
    bucket = _env_or_default("S3_MODEL_BUCKET", SETTINGS.s3_bucket)
    root = SETTINGS.s3_models_prefix or "models/"
    channel = Channel(args.channel)

    slots = (
        [_parse_slot(value, root=root) for value in args.slot]
        if args.slot
        else SETTINGS.prediction_model_slots
    )

    print("S3 model registry smoke test")
    print(f"  region : {region}")
    print(f"  bucket : {bucket}")
    print(f"  channel: {channel.value}")
    print(f"  profile: {SETTINGS.s3_aws_profile or '<none>'}\n")

    if not slots:
        # Nothing promoted yet is a valid state, and CI must not turn it into a
        # red build: there is genuinely nothing to smoke test.
        print("No model slots enabled in [PredictionModels] ENABLED_MODELS.")
        return 0

    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=region)

    failures = 0
    for slot in slots:
        print(slot.describe())
        try:
            check_slot(s3_client=s3, bucket=bucket, slot=slot, channel=channel)
        except Exception as exc:  # noqa: BLE001 - report every slot, not just the first
            failures += 1
            print(f"  FAILED    : {exc}", file=sys.stderr)
        print()

    if failures:
        print(f"{failures} of {len(slots)} slot(s) failed.", file=sys.stderr)
        return 1
    print(f"All {len(slots)} slot(s) loaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
