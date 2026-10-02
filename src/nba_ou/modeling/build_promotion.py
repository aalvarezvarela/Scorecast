"""Promote a staged build to production, and roll one back.

The daily counterpart to ``training_pipeline.promote``: that one adopts a
*configuration* (rare, human judgement), this one ships a *build* (every day,
unattended).

A promotion is one ``PutObject`` against ``channels/production.json``. Nothing
is copied and nothing is deleted, so the previous build stays exactly where it
was and a rollback is the same call naming an earlier ``fit_id``. The old
``staging -> production -> archive`` shuffle of megabytes is gone, and with it
the window in which a reader could see a half-moved bundle.
"""

from __future__ import annotations

from dataclasses import dataclass

from nba_ou.modeling.registry_models import BuildPointer, ModelSpec
from nba_ou.modeling.registry_paths import Channel, ModelSlot
from nba_ou.modeling.registry_store import (
    RegistryError,
    SlotNotPromotedError,
    list_fit_ids,
    read_channel,
    resolve_build,
    resolve_channel,
    resolve_config_spec,
    set_build_channel,
)


class BuildPromotionError(RegistryError):
    """A staged build cannot be promoted."""


@dataclass(frozen=True)
class PromotionOutcome:
    slot: ModelSlot
    promoted_fit_id: str | None
    previous_fit_id: str | None
    spec_id: str | None
    skipped_reason: str | None = None

    @property
    def promoted(self) -> bool:
        return self.promoted_fit_id is not None


def verify_staged_build(
    *, s3_client, bucket: str, slot: ModelSlot
) -> tuple[str, ModelSpec]:
    """Check a staged build is loadable and is the configuration we asked for.

    Two failures this catches before anything serves:

    * the staged build was fitted from a spec that is no longer the slot's
      configuration -- i.e. a refit raced a configuration change;
    * the booster on disk disagrees with its spec about the feature count, so
      the serving path would hand it the wrong matrix.
    """
    from xgboost import XGBRegressor

    staged = read_channel(
        s3_client=s3_client, bucket=bucket, slot=slot, channel=Channel.STAGING
    )
    if not isinstance(staged, BuildPointer):
        raise SlotNotPromotedError(f"{slot.describe()} has no staged build.")

    resolved = resolve_build(
        s3_client=s3_client, bucket=bucket, slot=slot, fit_id=staged.fit_id
    )
    config_spec = resolve_config_spec(s3_client=s3_client, bucket=bucket, slot=slot)
    if resolved.spec_id != config_spec.spec_id:
        raise BuildPromotionError(
            f"{slot.describe()}: staged build {resolved.fit_id} was fitted from "
            f"spec {resolved.spec_id}, but the slot's configuration is now "
            f"{config_spec.spec_id}. Refit before promoting, or the model that "
            "ships will not be the one the configuration describes."
        )

    model = XGBRegressor()
    model.load_model(bytearray(resolved.model_bytes))
    booster_features = model.get_booster().num_features()
    if booster_features != resolved.spec.features.n_features:
        raise BuildPromotionError(
            f"{slot.describe()}: staged booster expects {booster_features} "
            f"features but its spec declares "
            f"{resolved.spec.features.n_features}. Refusing to serve it."
        )
    return resolved.fit_id, resolved.spec


def promote_build(
    *,
    s3_client,
    bucket: str,
    slot: ModelSlot,
    dry_run: bool = True,
    set_by: str | None = None,
) -> PromotionOutcome:
    """Point production at the staged build, if there is a new one."""
    staged = read_channel(
        s3_client=s3_client, bucket=bucket, slot=slot, channel=Channel.STAGING
    )
    if not isinstance(staged, BuildPointer):
        return PromotionOutcome(
            slot=slot,
            promoted_fit_id=None,
            previous_fit_id=None,
            spec_id=None,
            skipped_reason="no staged build",
        )

    current = read_channel(
        s3_client=s3_client, bucket=bucket, slot=slot, channel=Channel.PRODUCTION
    )
    current_fit_id = current.fit_id if isinstance(current, BuildPointer) else None
    if current_fit_id == staged.fit_id:
        return PromotionOutcome(
            slot=slot,
            promoted_fit_id=None,
            previous_fit_id=current_fit_id,
            spec_id=None,
            skipped_reason="staged build is already in production",
        )

    fit_id, spec = verify_staged_build(s3_client=s3_client, bucket=bucket, slot=slot)

    if not dry_run:
        set_build_channel(
            s3_client=s3_client,
            bucket=bucket,
            slot=slot,
            channel=Channel.PRODUCTION,
            fit_id=fit_id,
            set_by=set_by,
            reason="promote-build",
        )
    return PromotionOutcome(
        slot=slot,
        promoted_fit_id=fit_id,
        previous_fit_id=current_fit_id,
        spec_id=spec.spec_id,
    )


def rollback_production(
    *,
    s3_client,
    bucket: str,
    slot: ModelSlot,
    fit_id: str | None = None,
    dry_run: bool = True,
    set_by: str | None = None,
) -> PromotionOutcome:
    """Point production at an earlier build.

    With no ``fit_id``, uses the pointer's own ``previous_fit_id`` -- the
    common case of undoing the last promotion. Any build still in the slot is a
    valid target; nothing was deleted to promote over it.
    """
    current = read_channel(
        s3_client=s3_client, bucket=bucket, slot=slot, channel=Channel.PRODUCTION
    )
    if not isinstance(current, BuildPointer):
        raise SlotNotPromotedError(
            f"{slot.describe()} has no production build to roll back."
        )

    target = fit_id or current.previous_fit_id
    if target is None:
        raise BuildPromotionError(
            f"{slot.describe()}: production build {current.fit_id} has no "
            "recorded predecessor. Name a build explicitly; "
            f"available: {list_fit_ids(s3_client=s3_client, bucket=bucket, slot=slot)}"
        )
    if target == current.fit_id:
        return PromotionOutcome(
            slot=slot,
            promoted_fit_id=None,
            previous_fit_id=current.fit_id,
            spec_id=None,
            skipped_reason="already serving that build",
        )

    # Resolve before repointing: a rollback target whose spec has gone missing
    # would take production down rather than restore it.
    resolved = resolve_build(
        s3_client=s3_client, bucket=bucket, slot=slot, fit_id=target
    )
    if not dry_run:
        set_build_channel(
            s3_client=s3_client,
            bucket=bucket,
            slot=slot,
            channel=Channel.PRODUCTION,
            fit_id=target,
            set_by=set_by,
            reason="rollback",
        )
    return PromotionOutcome(
        slot=slot,
        promoted_fit_id=target,
        previous_fit_id=current.fit_id,
        spec_id=resolved.spec_id,
    )


def describe_slot(*, s3_client, bucket: str, slot: ModelSlot) -> str:
    """One-line status, for the CLI and for eyeballing a slot."""
    lines = [slot.describe()]
    try:
        production = resolve_channel(
            s3_client=s3_client, bucket=bucket, slot=slot, channel=Channel.PRODUCTION
        )
        lines.append(
            f"  production : {production.fit_id} (spec {production.spec_id}) "
            f"{production.fit.model_name}"
        )
    except SlotNotPromotedError:
        lines.append("  production : -")

    staged = read_channel(
        s3_client=s3_client, bucket=bucket, slot=slot, channel=Channel.STAGING
    )
    lines.append(
        f"  staging    : {staged.fit_id}" if isinstance(staged, BuildPointer)
        else "  staging    : -"
    )

    config = read_channel(
        s3_client=s3_client, bucket=bucket, slot=slot, channel=Channel.CONFIG
    )
    lines.append(
        f"  config     : spec {config.spec_id}" if config is not None
        else "  config     : -"
    )
    return "\n".join(lines)
