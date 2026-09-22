"""Reading and writing a model slot in S3.

Two rules the rest of the system depends on, enforced here rather than by
convention:

1. **Specs and builds are immutable.** Writing one that already exists with
   different bytes raises. Writing an identical one is a no-op, so a repeated
   promotion is safe.
2. **A pointer is never set before what it points at exists.** ``write_build``
   uploads the model and the fit and returns; only a separate ``set_channel``
   makes it visible. A reader can therefore never catch a build half-written.

Resolution order for a served model (see ``docs/model_registry_reorg_plan.md``
section 2.3)::

    channels/production.json   -> fit_id
    builds/{fit_id}/fit.json   -> spec_id
    specs/{spec_id}.json       -> features, cleaning, identity
    builds/{fit_id}/model.json -> the booster

The spec a served model is interpreted by is the one its own fit names, never
``channels/config.json``. That is what lets a new configuration be promoted
without disturbing production.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from botocore.exceptions import ClientError
from nba_ou.modeling.registry_models import (
    BuildPointer,
    ConfigPointer,
    FitRecord,
    HistoryEntry,
    ModelSpec,
)
from nba_ou.modeling.registry_paths import (
    Channel,
    ModelSlot,
    validate_fit_id,
    validate_spec_id,
)
from nba_ou.utils.s3_models import (
    list_s3_objects,
    read_s3_object_bytes,
    upload_bytes_to_s3,
)

#: Object keys that S3 reports as absent rather than raising.
_MISSING_CODES = {"NoSuchKey", "404", "NotFound"}


class RegistryError(RuntimeError):
    """Something is wrong with a slot's contents."""


class SlotNotPromotedError(RegistryError):
    """A slot has no build on the requested channel yet."""


class ImmutableObjectError(RegistryError):
    """An immutable object was written twice with different contents."""


def _read_bytes_or_none(*, s3_client, bucket: str, key: str) -> bytes | None:
    try:
        return read_s3_object_bytes(s3_client=s3_client, bucket=bucket, key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in _MISSING_CODES:
            return None
        raise


def _write_immutable(
    *, s3_client, bucket: str, key: str, payload: bytes, what: str
) -> bool:
    """Write once. Returns True if it wrote, False if identical bytes existed.

    Single-writer by design (one promotion job, one retrain job), so
    check-then-write is sufficient; the check exists to catch a second
    promotion writing a *different* body under an id that is supposed to
    determine it.
    """
    existing = _read_bytes_or_none(s3_client=s3_client, bucket=bucket, key=key)
    if existing is None:
        upload_bytes_to_s3(s3_client=s3_client, bucket=bucket, key=key, data=payload)
        return True
    if existing != payload:
        raise ImmutableObjectError(
            f"{what} already exists at {key} with different contents. "
            "Specs and builds are immutable; a changed body means a changed id."
        )
    return False


def _dump(model) -> bytes:
    return model.model_dump_json(indent=2).encode("utf-8")


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


def write_spec(*, s3_client, bucket: str, slot: ModelSlot, spec: ModelSpec) -> str:
    """Store a spec under its own content address. Idempotent."""
    spec.check()
    spec.check_matches_slot(slot)
    key = slot.spec_key(spec.spec_id)
    _write_immutable(
        s3_client=s3_client,
        bucket=bucket,
        key=key,
        payload=_dump(spec),
        what=f"Spec {spec.spec_id}",
    )
    return key


def read_spec(*, s3_client, bucket: str, slot: ModelSlot, spec_id: str) -> ModelSpec:
    key = slot.spec_key(validate_spec_id(spec_id))
    raw = _read_bytes_or_none(s3_client=s3_client, bucket=bucket, key=key)
    if raw is None:
        raise RegistryError(
            f"Spec {spec_id} is missing from {slot.describe()} ({key}). A fit "
            "names it, so it should never have been removed."
        )
    spec = ModelSpec.model_validate_json(raw)
    spec.check()
    spec.check_matches_slot(slot)
    return spec


def list_spec_ids(*, s3_client, bucket: str, slot: ModelSlot) -> list[str]:
    objects = list_s3_objects(
        s3_client=s3_client, bucket=bucket, prefix=slot.specs_prefix
    )
    return sorted(
        key.rsplit("/", 1)[-1].removesuffix(".json")
        for key in (obj["Key"] for obj in objects)
        if key.endswith(".json")
    )


# ---------------------------------------------------------------------------
# Builds
# ---------------------------------------------------------------------------


def write_build(
    *,
    s3_client,
    bucket: str,
    slot: ModelSlot,
    fit: FitRecord,
    model_bytes: bytes,
    spec: ModelSpec | None = None,
) -> str:
    """Upload one build. Does NOT point any channel at it.

    The model goes up before the fit, and neither is referenced by a pointer
    until a later ``set_channel`` -- so a partially written build is invisible
    rather than servable.
    """
    if spec is not None:
        fit.check_against(spec)
    validate_fit_id(fit.fit_id)
    _write_immutable(
        s3_client=s3_client,
        bucket=bucket,
        key=slot.model_key(fit.fit_id),
        payload=model_bytes,
        what=f"Model for build {fit.fit_id}",
    )
    _write_immutable(
        s3_client=s3_client,
        bucket=bucket,
        key=slot.fit_key(fit.fit_id),
        payload=_dump(fit),
        what=f"Fit record {fit.fit_id}",
    )
    return slot.build_prefix(fit.fit_id)


def read_fit(*, s3_client, bucket: str, slot: ModelSlot, fit_id: str) -> FitRecord:
    key = slot.fit_key(validate_fit_id(fit_id))
    raw = _read_bytes_or_none(s3_client=s3_client, bucket=bucket, key=key)
    if raw is None:
        raise RegistryError(
            f"Build {fit_id} has no fit.json in {slot.describe()} ({key})."
        )
    return FitRecord.model_validate_json(raw)


def read_model_bytes(
    *, s3_client, bucket: str, slot: ModelSlot, fit_id: str
) -> bytes:
    key = slot.model_key(validate_fit_id(fit_id))
    raw = _read_bytes_or_none(s3_client=s3_client, bucket=bucket, key=key)
    if raw is None:
        raise RegistryError(
            f"Build {fit_id} has no model.json in {slot.describe()} ({key})."
        )
    return raw


def list_fit_ids(*, s3_client, bucket: str, slot: ModelSlot) -> list[str]:
    """Every build in the slot, oldest first.

    ``fit_id`` is timestamp-prefixed, so lexicographic order is chronological
    and no ``LastModified`` is consulted -- a copy would reshuffle that.
    """
    objects = list_s3_objects(
        s3_client=s3_client, bucket=bucket, prefix=slot.builds_prefix
    )
    return sorted(
        {
            obj["Key"][len(slot.builds_prefix):].split("/", 1)[0]
            for obj in objects
            if obj["Key"].endswith("/fit.json")
        }
    )


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


def read_channel(
    *, s3_client, bucket: str, slot: ModelSlot, channel: Channel | str
) -> ConfigPointer | BuildPointer | None:
    channel = Channel(channel)
    raw = _read_bytes_or_none(
        s3_client=s3_client, bucket=bucket, key=slot.channel_key(channel)
    )
    if raw is None:
        return None
    if channel is Channel.CONFIG:
        return ConfigPointer.model_validate_json(raw)
    return BuildPointer.model_validate_json(raw)


def set_config_channel(
    *, s3_client, bucket: str, slot: ModelSlot, spec_id: str, set_by: str | None = None
) -> ConfigPointer:
    """Point the slot's next refit at a spec.

    Deliberately has no effect on what is being served.
    """
    validate_spec_id(spec_id)
    # Fail before repointing rather than after: a config channel naming an
    # absent spec would break every future refit with a confusing error.
    read_spec(s3_client=s3_client, bucket=bucket, slot=slot, spec_id=spec_id)
    pointer = ConfigPointer(spec_id=spec_id, set_by=set_by)
    upload_bytes_to_s3(
        s3_client=s3_client,
        bucket=bucket,
        key=slot.channel_key(Channel.CONFIG),
        data=_dump(pointer),
    )
    return pointer


def set_build_channel(
    *,
    s3_client,
    bucket: str,
    slot: ModelSlot,
    channel: Channel | str,
    fit_id: str,
    set_by: str | None = None,
    reason: str | None = None,
    record_history: bool = True,
) -> BuildPointer:
    """Point a channel at an existing build. This is the atomic promotion.

    One ``PutObject`` of a few hundred bytes -- no megabytes are copied and
    nothing is deleted, so a rollback is the same call with an earlier
    ``fit_id``.
    """
    channel = Channel(channel)
    if channel is Channel.CONFIG:
        raise ValueError("Use set_config_channel for the config channel.")
    validate_fit_id(fit_id)
    fit = read_fit(s3_client=s3_client, bucket=bucket, slot=slot, fit_id=fit_id)

    current = read_channel(
        s3_client=s3_client, bucket=bucket, slot=slot, channel=channel
    )
    previous_fit_id = (
        current.fit_id if isinstance(current, BuildPointer) else None
    )
    pointer = BuildPointer(
        fit_id=fit_id, previous_fit_id=previous_fit_id, set_by=set_by
    )
    upload_bytes_to_s3(
        s3_client=s3_client,
        bucket=bucket,
        key=slot.channel_key(channel),
        data=_dump(pointer),
    )
    if record_history:
        entry = HistoryEntry(
            channel=channel.value,
            from_fit_id=previous_fit_id,
            to_fit_id=fit_id,
            spec_id=fit.spec_id,
            by=set_by,
            reason=reason,
        )
        upload_bytes_to_s3(
            s3_client=s3_client,
            bucket=bucket,
            key=slot.history_key(entry.at),
            data=_dump(entry),
        )
    return pointer


def read_history(
    *, s3_client, bucket: str, slot: ModelSlot, limit: int | None = None
) -> list[HistoryEntry]:
    """Every channel flip, newest first."""
    objects = list_s3_objects(
        s3_client=s3_client, bucket=bucket, prefix=slot.history_prefix
    )
    keys = sorted(
        (obj["Key"] for obj in objects if obj["Key"].endswith(".json")), reverse=True
    )
    if limit is not None:
        keys = keys[:limit]
    entries = []
    for key in keys:
        raw = _read_bytes_or_none(s3_client=s3_client, bucket=bucket, key=key)
        if raw is not None:
            entries.append(HistoryEntry.model_validate_json(raw))
    return entries


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedBuild:
    """A build and the spec it was produced by, read together."""

    slot: ModelSlot
    fit: FitRecord
    spec: ModelSpec
    model_bytes: bytes

    @property
    def fit_id(self) -> str:
        return self.fit.fit_id

    @property
    def spec_id(self) -> str:
        return self.spec.spec_id


def resolve_build(
    *, s3_client, bucket: str, slot: ModelSlot, fit_id: str
) -> ResolvedBuild:
    """Load one build together with the spec that produced it."""
    fit = read_fit(s3_client=s3_client, bucket=bucket, slot=slot, fit_id=fit_id)
    spec = read_spec(
        s3_client=s3_client, bucket=bucket, slot=slot, spec_id=fit.spec_id
    )
    fit.check_against(spec)
    model_bytes = read_model_bytes(
        s3_client=s3_client, bucket=bucket, slot=slot, fit_id=fit_id
    )
    return ResolvedBuild(slot=slot, fit=fit, spec=spec, model_bytes=model_bytes)


def resolve_channel(
    *,
    s3_client,
    bucket: str,
    slot: ModelSlot,
    channel: Channel | str = Channel.PRODUCTION,
) -> ResolvedBuild:
    """Load whatever a channel currently points at.

    Note what is NOT consulted: ``channels/config.json``. A served model is
    interpreted by the spec its own fit names, so adopting a new configuration
    cannot invalidate what is already in production.
    """
    channel = Channel(channel)
    pointer = read_channel(
        s3_client=s3_client, bucket=bucket, slot=slot, channel=channel
    )
    if pointer is None:
        raise SlotNotPromotedError(
            f"{slot.describe()} has no {channel.value} build yet. Promote a "
            "configuration to it first "
            "(python -m training_pipeline.promote <run_dir> --to-s3)."
        )
    if not isinstance(pointer, BuildPointer):
        raise RegistryError(
            f"{slot.describe()} channel {channel.value} is not a build pointer."
        )
    return resolve_build(
        s3_client=s3_client, bucket=bucket, slot=slot, fit_id=pointer.fit_id
    )


def resolve_config_spec(
    *, s3_client, bucket: str, slot: ModelSlot
) -> ModelSpec:
    """The spec the next refit must use."""
    pointer = read_channel(
        s3_client=s3_client, bucket=bucket, slot=slot, channel=Channel.CONFIG
    )
    if pointer is None:
        raise SlotNotPromotedError(
            f"{slot.describe()} has no config channel. Nothing has been "
            "promoted to this slot yet, so there is no configuration to refit."
        )
    if not isinstance(pointer, ConfigPointer):
        raise RegistryError(
            f"{slot.describe()} config channel is not a config pointer."
        )
    return read_spec(
        s3_client=s3_client, bucket=bucket, slot=slot, spec_id=pointer.spec_id
    )


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def unreferenced_fit_ids(
    *,
    s3_client,
    bucket: str,
    slot: ModelSlot,
    keep_recent: int = 5,
    keep_after: datetime | None = None,
) -> list[str]:
    """Builds that nothing points at and that retention no longer protects.

    Never returns a build that a channel names, whatever the cutoffs say.
    History entries are kept regardless -- the record of what served when is
    smaller and more useful than the booster it names.
    """
    all_fits = list_fit_ids(s3_client=s3_client, bucket=bucket, slot=slot)
    referenced = set()
    for channel in (Channel.PRODUCTION, Channel.STAGING):
        pointer = read_channel(
            s3_client=s3_client, bucket=bucket, slot=slot, channel=channel
        )
        if isinstance(pointer, BuildPointer):
            referenced.add(pointer.fit_id)

    protected = set(all_fits[-keep_recent:]) if keep_recent > 0 else set()
    cutoff_stamp = (
        keep_after.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ") if keep_after else None
    )
    return [
        fit_id
        for fit_id in all_fits
        if fit_id not in referenced
        and fit_id not in protected
        and (cutoff_stamp is None or fit_id.split("-", 1)[0] < cutoff_stamp)
    ]
