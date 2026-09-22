"""Where every model artefact lives in S3, and nothing else.

One slot is one ``(schema_version, target, horizon, variant)`` tuple, and it owns
a subtree:

    models/{schema_version}/{target}/{horizon}/{variant}/
        specs/{spec_id}.json            immutable
        builds/{fit_id}/model.json      immutable
        builds/{fit_id}/fit.json        immutable
        channels/config.json            -> spec_id the NEXT refit must use
        channels/staging.json           -> fit_id awaiting promotion
        channels/production.json        -> fit_id currently serving
        channels/history/{ts}.json      one object per production flip

Nothing written is ever rewritten. The only mutable objects are the three
channel pointers, and each is a single ``PutObject`` -- so a promotion is atomic
and a reader can never catch a model beside the wrong metadata.

See ``docs/model_registry_reorg_plan.md`` for why it is shaped this way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

#: Default root prefix inside the registry bucket. Matches ``S3.MODELS_PREFIX``.
DEFAULT_ROOT = "models/"

#: Horizon label for a model trained across every snapshot at once rather than
#: at one time-to-tip. Reserved, not yet produced by any campaign: see
#: ``scripts/create_train_data/slice_intermediate_snapshot.py``, which argues
#: that a pooled model is the one you actually bet with.
POOLED_HORIZON_LABEL = "tpool"

#: Minutes before tip-off. Four digits keeps lexicographic order chronological,
#: so listing a target reads in time order.
_HORIZON_DIGITS = 4
_MAX_HORIZON_MINUTES = 10**_HORIZON_DIGITS - 1

_SCHEMA_VERSION_RE = re.compile(r"^\d+_\d+$")
_VARIANT_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_HORIZON_RE = re.compile(rf"^t(\d{{{_HORIZON_DIGITS}}})$")
_SPEC_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_FIT_ID_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")


class ModelTarget(StrEnum):
    """What a model predicts.

    The string values are exactly ``training_pipeline.config.TargetFamily``'s,
    so a campaign config and a registry path use one word. Declared here rather
    than imported because ``nba_ou`` is the lower layer -- the serving path must
    not need the training pipeline to resolve a slot.

    ``TargetFamily.OVER_UNDER`` is deliberately absent: a classifier cannot be
    promoted (see ``training_pipeline.promote``), so it has no slot.
    """

    LINE_ERROR = "line_error"
    TOTAL_POINTS = "total_points"
    SPREAD_ERROR = "spread_error"


class Channel(StrEnum):
    """The three mutable pointers in a slot.

    ``CONFIG`` names the spec the next refit must use; it is never consulted by
    serving. ``PRODUCTION`` and ``STAGING`` name builds. Changing ``CONFIG``
    therefore cannot disturb what is currently being served -- a served model is
    always interpreted by the spec its own fit names.
    """

    CONFIG = "config"
    STAGING = "staging"
    PRODUCTION = "production"


class SlotPathError(ValueError):
    """A slot could not be built or parsed."""


def format_horizon(horizon_minutes: int | None) -> str:
    """Render a horizon as its path segment. ``None`` means pooled."""
    if horizon_minutes is None:
        return POOLED_HORIZON_LABEL
    if horizon_minutes < 0:
        raise SlotPathError(
            f"horizon_minutes is minutes BEFORE tip-off and cannot be negative: "
            f"{horizon_minutes}"
        )
    if horizon_minutes > _MAX_HORIZON_MINUTES:
        raise SlotPathError(
            f"horizon_minutes must fit in {_HORIZON_DIGITS} digits "
            f"(<= {_MAX_HORIZON_MINUTES}): {horizon_minutes}"
        )
    return f"t{horizon_minutes:0{_HORIZON_DIGITS}d}"


def parse_horizon(label: str) -> int | None:
    """Inverse of :func:`format_horizon`."""
    if label == POOLED_HORIZON_LABEL:
        return None
    match = _HORIZON_RE.match(label)
    if match is None:
        raise SlotPathError(
            f"Not a horizon segment: {label!r}. Expected "
            f"t + {_HORIZON_DIGITS} digits (e.g. 't0060') or "
            f"{POOLED_HORIZON_LABEL!r}."
        )
    return int(match.group(1))


def build_fit_id(spec_id: str, *, fitted_at: datetime | None = None) -> str:
    """``{UTC timestamp}-{spec_id[:8]}``: sortable, unique, names its spec.

    A build directory is identified by when it was fitted and from which
    configuration, which is everything you need to read a listing.
    """
    validate_spec_id(spec_id)
    moment = fitted_at or datetime.now(tz=UTC)
    stamp = moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{spec_id[:8]}"


def validate_spec_id(spec_id: str) -> str:
    if not _SPEC_ID_RE.match(spec_id):
        raise SlotPathError(
            f"spec_id must be 12 lowercase hex characters: {spec_id!r}"
        )
    return spec_id


def validate_fit_id(fit_id: str) -> str:
    if not _FIT_ID_RE.match(fit_id):
        raise SlotPathError(
            f"fit_id must look like '20260921T154203Z-a3f9c21d': {fit_id!r}"
        )
    return fit_id


@dataclass(frozen=True)
class ModelSlot:
    """One target at one horizon under one schema version.

    ``horizon_minutes`` is minutes BEFORE tip-off -- 0 is the close -- or
    ``None`` for a pooled model.
    """

    schema_version: str
    target: ModelTarget
    horizon_minutes: int | None
    variant: str = "main"
    root: str = DEFAULT_ROOT

    def __post_init__(self) -> None:
        if not _SCHEMA_VERSION_RE.match(self.schema_version):
            raise SlotPathError(
                f"schema_version must look like '2_5' (it is "
                f"TRAINING_DATA_SCHEMA_VERSION verbatim): {self.schema_version!r}"
            )
        # Accepts the enum or its value, and rejects anything else loudly. A
        # target that falls through to a default is the failure this whole
        # layout exists to prevent.
        try:
            object.__setattr__(self, "target", ModelTarget(self.target))
        except ValueError as exc:
            known = ", ".join(sorted(t.value for t in ModelTarget))
            raise SlotPathError(
                f"Unknown target {self.target!r}. Known targets: {known}. "
                "(over_under has no slot: a classifier cannot be promoted.)"
            ) from exc
        if not _VARIANT_RE.match(self.variant):
            raise SlotPathError(
                f"variant must be lowercase alphanumeric with underscores: "
                f"{self.variant!r}"
            )
        format_horizon(self.horizon_minutes)  # validates
        if self.root and not self.root.endswith("/"):
            object.__setattr__(self, "root", f"{self.root}/")

    # -- labels ------------------------------------------------------------

    @property
    def horizon_label(self) -> str:
        return format_horizon(self.horizon_minutes)

    @property
    def is_pooled(self) -> bool:
        return self.horizon_minutes is None

    def describe(self) -> str:
        """Short human form, e.g. ``2_5/line_error/t0060/main``."""
        return (
            f"{self.schema_version}/{self.target.value}/"
            f"{self.horizon_label}/{self.variant}"
        )

    # -- keys --------------------------------------------------------------

    @property
    def prefix(self) -> str:
        """The slot's subtree, with a trailing slash."""
        return f"{self.root}{self.describe()}/"

    def spec_key(self, spec_id: str) -> str:
        return f"{self.prefix}specs/{validate_spec_id(spec_id)}.json"

    @property
    def specs_prefix(self) -> str:
        return f"{self.prefix}specs/"

    def build_prefix(self, fit_id: str) -> str:
        return f"{self.prefix}builds/{validate_fit_id(fit_id)}/"

    def model_key(self, fit_id: str) -> str:
        return f"{self.build_prefix(fit_id)}model.json"

    def fit_key(self, fit_id: str) -> str:
        return f"{self.build_prefix(fit_id)}fit.json"

    @property
    def builds_prefix(self) -> str:
        return f"{self.prefix}builds/"

    def channel_key(self, channel: Channel | str) -> str:
        return f"{self.prefix}channels/{Channel(channel).value}.json"

    def history_key(self, moment: datetime | None = None) -> str:
        stamp = (moment or datetime.now(tz=UTC)).astimezone(UTC).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        return f"{self.prefix}channels/history/{stamp}.json"

    @property
    def history_prefix(self) -> str:
        return f"{self.prefix}channels/history/"

    # -- parsing -----------------------------------------------------------

    @classmethod
    def parse(cls, prefix: str, *, root: str = DEFAULT_ROOT) -> ModelSlot:
        """Round-trip :attr:`prefix`, or any key beneath a slot.

        Accepts the slot prefix itself and any object key under it, so a key
        read back from a listing can be attributed to its slot without
        bookkeeping.
        """
        root = f"{root}/" if root and not root.endswith("/") else root
        remainder = prefix[len(root):] if prefix.startswith(root) else prefix
        parts = [part for part in remainder.split("/") if part]
        if len(parts) < 4:
            raise SlotPathError(
                f"Not a slot prefix: {prefix!r}. Expected "
                f"{root}<schema_version>/<target>/<horizon>/<variant>/..."
            )
        schema_version, target, horizon_label, variant = parts[:4]
        try:
            parsed_target = ModelTarget(target)
        except ValueError as exc:
            known = ", ".join(sorted(t.value for t in ModelTarget))
            raise SlotPathError(
                f"Unknown target {target!r} in {prefix!r}. Known targets: {known}."
            ) from exc
        return cls(
            schema_version=schema_version,
            target=parsed_target,
            horizon_minutes=parse_horizon(horizon_label),
            variant=variant,
            root=root,
        )
