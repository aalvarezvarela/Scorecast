"""The documents stored in a model slot: specs, fits and channel pointers.

Deliberately separate from ``nba_ou.modeling.modeling``, which owns the legacy
``ModelBundleMetadata`` and pulls in xgboost, sklearn and tqdm. Serving,
promotion and the retirement sweep all need these shapes; none of them should
have to import a training stack to read a pointer.

The split that matters:

* a **spec** is a configuration. It is chosen once by a human from an
  experiment, and it is what every subsequent refit reads. It holds the feature
  list, the cleaning thresholds and the hyperparameters.
* a **fit** is one training run of that spec against one dataset build. It
  holds only what changed: dates, counts, and which bytes it was trained on.

Both are immutable and content- or time-addressed. Storing ``feature_names``
once per spec rather than once per fit is what keeps a slot's history small:
the previous layout wrote a 1,366-name list into all 348 archived bundles.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal

from nba_ou.modeling.registry_paths import (
    ModelSlot,
    ModelTarget,
    format_horizon,
    validate_spec_id,
)
from pydantic import BaseModel, ConfigDict, Field

#: Length of the truncated sha256 used as a spec id. Twelve hex characters is
#: 48 bits -- ample for the number of configurations one repository will ever
#: promote, and short enough to read in a path.
SPEC_ID_LENGTH = 12


class _Strict(BaseModel):
    """Reject unknown keys on the way in.

    A spec that silently drops a field it does not recognise is a spec that
    trains on something other than what was written down.
    """

    model_config = ConfigDict(extra="forbid")


class SpecIdentity(_Strict):
    """What this model is. Must agree with the slot it is stored under."""

    schema_version: str
    target: ModelTarget
    prediction_strategy: str
    #: Minutes before tip-off; 0 is the close, ``None`` is a pooled model.
    horizon_minutes: int | None
    variant: str = "main"
    #: ``closing_line`` or ``intermediate_line``. NOT derivable from the
    #: horizon: both datasets can produce a model at T-0, and they are
    #: different models.
    dataset_type: str

    def slot(self, *, root: str | None = None) -> ModelSlot:
        kwargs: dict[str, Any] = {
            "schema_version": self.schema_version,
            "target": self.target,
            "horizon_minutes": self.horizon_minutes,
            "variant": self.variant,
        }
        if root is not None:
            kwargs["root"] = root
        return ModelSlot(**kwargs)


class SpecFeatures(_Strict):
    """The feature matrix this model expects, in order.

    Order is part of the contract: XGBoost is handed a frame, not a mapping, so
    a reordered list is a different model. It is therefore hashed as given and
    never sorted.
    """

    feature_names: list[str]
    n_features: int

    def check(self) -> None:
        if self.n_features != len(self.feature_names):
            raise ValueError(
                f"n_features={self.n_features} but {len(self.feature_names)} "
                "names were given."
            )
        if not self.feature_names:
            raise ValueError("A spec must name at least one feature.")
        duplicates = {
            name for name in self.feature_names if self.feature_names.count(name) > 1
        }
        if duplicates:
            raise ValueError(f"Duplicate feature names: {sorted(duplicates)}")


class SpecCleaning(_Strict):
    nan_threshold: float
    max_na_per_row: int
    corr_threshold: float | None = None
    corr_threshold_overrides: dict[str, float] = Field(default_factory=dict)


class SpecTraining(_Strict):
    """Everything needed to refit, and nothing that changes when you do."""

    params: dict[str, Any]
    n_estimators: int
    sample_weight_lambda: float | None = None
    #: Counted in GAMES, not rows. On a pooled intermediate frame one game is
    #: several snapshot rows, so a row count multiplies the intended window.
    train_games: int | None = None
    refit_strategy: str = "rolling_window"
    required_line_col: str | None = None
    minimum_line_value: float | None = None


class SpecProvenance(_Strict):
    """Which experiment vouched for this configuration.

    Excluded from ``spec_id`` on purpose: re-promoting the same configuration
    from a different run must produce the same id, or the registry would fill
    with specs that differ only in who wrote them.
    """

    experiment_name: str | None = None
    comparison_group: str | None = None
    source_run: str | None = None
    training_version: str | None = None
    selected_trial_number: int | None = None
    #: Which concrete CSV generation the hyperparameters were tuned on, and its
    #: exact bytes. The schema version alone does not identify a dataset: see
    #: ``docs/model_registry_reorg_plan.md`` section 2.5.
    source_dataset_build_id: str | None = None
    source_dataset_checksum: str | None = None
    vouching_metrics: dict[str, float | None] = Field(default_factory=dict)
    promoted_at: datetime | None = None
    promoted_by: str | None = None


class ModelSpec(_Strict):
    """A promoted configuration. Immutable once written."""

    spec_id: str
    identity: SpecIdentity
    features: SpecFeatures
    cleaning: SpecCleaning
    training: SpecTraining
    provenance: SpecProvenance = Field(default_factory=SpecProvenance)

    def canonical_body(self) -> dict[str, Any]:
        """The part of the spec that defines it.

        ``spec_id`` is excluded because it is the output, and ``provenance``
        because it describes how the configuration was arrived at rather than
        what it is.
        """
        return self.model_dump(
            mode="json", exclude={"spec_id", "provenance"}, exclude_none=False
        )

    def compute_spec_id(self) -> str:
        return compute_spec_id(self.canonical_body())

    def check(self) -> None:
        """Validate what pydantic cannot: internal agreement."""
        self.features.check()
        validate_spec_id(self.spec_id)
        expected = self.compute_spec_id()
        if expected != self.spec_id:
            raise ValueError(
                f"spec_id {self.spec_id!r} does not match its contents "
                f"(recomputed {expected!r}). The document was edited after it "
                "was written; a spec is immutable by contract."
            )
        # A horizon that does not render is a slot that cannot be addressed.
        format_horizon(self.identity.horizon_minutes)

    def check_matches_slot(self, slot: ModelSlot) -> None:
        """Refuse a spec stored under a slot it does not describe."""
        actual = self.identity.slot(root=slot.root)
        if actual != slot:
            raise ValueError(
                f"Spec {self.spec_id} describes {actual.describe()} but is "
                f"stored under {slot.describe()}."
            )


def compute_spec_id(body: dict[str, Any]) -> str:
    """Content address for a spec body.

    ``sort_keys`` normalises mapping order so that two logically identical
    specs hash alike; list order is left untouched because ``feature_names``
    order is part of the model.
    """
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:SPEC_ID_LENGTH]


def finalize_spec(
    *,
    identity: SpecIdentity,
    features: SpecFeatures,
    cleaning: SpecCleaning,
    training: SpecTraining,
    provenance: SpecProvenance | None = None,
) -> ModelSpec:
    """Build a spec and stamp it with its own content address."""
    features.check()
    draft = ModelSpec(
        spec_id="0" * SPEC_ID_LENGTH,
        identity=identity,
        features=features,
        cleaning=cleaning,
        training=training,
        provenance=provenance or SpecProvenance(),
    )
    spec = draft.model_copy(update={"spec_id": draft.compute_spec_id()})
    spec.check()
    return spec


class FitRecord(_Strict):
    """One training run of one spec. Immutable once written."""

    fit_id: str
    spec_id: str
    #: Human-facing name, and the predictions table's ``model_name``. Carries
    #: the horizon because that table upserts on it: two horizons refit to the
    #: same date would otherwise collide.
    model_name: str
    train_date_min: datetime
    train_date_max: datetime
    n_train_games: int
    n_features: int
    #: Which dataset generation this was trained on, and its exact bytes. The
    #: only way a silent semantic change under an unchanged schema_version
    #: becomes visible after the fact -- the feature names still resolve.
    dataset_build_id: str | None = None
    dataset_checksum: str | None = None
    fitted_at: datetime
    fitted_by: str | None = None

    def check_against(self, spec: ModelSpec) -> None:
        if self.spec_id != spec.spec_id:
            raise ValueError(
                f"Fit {self.fit_id} was produced by spec {self.spec_id}, not "
                f"{spec.spec_id}."
            )
        if self.n_features != spec.features.n_features:
            raise ValueError(
                f"Fit {self.fit_id} has {self.n_features} features but its "
                f"spec declares {spec.features.n_features}."
            )


def build_model_name(
    *,
    target: ModelTarget,
    horizon_minutes: int | None,
    variant: str,
    schema_version: str,
    train_date_max: datetime,
) -> str:
    """``line_error_t0060_main_2_5_21_09_26``.

    Version-then-date, matching the training CSV convention. The date is always
    the trailing eight characters, so it parses unambiguously.

    ``train_date_max`` is the training data's last date, not today's: a model
    refitted on a stale snapshot must not look freshly trained.
    """
    return (
        f"{ModelTarget(target).value}"
        f"_{format_horizon(horizon_minutes)}"
        f"_{variant}"
        f"_{schema_version}"
        f"_{train_date_max.strftime('%d_%m_%y')}"
    )


class ConfigPointer(_Strict):
    """``channels/config.json``: the spec the NEXT refit must use.

    Never read by serving. Repointing it is how a new configuration is adopted,
    and it must not disturb what is currently in production -- that is the
    whole reason a served model resolves its spec through its own fit.
    """

    kind: Literal["config"] = "config"
    spec_id: str
    set_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    set_by: str | None = None


class BuildPointer(_Strict):
    """``channels/production.json`` or ``channels/staging.json``."""

    kind: Literal["build"] = "build"
    fit_id: str
    #: What this channel pointed at before. Makes a rollback target readable
    #: without listing the history.
    previous_fit_id: str | None = None
    set_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    set_by: str | None = None


class HistoryEntry(_Strict):
    """One production flip, written as its own object.

    An append-only log of small objects rather than one growing file: S3 has no
    append, and read-modify-write on a shared log loses entries.
    """

    channel: str
    from_fit_id: str | None
    to_fit_id: str
    spec_id: str
    at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    by: str | None = None
    reason: str | None = None
