"""Turn an experiment run into a registry spec.

``promote.py`` answers "refit this run's hyperparameters on fresher data". This
module answers the question that comes immediately after: *which slot does the
result belong to, and what does its configuration say*.

Everything here is derived from the run's own saved config, so a promotion
cannot describe a model differently from the experiment that justified it. The
one thing deliberately NOT taken from the current checkout is the schema
version -- see :func:`parse_schema_version`.
"""

from __future__ import annotations

import re
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nba_ou.data_processing.line_history.snapshots import DEFAULT_SNAPSHOT_GRID
from nba_ou.modeling.registry_models import (
    ModelSpec,
    SpecCleaning,
    SpecFeatures,
    SpecIdentity,
    SpecProvenance,
    SpecTraining,
    finalize_spec,
)
from nba_ou.modeling.registry_paths import (
    ModelSlot,
    ModelTarget,
    SlotPathError,
    parse_horizon,
)

from training_pipeline.config import DatasetType, ExperimentConfig, TargetFamily
from training_pipeline.reuse import RunHyperparameters

#: ``training_data_2_5_20260704.csv`` / ``intermediate_line_data_2_5_20260613.csv``
_SCHEMA_IN_FILENAME = re.compile(r"_(\d+_\d+)_\d{8}")

#: TargetFamily members that can be served. OVER_UNDER is absent because the
#: serving path reads ``model.predict()`` as a continuous value; promote.py
#: refuses classifier runs for the same reason.
_PROMOTABLE_TARGETS = {
    TargetFamily.LINE_ERROR: ModelTarget.LINE_ERROR,
    TargetFamily.TOTAL_POINTS: ModelTarget.TOTAL_POINTS,
    TargetFamily.SPREAD_ERROR: ModelTarget.SPREAD_ERROR,
}


class PromotionError(ValueError):
    """A run cannot be turned into a registry spec."""


def parse_schema_version(csv_path: str | Path) -> str:
    """Read the training-data schema version out of a dataset filename.

    Taken from the DATA, never from ``TRAINING_DATA_SCHEMA_VERSION``: a run
    trained on ``training_data_2_3_20260909.csv`` is a 2_3 model even when the
    checkout has moved on to 2_5. Promoting an older build deliberately is
    legitimate, and the per-row schema version in ``ENABLED_MODELS`` is what
    lets the result be enabled alongside newer slots.
    """
    match = _SCHEMA_IN_FILENAME.search(Path(csv_path).name)
    if match is None:
        raise PromotionError(
            f"Cannot read a schema version from {Path(csv_path).name!r}. "
            "Expected a name like 'training_data_2_5_20260704.csv'. Pass "
            "--schema-version to state it explicitly."
        )
    return match.group(1)


def dataset_build_id(csv_path: str | Path) -> str:
    """The concrete generation of a dataset: its filename without extension.

    Distinct from the schema version on purpose. Columns have been added and
    semantics changed under an unchanged label -- see the 2_5 entry in
    ``nba_ou.config.dataset_versions`` -- so "same schema version" does not
    mean "same columns". This plus the checksum is what makes such a change
    visible after the fact.
    """
    return Path(csv_path).stem


def resolve_target(config: ExperimentConfig) -> ModelTarget:
    family = config.target_family
    if family not in _PROMOTABLE_TARGETS:
        raise PromotionError(
            f"target_family={family.value!r} has no registry slot. Only "
            f"{', '.join(sorted(t.value for t in ModelTarget))} can be served."
        )
    return _PROMOTABLE_TARGETS[family]


def resolve_horizon_minutes(config: ExperimentConfig) -> int | None:
    """Minutes before tip-off this model is for.

    A closing-line run has no ``snapshot_minutes`` and is a T-0 model. An
    intermediate run without one is pooled across every snapshot, which is the
    reserved ``tpool`` slot.
    """
    if config.data.dataset_type is DatasetType.CLOSING_LINE:
        return 0
    return config.data.snapshot_minutes


def check_horizon_is_buildable(horizon_minutes: int | None) -> None:
    """Warn when the daily dataset build would not sample this horizon.

    The snapshot grid is frozen into the dataset at build time: removing a
    horizon is a filter, but adding one back means regenerating the whole file
    (``nba_ou.data_processing.line_history.snapshots``). So the promotable
    horizons are bounded by whatever grid the daily build uses.

    ``DEFAULT_SNAPSHOT_GRID`` now covers all 17 horizons the schema-2.5
    intermediate dataset carries, so every horizon a campaign has trained at is
    promotable and this no longer fires on ordinary work. It previously did:
    the default stopped at 720 while campaigns trained at 420, 540, 600, 660,
    840, 960 and 1080.

    Still a warning rather than an error, and for one reason only -- the daily
    intermediate build does not exist yet (plan section 5, work item 10), so
    the grid is a declared contract rather than an observed property of a
    running job. When that build lands it should report the grid it actually
    used, and this should check against THAT and raise.
    """
    if horizon_minutes is None or horizon_minutes in DEFAULT_SNAPSHOT_GRID:
        return
    warnings.warn(
        f"Horizon T-{horizon_minutes} is not in the committed default snapshot "
        f"grid {tuple(DEFAULT_SNAPSHOT_GRID)}. The daily dataset build must be "
        "configured to sample it (--snapshot-grid), or this slot will never "
        "refit.",
        stacklevel=2,
    )


def slot_from_config(
    config: ExperimentConfig,
    *,
    variant: str = "main",
    schema_version: str | None = None,
    root: str | None = None,
) -> ModelSlot:
    """The slot a run's configuration implies."""
    kwargs: dict[str, Any] = {
        "schema_version": schema_version or parse_schema_version(config.data.csv_path),
        "target": resolve_target(config),
        "horizon_minutes": resolve_horizon_minutes(config),
        "variant": variant,
    }
    if root is not None:
        kwargs["root"] = root
    return ModelSlot(**kwargs)


def check_schema_version_against_checkout(schema_version: str) -> None:
    """Warn, do not fail, when promoting a build older than the checkout."""
    from nba_ou.config.dataset_versions import TRAINING_DATA_SCHEMA_VERSION

    if schema_version != TRAINING_DATA_SCHEMA_VERSION:
        warnings.warn(
            f"Promoting a {schema_version} model while this checkout builds "
            f"{TRAINING_DATA_SCHEMA_VERSION} datasets. That is allowed -- the "
            "slot records its own schema version -- but the daily refit needs "
            f"a {schema_version} dataset to keep feeding it.",
            stacklevel=2,
        )


def build_spec_from_run(
    config: ExperimentConfig,
    *,
    hyperparameters: RunHyperparameters,
    feature_names: list[str],
    slot: ModelSlot,
    source_run: Path | str | None = None,
    dataset_checksum: str | None = None,
    vouching_metrics: dict[str, float | None] | None = None,
    promoted_by: str | None = None,
) -> ModelSpec:
    """Assemble the configuration half of a promotion.

    ``feature_names`` comes from the fitted matrix rather than from the config,
    because correlation pruning is data-dependent: what the model was actually
    handed is the contract, not what was asked for.
    """
    identity = SpecIdentity(
        schema_version=slot.schema_version,
        target=slot.target,
        prediction_strategy=config.strategy.value,
        horizon_minutes=slot.horizon_minutes,
        variant=slot.variant,
        dataset_type=config.data.dataset_type.value,
    )

    # The selected trial may have tuned the window; the config value is only
    # the fallback used when no trial-level value was chosen.
    train_games = (
        hyperparameters.train_games
        if hyperparameters.train_games is not None
        else config.walk_forward.train_games
    )

    training = SpecTraining(
        params=dict(hyperparameters.params),
        n_estimators=hyperparameters.n_estimators,
        sample_weight_lambda=hyperparameters.sample_weight_lambda,
        train_games=train_games,
        refit_strategy=config.refit.strategy.value,
        required_line_col=config.line_col,
    )

    cleaning = SpecCleaning(
        nan_threshold=float(config.cleaning.nan_threshold),
        max_na_per_row=int(config.cleaning.max_na_per_row),
        corr_threshold=config.cleaning.corr_threshold,
        corr_threshold_overrides=dict(config.cleaning.corr_threshold_overrides or {}),
    )

    provenance = SpecProvenance(
        experiment_name=config.experiment_name,
        comparison_group=getattr(config, "comparison_group", None),
        source_run=str(source_run) if source_run is not None else None,
        training_version=config.training_version,
        selected_trial_number=hyperparameters.trial_number,
        source_dataset_build_id=dataset_build_id(config.data.csv_path),
        source_dataset_checksum=dataset_checksum,
        vouching_metrics=vouching_metrics or {},
        promoted_at=datetime.now(tz=UTC),
        promoted_by=promoted_by,
    )

    spec = finalize_spec(
        identity=identity,
        features=SpecFeatures(
            feature_names=list(feature_names), n_features=len(feature_names)
        ),
        cleaning=cleaning,
        training=training,
        provenance=provenance,
    )
    spec.check_matches_slot(slot)
    return spec


def enabled_models_row(slot: ModelSlot) -> str:
    """The config.ini line that turns this slot on."""
    horizon = "tpool" if slot.is_pooled else str(slot.horizon_minutes)
    return (
        f"    {slot.schema_version} | {slot.target.value} | "
        f"{horizon} | {slot.variant}"
    )


def parse_slot_argument(value: str, *, root: str | None = None) -> ModelSlot:
    """Parse ``--slot 2_5/line_error/t0060/main`` (variant optional)."""
    parts = [part for part in value.split("/") if part]
    if len(parts) == 3:
        parts.append("main")
    if len(parts) != 4:
        raise PromotionError(
            f"--slot must be 'schema_version/target/horizon/variant' "
            f"(variant optional), e.g. '2_5/line_error/t0060/main'. Got {value!r}."
        )
    schema_version, target, horizon_label, variant = parts
    kwargs: dict[str, Any] = {
        "schema_version": schema_version,
        "target": target,
        "horizon_minutes": parse_horizon(horizon_label),
        "variant": variant,
    }
    if root is not None:
        kwargs["root"] = root
    try:
        return ModelSlot(**kwargs)
    except SlotPathError as exc:
        raise PromotionError(str(exc)) from exc
