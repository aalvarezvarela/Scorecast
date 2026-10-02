"""Path and document invariants for the S3 model registry.

These are the rules the rest of the registry assumes silently: that a key
round-trips to the slot it came from, that a spec is determined by its
contents, and that horizons sort chronologically.
"""

from datetime import UTC, datetime

import pytest
from nba_ou.modeling.registry_models import (
    SpecCleaning,
    SpecFeatures,
    SpecIdentity,
    SpecProvenance,
    SpecTraining,
    build_model_name,
    finalize_spec,
)
from nba_ou.modeling.registry_paths import (
    Channel,
    ModelSlot,
    ModelTarget,
    SlotPathError,
    build_fit_id,
    format_horizon,
    parse_horizon,
)


def _slot(**overrides) -> ModelSlot:
    kwargs = {
        "schema_version": "2_5",
        "target": ModelTarget.LINE_ERROR,
        "horizon_minutes": 60,
        "variant": "main",
    }
    kwargs.update(overrides)
    return ModelSlot(**kwargs)


def _spec(**identity_overrides):
    identity_kwargs = {
        "schema_version": "2_5",
        "target": ModelTarget.LINE_ERROR,
        "prediction_strategy": "line_error_regressor",
        "horizon_minutes": 60,
        "variant": "main",
        "dataset_type": "intermediate_line",
    }
    identity_kwargs.update(identity_overrides)
    return finalize_spec(
        identity=SpecIdentity(**identity_kwargs),
        features=SpecFeatures(feature_names=["A", "B", "C"], n_features=3),
        cleaning=SpecCleaning(nan_threshold=5.0, max_na_per_row=500),
        training=SpecTraining(params={"max_depth": 4}, n_estimators=82),
    )


# --- paths -----------------------------------------------------------------


def test_slot_prefix_is_schema_target_horizon_variant():
    assert _slot().prefix == "models/2_5/line_error/t0060/main/"


def test_any_key_under_a_slot_parses_back_to_that_slot():
    slot = _slot()
    for key in (
        slot.prefix,
        slot.spec_key("a3f9c21d4b0e"),
        slot.model_key("20260921T154203Z-a3f9c21d"),
        slot.channel_key(Channel.PRODUCTION),
        slot.history_key(datetime(2026, 9, 21, tzinfo=UTC)),
    ):
        assert ModelSlot.parse(key) == slot, key


def test_horizons_sort_chronologically_as_strings():
    """The reason for zero-padding: listing a target reads in time order."""
    labels = [format_horizon(minutes) for minutes in (0, 60, 180, 720, 1080)]
    assert labels == sorted(labels)
    assert labels[0] == "t0000"


def test_pooled_horizon_round_trips():
    slot = _slot(horizon_minutes=None)
    assert slot.horizon_label == "tpool"
    assert slot.is_pooled
    assert ModelSlot.parse(slot.prefix).horizon_minutes is None
    assert parse_horizon("tpool") is None


def test_unknown_target_is_rejected_rather_than_defaulted():
    """The failure the old serving path had: anything unrecognised became
    line_error. Here it cannot be spelled at all."""
    with pytest.raises(SlotPathError, match="Unknown target"):
        _slot(target="over_under")
    with pytest.raises(SlotPathError, match="Unknown target"):
        ModelSlot.parse("models/2_5/margin/t0000/main/")


def test_malformed_identifiers_are_rejected():
    with pytest.raises(SlotPathError):
        _slot(schema_version="25")
    with pytest.raises(SlotPathError):
        _slot(variant="Main")
    with pytest.raises(SlotPathError):
        _slot(horizon_minutes=-30)
    with pytest.raises(SlotPathError):
        _slot(horizon_minutes=20000)


def test_fit_id_is_timestamp_then_spec_prefix():
    fit_id = build_fit_id("a3f9c21d4b0e", fitted_at=datetime(2026, 9, 21, 15, 42, 3, tzinfo=UTC))
    assert fit_id == "20260921T154203Z-a3f9c21d"


def test_fit_ids_sort_chronologically():
    early = build_fit_id("a3f9c21d4b0e", fitted_at=datetime(2026, 9, 1, tzinfo=UTC))
    late = build_fit_id("b0000000000c", fitted_at=datetime(2026, 9, 21, tzinfo=UTC))
    assert sorted([late, early]) == [early, late]


# --- spec identity ---------------------------------------------------------


def test_spec_id_is_determined_by_contents():
    assert _spec().spec_id == _spec().spec_id


def test_provenance_does_not_change_the_spec_id():
    """Re-promoting the same configuration from another run must not create a
    second spec that differs only in who wrote it."""
    base = _spec()
    with_provenance = finalize_spec(
        identity=base.identity,
        features=base.features,
        cleaning=base.cleaning,
        training=base.training,
        provenance=SpecProvenance(experiment_name="whatever", promoted_by="someone"),
    )
    assert with_provenance.spec_id == base.spec_id


def test_changing_the_configuration_changes_the_spec_id():
    other = finalize_spec(
        identity=_spec().identity,
        features=SpecFeatures(feature_names=["A", "B", "C"], n_features=3),
        cleaning=SpecCleaning(nan_threshold=5.0, max_na_per_row=400),
        training=SpecTraining(params={"max_depth": 4}, n_estimators=82),
    )
    assert other.spec_id != _spec().spec_id


def test_feature_order_is_part_of_the_spec_id():
    """XGBoost is handed a frame, not a mapping: a reordered list is a
    different model."""
    reordered = finalize_spec(
        identity=_spec().identity,
        features=SpecFeatures(feature_names=["C", "B", "A"], n_features=3),
        cleaning=_spec().cleaning,
        training=_spec().training,
    )
    assert reordered.spec_id != _spec().spec_id


def test_edited_spec_fails_its_own_check():
    spec = _spec()
    tampered = spec.model_copy(
        update={"training": SpecTraining(params={"max_depth": 9}, n_estimators=82)}
    )
    with pytest.raises(ValueError, match="does not match its contents"):
        tampered.check()


def test_spec_stored_under_the_wrong_slot_is_rejected():
    with pytest.raises(ValueError, match="stored under"):
        _spec().check_matches_slot(_slot(horizon_minutes=360))


def test_inconsistent_feature_count_is_rejected():
    with pytest.raises(ValueError, match="n_features"):
        SpecFeatures(feature_names=["A", "B"], n_features=3).check()


def test_duplicate_feature_names_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        SpecFeatures(feature_names=["A", "A"], n_features=2).check()


# --- model name ------------------------------------------------------------


def test_model_name_carries_the_horizon():
    """The predictions table upserts on model_name, so two horizons refit to
    the same date must not collide."""
    names = {
        build_model_name(
            target=ModelTarget.LINE_ERROR,
            horizon_minutes=minutes,
            variant="main",
            schema_version="2_5",
            train_date_max=datetime(2026, 9, 21, tzinfo=UTC),
        )
        for minutes in (0, 60, 360)
    }
    assert len(names) == 3
    assert "line_error_t0060_main_2_5_21_09_26" in names
