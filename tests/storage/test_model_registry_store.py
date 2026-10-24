"""Behaviour of the S3 model registry: immutability, pointers, resolution.

The test that matters most is
``test_adopting_a_new_configuration_does_not_disturb_production``: the previous
design took production down whenever a new configuration was promoted, because
serving validated the running model against the *current* config rather than
against the spec its own fit named.
"""

from datetime import UTC, datetime, timedelta

import pytest
from nba_ou.modeling.registry_models import (
    FitRecord,
    SpecCleaning,
    SpecFeatures,
    SpecIdentity,
    SpecTraining,
    build_model_name,
    finalize_spec,
)
from nba_ou.modeling.registry_paths import (
    Channel,
    ModelSlot,
    ModelTarget,
    build_fit_id,
)
from nba_ou.modeling.registry_store import (
    ImmutableObjectError,
    RegistryError,
    SlotNotPromotedError,
    list_fit_ids,
    read_channel,
    read_history,
    resolve_channel,
    resolve_config_spec,
    set_build_channel,
    set_config_channel,
    unreferenced_fit_ids,
    write_build,
    write_spec,
)

from tests.fake_s3 import FakeS3Client

BUCKET = "test-registry"


@pytest.fixture
def s3():
    return FakeS3Client()


@pytest.fixture
def slot():
    return ModelSlot(
        schema_version="2_5",
        target=ModelTarget.LINE_ERROR,
        horizon_minutes=60,
    )


def make_spec(*, max_na_per_row: int = 500, features=("A", "B", "C")):
    return finalize_spec(
        identity=SpecIdentity(
            schema_version="2_5",
            target=ModelTarget.LINE_ERROR,
            prediction_strategy="line_error_regressor",
            horizon_minutes=60,
            variant="main",
            dataset_type="intermediate_line",
        ),
        features=SpecFeatures(feature_names=list(features), n_features=len(features)),
        cleaning=SpecCleaning(nan_threshold=5.0, max_na_per_row=max_na_per_row),
        training=SpecTraining(params={"max_depth": 4}, n_estimators=82),
    )


def make_fit(spec, *, fitted_at: datetime, n_features: int | None = None):
    return FitRecord(
        fit_id=build_fit_id(spec.spec_id, fitted_at=fitted_at),
        spec_id=spec.spec_id,
        model_name=build_model_name(
            target=ModelTarget.LINE_ERROR,
            horizon_minutes=60,
            variant="main",
            schema_version="2_5",
            train_date_max=fitted_at,
        ),
        train_date_min=fitted_at - timedelta(days=900),
        train_date_max=fitted_at,
        n_train_games=3500,
        n_features=n_features or spec.features.n_features,
        fitted_at=fitted_at,
    )


def publish(s3, slot, spec, *, fitted_at, model_bytes=b"booster", channel=None):
    """Write a spec and a build, optionally pointing a channel at it."""
    write_spec(s3_client=s3, bucket=BUCKET, slot=slot, spec=spec)
    fit = make_fit(spec, fitted_at=fitted_at)
    write_build(
        s3_client=s3,
        bucket=BUCKET,
        slot=slot,
        fit=fit,
        model_bytes=model_bytes,
        spec=spec,
    )
    if channel is not None:
        set_build_channel(
            s3_client=s3, bucket=BUCKET, slot=slot, channel=channel, fit_id=fit.fit_id
        )
    return fit


# --- immutability ----------------------------------------------------------


def test_writing_an_identical_spec_twice_is_a_no_op(s3, slot):
    spec = make_spec()
    write_spec(s3_client=s3, bucket=BUCKET, slot=slot, spec=spec)
    write_spec(s3_client=s3, bucket=BUCKET, slot=slot, spec=spec)
    assert s3.put_order.count(slot.spec_key(spec.spec_id)) == 1


def test_a_build_cannot_be_overwritten_with_different_bytes(s3, slot):
    spec = make_spec()
    fit = publish(s3, slot, spec, fitted_at=datetime(2026, 9, 21, tzinfo=UTC))
    with pytest.raises(ImmutableObjectError, match="immutable"):
        write_build(
            s3_client=s3,
            bucket=BUCKET,
            slot=slot,
            fit=fit,
            model_bytes=b"a different booster",
            spec=spec,
        )


# --- write ordering --------------------------------------------------------


def test_a_build_is_fully_written_before_any_pointer_names_it(s3, slot):
    """A reader must never catch a model beside the wrong metadata."""
    spec = make_spec()
    fit = publish(
        s3,
        slot,
        spec,
        fitted_at=datetime(2026, 9, 21, tzinfo=UTC),
        channel=Channel.PRODUCTION,
    )
    pointer_index = s3.put_order.index(slot.channel_key(Channel.PRODUCTION))
    for key in (slot.model_key(fit.fit_id), slot.fit_key(fit.fit_id)):
        assert s3.put_order.index(key) < pointer_index


def test_a_channel_cannot_point_at_a_build_that_does_not_exist(s3, slot):
    with pytest.raises(RegistryError):
        set_build_channel(
            s3_client=s3,
            bucket=BUCKET,
            slot=slot,
            channel=Channel.PRODUCTION,
            fit_id="20260921T154203Z-deadbeef",
        )


def test_config_channel_cannot_point_at_a_missing_spec(s3, slot):
    with pytest.raises(RegistryError):
        set_config_channel(
            s3_client=s3, bucket=BUCKET, slot=slot, spec_id="a3f9c21d4b0e"
        )


# --- resolution ------------------------------------------------------------


def test_production_resolves_through_the_fit_to_its_own_spec(s3, slot):
    spec = make_spec()
    fit = publish(
        s3,
        slot,
        spec,
        fitted_at=datetime(2026, 9, 21, tzinfo=UTC),
        channel=Channel.PRODUCTION,
    )
    resolved = resolve_channel(s3_client=s3, bucket=BUCKET, slot=slot)
    assert resolved.fit_id == fit.fit_id
    assert resolved.spec_id == spec.spec_id
    assert resolved.model_bytes == b"booster"


def test_adopting_a_new_configuration_does_not_disturb_production(s3, slot):
    """The bug the pointer layout exists to remove.

    promote-config writes a new spec, repoints the config channel and stages a
    build. Production is still serving the old build, and must keep resolving
    cleanly against the OLD spec until promote-build says otherwise.
    """
    old_spec = make_spec(max_na_per_row=500)
    old_fit = publish(
        s3,
        slot,
        old_spec,
        fitted_at=datetime(2026, 9, 1, tzinfo=UTC),
        channel=Channel.PRODUCTION,
    )
    set_config_channel(
        s3_client=s3, bucket=BUCKET, slot=slot, spec_id=old_spec.spec_id
    )

    new_spec = make_spec(max_na_per_row=80, features=("A", "B", "C", "D"))
    assert new_spec.spec_id != old_spec.spec_id
    publish(
        s3,
        slot,
        new_spec,
        fitted_at=datetime(2026, 9, 21, tzinfo=UTC),
        channel=Channel.STAGING,
    )
    set_config_channel(
        s3_client=s3, bucket=BUCKET, slot=slot, spec_id=new_spec.spec_id
    )

    served = resolve_channel(s3_client=s3, bucket=BUCKET, slot=slot)
    assert served.fit_id == old_fit.fit_id
    assert served.spec_id == old_spec.spec_id
    assert served.spec.features.feature_names == ["A", "B", "C"]
    # And the next refit picks up the new configuration.
    assert (
        resolve_config_spec(s3_client=s3, bucket=BUCKET, slot=slot).spec_id
        == new_spec.spec_id
    )


def test_an_unpromoted_slot_says_so(s3, slot):
    with pytest.raises(SlotNotPromotedError, match="no production build"):
        resolve_channel(s3_client=s3, bucket=BUCKET, slot=slot)
    with pytest.raises(SlotNotPromotedError, match="no config channel"):
        resolve_config_spec(s3_client=s3, bucket=BUCKET, slot=slot)


def test_a_fit_whose_feature_count_contradicts_its_spec_is_rejected(s3, slot):
    spec = make_spec()
    write_spec(s3_client=s3, bucket=BUCKET, slot=slot, spec=spec)
    fit = make_fit(spec, fitted_at=datetime(2026, 9, 21, tzinfo=UTC), n_features=99)
    write_build(
        s3_client=s3, bucket=BUCKET, slot=slot, fit=fit, model_bytes=b"booster"
    )
    set_build_channel(
        s3_client=s3,
        bucket=BUCKET,
        slot=slot,
        channel=Channel.PRODUCTION,
        fit_id=fit.fit_id,
    )
    with pytest.raises(ValueError, match="features"):
        resolve_channel(s3_client=s3, bucket=BUCKET, slot=slot)


# --- promotion and rollback ------------------------------------------------


def test_promotion_writes_one_pointer_and_copies_nothing(s3, slot):
    spec = make_spec()
    publish(s3, slot, spec, fitted_at=datetime(2026, 9, 1, tzinfo=UTC))
    newer = publish(s3, slot, spec, fitted_at=datetime(2026, 9, 21, tzinfo=UTC))

    before = len(s3.put_order)
    set_build_channel(
        s3_client=s3,
        bucket=BUCKET,
        slot=slot,
        channel=Channel.PRODUCTION,
        fit_id=newer.fit_id,
    )
    # One pointer plus one history entry; no model bytes moved.
    assert len(s3.put_order) - before == 2


def test_rollback_is_the_same_flip_with_an_earlier_build(s3, slot):
    spec = make_spec()
    first = publish(s3, slot, spec, fitted_at=datetime(2026, 9, 1, tzinfo=UTC))
    second = publish(s3, slot, spec, fitted_at=datetime(2026, 9, 21, tzinfo=UTC))
    for fit_id in (first.fit_id, second.fit_id, first.fit_id):
        set_build_channel(
            s3_client=s3,
            bucket=BUCKET,
            slot=slot,
            channel=Channel.PRODUCTION,
            fit_id=fit_id,
        )

    pointer = read_channel(
        s3_client=s3, bucket=BUCKET, slot=slot, channel=Channel.PRODUCTION
    )
    assert pointer.fit_id == first.fit_id
    assert pointer.previous_fit_id == second.fit_id
    assert resolve_channel(s3_client=s3, bucket=BUCKET, slot=slot).fit_id == first.fit_id


def test_history_records_every_flip_newest_first(s3, slot):
    spec = make_spec()
    first = publish(s3, slot, spec, fitted_at=datetime(2026, 9, 1, tzinfo=UTC))
    second = publish(s3, slot, spec, fitted_at=datetime(2026, 9, 21, tzinfo=UTC))
    for fit_id in (first.fit_id, second.fit_id):
        set_build_channel(
            s3_client=s3,
            bucket=BUCKET,
            slot=slot,
            channel=Channel.PRODUCTION,
            fit_id=fit_id,
        )

    entries = read_history(s3_client=s3, bucket=BUCKET, slot=slot)
    assert [entry.to_fit_id for entry in entries] == [second.fit_id, first.fit_id]
    assert entries[0].from_fit_id == first.fit_id
    assert entries[-1].from_fit_id is None


# --- listing and retention -------------------------------------------------


def test_builds_list_chronologically_without_consulting_last_modified(s3, slot):
    spec = make_spec()
    late = publish(s3, slot, spec, fitted_at=datetime(2026, 9, 21, tzinfo=UTC))
    early = publish(s3, slot, spec, fitted_at=datetime(2026, 9, 1, tzinfo=UTC))
    assert list_fit_ids(s3_client=s3, bucket=BUCKET, slot=slot) == [
        early.fit_id,
        late.fit_id,
    ]


def test_retention_never_proposes_a_referenced_build(s3, slot):
    spec = make_spec()
    fits = [
        publish(s3, slot, spec, fitted_at=datetime(2026, 9, day, tzinfo=UTC))
        for day in range(1, 8)
    ]
    set_build_channel(
        s3_client=s3,
        bucket=BUCKET,
        slot=slot,
        channel=Channel.PRODUCTION,
        fit_id=fits[0].fit_id,
    )
    deletable = unreferenced_fit_ids(
        s3_client=s3, bucket=BUCKET, slot=slot, keep_recent=2
    )
    assert fits[0].fit_id not in deletable
    assert fits[-1].fit_id not in deletable
    assert fits[1].fit_id in deletable
