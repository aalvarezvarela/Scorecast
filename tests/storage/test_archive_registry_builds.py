"""archive_registry_builds: what it may move, and what it must never touch."""

from datetime import UTC, datetime

import pytest
from nba_ou.modeling.registry_paths import Channel, ModelSlot, ModelTarget
from nba_ou.modeling.registry_store import (
    list_fit_ids,
    resolve_channel,
    set_build_channel,
    set_config_channel,
)

from scripts.archive_registry_builds import discover_slots, execute_moves, plan_slot
from tests.fake_s3 import FakeS3Client
from tests.test_model_registry_store import BUCKET, make_spec, publish

OLD = datetime(2026, 9, 24, tzinfo=UTC)
NEW = datetime(2026, 10, 5, 12, tzinfo=UTC)
CUTOFF = datetime(2026, 10, 5, tzinfo=UTC)


@pytest.fixture
def s3():
    return FakeS3Client()


@pytest.fixture
def slot():
    return ModelSlot(schema_version="2_5", target=ModelTarget.LINE_ERROR, horizon_minutes=60)


def promoted_twice(s3, slot):
    """The state after this campaign: old build superseded by a new one."""
    old_spec, new_spec = make_spec(max_na_per_row=500), make_spec(max_na_per_row=300)
    old = publish(s3, slot, old_spec, fitted_at=OLD, channel=Channel.PRODUCTION)
    set_build_channel(
        s3_client=s3, bucket=BUCKET, slot=slot, channel=Channel.STAGING, fit_id=old.fit_id
    )
    new = publish(s3, slot, new_spec, fitted_at=NEW, channel=Channel.STAGING)
    set_build_channel(
        s3_client=s3, bucket=BUCKET, slot=slot, channel=Channel.PRODUCTION, fit_id=new.fit_id
    )
    set_config_channel(s3_client=s3, bucket=BUCKET, slot=slot, spec_id=new_spec.spec_id)
    return old, new, old_spec, new_spec


def plan(s3, slot, **kwargs):
    kwargs.setdefault("include_rollback_target", False)
    return plan_slot(
        s3_client=s3, bucket=BUCKET, slot=slot, before=CUTOFF, stamp="20261105", **kwargs
    )


def test_the_rollback_target_is_kept_unless_asked(s3, slot):
    old, _, _, _ = promoted_twice(s3, slot)
    p = plan(s3, slot)
    assert p.fit_ids == [] and p.moves == []
    assert p.kept == [(old.fit_id, "production's rollback target")]


def test_old_build_and_its_spec_move_under_retired(s3, slot):
    old, new, old_spec, new_spec = promoted_twice(s3, slot)
    p = plan(s3, slot, include_rollback_target=True)
    assert p.fit_ids == [old.fit_id]
    assert p.spec_ids == [old_spec.spec_id]
    for source, target in p.moves:
        assert target == source.replace("models/", "models/retired/20261105/", 1)

    assert execute_moves(s3_client=s3, bucket=BUCKET, moves=p.moves, keep_source=False) == 0
    assert list_fit_ids(s3_client=s3, bucket=BUCKET, slot=slot) == [new.fit_id]
    # Production still resolves through its own fit to its own spec.
    served = resolve_channel(s3_client=s3, bucket=BUCKET, slot=slot, channel=Channel.PRODUCTION)
    assert served.fit.fit_id == new.fit_id and served.spec.spec_id == new_spec.spec_id
    assert any(k.startswith("models/retired/20261105/2_5/") for k in s3.store)


def test_a_build_a_channel_names_is_never_moved(s3, slot):
    spec = make_spec()
    fit = publish(s3, slot, spec, fitted_at=OLD, channel=Channel.PRODUCTION)
    p = plan(s3, slot, include_rollback_target=True)
    assert p.fit_ids == [] and p.spec_ids == []
    assert fit.fit_id in list_fit_ids(s3_client=s3, bucket=BUCKET, slot=slot)


def test_a_spec_still_named_by_config_stays(s3, slot):
    old, _, old_spec, _ = promoted_twice(s3, slot)
    set_config_channel(s3_client=s3, bucket=BUCKET, slot=slot, spec_id=old_spec.spec_id)
    p = plan(s3, slot, include_rollback_target=True)
    assert p.fit_ids == [old.fit_id]
    assert p.spec_ids == []


def test_builds_after_the_cutoff_are_left_alone(s3, slot):
    promoted_twice(s3, slot)
    p = plan_slot(
        s3_client=s3, bucket=BUCKET, slot=slot, before=OLD, stamp="20261105",
        include_rollback_target=True,
    )
    assert p.fit_ids == []


def test_nothing_is_deleted_when_a_copy_fails(s3, slot):
    promoted_twice(s3, slot)
    p = plan(s3, slot, include_rollback_target=True)
    before = dict(s3.store)
    moves = p.moves + [("models/does/not/exist.json", "models/retired/x.json")]
    assert execute_moves(s3_client=s3, bucket=BUCKET, moves=moves, keep_source=False) == 1
    assert all(key in s3.store for key in before)


def test_discovery_filters_by_target(s3, slot):
    promoted_twice(s3, slot)
    other = ModelSlot(schema_version="2_5", target=ModelTarget.TOTAL_POINTS, horizon_minutes=0)
    s3.put_object(Bucket=BUCKET, Key=other.spec_key("0" * 12), Body=b"{}")
    found = discover_slots(
        s3_client=s3, bucket=BUCKET, root="models/", schema_version="2_5",
        targets={"line_error"},
    )
    assert [s.describe() for s in found] == ["2_5/line_error/t0060/main"]
