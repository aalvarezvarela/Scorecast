import json

import pandas as pd
import pytest
import requests
from nba_ou.fetch_data.referees import load_assignments as loader
from nba_ou.fetch_data.referees.assignment_store import (
    PREFIX,
    AssignmentStore,
    normalize_assignments,
)
from nba_ou.fetch_data.referees.fetch_refs_data import ASSIGNMENT_COLUMNS

from tests.fake_s3 import FakeS3Client

DAY = "2026-10-16"
NOW = pd.Timestamp("2026-10-16T13:05:22.123456Z")


def assignments():
    return pd.DataFrame(
        {
            "Game": ["Boston @ New York", "Toronto @ Houston"],
            "Crew Chief": ["Ann One (#1)", "Bob Two (#2)"],
            "Referee": ["Cy Three (#3)", "Dee Four (#4)"],
            "Umpire": ["Eve Five (#5)", None],
        }
    )


class RecordingS3(FakeS3Client):
    def __init__(self):
        super().__init__()
        self.gets = []

    def get_object(self, Bucket, Key):
        self.gets.append(Key)
        return super().get_object(Bucket=Bucket, Key=Key)

    def get_paginator(self, name):
        raise AssertionError("Referee fallback must never list the bucket")


@pytest.fixture
def store(monkeypatch):
    value = AssignmentStore(RecordingS3(), "test")
    monkeypatch.setattr(loader, "_store", lambda: value)
    return value


def test_snapshot_round_trip_and_direct_two_get_lookup(store):
    assert store.save(DAY, assignments(), NOW)
    key = f"{PREFIX}/game_date={DAY}/fetched_at=2026-10-16T13-05-22.123456Z/assignments.parquet"
    manifest_key = f"{PREFIX}/game_date={DAY}/latest.json"
    assert store.s3.put_order == [key, manifest_key]
    store.s3.gets.clear()
    frame, meta = store.read(DAY)
    assert store.s3.gets == [manifest_key, key]
    pd.testing.assert_frame_equal(frame, normalize_assignments(assignments()))
    assert meta["fetched_at_utc"] == NOW.isoformat()
    assert frame.Umpire.isna().sum() == 1


def test_identical_crews_in_any_row_order_are_not_uploaded_again(store):
    store.save(DAY, assignments(), NOW)
    writes = list(store.s3.put_order)
    store.s3.gets.clear()
    assert not store.save(DAY, assignments().iloc[::-1], NOW + pd.Timedelta(minutes=5))
    assert store.s3.put_order == writes
    assert store.s3.gets == [f"{PREFIX}/game_date={DAY}/latest.json"]


def test_changed_crews_get_a_new_timestamp_and_update_latest(store):
    store.save(DAY, assignments(), NOW)
    changed = assignments()
    changed.loc[0, "Crew Chief"] = "Replacement (#99)"
    later = NOW + pd.Timedelta(minutes=5)
    assert store.save(DAY, changed, later)
    assert len(store.s3.store) == 3  # Both versions plus latest.json.
    frame, meta = store.read(DAY)
    assert meta["fetched_at_utc"] == later.isoformat()
    assert frame.loc[0, "Crew Chief"] == "Replacement (#99)"


def test_cache_miss_is_one_get_and_never_uses_another_day(store):
    store.save(DAY, assignments(), NOW)
    store.s3.gets.clear()
    assert store.read("2026-10-17") is None
    assert store.s3.gets == [f"{PREFIX}/game_date=2026-10-17/latest.json"]


def test_wrong_day_cannot_be_saved(store):
    with pytest.raises(ValueError, match="different game date"):
        store.save("2026-10-17", assignments(), NOW)
    assert not store.s3.put_order


def test_partial_upload_does_not_publish_a_pointer(store, monkeypatch):
    store.save(DAY, assignments(), NOW)
    manifest_key = f"{PREFIX}/game_date={DAY}/latest.json"
    original_manifest = store.s3.store[manifest_key]
    changed = assignments()
    changed.loc[0, "Crew Chief"] = "Replacement"

    def fail(**kwargs):
        raise OSError("Upload unavailable")

    monkeypatch.setattr(store.s3, "put_object", fail)
    with pytest.raises(OSError):
        store.save(DAY, changed, NOW + pd.Timedelta(minutes=5))
    assert store.s3.store[manifest_key] == original_manifest


def test_live_failure_uses_existing_snapshot_without_writing(
    store, monkeypatch, caplog
):
    store.save(DAY, assignments(), NOW)
    writes = list(store.s3.put_order)

    def fail(*args, **kwargs):
        raise requests.Timeout("NBA timed out")

    monkeypatch.setattr(loader, "fetch_nba_referee_assignments_today", fail)
    result = loader.load_referee_assignments(DAY)
    pd.testing.assert_frame_equal(result, normalize_assignments(assignments()))
    assert store.s3.put_order == writes
    assert "Using S3 referee assignments" in caplog.text
    assert NOW.isoformat() in caplog.text


@pytest.mark.parametrize(
    "cache_state",
    ["missing", "corrupt", "wrong_date", "missing_parquet", "unreachable"],
)
def test_both_sources_unavailable_leave_nan_assignments(
    store, monkeypatch, cache_state, caplog
):
    def fail(*args, **kwargs):
        raise requests.HTTPError("NBA unavailable")

    monkeypatch.setattr(loader, "fetch_nba_referee_assignments_today", fail)
    if cache_state != "missing":
        store.save(DAY, assignments(), NOW)
        meta_key = f"{PREFIX}/game_date={DAY}/latest.json"
        meta = json.loads(store.s3.store[meta_key])
        if cache_state == "corrupt":
            store.s3.store[meta["key"]] = b"corrupt"
        elif cache_state == "wrong_date":
            meta["assignment_date"] = "2026-10-15"
            store.s3.store[meta_key] = json.dumps(meta).encode()
        elif cache_state == "missing_parquet":
            del store.s3.store[meta["key"]]
        elif cache_state == "unreachable":
            monkeypatch.setattr(store.s3, "get_object", fail)
    result = loader.load_referee_assignments(DAY)
    assert result.empty
    assert list(result.columns) == ASSIGNMENT_COLUMNS
    assert "continuing with NaN features" in caplog.text


def test_successful_live_fetch_is_used_even_if_s3_write_fails(store, monkeypatch):
    monkeypatch.setattr(
        loader, "fetch_nba_referee_assignments_today", lambda *a, **kw: assignments()
    )

    def fail(*args, **kwargs):
        raise OSError("S3 unavailable")

    monkeypatch.setattr(store, "save", fail)
    result = loader.load_referee_assignments(DAY)
    pd.testing.assert_frame_equal(result, normalize_assignments(assignments()))


def test_successful_live_fetch_saves_with_the_requested_date(store, monkeypatch):
    called = []

    def live(url, *, expected_assignment_date):
        called.append(expected_assignment_date)
        return assignments()

    monkeypatch.setattr(loader, "fetch_nba_referee_assignments_today", live)
    monkeypatch.setattr(
        store, "save", lambda day, frame, fetched_at: called.append(day)
    )
    result = loader.load_referee_assignments(DAY)
    assert called == [DAY, DAY]
    pd.testing.assert_frame_equal(result, normalize_assignments(assignments()))


def test_successful_empty_response_does_not_revive_or_overwrite_cached_crews(
    store, monkeypatch
):
    store.save(DAY, assignments(), NOW)
    before = list(store.s3.put_order)
    store.s3.gets.clear()
    monkeypatch.setattr(
        loader,
        "fetch_nba_referee_assignments_today",
        lambda *a, **kw: pd.DataFrame(columns=ASSIGNMENT_COLUMNS),
    )
    assert loader.load_referee_assignments(DAY).empty
    assert not store.s3.gets
    assert store.s3.put_order == before
