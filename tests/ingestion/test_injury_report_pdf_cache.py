"""Archive and ingest share PDF bytes without treating local files as uploads."""

from unittest.mock import Mock

import pytest
from nba_ou.fetch_data.injury_reports.archive import storage

from tests.support.fake_s3 import FakeS3Client

KEY = "injury_reports/raw/season=2026-27/date=2026-10-25/report.pdf"


@pytest.fixture
def s3(monkeypatch):
    client = FakeS3Client()
    client.get_object = Mock(wraps=client.get_object)
    monkeypatch.setattr(storage, "make_s3_client", lambda **kw: client)
    return client


def cached(tmp_path, bucket="archive"):
    return storage.S3Storage(
        bucket=bucket, profile=None, region="eu-west-1", cache_dir=tmp_path
    )


def test_ingest_reuses_uploaded_pdf_across_storage_instances(tmp_path, s3):
    # The downloader runs in a subprocess; ingest opens a separate instance.
    cached(tmp_path).put(KEY, b"pdf from NBA", {})
    assert s3.store[KEY] == b"pdf from NBA"
    assert cached(tmp_path).get(KEY) == b"pdf from NBA"
    s3.get_object.assert_not_called()


def test_older_pdf_is_read_from_s3_only_once(tmp_path, s3):
    s3.store[KEY] = b"previously archived"
    assert cached(tmp_path).get(KEY) == b"previously archived"
    assert cached(tmp_path).get(KEY) == b"previously archived"
    s3.get_object.assert_called_once_with(Bucket="archive", Key=KEY)


def test_buckets_do_not_share_cached_bytes(tmp_path, s3):
    cached(tmp_path, "first").put(KEY, b"first bucket", {})
    s3.store[KEY] = b"second bucket"
    assert cached(tmp_path, "second").get(KEY) == b"second bucket"
    s3.get_object.assert_called_once_with(Bucket="second", Key=KEY)


def test_failed_upload_does_not_leave_a_cached_pdf(tmp_path, s3):
    s3.put_object = Mock(side_effect=OSError("upload failed"))
    with pytest.raises(OSError, match="upload failed"):
        cached(tmp_path).put(KEY, b"not archived", {})
    assert not list(tmp_path.rglob("*.pdf"))
    assert cached(tmp_path).get(KEY) is None


def test_missing_pdf_is_not_cached(tmp_path, s3):
    assert cached(tmp_path).get(KEY) is None
    s3.store[KEY] = b"now available"
    assert cached(tmp_path).get(KEY) == b"now available"


def test_cache_io_errors_do_not_prevent_s3_reads_and_writes(tmp_path, s3):
    store = cached(tmp_path)
    store.cache.get = Mock(side_effect=OSError("unreadable cache"))
    store.cache.put = Mock(side_effect=OSError("disk full"))
    store.put(KEY, b"archived", {})
    assert s3.store[KEY] == b"archived"
    assert store.get(KEY) == b"archived"


def test_cache_is_optional(tmp_path, s3):
    store = storage.S3Storage(bucket="archive", profile=None, region="eu-west-1")
    store.put(KEY, b"archived", {})
    assert store.get(KEY) == b"archived"
    assert store.get(KEY) == b"archived"
    assert s3.get_object.call_count == 2
