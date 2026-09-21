"""Backing the line-history store up to S3.

The partition handling is the part worth pinning down: ``lh_line`` is a
partitioned parent, and treating it like an ordinary table would write the
whole fact table twice.
"""

from datetime import UTC, datetime

import pandas as pd
import pytest
from nba_ou.postgre_db import backup_core
from nba_ou.postgre_db.line_history_aiven import backup as bk

# (relname, relkind, is_partition) as pg_class reports them.
PG_CLASS_ROWS = [
    ("lh_book", "r", False),
    ("lh_game", "r", False),
    ("lh_line", "p", False),
    ("lh_line_2021", "r", True),
    ("lh_line_2025", "r", True),
    ("lh_load_meta", "r", False),
    ("lh_market", "r", False),
]


class _Cursor:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        self.query = query

    def fetchall(self):
        return self._rows


class _Conn:
    def __init__(self, rows=PG_CLASS_ROWS):
        self._rows = rows

    def cursor(self):
        return _Cursor(self._rows)


class _S3:
    """Records puts instead of performing them."""

    def __init__(self):
        self.puts = []

    def put_object(self, Bucket, Key, Body, Tagging=None):  # noqa: N803 - boto3 casing
        self.puts.append((Bucket, Key, len(Body)))


class TestDiscovery:
    def test_partitioned_parent_is_skipped(self):
        # Its rows live in the partitions; exporting both would duplicate the
        # entire fact table.
        targets, skipped = bk.discover_backup_targets(_Conn())
        assert skipped == ["lh_line"]
        assert "lh_line" not in [t.table for t in targets]

    def test_partitions_are_exported_individually(self):
        targets, _ = bk.discover_backup_targets(_Conn())
        partitions = [t.table for t in targets if t.is_partition]
        assert partitions == ["lh_line_2021", "lh_line_2025"]

    def test_ordinary_tables_are_included(self):
        targets, _ = bk.discover_backup_targets(_Conn())
        plain = [t.table for t in targets if not t.is_partition]
        assert plain == ["lh_book", "lh_game", "lh_load_meta", "lh_market"]

    def test_every_relation_is_either_exported_or_skipped(self):
        targets, skipped = bk.discover_backup_targets(_Conn())
        assert len(targets) + len(skipped) == len(PG_CLASS_ROWS)

    def test_target_label_reads_naturally(self):
        targets, _ = bk.discover_backup_targets(_Conn())
        by_name = {t.table: t.label for t in targets}
        assert by_name["lh_line_2021"] == "partition"
        assert by_name["lh_book"] == "table"

    def test_empty_schema_yields_nothing(self):
        targets, skipped = bk.discover_backup_targets(_Conn(rows=[]))
        assert targets == [] and skipped == []


class TestBackupRun:
    def _patch_export(self, monkeypatch, payload=b"PARQUET", rows=100):
        monkeypatch.setattr(
            backup_core,
            "export_table_to_parquet",
            lambda conn, schema, table: (payload, rows),
        )

    def test_writes_one_object_per_target(self, monkeypatch):
        self._patch_export(monkeypatch)
        s3 = _S3()
        result = bk.backup_line_history(
            _Conn(), date_tag="2026-08-01", s3_client=s3, progress=False
        )
        # 4 plain tables + 2 partitions (parent skipped), plus the manifest.
        assert len(s3.puts) == 7
        assert len([k for _, k, _ in s3.puts if k.endswith("_manifest.json")]) == 1
        assert len(result.uploaded) == 6
        assert result.rows == 600

    def test_keys_are_namespaced_by_date_and_schema(self, monkeypatch):
        self._patch_export(monkeypatch)
        s3 = _S3()
        bk.backup_line_history(
            _Conn(), date_tag="2026-08-01", s3_client=s3, progress=False
        )
        keys = [key for _, key, _ in s3.puts]
        assert all(k.startswith("backups/db/2026-08-01/line_history/") for k in keys)
        assert "backups/db/2026-08-01/line_history/lh_line_2021.parquet" in keys

    def test_each_run_lands_under_its_own_date(self, monkeypatch):
        # Runs must never overwrite one another; a restore is "pick a date".
        self._patch_export(monkeypatch)
        first, second = _S3(), _S3()
        bk.backup_line_history(
            _Conn(), date_tag="2026-08-01", s3_client=first, progress=False
        )
        bk.backup_line_history(
            _Conn(), date_tag="2026-09-01", s3_client=second, progress=False
        )
        assert {k for _, k, _ in first.puts}.isdisjoint({k for _, k, _ in second.puts})

    def test_dry_run_uploads_nothing_but_reports_the_plan(self, monkeypatch):
        self._patch_export(monkeypatch)
        s3 = _S3()
        result = bk.backup_line_history(
            _Conn(), date_tag="2026-08-01", dry_run=True, s3_client=s3, progress=False
        )
        assert s3.puts == []
        assert len(result.uploaded) == 6
        assert result.rows == 0

    def test_date_tag_defaults_to_today_in_utc(self, monkeypatch):
        # UTC, not local time. The workflow pins TZ=UTC and both databases
        # share one date tag, so a runner in any timezone must land in the
        # same folder. Asserting against the local date fails for the hours
        # between local midnight and UTC midnight.
        self._patch_export(monkeypatch)
        s3 = _S3()
        result = bk.backup_line_history(_Conn(), s3_client=s3, progress=False)
        assert datetime.now(UTC).strftime("%Y-%m-%d") in result.prefix

    def test_empty_schema_uploads_nothing(self, monkeypatch):
        self._patch_export(monkeypatch)
        s3 = _S3()
        result = bk.backup_line_history(_Conn(rows=[]), s3_client=s3, progress=False)
        assert s3.puts == [] and result.uploaded == []


class TestParquetRoundTrip:
    def test_timezone_survives_the_round_trip(self, tmp_path):
        # line_ts is the column all the timezone work rests on; a backup that
        # silently dropped the offset would be worthless.
        frame = pd.DataFrame(
            {"line_ts": pd.to_datetime(["2025-04-04 23:05:00"], utc=True)}
        )
        path = tmp_path / "t.parquet"
        frame.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        restored = pd.read_parquet(path)
        assert str(restored["line_ts"].dtype) == "datetime64[ns, UTC]"
        assert restored["line_ts"][0] == frame["line_ts"][0]


class TestS3Settings:
    def test_blank_profile_becomes_none(self, monkeypatch):
        # CI supplies credentials via OIDC and sets S3_AWS_PROFILE="";
        # boto3 must not be handed an empty profile name.
        monkeypatch.setenv("S3_AWS_PROFILE", "")
        assert bk.get_s3_settings()["profile"] is None

    def test_bucket_can_be_overridden_by_env(self, monkeypatch):
        monkeypatch.setenv("S3_BACKUP_BUCKET", "some-other-bucket")
        assert bk.get_s3_settings()["bucket"] == "some-other-bucket"

    def test_prefix_is_shared_with_the_supabase_backup(self):
        # Both databases restore the same way.
        assert bk.S3_BACKUP_PREFIX == "backups/db"


@pytest.mark.parametrize("relkind,expected", [("p", True), ("r", False)])
def test_only_relkind_p_is_treated_as_a_parent(relkind, expected):
    targets, skipped = bk.discover_backup_targets(
        _Conn(rows=[("thing", relkind, False)])
    )
    assert bool(skipped) is expected
