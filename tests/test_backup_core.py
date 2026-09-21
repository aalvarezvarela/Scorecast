"""Shared backup machinery: schema coverage, manifests, verification.

Two bug classes are pinned down here, both of which shipped:

* a hand-maintained schema list silently stops covering new schemas -- three
  went unbacked for months while the list still named a long-empty one;
* ``information_schema`` cannot tell a partitioned parent from its leaves, so
  discovery must go through ``pg_class.relkind``.
"""

import hashlib
import json

import pytest
from nba_ou.postgre_db import backup_core as bc

# (nspname, relation_count) as the discovery query reports it.
NAMESPACE_ROWS = [
    ("auth", 27),
    ("extensions", 0),
    ("graphql", 0),
    ("lineups", 4),
    ("nba_all_star_voting", 1),
    ("nba_games", 1),
    ("nba_odds", 0),
    ("nba_predictions", 1),
    ("pg_catalog", 130),
    ("pg_toast", 50),
    ("public", 0),
    ("storage", 8),
    ("vault", 1),
]

# (relname, relkind, is_partition) as pg_class reports them.
LINEUPS_ROWS = [
    ("lu_game_status", "r", False),
    ("lu_lineup", "r", False),
    ("lu_stint", "p", False),
    ("lu_stint_2018", "r", True),
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
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _Cursor(self._rows)


class _S3:
    """Records puts and serves them back, so a run can be verified."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.tags: dict[str, str | None] = {}

    def put_object(self, Bucket, Key, Body, Tagging=None):  # noqa: N803 - boto3 casing
        self.objects[Key] = Body
        self.tags[Key] = Tagging

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)

        class _Body:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

        return {"Body": _Body(self.objects[Key])}

    def list_objects_v2(
        self, Bucket, Prefix, Delimiter=None, MaxKeys=None
    ):  # noqa: N803
        keys = [k for k in self.objects if k.startswith(Prefix)]
        result = {"KeyCount": len(keys)}
        if Delimiter:
            prefixes = set()
            for key in keys:
                rest = key[len(Prefix) :]
                if Delimiter in rest:
                    prefixes.add(Prefix + rest.split(Delimiter)[0] + Delimiter)
            result["CommonPrefixes"] = [{"Prefix": p} for p in sorted(prefixes)]
        return result

    def get_paginator(self, _name):
        outer = self

        class _Paginator:
            def paginate(self, **kwargs):
                return [outer.list_objects_v2(**kwargs)]

        return _Paginator()


class TestSchemaDiscovery:
    def test_project_schemas_are_discovered(self):
        assert bc.discover_schemas(_Conn(NAMESPACE_ROWS)) == [
            "lineups",
            "nba_all_star_voting",
            "nba_games",
            "nba_predictions",
        ]

    def test_schemas_that_were_silently_unbacked_are_now_in_scope(self):
        # The regression this refactor exists for: these three held live data
        # and appeared in no backup, because they were not in the old list.
        found = bc.discover_schemas(_Conn(NAMESPACE_ROWS))
        for schema in ("lineups", "nba_all_star_voting", "nba_predictions"):
            assert schema in found

    def test_empty_schemas_are_skipped(self):
        # nba_odds was named by the old list long after its last table went.
        assert "nba_odds" not in bc.discover_schemas(_Conn(NAMESPACE_ROWS))
        assert "public" not in bc.discover_schemas(_Conn(NAMESPACE_ROWS))

    @pytest.mark.parametrize(
        "schema", ["auth", "storage", "vault", "extensions", "graphql"]
    )
    def test_platform_schemas_are_excluded(self, schema):
        assert not bc.is_project_schema(schema)

    @pytest.mark.parametrize("schema", ["pg_catalog", "pg_toast", "pg_temp_3"])
    def test_postgres_internal_schemas_are_excluded(self, schema):
        assert not bc.is_project_schema(schema)

    def test_an_unknown_schema_is_in_scope_by_default(self):
        # The whole point of inverting the list: something added tomorrow is
        # covered without anyone editing this file.
        assert bc.is_project_schema("nba_something_new")


class TestPartitionHandling:
    def test_partitioned_parent_is_skipped(self):
        # lineups.lu_stint is a real partitioned parent that the old
        # information_schema-based Supabase script would have double-written.
        targets, skipped = bc.discover_backup_targets(_Conn(LINEUPS_ROWS), "lineups")
        assert skipped == ["lu_stint"]
        assert [t.table for t in targets] == [
            "lu_game_status",
            "lu_lineup",
            "lu_stint_2018",
        ]

    def test_rows_are_never_counted_twice(self, monkeypatch):
        monkeypatch.setattr(
            bc, "export_table_to_parquet", lambda conn, schema, table: (b"X", 100)
        )
        s3 = _S3()
        result = bc.backup_database(
            _Conn(LINEUPS_ROWS),
            schemas=["lineups"],
            date_tag="2026-09-21",
            s3_client=s3,
            progress=False,
        )
        # 3 exported relations, not 4: the parent contributes nothing.
        assert result.rows == 300


class TestManifest:
    def _run(self, monkeypatch, s3):
        monkeypatch.setattr(
            bc, "export_table_to_parquet", lambda conn, schema, table: (b"X", 100)
        )
        return bc.backup_database(
            _Conn(LINEUPS_ROWS),
            schemas=["lineups"],
            date_tag="2026-09-21",
            s3_client=s3,
            progress=False,
        )

    def test_manifest_is_written_beside_the_parquet(self, monkeypatch):
        s3 = _S3()
        self._run(monkeypatch, s3)
        assert "backups/db/2026-09-21/lineups/_manifest.json" in s3.objects

    def test_manifest_records_every_exported_relation(self, monkeypatch):
        s3 = _S3()
        self._run(monkeypatch, s3)
        manifest = json.loads(
            s3.objects["backups/db/2026-09-21/lineups/_manifest.json"]
        )
        assert {t["table"] for t in manifest["tables"]} == {
            "lu_game_status",
            "lu_lineup",
            "lu_stint_2018",
        }
        assert manifest["total_rows"] == 300
        assert manifest["skipped_parents"] == ["lu_stint"]

    def test_dry_run_writes_no_manifest(self, monkeypatch):
        monkeypatch.setattr(
            bc, "export_table_to_parquet", lambda conn, schema, table: (b"X", 100)
        )
        s3 = _S3()
        bc.backup_database(
            _Conn(LINEUPS_ROWS),
            schemas=["lineups"],
            date_tag="2026-09-21",
            dry_run=True,
            s3_client=s3,
            progress=False,
        )
        assert s3.objects == {}


class TestVerification:
    def _backed_up(self, monkeypatch, rows=LINEUPS_ROWS):
        monkeypatch.setattr(
            bc, "export_table_to_parquet", lambda conn, schema, table: (b"X", 100)
        )
        s3 = _S3()
        bc.backup_database(
            _Conn(rows),
            schemas=["lineups"],
            date_tag="2026-09-21",
            s3_client=s3,
            progress=False,
        )
        return s3

    def test_a_complete_backup_reports_no_issues(self, monkeypatch):
        s3 = self._backed_up(monkeypatch)
        issues = bc.verify_backup(
            _Conn(LINEUPS_ROWS), "2026-09-21", schemas=["lineups"], s3_client=s3
        )
        assert issues == []

    def test_a_new_partition_is_flagged(self, monkeypatch):
        # Exactly the 2019/2020 line-partition gap: a partition appears in the
        # database after the backup was taken and nothing notices.
        s3 = self._backed_up(monkeypatch)
        grown = LINEUPS_ROWS + [("lu_stint_2019", "r", True)]
        issues = bc.verify_backup(
            _Conn(grown), "2026-09-21", schemas=["lineups"], s3_client=s3
        )
        assert [i.kind for i in issues] == ["missing-table"]
        assert "lu_stint_2019" in issues[0].detail

    def test_an_entirely_unbacked_schema_is_flagged(self, monkeypatch):
        s3 = self._backed_up(monkeypatch)
        issues = bc.verify_backup(
            _Conn(LINEUPS_ROWS),
            "2026-09-21",
            schemas=["lineups", "nba_all_star_voting"],
            s3_client=s3,
        )
        assert [i.kind for i in issues] == ["missing-schema"]
        assert issues[0].schema == "nba_all_star_voting"

    def test_a_backup_without_a_manifest_cannot_be_verified(self, monkeypatch):
        s3 = self._backed_up(monkeypatch)
        del s3.objects["backups/db/2026-09-21/lineups/_manifest.json"]
        issues = bc.verify_backup(
            _Conn(LINEUPS_ROWS), "2026-09-21", schemas=["lineups"], s3_client=s3
        )
        assert [i.kind for i in issues] == ["missing-manifest"]


class TestLayout:
    def test_both_databases_share_one_prefix(self):
        assert bc.S3_BACKUP_PREFIX == "backups/db"

    def test_keys_are_namespaced_by_date_then_schema(self, monkeypatch):
        monkeypatch.setattr(
            bc, "export_table_to_parquet", lambda conn, schema, table: (b"X", 1)
        )
        s3 = _S3()
        bc.backup_database(
            _Conn(LINEUPS_ROWS),
            schemas=["lineups"],
            date_tag="2026-09-21",
            s3_client=s3,
            progress=False,
        )
        assert all(k.startswith("backups/db/2026-09-21/lineups/") for k in s3.objects)


class TestRetentionTagging:
    """Weekly in-season runs need tiered retention, not dedup.

    S3 lifecycle cannot express "keep the first backup of each month" by date,
    but it can filter on an object tag. So the tier is decided here, at write
    time, and the rule is a pure function of the date tag.
    """

    @pytest.mark.parametrize("tag", ["2026-10-01", "2026-10-05", "2026-10-07"])
    def test_first_week_of_a_month_is_kept_long_term(self, tag):
        assert bc.retention_class(tag) == bc.RETENTION_MONTHLY

    @pytest.mark.parametrize("tag", ["2026-10-08", "2026-10-12", "2026-10-31"])
    def test_later_weeks_age_out(self, tag):
        assert bc.retention_class(tag) == bc.RETENTION_WEEKLY

    def test_exactly_one_weekly_run_per_month_is_kept(self):
        # Weekly runs are 7 days apart, so whatever weekday they land on,
        # precisely one falls in days 1-7. Checked across a full season
        # (November-June), whichever weekday the schedule starts on.
        from datetime import date, timedelta

        for start in range(1, 8):
            day = date(2026, 11, start)
            kept_per_month: dict[tuple[int, int], int] = {}
            while day < date(2027, 7, 1):
                if bc.retention_class(day.isoformat()) == bc.RETENTION_MONTHLY:
                    key = (day.year, day.month)
                    kept_per_month[key] = kept_per_month.get(key, 0) + 1
                day += timedelta(days=7)
            assert set(kept_per_month.values()) == {1}

    def test_an_unparseable_tag_is_kept_rather_than_expired(self):
        # Failing safe matters: the cost of keeping a backup is pennies, the
        # cost of expiring one that should have been kept is the backup.
        assert bc.retention_class("not-a-date") == bc.RETENTION_MONTHLY

    def test_objects_are_tagged_for_lifecycle(self, monkeypatch):
        monkeypatch.setattr(
            bc, "export_table_to_parquet", lambda conn, schema, table: (b"X", 1)
        )
        s3 = _S3()
        bc.backup_database(
            _Conn(LINEUPS_ROWS),
            schemas=["lineups"],
            date_tag="2026-10-12",
            s3_client=s3,
            progress=False,
        )
        assert set(s3.tags.values()) == {"retention=weekly"}

    def test_the_manifest_is_tagged_like_its_data(self, monkeypatch):
        # A manifest that outlived its Parquet, or vice versa, would leave a
        # restore point that lies about itself.
        monkeypatch.setattr(
            bc, "export_table_to_parquet", lambda conn, schema, table: (b"X", 1)
        )
        s3 = _S3()
        bc.backup_database(
            _Conn(LINEUPS_ROWS),
            schemas=["lineups"],
            date_tag="2026-10-05",
            s3_client=s3,
            progress=False,
        )
        assert len(set(s3.tags.values())) == 1
        assert s3.tags["backups/db/2026-10-05/lineups/_manifest.json"] == (
            "retention=monthly"
        )


class TestContentHashes:
    def _manifest(self, monkeypatch, payload=b"PARQUET-BYTES"):
        monkeypatch.setattr(
            bc, "export_table_to_parquet", lambda conn, schema, table: (payload, 1)
        )
        s3 = _S3()
        bc.backup_database(
            _Conn(LINEUPS_ROWS),
            schemas=["lineups"],
            date_tag="2026-10-05",
            s3_client=s3,
            progress=False,
        )
        return json.loads(s3.objects["backups/db/2026-10-05/lineups/_manifest.json"])

    def test_every_table_records_its_hash(self, monkeypatch):
        manifest = self._manifest(monkeypatch)
        expected = hashlib.sha256(b"PARQUET-BYTES").hexdigest()
        assert all(t["sha256"] == expected for t in manifest["tables"])

    def test_identical_data_hashes_identically_across_runs(self, monkeypatch):
        # The precondition for ever adopting content-addressed dedup: exports
        # are byte-deterministic, so an unchanged table hashes the same.
        first = self._manifest(monkeypatch)
        second = self._manifest(monkeypatch)
        assert [t["sha256"] for t in first["tables"]] == [
            t["sha256"] for t in second["tables"]
        ]

    def test_changed_data_changes_the_hash(self, monkeypatch):
        a = self._manifest(monkeypatch, payload=b"ONE")
        b = self._manifest(monkeypatch, payload=b"TWO")
        assert a["tables"][0]["sha256"] != b["tables"][0]["sha256"]

    def test_manifest_records_its_retention_tier(self, monkeypatch):
        assert self._manifest(monkeypatch)["retention"] == "monthly"


class _TaggingDeniedS3(_S3):
    """S3 that accepts plain puts but refuses tagged ones.

    Models a role holding s3:PutObject but not s3:PutObjectTagging -- the exact
    shape of the real credentials, which fail a tagged put outright.
    """

    def __init__(self, deny=True):
        super().__init__()
        self.deny = deny
        self.tagged_attempts = 0

    def put_object(self, Bucket, Key, Body, Tagging=None):  # noqa: N803
        if Tagging is not None:
            self.tagged_attempts += 1
            if self.deny:
                raise _AccessDenied(
                    {
                        "Error": {
                            "Code": "AccessDenied",
                            "Message": (
                                "User: arn:aws:iam::1:user/x is not authorized to "
                                "perform: s3:PutObjectTagging on resource: ..."
                            ),
                        }
                    }
                )
        super().put_object(Bucket, Key, Body, Tagging)


class _AccessDenied(Exception):
    def __init__(self, response):
        super().__init__(response["Error"]["Message"])
        self.response = response


class _OtherFailure(Exception):
    def __init__(self):
        super().__init__("no such bucket")
        self.response = {"Error": {"Code": "NoSuchBucket", "Message": "nope"}}


class TestTaggingDegradesGracefully:
    """Losing the retention tier is recoverable; losing the backup is not."""

    def _run(self, monkeypatch, s3):
        monkeypatch.setattr(
            bc, "export_table_to_parquet", lambda conn, schema, table: (b"X", 1)
        )
        return bc.backup_database(
            _Conn(LINEUPS_ROWS),
            schemas=["lineups"],
            date_tag="2026-10-12",
            s3_client=s3,
            progress=False,
        )

    def test_backup_still_completes_without_tagging_permission(self, monkeypatch):
        s3 = _TaggingDeniedS3()
        result = self._run(monkeypatch, s3)
        assert len(result.uploaded) == 3
        assert "backups/db/2026-10-12/lineups/_manifest.json" in s3.objects

    def test_untagged_fallback_stores_the_real_bytes(self, monkeypatch):
        s3 = _TaggingDeniedS3()
        self._run(monkeypatch, s3)
        assert s3.objects["backups/db/2026-10-12/lineups/lu_lineup.parquet"] == b"X"

    def test_tagging_is_only_attempted_once_per_run(self, monkeypatch):
        # Having learned the permission is missing, it must not retry on every
        # object and double the API calls.
        s3 = _TaggingDeniedS3()
        self._run(monkeypatch, s3)
        assert s3.tagged_attempts == 1

    def test_tags_are_applied_when_permission_exists(self, monkeypatch):
        s3 = _TaggingDeniedS3(deny=False)
        self._run(monkeypatch, s3)
        assert set(s3.tags.values()) == {"retention=weekly"}

    def test_non_tagging_failures_still_propagate(self, monkeypatch):
        # Only the tagging permission is forgiven; a genuinely broken upload
        # must not be swallowed into a silently empty backup.
        class _Broken(_S3):
            def put_object(self, Bucket, Key, Body, Tagging=None):  # noqa: N803
                raise _OtherFailure()

        monkeypatch.setattr(
            bc, "export_table_to_parquet", lambda conn, schema, table: (b"X", 1)
        )
        with pytest.raises(_OtherFailure):
            bc.backup_database(
                _Conn(LINEUPS_ROWS),
                schemas=["lineups"],
                date_tag="2026-10-12",
                s3_client=_Broken(),
                progress=False,
            )

    def test_the_error_matcher_is_specific(self):
        assert bc.is_tagging_permission_error(
            _AccessDenied(
                {"Error": {"Code": "AccessDenied", "Message": "s3:PutObjectTagging"}}
            )
        )
        # A plain PutObject denial is a real failure, not a tagging one.
        assert not bc.is_tagging_permission_error(
            _AccessDenied(
                {"Error": {"Code": "AccessDenied", "Message": "s3:PutObject denied"}}
            )
        )
        assert not bc.is_tagging_permission_error(_OtherFailure())
        assert not bc.is_tagging_permission_error(ValueError("boom"))
