"""Bulk manifest writes must behave like a run of single ones."""

from __future__ import annotations

from nba_ou.fetch_data.nba_lineups.manifest import Manifest


def test_record_many_supersedes_earlier_rows_for_the_same_call(tmp_path):
    manifest = Manifest(tmp_path)
    manifest.record(2018, "0021800001", "playbyplayv3", "failed")
    manifest.record_many(
        2018,
        [("0021800001", "playbyplayv3", "ok", 10), ("0021800002", "playbyplayv3", "ok", 20)],
    )
    assert manifest.is_ok(2018, "0021800001", "playbyplayv3")
    assert manifest.is_ok(2018, "0021800002", "playbyplayv3")
    assert len(manifest.load(2018)) == 2
