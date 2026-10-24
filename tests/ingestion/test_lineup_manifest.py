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


def test_a_manifest_written_before_status_codes_still_loads(tmp_path):
    """Pre-2026-09-24 manifests have five columns; they must load and resume."""
    import pandas as pd

    legacy = pd.DataFrame(
        [("0021800001", "gamerotation", pd.Timestamp.now(tz="UTC"), "ok", 10)],
        columns=["game_id", "endpoint", "fetched_at", "status", "bytes"],
    )
    path = tmp_path / "season=2018" / "manifest.parquet"
    path.parent.mkdir(parents=True)
    legacy.to_parquet(path, index=False)

    manifest = Manifest(tmp_path)
    assert manifest.is_ok(2018, "0021800001", "gamerotation")
    frame = manifest.load(2018)
    assert frame["http_status"].isna().all()
    assert frame["elapsed_s"].isna().all()

    manifest.record(
        2018, "0021800002", "gamerotation", "server_timeout",
        http_status=500, elapsed_s=30.5,
    )
    reloaded = Manifest(tmp_path).load(2018).set_index("game_id")
    assert reloaded.loc["0021800002", "status"] == "server_timeout"
    assert reloaded.loc["0021800002", "http_status"] == 500
    assert reloaded.loc["0021800002", "elapsed_s"] == 30.5
    assert pd.isna(reloaded.loc["0021800001", "http_status"])


def test_record_many_keeps_its_four_field_rows(tmp_path):
    manifest = Manifest(tmp_path)
    manifest.record_many(2018, [("0021800001", "playbyplayv3", "ok", 10)])
    frame = Manifest(tmp_path).load(2018)
    assert frame.loc[0, "status"] == "ok"
    assert frame["http_status"].isna().all()


def test_unknown_status_is_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        Manifest(tmp_path).record(2018, "0021800001", "gamerotation", "timeout")


def test_coverage_separates_holes_timeouts_and_unverified_empties(tmp_path):
    import pandas as pd

    from scripts.lineups.report_lineup_coverage import coverage

    root = tmp_path / "nba_api_raw" / "manifest"
    legacy = pd.DataFrame(
        [("0021800001", "gamerotation", pd.Timestamp.now(tz="UTC"), "empty", 0)],
        columns=["game_id", "endpoint", "fetched_at", "status", "bytes"],
    )
    path = root / "season=2018" / "manifest.parquet"
    path.parent.mkdir(parents=True)
    legacy.to_parquet(path, index=False)

    manifest = Manifest(root)
    manifest.record(
        2018, "0021800002", "gamerotation", "empty", http_status=500, elapsed_s=0.4
    )
    manifest.record(
        2018, "0021800003", "gamerotation", "server_timeout",
        http_status=500, elapsed_s=30.5,
    )
    row = coverage(tmp_path).iloc[0]
    assert row.rotation_empty == 1
    assert row.rotation_empty_unverified == 1
    assert row.rotation_server_timeout == 1
