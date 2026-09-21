from pathlib import Path

import pytest
from nba_ou.fetch_data.nba_lineups.client import CircuitOpen
from nba_ou.fetch_data.nba_lineups.run_lock import (
    BackfillAlreadyRunning,
    lineup_run_lock,
)


def test_lineup_run_lock_rejects_a_concurrent_writer(tmp_path):
    with lineup_run_lock(tmp_path):
        path = tmp_path / "nba_api_raw" / "backfill.lock"
        owner = path.read_text()
        with pytest.raises(BackfillAlreadyRunning):
            with lineup_run_lock(tmp_path):
                pass
        assert path.read_text() == owner


def test_database_status_comparison_normalizes_success_reasons():
    from scripts.lineups.load_lineup_stints import _status_key

    assert _status_key("ok", "") == ("ok", None)
    assert _status_key("ok", float("nan")) == ("ok", None)
    assert _status_key("failed", "score_mismatch") == (
        "failed",
        "score_mismatch",
    )


def test_daily_update_builds_and_loads_work_completed_before_breaker(
    monkeypatch, tmp_path
):
    from scripts.lineups import update_lineups

    monkeypatch.setattr(
        update_lineups, "finished_games", lambda minimum: [(2018, "g1")]
    )
    monkeypatch.setattr(update_lineups, "fetch_game_statuses", lambda: {})
    monkeypatch.setattr(
        update_lineups,
        "backfill",
        lambda *args, **kwargs: (_ for _ in ()).throw(CircuitOpen("blocked")),
    )
    built = []
    loaded = []
    monkeypatch.setattr(
        update_lineups,
        "build_archived",
        lambda season, **kwargs: built.append(season) or {"ok": 1, "failed": 0},
    )
    monkeypatch.setattr(
        update_lineups,
        "load_season",
        lambda season, **kwargs: loaded.append(season) or 1,
    )

    with pytest.raises(CircuitOpen, match="completed work"):
        update_lineups.update(Path(tmp_path), load_to_db=True)
    assert built == [2018]
    assert loaded == [2018]


def test_stint_update_skips_games_already_validated(monkeypatch, tmp_path):
    from nba_ou.fetch_data.nba_lineups.archive import RawArchive
    from nba_ou.fetch_data.nba_lineups.manifest import Manifest

    from scripts.lineups import build_lineup_stints

    manifest = Manifest(tmp_path / "nba_api_raw" / "manifest")
    for endpoint in ("gamerotation", "playbyplayv3"):
        manifest.record(2018, "0021800001", endpoint, "ok", 10)
    target = tmp_path / "lineup_stints" / "season=2018"
    target.mkdir(parents=True)
    import pandas as pd

    pd.DataFrame([{"game_id": "0021800001", "status": "ok", "reason": ""}]).to_parquet(
        target / "game_status.parquet", index=False
    )
    monkeypatch.setattr(
        build_lineup_stints,
        "game_context",
        lambda ids: pytest.fail("no database read is needed for validated games"),
    )
    assert build_lineup_stints.build_archived(
        2018,
        archive=RawArchive(root=tmp_path),
        manifest=manifest,
        output_root=tmp_path / "lineup_stints",
    ) == {"ok": 0, "failed": 0}


def test_daily_update_only_processes_games_missing_from_database(monkeypatch, tmp_path):
    from scripts.lineups import update_lineups

    games = [(2018, "0021800001"), (2018, "0021800002")]
    monkeypatch.setattr(update_lineups, "finished_games", lambda minimum: games)
    monkeypatch.setattr(
        update_lineups,
        "fetch_game_statuses",
        lambda: {"0021800001": ("ok", None)},
    )
    fetched = []
    monkeypatch.setattr(
        update_lineups,
        "backfill",
        lambda pending, **kwargs: (
            fetched.extend(pending) or {"ok": 2, "empty": 0, "failed": 0, "skipped": 0}
        ),
    )
    built = []
    monkeypatch.setattr(
        update_lineups,
        "build_archived",
        lambda season, **kwargs: (
            built.append(kwargs["game_ids"]) or {"ok": 1, "failed": 0}
        ),
    )
    monkeypatch.setattr(update_lineups, "load_season", lambda *args, **kwargs: 1)

    update_lineups.update(tmp_path, load_to_db=True)

    assert fetched == [(2018, "0021800002")]
    assert built == [{"0021800002"}]


def test_daily_update_retries_a_previously_failed_game(monkeypatch, tmp_path):
    """A failed build must not be permanent: the raw archive is already paid."""
    from scripts.lineups import update_lineups

    monkeypatch.setattr(
        update_lineups,
        "finished_games",
        lambda minimum: [(2018, "0021800001"), (2018, "0021800002")],
    )
    monkeypatch.setattr(
        update_lineups,
        "fetch_game_statuses",
        lambda: {
            "0021800001": ("ok", None),
            "0021800002": ("failed", "points_mismatch"),
        },
    )
    requested = {}

    def fake_backfill(pending, **kwargs):
        requested["pending"] = list(pending)
        return {"ok": 0, "empty": 0, "failed": 0, "skipped": 0}

    monkeypatch.setattr(update_lineups, "backfill", fake_backfill)
    monkeypatch.setattr(
        update_lineups, "build_archived", lambda season, **kwargs: {"ok": 1, "failed": 0}
    )
    monkeypatch.setattr(update_lineups, "load_season", lambda season, **kwargs: 1)

    update_lineups.update(Path(tmp_path), load_to_db=True)
    assert requested["pending"] == [(2018, "0021800002")]


def test_daily_update_defaults_to_the_local_parquet_store(monkeypatch, tmp_path):
    """Without --load-db nothing reads or writes the lineups database."""
    import pandas as pd

    from scripts.lineups import update_lineups

    target = tmp_path / "lineup_stints" / "season=2018"
    target.mkdir(parents=True)
    pd.DataFrame(
        [
            {"game_id": "0021800001", "status": "ok", "reason": ""},
            {"game_id": "0021800002", "status": "failed", "reason": "points_mismatch"},
        ]
    ).to_parquet(target / "game_status.parquet", index=False)

    monkeypatch.setattr(
        update_lineups,
        "finished_games",
        lambda minimum: [
            (2018, "0021800001"),
            (2018, "0021800002"),
            (2018, "0021800003"),
        ],
    )
    monkeypatch.setattr(
        update_lineups,
        "fetch_game_statuses",
        lambda: pytest.fail("the default run must not touch the lineups database"),
    )
    monkeypatch.setattr(
        update_lineups,
        "load_season",
        lambda *args, **kwargs: pytest.fail("no database load without --load-db"),
    )
    requested = {}

    def fake_backfill(pending, **kwargs):
        requested["pending"] = list(pending)
        return {"ok": 2, "empty": 0, "failed": 0, "skipped": 0}

    monkeypatch.setattr(update_lineups, "backfill", fake_backfill)
    monkeypatch.setattr(
        update_lineups, "build_archived", lambda season, **kwargs: {"ok": 2, "failed": 0}
    )

    result = update_lineups.update(tmp_path)

    assert requested["pending"] == [(2018, "0021800002"), (2018, "0021800003")]
    assert "loaded" not in result
