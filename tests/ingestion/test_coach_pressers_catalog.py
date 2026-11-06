"""Step 1 of the coach presser archive: expected pressers and the manifest."""

from __future__ import annotations

import json

import pytest

from lab.coach_pressers import catalog
from lab.coach_pressers.manifest import checkpoint, load_manifest
from tests.support.coach_pressers import config_2021, game, schedule, season_2021


def test_two_entries_per_final_game_in_scope(tmp_path):
    payload = season_2021(
        game("0042100101", "2022-04-17", "SAC", "PHX"),
        game("0062100001", "2021-12-09", "PHX", "SAC"),
        game("0022100040", "2021-12-10", "PHX", "SAC", status=1),
    )
    manifest = catalog.prepare_manifest(config_2021(tmp_path), payload)
    entries = manifest["entries"]
    assert len(entries) == 2 * 6
    away = entries["0022100010:SAC"]
    assert away["opponent"] == "PHX" and away["is_home"] is False
    assert away["team_score"] == 100 and away["opponent_score"] == 110
    assert away["tip_utc"] == "2021-11-19T23:30:00Z"
    assert entries["0042100101:PHX"]["playoff_game_no"] == 1
    assert entries["0062100001:SAC"]["phase"] == "cup_final"
    future = entries["0022100040:SAC"]
    assert future["status"] == "not_final" and future["team_score"] is None
    assert manifest["final_games_by_phase"] == {
        "cup_final": 1,
        "playoffs": 1,
        "regular_season": 3,
    }


def test_team_filter_does_not_shrink_the_inventory(tmp_path):
    config = config_2021(tmp_path, teams=("SAC",))
    manifest = catalog.prepare_manifest(config, season_2021())
    assert {e["team"] for e in manifest["entries"].values()} == {"SAC", "PHX"}


def test_state_survives_a_refresh_and_identity_changes_are_refused(tmp_path):
    config = config_2021(tmp_path)
    manifest = catalog.prepare_manifest(config, season_2021())
    entry = manifest["entries"]["0022100010:SAC"]
    entry.update(status="selected", selection={"video_id": "abc"})
    checkpoint(config, manifest)

    again = catalog.prepare_manifest(config, season_2021())
    assert again["entries"]["0022100010:SAC"]["selection"] == {"video_id": "abc"}

    changed = season_2021()
    changed["leagueSchedule"]["gameDates"][0]["games"][0]["awayTeam"]["score"] = 99
    with pytest.raises(ValueError, match="identity changed"):
        catalog.prepare_manifest(config, changed)

    shrunk = schedule(
        "2021-22", season_2021()["leagueSchedule"]["gameDates"][0]["games"][1:]
    )
    with pytest.raises(ValueError, match="lost"):
        catalog.prepare_manifest(config, shrunk)


def test_a_game_that_becomes_final_turns_pending(tmp_path):
    config = config_2021(tmp_path)
    future = game("0022100040", "2021-12-10", "PHX", "SAC", status=1)
    checkpoint(config, catalog.prepare_manifest(config, season_2021(future)))
    future["gameStatus"] = 3
    manifest = catalog.prepare_manifest(config, season_2021(future))
    assert manifest["entries"]["0022100040:SAC"]["status"] == "pending"


def test_manifest_from_another_destination_is_refused(tmp_path):
    config = config_2021(tmp_path)
    checkpoint(config, catalog.prepare_manifest(config, season_2021()))
    other = config_2021(tmp_path, prefix="other/prefix")
    with pytest.raises(ValueError, match="s3_prefix"):
        load_manifest(other)
    promoted = config_2021(tmp_path, bucket="bucket")
    assert load_manifest(promoted)["bucket"] == ""


def test_count_check_warns_only_for_a_finished_season(tmp_path):
    config = config_2021(tmp_path)
    payload = season_2021()
    assert catalog.prepare_manifest(config, payload)["warnings"] == []
    payload = season_2021(game("0042100407", "2022-06-16", "PHX", "SAC"))
    warnings = catalog.prepare_manifest(config, payload)["warnings"]
    assert warnings == ["3 regular-season games, expected 1230"]


def test_finished_season_schedule_is_read_from_cache(tmp_path, monkeypatch):
    config = config_2021(tmp_path)
    payload = season_2021(game("0042100407", "2022-06-16", "PHX", "SAC"))
    path = catalog.schedule_path(config)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload))
    assert catalog.load_schedule(config) == payload
