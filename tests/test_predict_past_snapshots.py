"""The day's last run: every intermediate slot on its own, past snapshot."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from nba_ou.modeling.registry_paths import ModelSlot

import scripts.predict_past_snapshots as job


def _slot(horizon):
    return ModelSlot(schema_version="2_5", target="line_error", horizon_minutes=horizon)


def test_default_game_date_is_last_night_until_noon_eastern():
    et = "America/New_York"
    # After the late games' final buzzer, and the next morning: last night.
    assert (
        job.default_game_date(pd.Timestamp("2026-10-22 01:30", tz=et)) == "2026-10-21"
    )
    assert (
        job.default_game_date(pd.Timestamp("2026-10-22 11:00", tz=et)) == "2026-10-21"
    )
    # The evening itself, once the games have tipped.
    assert (
        job.default_game_date(pd.Timestamp("2026-10-21 23:30", tz=et)) == "2026-10-21"
    )
    # Any zone: Madrid 07:00 is 01:00 ET.
    madrid = pd.Timestamp("2026-10-22 07:00", tz="Europe/Madrid")
    assert job.default_game_date(madrid) == "2026-10-21"


def test_only_intermediate_slots_run():
    slots = [_slot(0), _slot(30), _slot(1080)]
    assert [s.horizon_minutes for s in job.intermediate_slots(slots)] == [30, 1080]


def test_snapshots_still_ahead_are_left_out():
    frame = pd.DataFrame(
        {
            "GAME_ID": ["early", "early", "late", "late"],
            "TIME_TO_MATCH_MIN": [30, 60, 30, 60],
            "SNAPSHOT_PASSED": [True, True, False, True],
        }
    )
    kept = job.keep_passed_snapshots(frame)
    assert kept[["GAME_ID", "TIME_TO_MATCH_MIN"]].values.tolist() == [
        ["early", 30],
        ["early", 60],
        ["late", 60],
    ]


def test_each_slot_reads_its_own_snapshot(monkeypatch):
    frame = pd.DataFrame(
        {
            "GAME_ID": ["g1", "g1"],
            "TIME_TO_MATCH_MIN": [30, 60],
            "SNAPSHOT_PASSED": [True, True],
        }
    )
    monkeypatch.setattr(
        job,
        "SETTINGS",
        SimpleNamespace(
            prediction_model_slots=[_slot(0), _slot(30), _slot(60)],
            s3_aws_profile=None,
            s3_aws_region="eu-west-1",
            s3_bucket="bucket",
            headless=True,
        ),
    )
    monkeypatch.setattr(
        job,
        "past_scheduled_data",
        lambda date: {"scheduled_games": pd.DataFrame({"GAME_ID": ["g1"]})},
    )
    monkeypatch.setattr(job, "tonight_intermediate_frame", lambda *a, **k: frame)
    monkeypatch.setattr(job, "make_s3_client", lambda **_: object())
    calls = []

    def predict(*, slot, frames, latest_input, persist, **_):
        calls.append((slot.horizon_minutes, latest_input, persist))
        assert set(frames) == {"intermediate_line"}
        return frames["intermediate_line"].iloc[:1]

    monkeypatch.setattr(job, "load_registry_model_and_predict", predict)
    monkeypatch.setattr(
        job,
        "upload_model_predictions_snapshot",
        lambda **_: pytest.fail("a dry run uploaded predictions"),
    )
    job.predict_past_snapshots("2026-10-21", persist=False, update_databases=False)
    # The closing slot is left out; no slot reads the latest snapshot.
    assert calls == [(30, False, False), (60, False, False)]
