"""The day's last run: every intermediate slot on its own, past snapshot."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from nba_ou.modeling.registry_paths import ModelSlot

import scripts.predict_past_snapshots as job


def _slot(horizon):
    return ModelSlot(schema_version="2_5", target="line_error", horizon_minutes=horizon)


@pytest.mark.parametrize("unavailable", [False, True])
def test_recreated_rows_fetch_day_openers_without_waiting_for_historical_insert(
    monkeypatch,
    unavailable,
    capsys,
):
    scheduled = pd.DataFrame({"GAME_ID": ["0022600145"]})
    day = pd.DataFrame(
        {"game_id": ["0022600145"], "total_consensus_opener_line_over": [238.5]}
    )
    monkeypatch.setattr(job, "replay_scheduled_games", lambda date: scheduled)
    monkeypatch.setattr(
        job, "process_scheduled_referee_assignments", lambda frame: pd.DataFrame()
    )
    monkeypatch.setattr(job, "load_live_rosters", lambda frame: pd.DataFrame())

    def odds(date, games):
        assert date == "2026-10-29" and games is scheduled
        if unavailable:
            raise OSError("SBR unavailable")
        return day

    monkeypatch.setattr(job, "get_sportsbook_opener_data", odds)
    result = job.past_scheduled_data("2026-10-29")
    if unavailable:
        assert result["df_odds_sportsbook_opener_scheduled"].empty
        assert "leaving them missing" in capsys.readouterr().out
    else:
        assert result["df_odds_sportsbook_opener_scheduled"] is day


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


class _NullRecorder:
    """The prediction store's recorder, accepting everything (it has its own
    tests in test_prediction_store.py)."""

    def __init__(self, **_):
        self.frames = {}

    def set_schedule(self, *_):
        pass

    def add_frame(self, name, frame, **_):
        self.frames[name] = frame

    def record_slot(self, *_, **__):
        pass

    def add_predictions(self, *_, **__):
        return None


@pytest.mark.parametrize("trained_on_evaluated_date", [False, True])
def test_each_slot_reads_its_own_snapshot(monkeypatch, trained_on_evaluated_date):
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
            report_required_max_horizon=None,
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
    live = SimpleNamespace(line_history=SimpleNamespace(ticks=pd.DataFrame()))
    monkeypatch.setattr(job, "fetch_live_game_day", lambda *a, **kw: live)
    monkeypatch.setattr(job, "make_s3_client", lambda **_: object())
    # Every build is read in the slot's turn, by the stubbed predictor.
    monkeypatch.setattr(
        job,
        "read_slot_builds",
        lambda **_: {
            _slot(0): (
                SimpleNamespace(fit=SimpleNamespace(train_date_max="2026-10-21"))
                if trained_on_evaluated_date
                else None
            )
        },
    )
    calls = []

    def predict(*, slot, frames, latest_input, **_):
        calls.append((slot.horizon_minutes, latest_input))
        if slot.horizon_minutes:
            assert set(frames) == {"intermediate_line"}
            return frames["intermediate_line"].iloc[:1]
        assert frames["closing_line"].TIME_TO_MATCH_MIN.tolist() == [480, 360, 240, 0]
        assert frames["closing_line"].INPUT_MODE.eq("replay").all()
        return frames["closing_line"]

    monkeypatch.setattr(job, "load_registry_model_and_predict", predict)
    monkeypatch.setattr(job, "PredictionRunRecorder", _NullRecorder)
    monkeypatch.setattr(job, "add_past_baselines", lambda *a: [])
    monkeypatch.setattr(
        job,
        "closing_replay_frame",
        lambda *a, **kw: pd.DataFrame(
            {
                "GAME_ID": ["g1"] * 4,
                "TIME_TO_MATCH_MIN": [480, 360, 240, 0],
                "INPUT_MODE": "replay",
            }
        ),
    )
    monkeypatch.setattr(
        job,
        "persist_prediction_run",
        lambda *_, **__: pytest.fail("a dry run stored predictions"),
    )
    if trained_on_evaluated_date:
        with pytest.raises(SystemExit) as exc:
            job.predict_past_snapshots(
                "2026-10-21", persist=False, update_databases=False
            )
        assert exc.value.code == 1
        assert calls == [(30, False), (60, False)]
    else:
        job.predict_past_snapshots("2026-10-21", persist=False, update_databases=False)
        # No intermediate slot reads the latest snapshot.
        assert calls == [(30, False), (60, False), (0, False)]
