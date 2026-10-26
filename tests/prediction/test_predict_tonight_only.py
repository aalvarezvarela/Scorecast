"""The prediction job on a tonight-only feature frame.

``create_df_to_predict(target_game_ids=...)`` returns tonight's rows alone, so
nothing downstream may read history from that frame: the historical-average
baseline takes it from the build's layer inputs instead.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from nba_ou.prediction import baseline_predictions
from nba_ou.prediction.baseline_predictions import (
    HISTORICAL_AVERAGE_LINE_ERROR_BASELINE_MODEL_NAME,
    baseline_history_window,
)
from nba_ou.prediction.line_source import LineSource

import scripts.predict_nba_games as job


class _NullRecorder:
    """The prediction store's recorder, accepting everything (it has its own
    tests in test_prediction_store.py)."""

    def __init__(self, **_):
        self.outputs = []

    def set_schedule(self, *_):
        pass

    def add_frame(self, *_, **__):
        pass

    def record_slot(self, *_, **__):
        pass

    def add_predictions(self, predictions, **_):
        self.outputs.append(predictions)

    def add_baseline(self, name, predictions, **_):
        self.outputs.append(predictions)


def _sbr_lines(game_date, **_):
    """The feed check of a day SBR is fresh: no network."""
    return LineSource(game_date)


def _history() -> pd.DataFrame:
    dates = pd.to_datetime(
        [
            "2024-04-10",
            "2024-10-25",
            "2025-03-01",
            "2025-10-22",
            "2025-11-02",
            "2025-11-03",
        ]
    )
    return pd.DataFrame(
        {
            "GAME_ID": [f"g{i}" for i in range(len(dates))],
            "GAME_DATE": dates,
            "TOTAL_POINTS": [200.0, 210.0, 220.0, 230.0, 240.0, np.nan],
        }
    )


def test_window_is_this_season_and_the_one_before_until_tonight():
    window = baseline_history_window(_history(), "2025-11-03")
    assert window["GAME_ID"].tolist() == ["g1", "g2", "g3", "g4"]
    # On opening night only last season remains.
    opener = baseline_history_window(_history(), "2025-10-22")
    assert opener["GAME_ID"].tolist() == ["g1", "g2"]


def test_baselines_read_history_from_the_layer_inputs(monkeypatch):
    tonight = pd.DataFrame(
        {
            "GAME_ID": ["g5"],
            "GAME_DATE": [pd.Timestamp("2025-11-03")],
            "GAME_TIME": [pd.Timestamp("2025-11-03 19:00", tz="US/Eastern")],
            "ODDS_TOTAL_LINE_bet365": [225.5],
            "TOTAL_POINTS": [np.nan],
        }
    )
    history = _history()
    monkeypatch.setattr(
        job,
        "SETTINGS",
        SimpleNamespace(
            prediction_model_slots=[],
            intermediate_input_mode="own",
            report_required_max_horizon=None,
            headless=True,
            nba_injury_reports_url="",
            s3_aws_profile=None,
            s3_aws_region="eu-west-1",
            s3_bucket="bucket",
        ),
    )
    monkeypatch.setattr(job, "update_all_databases", lambda **_: None)
    monkeypatch.setattr(job, "check_line_feed", _sbr_lines)
    monkeypatch.setattr(
        job,
        "get_all_info_for_scheduled_games",
        lambda **_: {"scheduled_games": tonight[["GAME_ID"]]},
    )
    built = {}

    def build(**kwargs):
        built.update(kwargs)
        return tonight.copy(), history

    monkeypatch.setattr(job, "create_df_to_predict", build)
    monkeypatch.setattr(job, "make_s3_client", lambda **_: object())
    monkeypatch.setattr(job, "PredictionRunRecorder", _NullRecorder)
    monkeypatch.setattr(job, "persist_prediction_run", lambda *_, **__: [])
    uploaded = {}

    def upload(predictions):
        uploaded[predictions["MODEL_NAME"].iloc[0]] = predictions

    monkeypatch.setattr(job, "upload_predictions_to_postgre", upload)
    monkeypatch.setattr(
        baseline_predictions,
        "upload_predictions_to_postgre",
        lambda df: pytest.fail("baselines stored from inside the predictor"),
    )

    class _Now:
        @staticmethod
        def now(tz=None):
            return pd.Timestamp("2025-11-03 12:00", tz=tz).to_pydatetime()

    monkeypatch.setattr(job, "datetime", _Now)
    job.predict_nba_games()

    assert built["target_game_ids"] == ["g5"]
    average = uploaded[HISTORICAL_AVERAGE_LINE_ERROR_BASELINE_MODEL_NAME]
    assert average["PRED_TOTAL_POINTS"].tolist() == [
        np.mean([210.0, 220.0, 230.0, 240.0])
    ]


def test_a_failing_slot_does_not_stop_the_others(monkeypatch, capsys):
    """Every slot and the baselines get their turn; the run then exits 1."""
    from nba_ou.modeling.registry_paths import ModelSlot

    tonight = pd.DataFrame(
        {
            "GAME_ID": ["g5"],
            "GAME_DATE": [pd.Timestamp("2025-11-03")],
            "GAME_TIME": [pd.Timestamp("2025-11-03 19:00", tz="US/Eastern")],
            "ODDS_TOTAL_LINE_bet365": [225.5],
            "TOTAL_POINTS": [np.nan],
        }
    )
    slots = [
        ModelSlot(
            schema_version="2_5", target=target, horizon_minutes=0, variant="main"
        )
        for target in ("line_error", "total_points")
    ]
    monkeypatch.setattr(
        job,
        "SETTINGS",
        SimpleNamespace(
            prediction_model_slots=slots,
            intermediate_input_mode="own",
            report_required_max_horizon=None,
            headless=True,
            nba_injury_reports_url="",
            s3_aws_profile=None,
            s3_aws_region="eu-west-1",
            s3_bucket="bucket",
        ),
    )
    monkeypatch.setattr(job, "update_all_databases", lambda **_: None)
    monkeypatch.setattr(job, "check_line_feed", _sbr_lines)
    monkeypatch.setattr(
        job,
        "get_all_info_for_scheduled_games",
        lambda **_: {"scheduled_games": tonight[["GAME_ID"]]},
    )
    monkeypatch.setattr(
        job, "create_df_to_predict", lambda **_: (tonight.copy(), _history())
    )
    monkeypatch.setattr(job, "make_s3_client", lambda **_: object())
    monkeypatch.setattr(job, "PredictionRunRecorder", _NullRecorder)
    monkeypatch.setattr(
        job,
        "persist_prediction_run",
        lambda *_, **__: pytest.fail("a dry run stored predictions"),
    )
    monkeypatch.setattr(
        job,
        "upload_predictions_to_postgre",
        lambda df: pytest.fail("a dry run stored predictions in Supabase"),
    )
    monkeypatch.setattr(
        baseline_predictions,
        "upload_predictions_to_postgre",
        lambda df: pytest.fail("a dry run stored baselines"),
    )
    served = []

    def serve(*, slot, persist, **_):
        assert persist is False
        served.append(slot.target)
        if slot.target == "line_error":
            raise RuntimeError("booster unreadable")
        return pd.DataFrame({"GAME_ID": ["g5"]})

    monkeypatch.setattr(job, "load_registry_model_and_predict", serve)

    class _Now:
        @staticmethod
        def now(tz=None):
            return pd.Timestamp("2025-11-03 12:00", tz=tz).to_pydatetime()

    monkeypatch.setattr(job, "datetime", _Now)
    with pytest.raises(SystemExit) as exit_info:
        job.predict_nba_games(persist=False)

    assert exit_info.value.code == 1
    assert served == ["line_error", "total_points"]
    out = capsys.readouterr().out
    assert "1 slot(s) predicted" in out
    assert "booster unreadable" in out
    assert "Baseline predictions generated" in out


def test_a_failing_supabase_write_does_not_cost_the_prediction_store(
    monkeypatch, capsys
):
    """The old table and the store are written independently: a Supabase
    failure is reported, and the store still gets every slot's output."""
    from nba_ou.modeling.registry_paths import ModelSlot

    tonight = pd.DataFrame(
        {
            "GAME_ID": ["g5"],
            "GAME_DATE": [pd.Timestamp("2025-11-03")],
            "GAME_TIME": [pd.Timestamp("2025-11-03 19:00", tz="US/Eastern")],
            "ODDS_TOTAL_LINE_bet365": [225.5],
            "TOTAL_POINTS": [np.nan],
        }
    )
    monkeypatch.setattr(
        job,
        "SETTINGS",
        SimpleNamespace(
            prediction_model_slots=[
                ModelSlot(
                    schema_version="2_5",
                    target="line_error",
                    horizon_minutes=0,
                    variant="main",
                )
            ],
            intermediate_input_mode="own",
            report_required_max_horizon=None,
            headless=True,
            nba_injury_reports_url="",
            s3_aws_profile=None,
            s3_aws_region="eu-west-1",
            s3_bucket="bucket",
        ),
    )
    monkeypatch.setattr(job, "update_all_databases", lambda **_: None)
    monkeypatch.setattr(job, "check_line_feed", _sbr_lines)
    monkeypatch.setattr(
        job,
        "get_all_info_for_scheduled_games",
        lambda **_: {"scheduled_games": tonight[["GAME_ID"]]},
    )
    monkeypatch.setattr(
        job, "create_df_to_predict", lambda **_: (tonight.copy(), _history())
    )
    monkeypatch.setattr(job, "make_s3_client", lambda **_: object())
    monkeypatch.setattr(
        job,
        "load_registry_model_and_predict",
        lambda *, persist, **_: pd.DataFrame(
            {"GAME_ID": ["g5"], "MODEL_NAME": ["m"], "persisted_inside": [persist]}
        ),
    )
    monkeypatch.setattr(job, "PredictionRunRecorder", _NullRecorder)

    def supabase_down(predictions):
        raise ConnectionError("supabase down")

    monkeypatch.setattr(job, "upload_predictions_to_postgre", supabase_down)
    stored = {}

    def persist(recorder, *, partial, **_):
        stored["outputs"] = recorder.outputs
        stored["partial"] = partial
        return []

    monkeypatch.setattr(job, "persist_prediction_run", persist)

    class _Now:
        @staticmethod
        def now(tz=None):
            return pd.Timestamp("2025-11-03 12:00", tz=tz).to_pydatetime()

    monkeypatch.setattr(job, "datetime", _Now)
    with pytest.raises(SystemExit) as exit_info:
        job.predict_nba_games()

    assert exit_info.value.code == 1
    # The slot and both baselines reached the store, the slot without having
    # been written to Supabase from inside the predictor.
    assert len(stored["outputs"]) == 3
    assert stored["outputs"][0]["persisted_inside"].tolist() == [False]
    # Supabase is not the store: its failure does not make the run partial.
    assert stored["partial"] is False
    assert "supabase down" in capsys.readouterr().out


def test_no_filed_report_skips_the_closing_rows_not_the_run(monkeypatch, capsys):
    """Before any team has filed, no closing row is built: the closing slots
    and the baselines skip, and the intermediate slots still predict."""
    from nba_ou.modeling.registry_paths import ModelSlot

    slots = [
        ModelSlot(
            schema_version="2_5",
            target="line_error",
            horizon_minutes=horizon,
            variant="main",
        )
        for horizon in (0, 60)
    ]
    monkeypatch.setattr(
        job,
        "SETTINGS",
        SimpleNamespace(
            prediction_model_slots=slots,
            intermediate_input_mode="own",
            report_required_max_horizon=300,
            headless=True,
            nba_injury_reports_url="",
            s3_aws_profile=None,
            s3_aws_region="eu-west-1",
            s3_bucket="bucket",
        ),
    )
    monkeypatch.setattr(job, "update_all_databases", lambda **_: None)
    monkeypatch.setattr(job, "check_line_feed", _sbr_lines)
    monkeypatch.setattr(
        job,
        "get_all_info_for_scheduled_games",
        lambda **_: {
            "scheduled_games": pd.DataFrame({"GAME_ID": ["g5"]}),
            "games_not_updated": ["g5"],
        },
    )
    monkeypatch.setattr(
        job,
        "create_df_to_predict",
        lambda **_: pytest.fail("closing rows built with no report filed"),
    )
    intermediate = pd.DataFrame(
        {"GAME_ID": ["g5"], "TIME_TO_MATCH_MIN": [60], "SNAPSHOT_PASSED": [False]}
    )
    monkeypatch.setattr(
        job, "tonight_intermediate_frame", lambda *_, **__: intermediate
    )
    monkeypatch.setattr(job, "make_s3_client", lambda **_: object())
    monkeypatch.setattr(job, "PredictionRunRecorder", _NullRecorder)
    monkeypatch.setattr(
        job,
        "load_baseline_predictions_for_nba_games",
        lambda **_: pytest.fail("baselines run without closing rows"),
    )
    served = {}

    def serve(*, slot, frames, report_required_max_horizon, **_):
        served[slot.horizon_minutes] = (sorted(frames), report_required_max_horizon)
        if "closing_line" not in frames and slot.horizon_minutes == 0:
            return pd.DataFrame()
        return pd.DataFrame({"GAME_ID": ["g5"]})

    monkeypatch.setattr(job, "load_registry_model_and_predict", serve)

    class _Now:
        @staticmethod
        def now(tz=None):
            return pd.Timestamp("2025-11-03 12:00", tz=tz).to_pydatetime()

    monkeypatch.setattr(job, "datetime", _Now)
    job.predict_nba_games(persist=False)

    assert served == {0: (["intermediate_line"], 300), 60: (["intermediate_line"], 300)}
    out = capsys.readouterr().out
    assert "1 slot(s) predicted" in out
    assert "baselines skipped" in out
    assert "0 failure(s)" in out
