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
    # Every build is read in the slot's turn, by the stubbed predictor.
    monkeypatch.setattr(job, "read_slot_builds", lambda **_: {})
    monkeypatch.setattr(job, "PredictionRunRecorder", _NullRecorder)
    stored = {}

    def persist(recorder, **_):
        for predictions in recorder.outputs:
            stored[predictions["MODEL_NAME"].iloc[0]] = predictions
        return []

    monkeypatch.setattr(job, "persist_prediction_run", persist)

    class _Now:
        @staticmethod
        def now(tz=None):
            return pd.Timestamp("2025-11-03 12:00", tz=tz).to_pydatetime()

    monkeypatch.setattr(job, "datetime", _Now)
    job.predict_nba_games()

    assert built["target_game_ids"] == ["g5"]
    average = stored[HISTORICAL_AVERAGE_LINE_ERROR_BASELINE_MODEL_NAME]
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
    # Every build is read in the slot's turn, by the stubbed predictor.
    monkeypatch.setattr(job, "read_slot_builds", lambda **_: {})
    monkeypatch.setattr(job, "PredictionRunRecorder", _NullRecorder)
    monkeypatch.setattr(
        job,
        "persist_prediction_run",
        lambda *_, **__: pytest.fail("a dry run stored predictions"),
    )
    served = []

    def serve(*, slot, **_):
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


def test_a_run_stores_every_slot_and_baseline_in_the_prediction_store(monkeypatch):
    """The prediction store (Aiven and its S3 archive) is the only place a run
    writes predictions: it gets the slot and both baselines, in one run."""
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
    # Every build is read in the slot's turn, by the stubbed predictor.
    monkeypatch.setattr(job, "read_slot_builds", lambda **_: {})
    monkeypatch.setattr(
        job,
        "load_registry_model_and_predict",
        lambda **_: pd.DataFrame({"GAME_ID": ["g5"], "MODEL_NAME": ["m"]}),
    )
    monkeypatch.setattr(job, "PredictionRunRecorder", _NullRecorder)
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
    job.predict_nba_games()

    assert len(stored["outputs"]) == 3
    assert stored["partial"] is False


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
    # Every build is read in the slot's turn, by the stubbed predictor.
    monkeypatch.setattr(job, "read_slot_builds", lambda **_: {})
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


def _side_by_side_job(monkeypatch, *, parallel_builds: bool, order: list):
    """A run with a closing and an intermediate slot and one filed game."""
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
            parallel_builds=parallel_builds,
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
        lambda **_: {"scheduled_games": pd.DataFrame({"GAME_ID": ["g5"]})},
    )
    tonight = pd.DataFrame(
        {
            "GAME_ID": ["g5"],
            "GAME_DATE": ["2025-11-03"],
            "TOTAL_POINTS": [np.nan],
        }
    )

    def closing(**_):
        order.append("closing")
        return tonight.copy(), _history()

    monkeypatch.setattr(job, "create_df_to_predict", closing)
    intermediate = pd.DataFrame(
        {"GAME_ID": ["g5"], "TIME_TO_MATCH_MIN": [60], "SNAPSHOT_PASSED": [False]}
    )

    def intermediate_rows(*_, **__):
        order.append("intermediate")
        return intermediate

    monkeypatch.setattr(job, "tonight_intermediate_frame", intermediate_rows)
    monkeypatch.setattr(job, "side_by_side_reason", lambda: None)
    monkeypatch.setattr(job, "make_s3_client", lambda **_: object())
    monkeypatch.setattr(job, "read_slot_builds", lambda **_: {})
    monkeypatch.setattr(job, "PredictionRunRecorder", _NullRecorder)
    monkeypatch.setattr(job, "load_baseline_predictions_for_nba_games", lambda **_: {})
    served = {}

    def serve(*, slot, frames, df, **_):
        served[slot.horizon_minutes] = (sorted(frames), df["GAME_ID"].tolist())
        return pd.DataFrame({"GAME_ID": ["g5"]})

    monkeypatch.setattr(job, "load_registry_model_and_predict", serve)

    class _Now:
        @staticmethod
        def now(tz=None):
            return pd.Timestamp("2025-11-03 12:00", tz=tz).to_pydatetime()

    monkeypatch.setattr(job, "datetime", _Now)
    return served


def test_closing_rows_build_beside_the_intermediate_rows(monkeypatch, capsys):
    """The side build starts before the intermediate rows and is collected
    after them; every slot then sees both frames, as in a sequential run."""
    from nba_ou.prediction.side_build import SideResult

    order = []
    served = _side_by_side_job(monkeypatch, parallel_builds=True, order=order)

    class _InlineSide:
        """The side process, run in this one when collected."""

        def __init__(self, fn, args, kwargs, label):
            self.call, self.label = (fn, args, kwargs), label
            self.gate = SimpleNamespace(cold_cause=None, waited=0.0)
            self.released = False
            order.append("side started")

        @classmethod
        def start(cls, fn, *args, current_season, label, own_label, **kwargs):
            assert current_season == 2025
            assert own_label == "intermediate rows"
            return cls(fn, args, kwargs, label)

        def release_gate(self):
            self.released = True

        def result(self):
            assert self.released
            fn, args, kwargs = self.call
            return SideResult(fn(*args, **kwargs), 1.0, None, 0.0)

    monkeypatch.setattr(job, "SideBuild", _InlineSide)
    job.predict_nba_games(persist=False)

    assert order == ["side started", "intermediate", "closing"]
    both = ["closing_line", "intermediate_line"]
    assert served == {0: (both, ["g5"]), 60: (both, ["g5"])}
    out = capsys.readouterr().out
    assert "[time] closing rows (side process)" in out
    assert "[time] closing and intermediate rows, side by side" in out


def test_turned_off_the_builds_run_one_after_the_other(monkeypatch, capsys):
    order = []
    served = _side_by_side_job(monkeypatch, parallel_builds=False, order=order)
    monkeypatch.setattr(
        job.SideBuild, "start", lambda *_, **__: pytest.fail("side build started")
    )
    job.predict_nba_games(persist=False)

    assert order == ["closing", "intermediate"]
    assert served[0] == (["closing_line", "intermediate_line"], ["g5"])
    assert "one after the other: turned off" in capsys.readouterr().out


# --- TabPFN jobs after the predictions ----------------------------------------


def _tabpfn_job(**changes):
    from nba_ou.prediction.tabpfn_run import TabPFNJob

    return TabPFNJob(
        **{
            "dataset_type": "intermediate_line",
            "horizon": 720,
            "as_of": pd.Timestamp("2025-11-03 11:00", tz="UTC"),
            "snapshot_games": ("g5",),
            **changes,
        }
    )


def _with_tabpfn(monkeypatch, order: list, *, failing: set[int] = frozenset()):
    """The side-by-side job, sequential, with the store and the TabPFN run's
    parts faked: what is under test is where and on what each job runs."""
    from unittest.mock import Mock

    from nba_ou.prediction.tabpfn_run import JobOutcome

    _side_by_side_job(monkeypatch, parallel_builds=False, order=order)
    scheduled = {"scheduled_games": pd.DataFrame({"GAME_ID": ["g5"]})}
    fetch = Mock(return_value=scheduled)
    monkeypatch.setattr(job, "get_all_info_for_scheduled_games", fetch)

    def stored(*_, **__):
        order.append("stored")
        return []

    monkeypatch.setattr(job, "persist_prediction_run", stored)
    calls = {"rows": [], "fits": []}

    def load_fit_inputs(tabpfn_job, **_):
        if tabpfn_job.horizon in failing:
            raise RuntimeError("partitions too old")
        return SimpleNamespace(specs=[f"spec {tabpfn_job.horizon}"])

    def tonight_rows(tabpfn_job, data, specs, **_):
        calls["rows"].append((tabpfn_job, data))
        return SimpleNamespace(df=pd.DataFrame())

    def fit_and_store(tabpfn_job, inputs, rows, data, **kwargs):
        order.append(f"TabPFN T-{tabpfn_job.horizon}")
        calls["fits"].append(kwargs["persist"])
        return JobOutcome(
            predicted=[f"2_5/line_error/t{tabpfn_job.horizon:04d}/tabpfn_v3_5"],
            failures=[],
            n_fits=1,
        )

    monkeypatch.setattr(job, "load_fit_inputs", load_fit_inputs)
    monkeypatch.setattr(job, "tonight_rows", tonight_rows)
    monkeypatch.setattr(job, "fit_and_store", fit_and_store)
    return fetch, scheduled, calls


def test_tabpfn_jobs_run_once_the_predictions_are_stored(monkeypatch, capsys):
    """On the run's own schedule and reports, at each job's snapshot."""
    order = []
    fetch, scheduled, calls = _with_tabpfn(monkeypatch, order)
    jobs = [_tabpfn_job(), _tabpfn_job(horizon=480)]
    job.predict_nba_games(persist=True, tabpfn_jobs=jobs)

    assert order[-3:] == ["stored", "TabPFN T-720", "TabPFN T-480"]
    fetch.assert_called_once()
    assert [(got, data) for got, data in calls["rows"]] == [
        (jobs[0], scheduled),
        (jobs[1], scheduled),
    ]
    assert calls["fits"] == [True, True]
    assert "2 TabPFN model(s) predicted" in capsys.readouterr().out


def test_refreshed_tabpfn_jobs_fetch_again_and_predict_at_this_moment(monkeypatch):
    order = []
    fetch, _, calls = _with_tabpfn(monkeypatch, order)
    job.predict_nba_games(
        persist=False, tabpfn_jobs=[_tabpfn_job()], tabpfn_refresh=True
    )

    assert fetch.call_count == 2
    assert fetch.call_args.kwargs["dataset_type"] == "intermediate_line"
    ((refreshed, _),) = calls["rows"]
    assert refreshed.as_of is None and refreshed.snapshot_games is None
    assert refreshed.horizon == 720
    assert calls["fits"] == [False]


def test_a_failed_tabpfn_job_costs_only_itself_and_fails_the_run(monkeypatch, capsys):
    order = []
    _with_tabpfn(monkeypatch, order, failing={720})
    with pytest.raises(SystemExit) as exc:
        job.predict_nba_games(
            persist=True, tabpfn_jobs=[_tabpfn_job(), _tabpfn_job(horizon=480)]
        )
    assert exc.value.code == 1
    # The predictions were stored before; the next job still ran.
    assert order[-2:] == ["stored", "TabPFN T-480"]
    assert "TabPFN t0720: RuntimeError: partitions too old" in capsys.readouterr().out


def test_a_run_without_tabpfn_jobs_fits_nothing(monkeypatch):
    order = []
    _, _, calls = _with_tabpfn(monkeypatch, order)
    job.predict_nba_games(persist=True)
    assert calls == {"rows": [], "fits": []}
    assert order[-1] == "stored"
