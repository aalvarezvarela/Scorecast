"""Daily baselines replay pregame lines without learning the evaluated results."""

from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from nba_ou.config.odds_columns import total_line_col
from nba_ou.modeling.registry_paths import ModelSlot
from nba_ou.postgre_db.predictions_aiven import schema
from nba_ou.prediction import baseline_predictions as baseline

import scripts.predict_past_snapshots as job

DAY = "2026-10-21"
GAME_IDS = ["0022600001", "0022600002"]


def _schedule():
    return pd.DataFrame(
        {
            "GAME_ID": GAME_IDS,
            "GAME_TIME": pd.to_datetime(["2026-10-21T23:00Z", "2026-10-22T02:00Z"]),
            "HOME_TEAM_ID": [1610612738, 1610612747],
            "VISITOR_TEAM_ID": [1610612752, 1610612744],
        }
    )


def _frame():
    rows = []
    for game in _schedule().itertuples():
        for horizon in (0, 30):
            rows.append(
                {
                    "GAME_ID": game.GAME_ID,
                    "GAME_DATE": DAY,
                    "GAME_TIME": game.GAME_TIME,
                    "TIME_TO_MATCH_MIN": horizon,
                    "SNAPSHOT_TS_UTC": game.GAME_TIME - pd.Timedelta(minutes=horizon),
                    "SNAPSHOT_PASSED": True,
                    total_line_col(): 225.0 if horizon == 0 else 220.0,
                    "ODDS_SNAP_TOT_BET365_HAS_QUOTE": 1,
                    "ODDS_SNAP_TOT_BET365_RAW_LINE": 225.5,
                    "ODDS_SNAP_TOT_BET365_PRICE_LEFT": -110,
                    "ODDS_SNAP_TOT_BET365_PRICE_RIGHT": -110,
                    "LINE_SOURCE": "sbr",
                }
            )
    return pd.DataFrame(rows)


def _raw_history():
    rows = []
    for game, date, points in (
        ("0022400001", "2024-11-01", (100, 105)),  # outside the two-season window
        ("0022500001", "2025-11-01", (110, 120)),
        ("0022600003", "2026-10-20", (120, 130)),
        (GAME_IDS[0], DAY, (150, 150)),  # the evaluated result is already stored
        ("0022600004", "2026-10-22", (160, 160)),
        ("0012600001", "2026-10-10", (170, 170)),  # preseason
        ("0022600005", "2026-10-19", (110, None)),  # incomplete result
    ):
        for team, pts in enumerate(points, 1):
            rows.append(
                {
                    "game_id": game,
                    "game_date": date,
                    "team_id": team,
                    "pts": pts,
                    "min": 240,
                }
            )
    rows.append(rows[2].copy())  # repeated team must not increase its game's total
    return pd.DataFrame(rows)


def test_average_history_excludes_current_future_exhibition_and_incomplete_games(
    monkeypatch,
):
    load = Mock(return_value=_raw_history())
    monkeypatch.setattr(baseline, "load_games_from_db", load)
    history = baseline.load_baseline_history(DAY)
    load.assert_called_once_with(seasons=["2025-26", "2026-27"])
    assert history["GAME_ID"].tolist() == ["0022500001", "0022600003"]
    assert history["TOTAL_POINTS"].tolist() == [230.0, 250.0]
    assert history["TOTAL_POINTS"].mean() == 240


def test_seeded_random_picks_survive_reordering_and_partial_coverage():
    frame = _frame().query("TIME_TO_MATCH_MIN == 0")
    kwargs = {"random_seed": 20261021}
    full = baseline.build_random_line_error_baseline_predictions(frame, **kwargs)
    reordered = baseline.build_random_line_error_baseline_predictions(
        frame.iloc[::-1], **kwargs
    )
    partial = baseline.build_random_line_error_baseline_predictions(
        frame.iloc[1:], **kwargs
    )
    expected = full.set_index("GAME_ID")["PRED_PICK"]
    assert reordered.set_index("GAME_ID")["PRED_PICK"].to_dict() == expected.to_dict()
    assert partial["PRED_PICK"].iloc[0] == expected.loc[GAME_IDS[1]]
    assert set(full["PRED_LINE_ERROR"]) <= {-1, 1}


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.setattr(
        job,
        "SETTINGS",
        SimpleNamespace(
            prediction_model_slots=[],
            report_required_max_horizon=None,
            s3_aws_profile=None,
            s3_aws_region="eu-west-1",
            s3_bucket="bucket",
        ),
    )
    monkeypatch.setattr(
        job, "past_scheduled_data", lambda date: {"scheduled_games": _schedule()}
    )
    monkeypatch.setattr(job, "tonight_intermediate_frame", lambda *a, **kw: _frame())
    monkeypatch.setattr(job, "make_s3_client", lambda **kw: object())
    monkeypatch.setattr(job, "read_slot_builds", lambda **kw: {})
    monkeypatch.setattr(baseline, "load_games_from_db", lambda **kw: _raw_history())
    stored = []

    def store(recorder, *, partial, **kw):
        tables = recorder.tables(
            status=schema.RUN_STATUS_PARTIAL if partial else schema.RUN_STATUS_OK
        )
        stored.append((recorder, tables))
        return []

    monkeypatch.setattr(job, "persist_prediction_run", store)
    return stored


def test_final_run_stores_two_baselines_for_every_game_using_pregame_lines(replay):
    job.predict_past_snapshots(DAY, update_databases=False)
    ((recorder, tables),) = replay
    assert tables.run["workflow"] == schema.WORKFLOW_PAST_SNAPSHOTS
    assert tables.run["status"] == schema.RUN_STATUS_OK
    assert len(tables.predictions) == 4
    assert len(tables.inputs) == 2
    assert set(tables.inputs["game_id"]) == {int(game) for game in GAME_IDS}
    assert set(tables.model_builds["kind"]) == {schema.KIND_BASELINE}
    assert set(tables.model_builds["variant"]) == {
        baseline.RANDOM_LINE_ERROR_BASELINE_MODEL_NAME,
        baseline.HISTORICAL_AVERAGE_LINE_ERROR_BASELINE_MODEL_NAME,
    }
    assert set(tables.inputs["total_model_line"]) == {225.0}
    assert set(tables.inputs["total_bet_line_x2"]) == {451}
    assert tables.predictions["time_to_tipoff_seconds"].lt(0).all()
    average = tables.model_builds.loc[
        tables.model_builds["variant"].eq(
            baseline.HISTORICAL_AVERAGE_LINE_ERROR_BASELINE_MODEL_NAME
        )
    ]
    assert average["fit_id"].tolist() == ["1.0-20261020"]
    assert set(recorder.outputs[1]["prediction_value"]) == {15.0}  # 240 - 225


@pytest.mark.parametrize("history_fails", [False, True])
def test_model_snapshots_and_daily_baselines_share_the_final_run(
    replay, monkeypatch, history_fails
):
    job.SETTINGS.prediction_model_slots = [
        ModelSlot(schema_version="2_5", target="line_error", horizon_minutes=30)
    ]
    if history_fails:
        monkeypatch.setattr(baseline, "load_games_from_db", lambda **kw: None)

    def predict(**kwargs):
        assert kwargs["latest_input"] is False
        frame = kwargs["frames"]["intermediate_line"].query("TIME_TO_MATCH_MIN == 30")
        return pd.DataFrame(
            {
                "GAME_ID": frame["GAME_ID"].to_numpy(),
                "PRED_LINE_ERROR": 2.0,
                "PRED_PICK": "OVER",
                "TOTAL_OVER_UNDER_LINE": frame[total_line_col()].to_numpy(),
                "PREDICTED_AT_UTC": pd.Timestamp.now(tz="UTC"),
                "MODEL_TARGET": "line_error",
                "MODEL_DATASET_TYPE": "intermediate_line",
                "MODEL_HORIZON_MINUTES": 30,
                "MODEL_SCHEMA_VERSION": "2_5",
                "MODEL_VARIANT": "main",
                "SPEC_ID": "test-spec",
                "FIT_ID": "test-fit",
                "INPUT_MODE": "own",
                "TIME_TO_MATCH_MIN": 30,
            }
        )

    monkeypatch.setattr(job, "load_registry_model_and_predict", predict)
    if history_fails:
        with pytest.raises(SystemExit):
            job.predict_past_snapshots(DAY, update_databases=False)
    else:
        job.predict_past_snapshots(DAY, update_databases=False)
    ((_, tables),) = replay
    assert set(tables.model_builds["kind"]) == {
        schema.KIND_REGISTRY,
        schema.KIND_BASELINE,
    }
    assert len(tables.predictions) == (4 if history_fails else 6)
    assert len(tables.inputs) == 4
    model_inputs = tables.inputs.loc[
        tables.inputs["input_mode"].eq(schema.INPUT_MODE_OWN)
    ]
    assert set(model_inputs["snapshot_horizon_minutes"]) == {30}
    assert set(model_inputs["total_model_line"]) == {220.0}
    baseline_inputs = tables.inputs.loc[
        tables.inputs["input_mode"].eq(schema.INPUT_MODE_CLOSING)
    ]
    assert set(baseline_inputs["total_model_line"]) == {225.0}


def test_a_failed_history_read_preserves_random_picks(replay, monkeypatch):
    monkeypatch.setattr(baseline, "load_games_from_db", lambda **kw: None)
    with pytest.raises(SystemExit):
        job.predict_past_snapshots(DAY, update_databases=False)
    ((_, tables),) = replay
    assert tables.run["status"] == schema.RUN_STATUS_PARTIAL
    assert tables.model_builds["variant"].tolist() == [
        baseline.RANDOM_LINE_ERROR_BASELINE_MODEL_NAME
    ]
    assert len(tables.predictions) == 2


def test_missing_closing_coverage_is_reported_without_fabricating_a_line(
    replay, monkeypatch
):
    frame = _frame()
    frame.loc[frame["GAME_ID"].eq(GAME_IDS[1]), total_line_col()] = float("nan")
    monkeypatch.setattr(job, "tonight_intermediate_frame", lambda *a, **kw: frame)
    with pytest.raises(SystemExit):
        job.predict_past_snapshots(DAY, update_databases=False)
    ((_, tables),) = replay
    assert tables.run["status"] == schema.RUN_STATUS_PARTIAL
    assert len(tables.predictions) == 2
    assert set(tables.inputs["game_id"]) == {int(GAME_IDS[0])}


def test_final_baselines_respect_no_persist(replay):
    job.predict_past_snapshots(DAY, persist=False, update_databases=False)
    assert replay == []
