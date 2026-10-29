"""Closing and intermediate models share pregame inputs, never final quotes."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
from nba_ou.modeling.registry_models import finalize_spec
from nba_ou.modeling.registry_paths import ModelSlot
from nba_ou.prediction import historical_backfill as backfill
from nba_ou.prediction.report_requirement import COVERED_COLUMNS

import scripts.backfill_past_predictions as cli
from tests.prediction import test_historical_backfill as registry_tests
from tests.prediction import test_historical_backfill_tabpfn as tabpfn_tests
from tests.prediction.test_historical_backfill import (
    DAY,
    GAME,
    TIP,
    prediction_frame,
    schedule,
)
from tests.support.historical_predictions import quoted_frame, saved_store

# Reuse the isolated service fixtures while keeping real prediction/recording.
pipeline = registry_tests.pipeline
tabpfn_pipeline = tabpfn_tests.tabpfn_pipeline


def closing_spec(spec):
    return finalize_spec(
        identity=spec.identity.model_copy(
            update={"dataset_type": "closing_line", "horizon_minutes": 0}
        ),
        features=spec.features,
        cleaning=spec.cleaning,
        training=spec.training,
    )


@pytest.mark.parametrize("persist", [False, True])
def test_six_standard_tabpfn_fits_share_four_hour_inputs(
    tabpfn_pipeline, monkeypatch, persist
):
    p = tabpfn_pipeline
    for slot, spec in list(p.specs.items()):
        p.specs[replace(slot, horizon_minutes=0)] = closing_spec(spec)
    p.settings.prediction_model_slots = list(p.specs)
    assert (
        backfill.run_backfill(
            DAY,
            slots=[],
            include_tabpfn=True,
            include_closing=True,
            persist=persist,
            output_dir=p.output,
        )
        == 0
    )
    assert len(p.fits) == 6
    assert {mode for _, _, mode in p.fits} == {"standard"}
    inputs = pd.read_csv(next(p.output.glob("*/*.inputs.csv")))
    outputs = pd.read_csv(next(p.output.glob("*/*.predictions.csv")))
    builds = pd.read_csv(next(p.output.glob("*/*.models.csv")))
    assert set(inputs.input_mode) == {0, 1}
    assert inputs.snapshot_horizon_minutes.eq(240).all()
    assert set(builds.dataset_type) == {1, 2}
    assert set(builds.horizon_minutes) == {0, 240}
    assert len(outputs) == 6 and set(outputs.target) == {1, 2, 3}
    assert all(
        training.train_date_max == pd.Timestamp("2026-10-23")
        for training, _, _ in p.fits
    )
    if persist:
        assert len(p.written) == 1 and len(p.written[0].predictions) == 6
    else:
        assert not p.written and not p.s3.put_order


def test_real_closing_registry_refit_records_four_hour_snapshot(pipeline, monkeypatch):
    p = pipeline
    slot = replace(p.slot, horizon_minutes=0)
    spec = closing_spec(p.spec)
    monkeypatch.setattr(backfill, "resolve_config_spec", lambda **k: spec)
    frame = prediction_frame().assign(
        TIME_TO_MATCH_MIN=240, SNAPSHOT_TS_UTC=TIP - pd.Timedelta(minutes=235)
    )
    frame = quoted_frame(frame)
    for column in COVERED_COLUMNS:
        frame[column] = 1
    monkeypatch.setattr(backfill, "SeasonPartitions", lambda **k: saved_store(frame))
    monkeypatch.setattr(
        backfill,
        "fetch_games",
        lambda *a: pd.DataFrame(
            {"game_id": [GAME], "tipoff_utc": [TIP + pd.Timedelta(minutes=5)]}
        ),
    )
    assert (
        backfill.run_backfill(
            DAY, slots=[slot], include_closing=True, persist=True, output_dir=p.output
        )
        == 0
    )
    tables = p.written[0]
    assert tables.inputs.input_mode.tolist() == [0]
    assert tables.inputs.snapshot_horizon_minutes.tolist() == [240]
    assert tables.inputs.tipoff_at.tolist() == [TIP + pd.Timedelta(minutes=5)]
    assert tables.model_builds.horizon_minutes.tolist() == [0]
    assert tables.model_builds.train_date_max.tolist() == [
        pd.Timestamp("2026-10-23").date()
    ]
    assert len(tables.predictions) == 1


def test_closing_coverage_requires_same_input_timepoint():
    slot = ModelSlot(schema_version="2_5", target="line_error", horizon_minutes=0)
    row = dict(
        game_id=int(GAME),
        spec_id="a",
        schema_version="2_5",
        target=1,
        horizon_minutes=0,
        variant="main",
        kind=1,
        dataset_type=1,
        input_horizon_minutes=240,
        train_date_max="2026-10-23",
    )
    options = dict(input_horizon_minutes=240)
    assert not backfill.missing_games(
        schedule(), pd.DataFrame([row]), slot, "a", **options
    )
    for change in [
        dict(input_horizon_minutes=360),
        dict(input_horizon_minutes=None),
        dict(dataset_type=2),
        dict(train_date_max=DAY),
    ]:
        assert backfill.missing_games(
            schedule(), pd.DataFrame([{**row, **change}]), slot, "a", **options
        ) == {GAME}


def test_action_matrix_groups_closing_and_tabpfn_at_same_timepoint(
    tabpfn_pipeline, monkeypatch, capsys
):
    p = tabpfn_pipeline
    slots = [*p.specs, *(replace(slot, horizon_minutes=0) for slot in p.specs)]
    monkeypatch.setattr(
        cli,
        "SETTINGS",
        SimpleNamespace(prediction_model_slots=slots, s3_models_prefix="models/"),
    )
    monkeypatch.setattr(cli, "backfill_dates", lambda *a: [DAY])
    assert (
        cli.main(["--start-date", DAY, "--include-closing", "--tabpfn", "--job-matrix"])
        == 0
    )
    jobs = json.loads(capsys.readouterr().out)["include"]
    assert len(jobs) == 1 and jobs[0]["horizon"] == 240
    assert len(jobs[0]["slots"].split()) == 6
    assert jobs[0]["tabpfn"] and not jobs[0]["tabpfn_only"]
    assert cli.parse_slot("2_5/line_error/t0000/main").horizon_minutes == 0
