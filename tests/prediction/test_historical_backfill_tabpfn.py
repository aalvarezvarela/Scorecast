"""A single standard TabPFN horizon, all targets, with past-only context."""

import json
import os
import subprocess
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml
from nba_ou.config.odds_columns import spread_line_home_col, total_line_col
from nba_ou.modeling.refit import TrainingFrame
from nba_ou.modeling.registry_models import finalize_spec
from nba_ou.modeling.registry_paths import ModelSlot, ModelTarget
from nba_ou.postgre_db.predictions_aiven.schema import KIND_TABPFN
from nba_ou.prediction import historical_backfill as backfill
from nba_ou.prediction.prediction_tabpfn_client import FittedTabPFN, model_variant
from nba_ou.prediction.report_requirement import COVERED_COLUMNS

import scripts.backfill_past_predictions as cli
from tests.prediction.test_historical_backfill import (
    DAY,
    GAME,
    NOW,
    TIP,
    prediction_frame,
    schedule,
)
from tests.support.fake_s3 import FakeS3Client
from tests.support.historical_predictions import quoted_frame, saved_store
from tests.support.model_registry import make_spec
from tests.support.paths import REPO_ROOT


@pytest.fixture
def tabpfn_pipeline(monkeypatch, tmp_path):
    base = make_spec(features=("A", total_line_col()))
    specs = {}
    for target in ModelTarget:
        slot = ModelSlot(schema_version="2_5", target=target, horizon_minutes=240)
        specs[slot] = finalize_spec(
            identity=base.identity.model_copy(
                update={"target": target, "horizon_minutes": 240}
            ),
            features=base.features,
            cleaning=base.cleaning,
            training=base.training.model_copy(
                update={"train_games": 3, "device": "cuda"}
            ),
        )
    settings = SimpleNamespace(
        prediction_model_slots=list(specs),
        report_required_max_horizon=300,
        s3_aws_profile=None,
        s3_aws_region="eu-west-1",
        s3_bucket="test",
        s3_models_prefix="models/",
    )
    monkeypatch.setattr(backfill, "SETTINGS", settings)
    s3 = FakeS3Client()
    monkeypatch.setattr(backfill, "make_s3_client", lambda **_: s3)
    monkeypatch.setattr(
        backfill, "resolve_config_spec", lambda *, slot, **k: specs[slot]
    )
    dates = backfill.backfill_dates
    monkeypatch.setattr(
        backfill, "backfill_dates", lambda start, end=None: dates(start, end, now=NOW)
    )
    monkeypatch.setattr(backfill, "historical_coverage", lambda *a: pd.DataFrame())
    monkeypatch.setattr(backfill, "replay_scheduled_games", lambda day: schedule())
    monkeypatch.setattr(
        backfill,
        "fetch_games",
        lambda *a: pd.DataFrame({"game_id": [GAME], "tipoff_utc": [TIP]}),
    )
    frame = prediction_frame().assign(
        TIME_TO_MATCH_MIN=240, SNAPSHOT_TS_UTC=TIP - pd.Timedelta(minutes=240)
    )
    frame[spread_line_home_col()] = 3.5
    for column in COVERED_COLUMNS:
        frame[column] = 1
    frame = quoted_frame(frame)
    partitions = saved_store(frame)
    monkeypatch.setattr(backfill, "SeasonPartitions", lambda **k: partitions)
    history = pd.DataFrame(
        {
            "GAME_ID": [f"00226000{i:02}" for i in range(2, 8)],
            "GAME_DATE": pd.to_datetime(
                [
                    "2026-10-20",
                    "2026-10-21",
                    "2026-10-22",
                    "2026-10-23",
                    DAY,
                    "2026-10-25",
                ]
            ),
            "TIME_TO_MATCH_MIN": [240] * 6,
            "TOTAL_POINTS": [220.0, 230.0, 225.0, 240.0, 350.0, 350.0],
            total_line_col(): [225.0] * 6,
            "A": [0.0, 1.0, 2.0, 3.0, 999.0, 9999.0],
            "HOME_MARGIN": [1.0, 2.0, 3.0, 4.0, 100.0, 100.0],
            spread_line_home_col(): [3.5] * 6,
        }
    )
    monkeypatch.setattr(
        backfill,
        "resolve_training_frame",
        lambda *a, **k: TrainingFrame(history, "partition", "unfiltered-checksum"),
    )
    import nba_ou.modeling.registry_store as registry

    monkeypatch.setattr(
        registry,
        "resolve_channel",
        lambda **k: pytest.fail("A production booster was read"),
    )
    monkeypatch.setattr(
        backfill,
        "require_device",
        lambda *a: pytest.fail("TabPFN-only required a local GPU"),
    )
    monkeypatch.setattr(
        backfill,
        "refit_from_spec",
        lambda *a, **k: pytest.fail("TabPFN-only fitted XGBoost"),
    )
    fits = []

    def fit(training, *, model_version, mode):
        fits.append((training, model_version, mode))
        assert training.train_date_max == pd.Timestamp("2026-10-23")
        assert training.X.A.max() == 3.0
        assert training.X["SEASON_YEAR"].tolist() == [2026.0] * 4
        assert training.n_games == 4
        assert training.train_date_min == pd.Timestamp("2026-10-20")

        def predict(X):
            assert X.columns.tolist() == training.X.columns.tolist()
            assert X["SEASON_YEAR"].tolist() == [2026.0] * len(X)
            return np.full(len(X), float(training.y.mean()))

        regressor = SimpleNamespace(predict=predict)
        return FittedTabPFN(
            regressor, model_version=model_version, client_version="test", mode=mode
        )

    monkeypatch.setattr(backfill, "fit_tabpfn", fit)
    import nba_ou.postgre_db.config.db_config as db_config
    from nba_ou.postgre_db.predictions_aiven import store

    written = []
    monkeypatch.setattr(db_config, "connect_nba_db", lambda **k: MagicMock())
    monkeypatch.setattr(
        store, "write_run", lambda conn, tables: written.append(tables) or True
    )
    return SimpleNamespace(
        specs=specs,
        settings=settings,
        fits=fits,
        fit=fit,
        written=written,
        s3=s3,
        output=tmp_path,
        store=partitions,
        frame=frame,
    )


@pytest.mark.parametrize("persist", [False, True])
def test_all_targets_standard_at_only_four_hours_with_historical_training(
    tabpfn_pipeline, persist
):
    p = tabpfn_pipeline
    assert (
        backfill.run_backfill(
            DAY, slots=[], include_tabpfn=True, persist=persist, output_dir=p.output
        )
        == 0
    )
    assert p.store.read.call_count == 1
    assert len(p.fits) == 3
    assert {training.y.name for training, _, _ in p.fits} == {
        "LINE_ERROR",
        "TOTAL_POINTS",
        "SPREAD_ERROR",
    }
    assert {mode for _, _, mode in p.fits} == {"standard"}
    predictions = pd.read_csv(next(p.output.glob("*/2026-10-24.predictions.csv")))
    assert set(predictions.kind) == {KIND_TABPFN}
    assert set(predictions.target) == {1, 2, 3}
    assert set(predictions.horizon_minutes) == {240}
    assert set(predictions.variant) == {model_variant(backfill.DEFAULT_MODEL_VERSION)}
    manifest = json.loads(next(p.output.glob("*/2026-10-24.json")).read_text())
    assert all(
        slot["training_checksum"].startswith("sha256:") for slot in manifest["slots"]
    )
    if persist:
        assert len(p.written) == 1
        assert p.written[0].run["workflow"] == 4
        assert len(p.written[0].predictions) == 3
        assert all(key.startswith("prediction_snapshots/") for key in p.s3.put_order)
    else:
        assert not p.written and not p.s3.put_order


def test_registry_and_thinking_predictions_do_not_cover_standard_tabpfn(
    tabpfn_pipeline,
):
    slot, spec = next(iter(tabpfn_pipeline.specs.items()))
    row = dict(
        game_id=int(GAME),
        kind=1,
        spec_id=spec.spec_id,
        schema_version=slot.schema_version,
        target=1,
        horizon_minutes=240,
        variant=model_variant(backfill.DEFAULT_MODEL_VERSION),
        train_date_max="2026-10-23",
    )
    options = dict(
        kind=KIND_TABPFN, variant=model_variant(backfill.DEFAULT_MODEL_VERSION)
    )
    assert backfill.missing_games(
        schedule(), pd.DataFrame([row]), slot, spec.spec_id, **options
    ) == {GAME}
    row.update(
        kind=3, variant=model_variant(backfill.DEFAULT_MODEL_VERSION, "thinking_high")
    )
    assert backfill.missing_games(
        schedule(), pd.DataFrame([row]), slot, spec.spec_id, **options
    ) == {GAME}
    row["variant"] = model_variant(backfill.DEFAULT_MODEL_VERSION)
    assert not backfill.missing_games(
        schedule(), pd.DataFrame([row]), slot, spec.spec_id, **options
    )


def test_tabpfn_failure_does_not_lose_other_targets(tabpfn_pipeline, monkeypatch):
    calls = []

    def fit(training, **kwargs):
        calls.append(training.y.name)
        if training.y.name == "TOTAL_POINTS":
            raise RuntimeError("Hosted service unavailable for this fit")
        return tabpfn_pipeline.fit(training, **kwargs)

    monkeypatch.setattr(backfill, "fit_tabpfn", fit)
    p = tabpfn_pipeline
    assert (
        backfill.run_backfill(
            DAY, slots=[], include_tabpfn=True, persist=True, output_dir=p.output
        )
        == 1
    )
    assert set(calls) == {"LINE_ERROR", "TOTAL_POINTS", "SPREAD_ERROR"}
    assert len(p.written[0].predictions) == 2
    assert p.written[0].run["status"] == 2


def test_registry_fit_failure_still_saves_all_tabpfn_targets(
    tabpfn_pipeline, monkeypatch
):
    monkeypatch.setattr(backfill, "require_device", lambda *a: None)

    def fail(*a, **k):
        raise RuntimeError("Registry model refit failed")

    monkeypatch.setattr(backfill, "refit_from_spec", fail)
    p = tabpfn_pipeline
    assert (
        backfill.run_backfill(
            DAY,
            slots=[next(iter(p.specs))],
            include_tabpfn=True,
            persist=True,
            output_dir=p.output,
        )
        == 1
    )
    assert len(p.fits) == 3
    assert set(p.written[0].predictions.kind) == {KIND_TABPFN}
    assert len(p.written[0].predictions) == 3


def test_existing_standard_tabpfn_rows_skip_hosted_calls(tabpfn_pipeline, monkeypatch):
    p = tabpfn_pipeline
    rows = [
        dict(
            game_id=int(GAME),
            kind=KIND_TABPFN,
            spec_id=spec.spec_id,
            schema_version=slot.schema_version,
            target=backfill.TARGET_CODES[slot.target],
            horizon_minutes=240,
            variant=model_variant(backfill.DEFAULT_MODEL_VERSION),
            train_date_max="2026-10-23",
        )
        for slot, spec in p.specs.items()
    ]
    monkeypatch.setattr(backfill, "historical_coverage", lambda *a: pd.DataFrame(rows))
    monkeypatch.setattr(
        backfill,
        "fetch_games",
        lambda *a: pytest.fail("Covered TabPFN inputs were read"),
    )
    assert (
        backfill.run_backfill(
            DAY, slots=[], include_tabpfn=True, persist=True, output_dir=p.output
        )
        == 0
    )
    assert not p.fits and not p.written and not p.s3.put_order


def test_cutoff_is_checked_before_contacting_tabpfn(tabpfn_pipeline, monkeypatch):
    prepare = backfill.tabpfn_training_set
    monkeypatch.setattr(
        backfill,
        "tabpfn_training_set",
        lambda *a, **k: replace(prepare(*a, **k), train_date_max=pd.Timestamp(DAY)),
    )
    assert (
        backfill.run_backfill(
            DAY, slots=[], include_tabpfn=True, output_dir=tabpfn_pipeline.output
        )
        == 1
    )
    assert not tabpfn_pipeline.fits


def test_missing_historical_injury_reports_prevent_hosted_fit(tabpfn_pipeline):
    tabpfn_pipeline.frame[COVERED_COLUMNS[0]] = 0
    assert (
        backfill.run_backfill(
            DAY, slots=[], include_tabpfn=True, output_dir=tabpfn_pipeline.output
        )
        == 1
    )
    assert not tabpfn_pipeline.fits


def test_tabpfn_only_cli_and_matrix_select_three_targets_at_one_timepoint(
    tabpfn_pipeline, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "SETTINGS", tabpfn_pipeline.settings)
    monkeypatch.setattr(cli, "backfill_dates", lambda *a: [DAY, "2026-10-25"])
    calls = []
    monkeypatch.setattr(cli, "run_backfill", lambda *a, **k: calls.append(k) or 0)
    assert cli.main(["--start-date", DAY, "--tabpfn-only", "--dry-run"]) == 0
    assert calls[-1]["include_tabpfn"] and calls[-1]["slots"] == []
    assert calls[-1]["tabpfn_horizon_minutes"] == 240
    assert cli.main(["--start-date", DAY, "--tabpfn-only", "--job-matrix"]) == 0
    jobs = json.loads(capsys.readouterr().out)["include"]
    assert len(jobs) == 2
    assert all(
        job["horizon"] == 240
        and job["tabpfn"]
        and job["tabpfn_only"]
        and job["slots"] == ""
        for job in jobs
    )
    for horizon in (0, -240, 241):
        with pytest.raises(SystemExit):
            cli.main(
                [
                    "--start-date",
                    DAY,
                    "--tabpfn-only",
                    "--tabpfn-horizon-minutes",
                    str(horizon),
                    "--job-matrix",
                ]
            )


def test_matrix_adds_tabpfn_only_to_its_selected_horizon(
    tabpfn_pipeline, monkeypatch, capsys
):
    slots = [
        *tabpfn_pipeline.specs,
        ModelSlot(schema_version="2_5", target="line_error", horizon_minutes=60),
    ]
    monkeypatch.setattr(cli, "SETTINGS", SimpleNamespace(prediction_model_slots=slots))
    monkeypatch.setattr(cli, "backfill_dates", lambda *a: [DAY])
    assert cli.main(["--start-date", DAY, "--tabpfn", "--job-matrix"]) == 0
    jobs = json.loads(capsys.readouterr().out)["include"]
    assert [job["horizon"] for job in jobs if job["tabpfn"]] == [240]
    assert jobs[0]["horizon"] == 60 and not jobs[0]["tabpfn_only"]


@pytest.mark.parametrize(
    "only,include,horizon", [(True, True, 240), (False, True, 240), (False, False, 60)]
)
def test_action_forwards_tabpfn_only_on_the_selected_job(
    tabpfn_pipeline, monkeypatch, tmp_path, only, include, horizon
):
    workflow = yaml.load(
        (REPO_ROOT / ".github/workflows/nba_backfill_predictions.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    step = next(
        step
        for step in workflow["jobs"]["backfill"]["steps"]
        if step.get("name") == "Refit and replay past snapshots"
    )
    capture = tmp_path / "args"
    script = 'python() { printf "%s\\n" "$@" > "$CAPTURE_ARGS"; }\n' + step["run"]
    subprocess.run(
        ["bash", "-c", script],
        env={
            **os.environ,
            "CAPTURE_ARGS": str(capture),
            "START_DATE": DAY,
            "SLOTS": "" if only else f"2_5/line_error/t{horizon:04d}/main",
            "TABPFN_ONLY": str(only).lower(),
            "INCLUDE_TABPFN": str(include).lower(),
            "TABPFN_HORIZON": "240",
            "PERSIST": "false",
        },
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(cli, "SETTINGS", tabpfn_pipeline.settings)
    calls = []
    monkeypatch.setattr(cli, "run_backfill", lambda *a, **k: calls.append(k) or 0)
    args = capture.read_text().splitlines()
    assert args[0] == "scripts/backfill_past_predictions.py"
    assert cli.main(args[1:]) == 0
    assert calls[-1]["include_tabpfn"] == include
    assert calls[-1]["tabpfn_horizon_minutes"] == 240
    assert calls[-1]["persist"] is False
    if only:
        assert calls[-1]["slots"] == []
    else:
        assert calls[-1]["slots"][0].horizon_minutes == horizon
