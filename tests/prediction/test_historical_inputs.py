"""Saved retrain rows are used for prediction without reading final outcomes."""

import json
from dataclasses import replace

import pandas as pd
import pytest
from nba_ou.config.odds_columns import spread_col, spread_line_home_col, total_line_col
from nba_ou.prediction import historical_backfill as backfill
from nba_ou.prediction.historical_inputs import (
    StoredInputsUnavailable,
    StoredPredictionInputs,
    closing_from_snapshot,
)

from tests.prediction import test_historical_backfill as registry_tests
from tests.prediction import test_historical_backfill_tabpfn as tabpfn_tests
from tests.prediction.test_historical_backfill import (
    DAY,
    GAME,
    TIP,
    prediction_frame,
    schedule,
)
from tests.prediction.test_historical_backfill_closing import closing_spec
from tests.support.historical_predictions import quoted_frame, saved_store

pipeline = registry_tests.pipeline
tabpfn_pipeline = tabpfn_tests.tabpfn_pipeline


def tipoffs(games=None):
    return (
        pd.DataFrame({"game_id": [GAME], "tipoff_utc": [TIP]})
        if games is None
        else games
    )


def forbid_rebuild(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail(
            "Stored inputs unexpectedly rebuilt historical features or fetched ticks"
        )

    import importlib

    for module, function in [
        ("nba_ou.create_training_data.create_df_to_predict", "create_df_to_predict"),
        (
            "nba_ou.create_training_data.create_intermediate_line_df",
            "create_intermediate_line_df",
        ),
        (
            "nba_ou.create_training_data.create_base_game_features",
            "create_base_game_features",
        ),
    ]:
        monkeypatch.setattr(importlib.import_module(module), function, fail)


@pytest.mark.parametrize("persist", [False, True])
def test_real_registry_fit_predicts_saved_rows_without_reconstruction(
    pipeline, monkeypatch, persist
):
    p = pipeline
    source = prediction_frame().assign(
        TOTAL_POINTS=9999.0, LINE_ERROR=9999.0, IS_OVERTIME=1
    )
    store = saved_store(source)
    monkeypatch.setattr(backfill, "SeasonPartitions", lambda **k: store)
    monkeypatch.setattr(backfill, "fetch_games", lambda *a: tipoffs())
    forbid_rebuild(monkeypatch)
    predict = backfill.load_registry_model_and_predict

    def check_inputs(**kwargs):
        frame = kwargs["frames"]["intermediate_line"]
        assert not {"TOTAL_POINTS", "LINE_ERROR", "IS_OVERTIME"}.intersection(
            frame.columns
        )
        assert frame.A.tolist() == [2.0]
        return predict(**kwargs)

    monkeypatch.setattr(backfill, "load_registry_model_and_predict", check_inputs)
    assert backfill.run_backfill(DAY, persist=persist, output_dir=p.output) == 0
    assert store.read.call_count == store.manifests.call_count == 1
    assert len(pd.read_csv(next(p.output.glob("*/*.predictions.csv")))) == 1
    manifest = json.loads(next(p.output.glob("*/2026-10-24.json")).read_text())
    assert manifest["input_sources"]["60"][0]["partition_checksum"] == "sha256:saved"
    assert bool(p.written) is persist
    assert source.TOTAL_POINTS.tolist() == [9999.0]


def test_all_six_tabpfn_models_share_saved_snapshot_rows(tabpfn_pipeline, monkeypatch):
    p = tabpfn_pipeline
    for slot, spec in list(p.specs.items()):
        p.specs[replace(slot, horizon_minutes=0)] = closing_spec(spec)
    p.settings.prediction_model_slots = list(p.specs)
    store = saved_store(quoted_frame(p.frame).assign(TOTAL_POINTS=9999.0))
    monkeypatch.setattr(backfill, "SeasonPartitions", lambda **k: store)
    monkeypatch.setattr(backfill, "fetch_games", lambda *a: tipoffs())
    forbid_rebuild(monkeypatch)
    assert (
        backfill.run_backfill(
            DAY,
            slots=[],
            include_tabpfn=True,
            include_closing=True,
            persist=True,
            output_dir=p.output,
        )
        == 0
    )
    assert len(p.fits) == 6
    assert store.read.call_count == 1
    assert len(p.written[0].predictions) == 6
    assert p.written[0].inputs.snapshot_horizon_minutes.eq(240).all()
    assert p.written[0].inputs.total_bet_line_x2.eq(450).all()
    assert p.written[0].inputs.spread_bet_line_x2.eq(7).all()


def test_missing_saved_game_is_reported_without_building_it(pipeline, monkeypatch):
    p = pipeline
    second = "0022600009"
    scheduled = pd.concat(
        [schedule(), schedule().assign(GAME_ID=second)], ignore_index=True
    )
    store = saved_store(prediction_frame())
    monkeypatch.setattr(backfill, "SeasonPartitions", lambda **k: store)
    monkeypatch.setattr(backfill, "fetch_games", lambda *a: tipoffs())
    monkeypatch.setattr(backfill, "replay_scheduled_games", lambda day: scheduled)
    forbid_rebuild(monkeypatch)
    assert backfill.run_backfill(DAY, output_dir=p.output) == 1
    assert len(pd.read_csv(next(p.output.glob("*/*.predictions.csv")))) == 1
    report = json.loads(next(p.output.glob("*/2026-10-24.json")).read_text())
    assert any(second in reason for reason in report["failures"])


def test_stored_only_missing_features_fail_without_rebuild_or_fit(
    pipeline, monkeypatch
):
    p = pipeline
    store = saved_store(prediction_frame().drop(columns="A"))
    monkeypatch.setattr(backfill, "SeasonPartitions", lambda **k: store)
    monkeypatch.setattr(backfill, "fetch_games", lambda *a: tipoffs())
    forbid_rebuild(monkeypatch)
    monkeypatch.setattr(
        backfill,
        "refit_from_spec",
        lambda *a, **k: pytest.fail("Fit without valid prediction inputs"),
    )
    assert backfill.run_backfill(DAY, output_dir=p.output) == 1
    report = json.loads(next(p.output.glob("*/2026-10-24.json")).read_text())
    assert "required features" in report["failures"][0]


def test_saved_early_snapshot_rows_do_not_expose_overnight_results():
    second = "0022600009"
    source = pd.concat(
        [prediction_frame(), prediction_frame().assign(GAME_ID=second)],
        ignore_index=True,
    ).assign(TIME_TO_MATCH_MIN=1080)
    store = saved_store(source)
    loader = StoredPredictionInputs(
        {"intermediate_line": store},
        tipoff_loader=lambda *a: pd.DataFrame(
            {
                "game_id": [GAME, second],
                "tipoff_utc": [TIP, TIP + pd.Timedelta(hours=6)],
            }
        ),
    )
    # GAME snapshots at 01:00 ET; second snapshots at 07:00 ET.
    result = loader.read(DAY, 1080, {GAME, second})
    assert result.GAME_ID.tolist() == [second]
    with pytest.raises(StoredInputsUnavailable, match="overnight"):
        loader.read(DAY, 1080, {GAME})


def test_closing_market_adapter_uses_snapshot_quotes_and_preserves_signs():
    frame = quoted_frame(prediction_frame()).assign(
        TOTAL_POINTS=9999.0,
        LINE_ERROR=9999.0,
        IS_OVERTIME=1,
        IMPLIED_PTS_HOME_BEFORE=9999.0,
        ODDS_total_pct_bets_over=99.0,
        ODDS_CLOSING_TOTAL_LINE=300.0,
    )
    converted = closing_from_snapshot(frame)
    assert converted[total_line_col()].tolist() == [225.0]
    assert converted[spread_line_home_col()].tolist() == [3.5]
    assert converted[f"{spread_col()}_TEAM_HOME"].tolist() == [-3.5]
    assert converted.IMPLIED_PTS_HOME_BEFORE.tolist() == [(225.0 + 3.5) / 2]
    assert converted.ODDS_total_pct_bets_over.isna().all()
    assert not {
        "TOTAL_POINTS",
        "LINE_ERROR",
        "IS_OVERTIME",
        "ODDS_CLOSING_TOTAL_LINE",
    }.intersection(converted.columns)
    assert not converted.columns.duplicated().any()
    assert frame.IMPLIED_PTS_HOME_BEFORE.tolist() == [9999.0]


def test_closing_registry_model_also_uses_saved_four_hour_row(pipeline, monkeypatch):
    from nba_ou.prediction.report_requirement import COVERED_COLUMNS

    p = pipeline
    slot = replace(p.slot, horizon_minutes=0)
    spec = closing_spec(p.spec)
    frame = quoted_frame(prediction_frame()).assign(TIME_TO_MATCH_MIN=240)
    for column in COVERED_COLUMNS:
        frame[column] = 1
    store = saved_store(frame)
    monkeypatch.setattr(backfill, "SeasonPartitions", lambda **k: store)
    monkeypatch.setattr(backfill, "fetch_games", lambda *a: tipoffs())
    monkeypatch.setattr(backfill, "resolve_config_spec", lambda **k: spec)
    forbid_rebuild(monkeypatch)
    assert (
        backfill.run_backfill(
            DAY, slots=[slot], include_closing=True, persist=True, output_dir=p.output
        )
        == 0
    )
    assert store.read.call_count == 1
    assert p.written[0].model_builds.dataset_type.tolist() == [1]
    assert p.written[0].inputs.input_mode.tolist() == [0]
    assert p.written[0].inputs.snapshot_horizon_minutes.tolist() == [240]
    assert p.written[0].model_builds.train_date_max.tolist() == [
        pd.Timestamp("2026-10-23").date()
    ]


def test_no_filed_reports_never_uses_final_injuries_or_rebuilds_to_find_them(
    tabpfn_pipeline, monkeypatch
):
    from nba_ou.prediction.report_requirement import COVERED_COLUMNS

    p = tabpfn_pipeline
    for slot, spec in list(p.specs.items()):
        p.specs[replace(slot, horizon_minutes=0)] = closing_spec(spec)
    p.settings.prediction_model_slots = list(p.specs)
    frame = quoted_frame(p.frame)
    frame[COVERED_COLUMNS[0]] = 0
    monkeypatch.setattr(backfill, "SeasonPartitions", lambda **k: saved_store(frame))
    monkeypatch.setattr(backfill, "fetch_games", lambda *a: tipoffs())
    forbid_rebuild(monkeypatch)
    assert (
        backfill.run_backfill(
            DAY,
            slots=[],
            include_tabpfn=True,
            include_closing=True,
            output_dir=p.output,
        )
        == 1
    )
    assert not p.fits


def test_saved_selection_excludes_later_date_and_keeps_original_training_rows():
    stored = pd.concat(
        [
            prediction_frame(),
            prediction_frame().assign(
                GAME_DATE=pd.Timestamp(DAY) + pd.Timedelta(days=1), A=9999.0
            ),
        ],
        ignore_index=True,
    )
    store = saved_store(stored)
    loader = StoredPredictionInputs(
        {"intermediate_line": store}, tipoff_loader=lambda *a: tipoffs()
    )
    selected = loader.read(DAY, 60, {GAME}, required_features=["A", total_line_col()])
    assert selected.A.tolist() == [2.0]
    assert stored.A.tolist() == [2.0, 9999.0]
