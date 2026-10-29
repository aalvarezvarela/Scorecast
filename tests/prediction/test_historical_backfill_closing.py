"""Closing and intermediate models share pregame inputs, never final quotes."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
from nba_ou.data_processing.injury_status.report_state import InjuryReportState
from nba_ou.modeling.registry_models import finalize_spec
from nba_ou.modeling.registry_paths import ModelSlot
from nba_ou.prediction import historical_backfill as backfill
from nba_ou.prediction import historical_closing as closing

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
    monkeypatch.setattr(
        backfill,
        "build_closing_frame",
        lambda *a: (p.frame.copy(), pd.Timestamp("2026-10-23")),
    )
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
    monkeypatch.setattr(
        backfill,
        "build_closing_frame",
        lambda *a: (frame.copy(), pd.Timestamp("2026-10-23")),
    )
    monkeypatch.setattr(
        backfill,
        "build_horizon_frame",
        lambda *a: (frame.copy(), pd.Timestamp("2026-10-23")),
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


def games_frame():
    return pd.DataFrame(
        [
            dict(
                game_id=GAME,
                game_date=pd.Timestamp(DAY),
                season_year=2026,
                tipoff_utc=TIP,
                team_home="Boston Celtics",
                team_away="New York Knicks",
            )
        ]
    )


def test_closing_odds_use_only_last_valid_quote_before_snapshot():
    template = pd.DataFrame(
        [
            dict(
                game_id=GAME,
                game_date=pd.Timestamp(DAY),
                season_year=2026,
                team_home="Boston Celtics",
                team_away="New York Knicks",
                total_fanduel_line_over=280.0,
                total_fanduel_line_under=280.0,
                total_fanduel_price_over=-110.0,
                total_fanduel_price_under=-110.0,
                total_betmgm_line_over=290.0,
                total_betmgm_line_under=290.0,
            )
        ]
    )
    ticks = pd.DataFrame(
        [
            dict(
                game_id=GAME,
                market="totals",
                book="fanduel",
                line_ts=TIP - pd.Timedelta(minutes=minutes),
                minutes_before_tip=minutes,
                left_line=line,
                right_line=line,
                left_price=-110,
                right_price=-110,
            )
            for minutes, line in [(300, 225.0), (240, 226.0), (239, 260.0), (1, 280.0)]
        ]
    )
    result = closing.sportsbook_odds_at_snapshots(
        template, ticks, games_frame(), {GAME: TIP - pd.Timedelta(minutes=240)}
    )
    assert result.total_fanduel_line_over.tolist() == [226.0]
    assert result.total_betmgm_line_over.isna().all()
    assert result.total_betmgm_line_under.isna().all()


def test_closing_feature_pipeline_receives_asof_reports_rosters_and_odds(monkeypatch):
    state = InjuryReportState.empty()
    state.filings = pd.DataFrame(
        [
            dict(game_id=GAME, team_id=str(team), submitted=True)
            for team in [1610612738, 1610612752]
        ]
    )
    cutoff = TIP - pd.Timedelta(minutes=240)
    seen = {}

    def reports(cutoffs):
        assert cutoffs == {GAME: cutoff}
        return state

    monkeypatch.setattr(closing, "reports_at_own_snapshots", reports)
    monkeypatch.setattr(
        closing,
        "load_odds_sportsbook_from_db",
        lambda *a: pd.DataFrame(columns=["game_id", "game_date"]),
    )
    monkeypatch.setattr(
        closing,
        "sportsbook_odds_at_snapshots",
        lambda *a: pd.DataFrame({"game_id": [GAME]}),
    )

    def build(**kwargs):
        seen.update(kwargs)
        return prediction_frame()

    monkeypatch.setattr(closing, "create_df_to_predict", build)
    rosters = pd.DataFrame(
        dict(
            TEAM_ID=["1610612738"] * 2,
            PLAYER_ID=["before", "after"],
            FETCHED_AT_UTC=[
                cutoff - pd.Timedelta(minutes=1),
                cutoff + pd.Timedelta(minutes=1),
            ],
        )
    )
    live = SimpleNamespace(
        game_date=pd.Timestamp(DAY),
        scheduled_games=schedule(),
        line_history=SimpleNamespace(games=games_frame(), ticks=pd.DataFrame()),
        df_referees_scheduled=pd.DataFrame(),
        roster_snapshots=rosters,
    )
    frame, limit = closing.build_closing_frame(live, 240, {GAME})
    assert limit == pd.Timestamp("2026-10-23")
    assert seen["strict_history_cutoff"] is True
    assert seen["scheduled_data"]["injury_report_state"] is state
    assert seen["scheduled_data"]["roster_snapshots"].PLAYER_ID.tolist() == ["before"]
    assert seen["scheduled_data"]["df_odds_yahoo_scheduled"].empty
    assert frame.TIME_TO_MATCH_MIN.tolist() == [240]
    assert frame.SNAPSHOT_TS_UTC.tolist() == [cutoff]
