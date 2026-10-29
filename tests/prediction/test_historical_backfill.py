"""Historical backfills refit past-only rows and never read served boosters."""

import json
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest
from nba_ou.config.odds_columns import total_line_col
from nba_ou.create_training_data.live_game_day import LiveGameDay
from nba_ou.modeling.refit import TrainingFrame
from nba_ou.modeling.registry_models import SpecTraining
from nba_ou.modeling.registry_paths import ModelSlot
from nba_ou.postgre_db.line_history_aiven.live import LiveLineHistory
from nba_ou.postgre_db.predictions_aiven import coverage as coverage_module
from nba_ou.postgre_db.predictions_aiven.schema import WORKFLOW_HISTORICAL_BACKFILL
from nba_ou.prediction import historical_backfill as backfill

import scripts.backfill_past_predictions as cli
from tests.support.fake_s3 import FakeS3Client
from tests.support.model_registry import make_spec

GAME = "0022600001"
DAY = "2026-10-24"
TIP = pd.Timestamp("2026-10-24T23:00:00Z")
NOW = pd.Timestamp("2026-10-29T12:00:00Z")


def schedule():
    return pd.DataFrame(
        {
            "GAME_ID": [GAME],
            "GAME_DATE_EST": [DAY],
            "GAME_TIME": [TIP],
            "HOME_TEAM_ID": [1610612738],
            "VISITOR_TEAM_ID": [1610612752],
        }
    )


def prediction_frame():
    return pd.DataFrame(
        {
            "GAME_ID": [GAME],
            "GAME_DATE": [pd.Timestamp(DAY)],
            "GAME_TIME": [TIP],
            "SEASON_TYPE": ["Regular Season"],
            "TEAM_NAME_TEAM_HOME": ["Boston Celtics"],
            "TEAM_NAME_TEAM_AWAY": ["New York Knicks"],
            "MATCHUP_TEAM_HOME": ["BOS vs. NYK"],
            "TIME_TO_MATCH_MIN": [60],
            "SNAPSHOT_TS_UTC": [TIP - pd.Timedelta(minutes=60)],
            "SNAPSHOT_PASSED": [True],
            total_line_col(): [225.0],
            "A": [2.0],
            "TOTAL_POINTS": [float("nan")],
        }
    )


def test_dates_are_past_eastern_dates_of_current_season():
    assert backfill.backfill_dates("2026-10-21", "2026-10-23", now=NOW) == [
        "2026-10-21",
        "2026-10-22",
        "2026-10-23",
    ]
    for start, end in [
        ("2026-10-29", None),
        ("2026-10-30", None),
        ("2026-06-01", None),
        ("2026-10-25", "2026-10-24"),
    ]:
        with pytest.raises(ValueError):
            backfill.backfill_dates(start, end, now=NOW)


def test_training_is_cut_before_cleaning_and_checksum_ignores_future_rows():
    rows = pd.DataFrame(
        {
            "GAME_DATE": ["2026-10-22", "2026-10-23", DAY, "2026-10-25"],
            "A": [1, 2, 999, 9999],
        }
    )
    frame = TrainingFrame(rows, "all-history", "original")
    historical = backfill.past_training_frame(frame, pd.Timestamp("2026-10-23"))
    assert historical.df.A.tolist() == [1, 2]
    assert len(frame.df) == 4
    assert historical.build_id.endswith("through=2026-10-23")
    changed = rows.copy()
    changed.loc[2:, "A"] = -1000
    assert (
        backfill.past_training_frame(
            replace(frame, df=changed), pd.Timestamp("2026-10-23")
        ).checksum
        == historical.checksum
    )
    with pytest.raises(ValueError, match="No training games"):
        backfill.past_training_frame(frame, pd.Timestamp("2026-10-01"))


def test_early_snapshots_do_not_read_the_previous_overnight_slate():
    assert backfill.history_limit(
        DAY, pd.Series(["2026-10-24T22:00Z"])
    ) == pd.Timestamp("2026-10-23")
    # 01:00 ET: the previous day's West Coast games may still be running.
    assert backfill.history_limit(
        DAY, pd.Series(["2026-10-24T05:00Z"])
    ) == pd.Timestamp("2026-10-22")
    assert backfill.history_limit(
        DAY, pd.Series(["2026-10-24T10:00Z"])
    ) == pd.Timestamp("2026-10-23")
    # 05:30 EST on the fall-back day is still before the 06:00 wall-clock boundary.
    assert backfill.history_limit(
        "2026-11-01", pd.Series(["2026-11-01T10:30Z"])
    ) == pd.Timestamp("2026-10-30")


def test_rosters_are_filtered_per_game_before_building_any_features():
    games = pd.DataFrame(
        {
            "GAME_ID": ["a", "b"],
            "HOME_TEAM_ID": ["h1", "h2"],
            "VISITOR_TEAM_ID": ["a1", "a2"],
        }
    )
    snapshots = pd.DataFrame(
        {
            "TEAM_ID": ["h1", "h1", "h2", "h2"],
            "FETCHED_AT_UTC": pd.to_datetime(
                [
                    "2026-10-24T14:00Z",
                    "2026-10-24T16:00Z",
                    "2026-10-24T16:00Z",
                    "2026-10-24T18:00Z",
                ]
            ),
            "PLAYER_ID": ["before_a", "after_a", "before_b", "after_b"],
        }
    )
    filtered = backfill.snapshot_rosters(
        games,
        snapshots,
        {
            "a": pd.Timestamp("2026-10-24T15:00Z"),
            "b": pd.Timestamp("2026-10-24T17:00Z"),
        },
    )
    assert filtered.PLAYER_ID.tolist() == ["before_a", "before_b"]


def test_only_same_spec_exact_horizon_predictions_count_as_coverage():
    slot = ModelSlot(schema_version="2_5", target="line_error", horizon_minutes=60)
    row = dict(
        game_id=int(GAME),
        kind=1,
        spec_id="a",
        schema_version="2_5",
        target=1,
        horizon_minutes=60,
        variant="main",
        train_date_max="2026-10-23",
    )
    assert not backfill.missing_games(schedule(), pd.DataFrame([row]), slot, "a")
    for changed in [
        dict(spec_id="b"),
        dict(horizon_minutes=30),
        dict(variant="other"),
        dict(target=2),
    ]:
        assert backfill.missing_games(
            schedule(), pd.DataFrame([{**row, **changed}]), slot, "a"
        ) == {GAME}


def test_early_horizon_does_not_skip_a_prediction_trained_on_the_overnight_slate():
    slot = ModelSlot(schema_version="2_5", target="line_error", horizon_minutes=1080)
    row = dict(
        game_id=int(GAME),
        kind=1,
        spec_id="a",
        schema_version="2_5",
        target=1,
        horizon_minutes=1080,
        variant="main",
        train_date_max="2026-10-23",
    )
    assert backfill.missing_games(schedule(), pd.DataFrame([row]), slot, "a") == {GAME}
    row["train_date_max"] = "2026-10-22"
    assert not backfill.missing_games(schedule(), pd.DataFrame([row]), slot, "a")


def test_coverage_read_is_read_only_and_excludes_models_trained_on_target_or_later(
    monkeypatch,
):
    conn = MagicMock()
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.description = [SimpleNamespace(name="game_id")]
    cursor.fetchall.return_value = []
    monkeypatch.setattr(
        coverage_module, "connect_nba_db", lambda env: nullcontext(conn)
    )
    coverage_module.historical_coverage("2026-10-21", DAY)
    assert conn.read_only is True
    query, params = cursor.execute.call_args.args
    assert params == ("2026-10-21", DAY)
    assert "m.train_date_max < g.game_date" in query
    assert "i.snapshot_horizon_minutes = m.horizon_minutes" in query
    assert "i.input_mode = 1" in query


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    spec = make_spec(features=("A", total_line_col()))
    # Preserve the production refit/predict functions: fit a real tiny CPU
    # booster and carry it through the actual recorder and S3 archive.
    training = SpecTraining(
        params={"max_depth": 1, "n_jobs": 1},
        n_estimators=2,
        train_games=3,
        random_state=16,
        device="cpu",
    )
    from nba_ou.modeling.registry_models import finalize_spec

    spec = finalize_spec(
        identity=spec.identity,
        features=spec.features,
        cleaning=spec.cleaning,
        training=training,
    )
    slot = ModelSlot(schema_version="2_5", target="line_error", horizon_minutes=60)
    s3 = FakeS3Client()
    monkeypatch.setattr(
        backfill,
        "SETTINGS",
        SimpleNamespace(
            prediction_model_slots=[slot],
            report_required_max_horizon=None,
            s3_aws_profile=None,
            s3_aws_region="eu-west-1",
            s3_bucket="test",
        ),
    )
    real_dates = backfill.backfill_dates
    monkeypatch.setattr(
        backfill,
        "backfill_dates",
        lambda start, end=None: real_dates(start, end, now=NOW),
    )
    monkeypatch.setattr(backfill, "make_s3_client", lambda **_: s3)
    monkeypatch.setattr(backfill, "resolve_config_spec", lambda **_: spec)
    monkeypatch.setattr(backfill, "historical_coverage", lambda *a: pd.DataFrame())
    monkeypatch.setattr(backfill, "SeasonPartitions", lambda **_: object())
    monkeypatch.setattr(backfill, "replay_scheduled_games", lambda day: schedule())
    monkeypatch.setattr(
        backfill,
        "stored_day",
        lambda *a: SimpleNamespace(
            line_history=SimpleNamespace(games=pd.DataFrame({"game_id": [GAME]}))
        ),
    )
    monkeypatch.setattr(
        backfill,
        "build_horizon_frame",
        lambda *a: (prediction_frame(), pd.Timestamp("2026-10-23")),
    )
    rows = pd.DataFrame(
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
            "TOTAL_POINTS": [220.0, 230.0, 225.0, 240.0, 350.0, 350.0],
            total_line_col(): [225.0] * 6,
            "A": [0.0, 1.0, 2.0, 3.0, 999.0, 9999.0],
        }
    )
    monkeypatch.setattr(
        backfill,
        "resolve_training_frame",
        lambda *a, **k: TrainingFrame(rows, "partition", "full-checksum"),
    )
    import nba_ou.modeling.registry_store as registry

    monkeypatch.setattr(
        registry,
        "resolve_channel",
        lambda **_: pytest.fail("A historical backfill read a served booster"),
    )
    import nba_ou.postgre_db.config.db_config as db_config
    from nba_ou.postgre_db.predictions_aiven import store

    written = []
    monkeypatch.setattr(db_config, "connect_nba_db", lambda **_: MagicMock())
    monkeypatch.setattr(
        store, "write_run", lambda conn, tables: written.append(tables) or True
    )
    return SimpleNamespace(
        s3=s3, slot=slot, spec=spec, output=tmp_path, written=written
    )


def test_dry_run_performs_real_historical_fit_and_prediction_without_external_writes(
    pipeline,
):
    assert backfill.run_backfill(DAY, output_dir=pipeline.output) == 0
    assert not pipeline.s3.put_order and not pipeline.written
    models = pd.read_csv(next(pipeline.output.glob("*/2026-10-24.models.csv")))
    assert models.train_date_max.tolist() == ["2026-10-23"]
    predictions = pd.read_csv(
        next(pipeline.output.glob("*/2026-10-24.predictions.csv"))
    )
    assert predictions.game_id.tolist() == [int(GAME)]
    assert predictions.snapshot_horizon_minutes.tolist() == [60]
    assert predictions.time_to_tipoff_seconds.lt(0).all()  # execution time stays real


def test_persist_archives_isolated_build_and_inserts_historical_predictions(pipeline):
    assert backfill.run_backfill(DAY, persist=True, output_dir=pipeline.output) == 0
    assert len(pipeline.written) == 1
    tables = pipeline.written[0]
    assert tables.run["workflow"] == WORKFLOW_HISTORICAL_BACKFILL
    assert len(tables.predictions) == 1
    assert tables.model_builds.train_date_max.tolist() == [
        pd.Timestamp("2026-10-23").date()
    ]
    assert any(
        key.startswith("models/historical_backfills/") and key.endswith("model.json")
        for key in pipeline.s3.put_order
    )
    assert any(
        "historical_backfill" in key and key.endswith("manifest.json")
        for key in pipeline.s3.put_order
    )
    assert all(
        "/channels/" not in key and not key.startswith("train_data/")
        for key in pipeline.s3.put_order
    )


def test_training_failure_never_falls_back_to_production(pipeline, monkeypatch):
    def fail(*a, **k):
        raise ValueError("No usable past training rows")

    monkeypatch.setattr(backfill, "refit_from_spec", fail)
    assert backfill.run_backfill(DAY, persist=True, output_dir=pipeline.output) == 1
    assert not any(key.endswith("model.json") for key in pipeline.s3.put_order)
    assert pipeline.written[0].predictions.empty
    assert pipeline.written[0].run["status"] == 2


def test_cutoff_violation_is_rejected_before_prediction_or_storage(
    pipeline, monkeypatch
):
    real_refit = backfill.refit_from_spec

    def invalid(*a, **k):
        result = real_refit(*a, **k)
        return replace(
            result,
            fit=result.fit.model_copy(
                update={"train_date_max": pd.Timestamp(DAY).to_pydatetime()}
            ),
        )

    monkeypatch.setattr(backfill, "refit_from_spec", invalid)
    assert backfill.run_backfill(DAY, persist=True, output_dir=pipeline.output) == 1
    assert pipeline.written[0].predictions.empty
    assert not any(key.endswith("model.json") for key in pipeline.s3.put_order)


def test_already_covered_slots_do_not_refit_or_insert(pipeline, monkeypatch):
    row = dict(
        game_id=int(GAME),
        kind=1,
        spec_id=pipeline.spec.spec_id,
        schema_version="2_5",
        target=1,
        horizon_minutes=60,
        variant="main",
        train_date_max="2026-10-23",
    )
    monkeypatch.setattr(backfill, "historical_coverage", lambda *a: pd.DataFrame([row]))
    monkeypatch.setattr(
        backfill,
        "stored_day",
        lambda *a: pytest.fail("Already covered date was rebuilt"),
    )
    assert backfill.run_backfill(DAY, persist=True, output_dir=pipeline.output) == 0
    assert not pipeline.s3.put_order and not pipeline.written


def test_cli_defaults_to_dry_run_and_modes_are_exclusive(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "run_backfill", lambda *a, **k: calls.append(k) or 0)
    assert cli.main(["--start-date", DAY, "--output-dir", str(tmp_path)]) == 0
    assert calls[-1]["persist"] is False
    assert cli.main(["--start-date", DAY, "--persist"]) == 0
    assert calls[-1]["persist"] is True
    with pytest.raises(SystemExit):
        cli.main(["--start-date", DAY, "--dry-run", "--persist"])


def test_action_matrix_groups_slots_by_date_and_horizon_without_running_backfill(
    monkeypatch, capsys
):
    slots = [
        ModelSlot(schema_version="2_5", target=target, horizon_minutes=horizon)
        for target, horizon in [
            ("line_error", 60),
            ("total_points", 60),
            ("line_error", 360),
        ]
    ]
    monkeypatch.setattr(cli, "SETTINGS", SimpleNamespace(prediction_model_slots=slots))
    monkeypatch.setattr(cli, "backfill_dates", lambda *a: [DAY, "2026-10-25"])
    monkeypatch.setattr(
        cli,
        "run_backfill",
        lambda *a, **k: pytest.fail("Matrix planning trained a model"),
    )
    assert cli.main(["--start-date", DAY, "--job-matrix"]) == 0
    jobs = json.loads(capsys.readouterr().out)["include"]
    assert len(jobs) == 4
    assert jobs[0] == {
        "game_date": DAY,
        "horizon": 60,
        "slots": "2_5/line_error/t0060/main 2_5/total_points/t0060/main",
    }
    assert jobs[3]["game_date"] == "2026-10-25"


def test_date_range_refits_each_cutoff_and_cannot_collide_with_same_second_fits(
    pipeline, monkeypatch
):
    def daily_schedule(day):
        return schedule().assign(
            GAME_ID=f"00226000{day[-2:]}",
            GAME_DATE_EST=day,
            GAME_TIME=pd.Timestamp(f"{day}T23:00Z"),
        )

    monkeypatch.setattr(backfill, "replay_scheduled_games", daily_schedule)
    monkeypatch.setattr(
        backfill,
        "stored_day",
        lambda day, games: SimpleNamespace(
            game_date=day,
            line_history=SimpleNamespace(
                games=pd.DataFrame({"game_id": games.GAME_ID})
            ),
        ),
    )

    def daily_frame(live, horizon, ids):
        tip = pd.Timestamp(f"{live.game_date}T23:00Z")
        frame = prediction_frame().assign(
            GAME_ID=next(iter(ids)),
            GAME_DATE=pd.Timestamp(live.game_date),
            GAME_TIME=tip,
            SNAPSHOT_TS_UTC=tip - pd.Timedelta(minutes=horizon),
        )
        return frame, pd.Timestamp(live.game_date) - pd.Timedelta(days=1)

    monkeypatch.setattr(backfill, "build_horizon_frame", daily_frame)
    refit = backfill.refit_from_spec

    def same_second_refit(*a, **k):
        result = refit(*a, **k)
        return replace(
            result,
            fit=result.fit.model_copy(update={"fit_id": "20261029T120000Z-aaaaaaaa"}),
        )

    monkeypatch.setattr(backfill, "refit_from_spec", same_second_refit)
    assert (
        backfill.run_backfill(
            DAY, "2026-10-25", persist=True, output_dir=pipeline.output
        )
        == 0
    )
    assert len(pipeline.written) == 2
    models = pd.concat([tables.model_builds for tables in pipeline.written])
    assert models.train_date_max.astype(str).tolist() == ["2026-10-23", DAY]
    assert models.fit_id.nunique() == 2


def test_replay_passes_roster_and_history_cutoffs_to_feature_builder(monkeypatch):
    games = schedule()
    ticks = pd.DataFrame(
        {
            "game_id": [GAME, GAME],
            "line_ts": pd.to_datetime(["2026-10-24T21:00Z", "2026-10-24T22:30Z"]),
        }
    )
    snapshots = pd.DataFrame(
        {
            "TEAM_ID": [str(games.HOME_TEAM_ID.iloc[0])] * 2,
            "FETCHED_AT_UTC": pd.to_datetime(
                ["2026-10-24T21:00Z", "2026-10-24T22:30Z"]
            ),
            "PLAYER_ID": ["known", "future"],
        }
    )
    live = LiveGameDay(
        game_date=pd.Timestamp(DAY),
        as_of=TIP,
        scheduled_games=games,
        line_history=LiveLineHistory(
            ticks=ticks,
            games=pd.DataFrame({"game_id": [GAME], "tipoff_utc": [TIP]}),
            as_of=TIP,
        ),
        roster_snapshots=snapshots,
    )

    def build(**kwargs):
        replay = kwargs["live"]
        assert replay.roster_snapshots.PLAYER_ID.tolist() == ["known"]
        assert replay.as_of == TIP - pd.Timedelta(minutes=60)
        assert len(replay.line_history.ticks) == 1
        assert kwargs["recent_limit_to_include"] == pd.Timestamp("2026-10-23")
        assert kwargs["target_snapshots"] == {GAME: 60}
        assert kwargs["strict_history_cutoff"] is True
        frame = prediction_frame().drop(
            columns=["SNAPSHOT_TS_UTC", "SNAPSHOT_PASSED", "GAME_TIME"]
        )
        return (
            frame,
            prediction_frame()[["GAME_ID", "TIME_TO_MATCH_MIN", "SNAPSHOT_TS_UTC"]],
        )

    monkeypatch.setattr(backfill, "create_intermediate_line_df", build)
    frame, limit = backfill.build_horizon_frame(live, 60, {GAME})
    assert frame.SNAPSHOT_PASSED.all()
    assert frame.GAME_TIME.iloc[0] == TIP
    assert limit == pd.Timestamp("2026-10-23")
