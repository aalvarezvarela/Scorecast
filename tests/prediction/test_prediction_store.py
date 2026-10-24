"""The prediction store: run rows, S3 run archive, results and the Aiven writer.

The writer tests need a disposable PostgreSQL (15+, for NULLS NOT DISTINCT):
set ``NBA_OU_TEST_PG_DSN`` to run them. They create and drop the
``prediction`` schema in that database.
"""

from __future__ import annotations

import io
import json
import os
from datetime import timedelta

import pandas as pd
import pyarrow.parquet as pq
import pytest
from nba_ou.postgre_db.predictions_aiven import schema as store_schema
from nba_ou.postgre_db.predictions_aiven.results import (
    drop_unfinished,
    final_scores,
)
from nba_ou.prediction.prediction_store import (
    PredictionRunRecorder,
    bet_line_columns,
    dropped_na_games,
    persist_prediction_run,
)
from nba_ou.utils.s3_prediction_runs import upload_prediction_run

from tests.support.fake_s3 import FakeS3Client

BOOK = "bet365"
GAME_1 = "0022600001"
GAME_2 = "0022600002"
TIPOFF_1 = pd.Timestamp("2026-10-21 19:30", tz="America/New_York")
TIPOFF_2 = pd.Timestamp("2026-10-21 22:00", tz="America/New_York")
PREDICTED_AT = pd.Timestamp("2026-10-21 18:43:00", tz="UTC")


def schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": [GAME_1, GAME_2],
            # The NBA's own date: it is the stored game date, not the run's.
            "GAME_DATE_EST": ["2026-10-21T00:00:00", "2026-10-21T00:00:00"],
            "GAME_TIME": [TIPOFF_1, TIPOFF_2],
            "HOME_TEAM_ID": [1610612738, 1610612747],
            "VISITOR_TEAM_ID": [1610612752, 1610612744],
        }
    )


def closing_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": [GAME_1, GAME_2],
            f"ODDS_TOTAL_LINE_{BOOK}": [224.5, 230.3],
            f"ODDS_SPREAD_LINE_HOME_{BOOK}": [-3.5, 2.0],
        }
    )


def intermediate_frame() -> pd.DataFrame:
    rows = []
    for game, tipoff, horizons in (
        (GAME_1, TIPOFF_1, (360, 47)),
        (GAME_2, TIPOFF_2, (360, 197)),
    ):
        for minutes in horizons:
            rows.append(
                {
                    "GAME_ID": game,
                    "TIME_TO_MATCH_MIN": minutes,
                    "SNAPSHOT_TS_UTC": tipoff.tz_convert("UTC")
                    - timedelta(minutes=minutes),
                    f"ODDS_TOTAL_LINE_{BOOK}": 225.0 + minutes / 1000,
                    f"ODDS_SPREAD_LINE_HOME_{BOOK}": -3.0,
                    "ODDS_SNAP_TOT_BET365_HAS_QUOTE": 1.0,
                    "ODDS_SNAP_TOT_BET365_RAW_LINE": 225.5,
                    "ODDS_SNAP_TOT_BET365_PRICE_LEFT": -115.0,
                    "ODDS_SNAP_TOT_BET365_PRICE_RIGHT": -105.0,
                    "ODDS_SNAP_TOT_BET365_LINE_AGE_MINUTES": 4.0,
                    "ODDS_SNAP_SPR_BET365_HAS_QUOTE": 1.0,
                    "ODDS_SNAP_SPR_BET365_RAW_LINE": -3.5,
                    "ODDS_SNAP_SPR_BET365_PRICE_LEFT": -120.0,
                    "ODDS_SNAP_SPR_BET365_PRICE_RIGHT": 100.0,
                    "ODDS_SNAP_SPR_BET365_LINE_AGE_MINUTES": 9.0,
                    "LATEST_INPUT": minutes != 360,
                }
            )
    return pd.DataFrame(rows)


def output(
    frame: pd.DataFrame,
    *,
    dataset_type: str,
    target: str,
    horizon: int,
    mode: str | None = None,
    value: float = 1.5,
    fit_id: str = "fit-a",
) -> pd.DataFrame:
    """A slot's output as ``load_registry_model_and_predict`` returns it."""
    line_column = (
        f"ODDS_SPREAD_LINE_HOME_{BOOK}"
        if target == "spread_error"
        else f"ODDS_TOTAL_LINE_{BOOK}"
    )
    out = pd.DataFrame({"GAME_ID": frame["GAME_ID"].to_numpy()})
    line = frame[line_column].to_numpy()
    if target == "spread_error":
        out["SPREAD_LINE_AT_PREDICTION"] = line
        out["PRED_SPREAD_ERROR"] = value
        out["PRED_PICK"] = "HOME" if value > 0 else "AWAY"
    else:
        out["TOTAL_OVER_UNDER_LINE"] = line
        out["PRED_LINE_ERROR"] = value
        out["PRED_TOTAL_POINTS"] = line + value
        out["PRED_PICK"] = "OVER" if value > 0 else "UNDER"
    if dataset_type == "intermediate_line":
        out["TIME_TO_MATCH_MIN"] = frame["TIME_TO_MATCH_MIN"].to_numpy()
        out["INPUT_MODE"] = mode
        if mode == "latest":
            out["INPUT_LIMITED_COUNT"] = 3
    out["NA_COLUMNS_COUNT"] = 2
    out["NA_COLUMNS_NAMES"] = "A,B"
    out["PREDICTED_AT_UTC"] = PREDICTED_AT
    out["MODEL_TARGET"] = target
    out["MODEL_DATASET_TYPE"] = dataset_type
    out["MODEL_HORIZON_MINUTES"] = horizon
    out["MODEL_SCHEMA_VERSION"] = "2_5"
    out["MODEL_VARIANT"] = "main"
    out["SPEC_ID"] = f"spec-{target}-{horizon}"
    out["FIT_ID"] = fit_id
    out["TRAINING_CODE_TAG"] = "2.0"
    out["TRAIN_DATE_MIN"] = "2019-10-22"
    out["TRAIN_DATE_MAX"] = "2026-06-20"
    out["MODEL_BUILD_PREFIX"] = f"models/2_5/{target}/t{horizon}/main/builds/{fit_id}/"
    return out


def new_recorder() -> PredictionRunRecorder:
    recorder = PredictionRunRecorder(workflow=1, game_date="2026-10-20", book=BOOK)
    recorder.set_schedule(schedule())
    recorder.add_frame("closing_line", closing_frame(), version="2_7")
    recorder.add_frame("intermediate_line", intermediate_frame(), version="2_5")
    return recorder


def latest_rows() -> pd.DataFrame:
    frame = intermediate_frame()
    return frame[frame["LATEST_INPUT"]]


def own_row(minutes: int = 360) -> pd.DataFrame:
    frame = intermediate_frame()
    return frame[frame["TIME_TO_MATCH_MIN"].eq(minutes)].head(1)


def baseline_output(train_date_max=None) -> pd.DataFrame:
    closing = closing_frame()
    return closing[["GAME_ID"]].assign(
        TOTAL_OVER_UNDER_LINE=closing[f"ODDS_TOTAL_LINE_{BOOK}"],
        PRED_LINE_ERROR=-1.0,
        PRED_PICK="UNDER",
        NA_COLUMNS_COUNT=0,
        MODEL_VERSION="1.0",
        TRAIN_DATE_MAX=train_date_max,
        PREDICTION_DATETIME=PREDICTED_AT.tz_convert("Europe/Madrid"),
    )


def daily_recorder() -> PredictionRunRecorder:
    """A daily run: closing slots, latest-input intermediate slots, a baseline,
    and one intermediate slot on its own (T-360) row."""
    recorder = new_recorder()
    closing = closing_frame()
    accepted = [
        recorder.add_predictions(
            output(
                closing, dataset_type="closing_line", target="line_error", horizon=0
            ),
            label="closing line_error",
        ),
        recorder.add_predictions(
            output(
                closing,
                dataset_type="closing_line",
                target="spread_error",
                horizon=0,
                value=-2.0,
            ),
            label="closing spread_error",
        ),
        *(
            recorder.add_predictions(
                output(
                    latest_rows(),
                    dataset_type="intermediate_line",
                    target="line_error",
                    horizon=horizon,
                    mode="latest",
                ),
                label=f"t{horizon} line_error",
            )
            for horizon in (360, 60)
        ),
        recorder.add_predictions(
            output(
                own_row(),
                dataset_type="intermediate_line",
                target="spread_error",
                horizon=360,
                mode="own",
                value=0.8,
            ),
            label="t360 spread_error",
        ),
        recorder.add_baseline(
            "random_line_error_baseline", baseline_output(), schema_version="2_5"
        ),
    ]
    assert accepted == [None] * 6
    return recorder


# -- run rows -----------------------------------------------------------------


def test_inputs_one_row_per_frame_row_and_mode():
    tables = daily_recorder().tables(status=1)
    inputs = tables.inputs.sort_values(list(tables.inputs.columns[:3]))
    keys = list(
        inputs[["game_id", "input_mode", "snapshot_horizon_minutes"]]
        .fillna(-1)
        .itertuples(index=False, name=None)
    )
    # -1: no snapshot (closing rows).
    assert keys == [
        (22600001, 0, -1),
        (22600001, 1, 360),
        (22600001, 2, 47),
        (22600002, 0, -1),
        (22600002, 2, 197),
    ]
    # Two closing slots, two latest slots, one own slot and a baseline.
    assert len(tables.predictions) == 2 * 2 + 2 * 2 + 1 + 2


def test_game_date_is_the_schedule_date():
    games = daily_recorder().tables(status=1).games
    # The run was dated 2026-10-20; the schedule says the games are the 21st.
    assert set(games["game_date"].astype(str)) == {"2026-10-21"}
    assert set(games["season_year"]) == {2026}
    assert set(games["season_type"]) == {2}
    assert games.set_index("game_id").loc[22600001, "home_team"] == "BOS"


def test_bet_lines_follow_the_line_history_sides():
    lines = bet_line_columns(intermediate_frame(), "intermediate_line", BOOK).iloc[0]
    assert lines["total_bet_line_x2"] == 451
    assert (lines["total_price_over"], lines["total_price_under"]) == (-115, -105)
    assert lines["spread_bet_line_x2"] == -7
    # Spread: left is the away side, right the home side.
    assert (lines["spread_price_home"], lines["spread_price_away"]) == (100, -120)
    assert lines["spread_line_age_minutes"] == 9.0


def test_closing_rows_bet_the_latest_snapshot_quote():
    """The closing frame is normalized in place, so a closing row's quote is
    its game's latest intermediate row's, the quote at the run's moment."""
    inputs = daily_recorder().tables(status=1).inputs
    closing = inputs[inputs["input_mode"].eq(0)].set_index("game_id")
    assert closing.loc[22600001, "total_bet_line_x2"] == 451
    assert closing.loc[22600001, "total_price_over"] == -115
    assert closing.loc[22600002, "spread_price_home"] == 100
    assert closing.loc[22600001, "total_model_line"] == 224.5


def test_closing_rows_without_a_latest_snapshot_have_no_bet_line():
    recorder = daily_recorder()
    recorder.frames["intermediate_line"] = recorder.frames["intermediate_line"].drop(
        columns="LATEST_INPUT"
    )
    inputs = recorder.tables(status=1).inputs
    closing = inputs[inputs["input_mode"].eq(0)]
    assert closing["total_bet_line_x2"].isna().all()
    assert closing["spread_price_home"].isna().all()


def test_unquoted_snapshot_has_no_bet_line():
    frame = intermediate_frame().assign(ODDS_SNAP_TOT_BET365_HAS_QUOTE=0.0)
    lines = bet_line_columns(frame, "intermediate_line", BOOK)
    assert lines["total_bet_line_x2"].isna().all()
    assert lines["total_price_over"].isna().all()


def test_prediction_rows_time_and_values():
    tables = daily_recorder().tables(status=1)
    rows = tables.predictions
    own = rows[rows["input_mode"].eq(1)].iloc[0]
    tipoff = TIPOFF_1.tz_convert("UTC")
    assert own["time_to_tipoff_seconds"] == int((tipoff - PREDICTED_AT).total_seconds())
    assert own["pick_side"] == 1
    assert own["prediction_value"] == pytest.approx(0.8)
    assert own["slot"] == "t360 spread_error"
    latest = rows[rows["input_mode"].eq(2)]
    assert set(latest["input_limited_count"]) == {3}
    assert set(latest["horizon_minutes"]) == {360, 60}
    baseline = rows[rows["kind"].eq(store_schema.KIND_BASELINE)]
    assert set(baseline["pick_side"]) == {-1}


def test_model_builds_once_per_build():
    builds = daily_recorder().tables(status=1).model_builds
    assert len(builds) == 6
    assert set(builds["anchor_book"]) == {BOOK}


@pytest.mark.parametrize("target", ["line_error", "total_points", "spread_error"])
def test_a_run_of_one_market_is_stored(target):
    """Each output carries only its own market's columns."""
    recorder = new_recorder()
    assert (
        recorder.add_predictions(
            output(
                closing_frame(), dataset_type="closing_line", target=target, horizon=0
            ),
            label=target,
        )
        is None
    )
    tables = recorder.tables(status=1)
    assert len(tables.predictions) == 2
    assert set(tables.predictions["target"]) == {store_schema.TARGET_CODES[target]}


def test_a_run_with_no_prediction_keeps_its_run_row():
    recorder = new_recorder()
    recorder.record_slot("t60 line_error", "failed", reason="booster unreadable")
    tables = recorder.tables(status=2)
    assert tables.predictions.empty and tables.inputs.empty and tables.games.empty
    assert tables.run["status"] == 2


def _rejected(recorder, frame, **kwargs) -> str | None:
    return recorder.add_predictions(output(frame, **kwargs), label="slot")


def test_a_rejected_output_costs_only_its_slot():
    recorder = daily_recorder()
    reason = _rejected(
        recorder,
        closing_frame().assign(**{f"ODDS_TOTAL_LINE_{BOOK}": 200.0}),
        dataset_type="closing_line",
        target="total_points",
        horizon=0,
    )
    assert "line differs" in reason
    assert recorder.slots[-1]["status"] == "failed"
    # The run's other outputs are still stored.
    assert len(recorder.tables(status=2).predictions) == 11


def test_an_own_row_must_be_the_models_horizon():
    reason = _rejected(
        new_recorder(),
        own_row(360),
        dataset_type="intermediate_line",
        target="line_error",
        horizon=60,
        mode="own",
    )
    assert "not the model's horizon" in reason


def test_the_same_build_twice_on_one_row_is_rejected():
    recorder = daily_recorder()
    reason = _rejected(
        recorder,
        closing_frame(),
        dataset_type="closing_line",
        target="line_error",
        horizon=0,
    )
    assert "already predicted" in reason


def test_an_intermediate_output_without_input_mode_is_rejected():
    reason = _rejected(
        new_recorder(),
        latest_rows(),
        dataset_type="intermediate_line",
        target="line_error",
        horizon=60,
    )
    assert "INPUT_MODE" in reason


def test_a_row_outside_the_frame_is_rejected():
    reason = _rejected(
        new_recorder(),
        own_row().assign(TIME_TO_MATCH_MIN=60),
        dataset_type="intermediate_line",
        target="line_error",
        horizon=60,
        mode="own",
    )
    assert "not in the intermediate_line frame" in reason


def test_an_unusable_schedule_rejects_outputs_without_raising():
    recorder = PredictionRunRecorder(workflow=1, game_date="2026-10-21", book=BOOK)
    recorder.set_schedule(pd.DataFrame({"GAME_ID": ["g5"]}))
    recorder.add_frame("closing_line", closing_frame())
    reason = _rejected(
        recorder,
        closing_frame(),
        dataset_type="closing_line",
        target="line_error",
        horizon=0,
    )
    assert "schedule is unusable" in reason
    assert "schedule" in recorder.setup_errors


def test_nan_budget_drops_are_recorded_per_slot():
    recorder = new_recorder()
    predictions = output(
        closing_frame().head(1),
        dataset_type="closing_line",
        target="line_error",
        horizon=0,
    )
    predictions.attrs["dropped_na_games"] = [GAME_2]
    recorder.add_predictions(predictions, label="closing line_error")
    empty = pd.DataFrame()
    empty.attrs["dropped_na_games"] = [GAME_1]
    recorder.add_predictions(empty, label="closing total_points")
    assert dropped_na_games(empty) == [GAME_1]
    assert [(s["status"], s["dropped_na_games"]) for s in recorder.slots] == [
        ("predicted", [GAME_2]),
        ("skipped", [GAME_1]),
    ]


def test_historical_baseline_is_a_build_per_window():
    recorder = new_recorder()
    recorder.add_baseline(
        "historical_average_line_error_baseline",
        baseline_output(train_date_max="2026-10-20"),
        schema_version="2_5",
    )
    recorder.add_baseline(
        "random_line_error_baseline", baseline_output(), schema_version="2_5"
    )
    builds = recorder.tables(status=1).model_builds.set_index("variant")
    assert builds.loc["historical_average_line_error_baseline", "fit_id"] == (
        "1.0-20261020"
    )
    assert builds.loc["random_line_error_baseline", "fit_id"] == "1.0"


# -- S3 run archive -----------------------------------------------------------


def test_run_archive_writes_each_input_row_once():
    recorder = daily_recorder()
    tables = recorder.tables(status=1)
    s3 = FakeS3Client()
    base, failures = upload_prediction_run(
        s3_client=s3, bucket="b", recorder=recorder, tables=tables
    )
    assert failures == []
    assert base == f"prediction_snapshots/v2/2026-10-20/{recorder.run_uuid}/"
    intermediate = pq.read_table(
        io.BytesIO(s3.store[f"{base}inputs_intermediate_line.parquet"])
    )
    # Of the four frame rows, the three a model read; each once, although the
    # T-360 row of game 1 was read by one model and the latest rows by two.
    assert intermediate.num_rows == 3
    assert intermediate.schema.metadata is None
    predictions = pq.read_table(
        io.BytesIO(s3.store[f"{base}predictions.parquet"])
    ).to_pandas()
    assert len(predictions) == len(tables.predictions)
    assert set(predictions["game_date"].astype(str)) == {"2026-10-21"}
    manifest = json.loads(s3.store[f"{base}manifest.json"])
    assert manifest["run_uuid"] == str(recorder.run_uuid)
    assert manifest["files"]["inputs_closing_line.parquet"] == {
        "rows": 2,
        "uploaded": True,
    }
    assert manifest["frame_versions"] == {
        "closing_line": "2_7",
        "intermediate_line": "2_5",
    }
    assert len(manifest["support_table_version"]) == 12
    assert [s["status"] for s in manifest["slots"]] == ["predicted"] * 6


def test_manifest_is_written_when_a_file_fails():
    class FailingInputs(FakeS3Client):
        def put_object(self, Bucket, Key, Body, **kwargs):  # noqa: N803
            if "inputs_closing" in Key:
                raise OSError("throttled")
            return super().put_object(Bucket=Bucket, Key=Key, Body=Body, **kwargs)

    recorder = daily_recorder()
    s3 = FailingInputs()
    base, failures = upload_prediction_run(
        s3_client=s3, bucket="b", recorder=recorder, tables=recorder.tables(status=1)
    )
    assert len(failures) == 1 and "throttled" in failures[0]
    manifest = json.loads(s3.store[f"{base}manifest.json"])
    assert manifest["files"]["inputs_closing_line.parquet"]["uploaded"] is False
    assert manifest["files"]["predictions.parquet"]["uploaded"] is True


# -- results ------------------------------------------------------------------


def test_final_scores_need_both_results_and_points():
    team_games = pd.DataFrame(
        {
            "GAME_ID": [GAME_1, GAME_1, GAME_2, GAME_2, "0022600003", "0022600003"],
            "MATCHUP": [
                "BOS vs. NYK",
                "NYK @ BOS",
                "LAL vs. GSW",
                "GSW @ LAL",
                "MIA vs. ORL",
                "ORL @ MIA",
            ],
            "PTS": [120, 110, 50, 48, 101, None],
            "WL": ["W", "L", None, None, "W", "L"],
        }
    )
    scores = final_scores(team_games)
    assert scores.to_dict("records") == [
        {"game_id": 22600001, "home_points": 120, "away_points": 110}
    ]


def test_games_in_progress_are_never_written():
    scores = pd.DataFrame(
        {
            "game_id": [22600001, 22600002, 22600003],
            "home_points": [120, 61, 99],
            "away_points": [110, 58, 97],
        }
    )
    now = pd.Timestamp("2026-10-22 03:00", tz="UTC")
    games = pd.DataFrame(
        {
            "game_id": [22600001, 22600002, 22600003],
            "tipoff_at": [
                now - pd.Timedelta(hours=6),
                now - pd.Timedelta(hours=1),
                now - pd.Timedelta(hours=3),
            ],
        }
    )
    # The live scoreboard says game 2 is on: only it is left out.
    kept = drop_unfinished(scores, games, live={22600002}, now=now)
    assert kept["game_id"].tolist() == [22600001, 22600003]
    # No scoreboard: every game tipped within the last 4 hours is left out.
    kept = drop_unfinished(scores, games, live=None, now=now)
    assert kept["game_id"].tolist() == [22600001]


# -- the Aiven writer, against a real PostgreSQL ------------------------------

DSN = os.getenv("NBA_OU_TEST_PG_DSN")
needs_db = pytest.mark.skipif(not DSN, reason="NBA_OU_TEST_PG_DSN not set")


@pytest.fixture
def conn():
    import psycopg

    with psycopg.connect(DSN) as connection:
        with connection.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS prediction CASCADE")
        connection.commit()
        store_schema.create_schema(connection)
        yield connection
        connection.rollback()
        with connection.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS prediction CASCADE")
        connection.commit()


def _scalar(conn, query: str, params=()):
    with conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchone()[0]


def _rows(conn, query: str) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(query)
        return pd.DataFrame(
            cur.fetchall(), columns=[column.name for column in cur.description]
        )


@needs_db
def test_writer_stores_a_run_once(conn):
    from nba_ou.postgre_db.predictions_aiven.store import write_run

    recorder = daily_recorder()
    tables = recorder.tables(status=1)
    assert write_run(conn, tables) is True
    assert write_run(conn, tables) is False
    assert _scalar(conn, "SELECT count(*) FROM prediction.predictions") == len(
        tables.predictions
    )
    assert _scalar(conn, "SELECT count(*) FROM prediction.inputs") == 5
    assert _scalar(conn, "SELECT count(*) FROM prediction.games") == 2

    # A second run reuses the games and the builds.
    second = daily_recorder()
    assert write_run(conn, second.tables(status=1)) is True
    assert _scalar(conn, "SELECT count(*) FROM prediction.model_builds") == 6
    assert _scalar(conn, "SELECT count(*) FROM prediction.games") == 2
    assert _scalar(conn, "SELECT count(*) FROM prediction.runs") == 2


@needs_db
def test_writer_stores_single_market_and_empty_runs(conn):
    from nba_ou.postgre_db.predictions_aiven.store import write_run

    spread_only = new_recorder()
    spread_only.add_predictions(
        output(
            closing_frame(),
            dataset_type="closing_line",
            target="spread_error",
            horizon=0,
        ),
        label="closing spread_error",
    )
    assert write_run(conn, spread_only.tables(status=1)) is True
    assert write_run(conn, new_recorder().tables(status=2)) is True
    assert _scalar(conn, "SELECT count(*) FROM prediction.runs") == 2
    assert _scalar(conn, "SELECT count(*) FROM prediction.predictions") == 2


@needs_db
def test_view_grades_against_the_bet_line(conn):
    from nba_ou.postgre_db.predictions_aiven.store import (
        pending_games,
        stale_pending_games,
        update_game_results,
        write_run,
    )

    write_run(conn, daily_recorder().tables(status=1))
    assert _scalar(
        conn, "SELECT count(*) FROM prediction.predictions_full WHERE status = 0"
    ) == _scalar(conn, "SELECT count(*) FROM prediction.predictions")
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE prediction.games SET tipoff_at = now() - interval '3 hours' "
            "WHERE game_id = 22600001"
        )
        cur.execute(
            "UPDATE prediction.games SET tipoff_at = now() - interval '5 days' "
            "WHERE game_id = 22600002"
        )
    conn.commit()
    assert pending_games(conn)["game_id"].tolist() == [22600001, 22600002]
    assert stale_pending_games(conn, 3)["game_id"].tolist() == [22600002]
    results = pd.DataFrame(
        {"game_id": [22600001], "home_points": [120], "away_points": [110]}
    )
    assert update_game_results(conn, results) == 1
    assert update_game_results(conn, results) == 0

    graded = _rows(
        conn, "SELECT * FROM prediction.predictions_full WHERE game_id = 22600001"
    )
    closing_total = graded[(graded["input_mode"] == 0) & (graded["target"] == 1)]
    model = closing_total[closing_total["kind"] == 1].iloc[0]
    # Bet line 225.5 (the latest snapshot's quote), actual 230: the OVER pick
    # wins; the baseline's UNDER loses.
    assert model["bet_line_x2"] == 451
    assert model["pick_result"] == 1
    assert model["predicted_value"] == pytest.approx(226.0)
    assert model["model_edge"] == pytest.approx(1.5)
    assert model["season_type"] == 2 and model["season_year"] == 2026
    assert model["run_status"] == 1 and model["workflow"] == 1
    baseline = closing_total[closing_total["kind"] == 2].iloc[0]
    assert baseline["pick_result"] == -1
    # Spread at -3.5 (home margin), actual +10: the AWAY pick loses.
    spread = graded[(graded["input_mode"] == 0) & (graded["target"] == 3)].iloc[0]
    assert spread["pick_result"] == -1
    assert spread["bet_side"] == -1
    # Own-snapshot row: on its horizon, with the snapshot's live quote.
    own = graded[graded["input_mode"] == 1].iloc[0]
    assert bool(own["on_horizon"]) is True
    assert own["bet_price_over_home"] == 100
    assert pd.Timestamp(own["predicted_at"]) == PREDICTED_AT
    assert pd.Timestamp(own["tipoff_at_prediction"]) == TIPOFF_1
    # Game 2 has no result: nothing graded.
    assert (
        _scalar(
            conn,
            "SELECT count(pick_result) FROM prediction.predictions_full "
            "WHERE game_id = 22600002",
        )
        == 0
    )


@needs_db
def test_archived_builds_are_retargeted(conn):
    from nba_ou.postgre_db.predictions_aiven.store import (
        retarget_build_prefixes,
        write_run,
    )

    write_run(conn, daily_recorder().tables(status=1))
    old = "models/2_5/line_error/t60/main/builds/fit-a/"
    new = "models/retired/20261101/2_5/line_error/t60/main/builds/fit-a/"
    assert retarget_build_prefixes(conn, {old: new}) == 1
    assert (
        _scalar(
            conn,
            "SELECT count(*) FROM prediction.model_builds WHERE s3_build_prefix = %s",
            (new,),
        )
        == 1
    )


@needs_db
def test_persist_writes_s3_then_the_store(conn):
    recorder = daily_recorder()
    s3 = FakeS3Client()
    failures = persist_prediction_run(
        recorder, s3_client=s3, bucket="b", partial=True, conn=conn
    )
    assert failures == []
    assert _scalar(conn, "SELECT status FROM prediction.runs") == (
        store_schema.RUN_STATUS_PARTIAL
    )
    assert any(key.endswith("manifest.json") for key in s3.store)
