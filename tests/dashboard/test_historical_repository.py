"""Read bounds/security plus SQL timing/source semantics on synthetic PostgreSQL rows."""

import os
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import psycopg
import pytest
from nba_ou.dashboard.repositories import analysis
from nba_ou.dashboard.repositories.predictions import DashboardDataError
from psycopg.types.json import Jsonb

from .conftest import NOW


def test_historical_read_binds_filters_and_bounds_rows_in_one_read_only_snapshot(
    monkeypatch,
):
    conn = MagicMock()
    conn.__enter__.return_value = conn
    cursor = conn.cursor.return_value.__enter__.return_value
    cursor.description = [SimpleNamespace(name="game_id")]
    cursor.fetchall.return_value = [(1,)]
    monkeypatch.setattr(analysis, "open_connection", MagicMock(return_value=conn))
    start, end = NOW.date() - timedelta(days=7), NOW.date()
    data = analysis.read_history_range(start, end, 3, 4)
    assert conn.read_only is True
    assert conn.isolation_level == psycopg.IsolationLevel.REPEATABLE_READ
    assert len(data.games) == 1 and len(data.predictions) == 1
    commands = cursor.execute.call_args_list
    assert commands[0].args[0] == "SET LOCAL statement_timeout = 45000"
    assert commands[1].args[1] == (start, end, [4])
    assert commands[2].args[1] == (
        start,
        end,
        3,
        [4],
        "2_5",
        data.fetched_at,
        data.fetched_at,
        analysis.MAX_ANALYSIS_ROWS + 1,
    )
    assert "SELECT v.*" not in commands[2].args[0]
    assert "ROW_NUMBER()" in commands[2].args[0]


def test_oversized_read_never_returns_a_truncated_sample(monkeypatch):
    conn = MagicMock()
    conn.__enter__.return_value = conn
    monkeypatch.setattr(analysis, "open_connection", MagicMock(return_value=conn))
    monkeypatch.setattr(analysis, "MAX_ANALYSIS_ROWS", 1)
    monkeypatch.setattr(
        analysis,
        "_frame",
        MagicMock(side_effect=[pd.DataFrame(), pd.DataFrame({"game_id": [1, 2]})]),
    )
    with pytest.raises(DashboardDataError, match="no partial results"):
        analysis.read_history_range(NOW.date(), NOW.date(), 1)


@pytest.mark.parametrize(
    "error,phrase",
    [
        (psycopg.errors.QueryCanceled("sensitive detail"), "shorter date range"),
        (psycopg.OperationalError("sensitive detail"), "Aiven"),
        (psycopg.errors.UndefinedColumn("sensitive detail"), "schema needs updating"),
        (psycopg.errors.InsufficientPrivilege("sensitive detail"), "read access"),
        (ValueError("sensitive detail"), "credentials"),
    ],
)
def test_historical_read_sanitizes_errors(monkeypatch, error, phrase):
    monkeypatch.setattr(analysis, "open_connection", MagicMock(side_effect=error))
    with pytest.raises(DashboardDataError) as caught:
        analysis.read_history_range(NOW.date(), NOW.date(), 1)
    assert phrase in str(caught.value)
    assert "sensitive" not in str(caught.value)


def test_invalid_filters_do_not_open_a_connection(monkeypatch):
    opener = MagicMock()
    monkeypatch.setattr(analysis, "open_connection", opener)
    for start, end, target, season_type in [
        (NOW.date(), NOW.date() - timedelta(days=1), 1, 2),
        (NOW.date(), NOW.date(), 99, 2),
        (NOW.date(), NOW.date(), 1, 99),
    ]:
        with pytest.raises(ValueError):
            analysis.read_history_range(start, end, target, season_type)
    opener.assert_not_called()


# Only CTE/SELECT statements: these fixtures can also validate the production
# SQL on a read-only connection without creating or changing any database table.
FIXTURE_COLUMNS = """
    game_id integer, kind integer, schema_version text, target integer,
    dataset_type integer, model_horizon_minutes integer, variant text,
    workflow integer, input_mode integer, snapshot_horizon_minutes integer,
    prediction_value double precision, model_line double precision,
    bet_line_x2 integer, bet_price_over_home integer, bet_price_under_away integer,
    home_points integer, away_points integer, status integer,
    predicted_at timestamptz, tipoff_at timestamptz,
    tipoff_at_prediction timestamptz, time_to_tipoff_seconds integer,
    run_id integer, input_id integer, model_build_id integer,
    run_started_at timestamptz, game_date date, season_type integer
"""


def synthetic_query():
    tipoff = NOW - pd.Timedelta(hours=6)
    inputs = []
    runs = []

    def add(r=360, **changes):
        row = dict(
            game_id=1,
            kind=1,
            schema_version="2_5",
            target=2,
            dataset_type=2,
            model_horizon_minutes=360,
            variant="main",
            workflow=1,
            input_mode=2,
            snapshot_horizon_minutes=r,
            prediction_value=238.0,
            model_line=230,
            bet_line_x2=480,
            bet_price_over_home=-110,
            bet_price_under_away=-110,
            home_points=121,
            away_points=114,
            status=1,
            predicted_at=tipoff - pd.Timedelta(minutes=r - 5),
            tipoff_at=tipoff,
            tipoff_at_prediction=tipoff,
            time_to_tipoff_seconds=(r - 5) * 60,
            run_id=len(inputs) + 1,
            input_id=len(inputs) + 1,
            model_build_id=len(inputs) + 1,
            run_started_at=tipoff - pd.Timedelta(minutes=r),
            game_date=tipoff.date(),
            season_type=2,
        )
        row.update(changes)
        finished = row.pop(
            "finished_at", row["predicted_at"] + pd.Timedelta(minutes=15)
        )
        inputs.append(row)
        runs.append(dict(run_id=row["run_id"], finished_at=finished))
        return row

    add()  # Exact horizon beats the later but off-grid input in this cell.
    add(r=359, prediction_value=239, finished_at=tipoff - pd.Timedelta(minutes=330))
    refit = add(prediction_value=240, finished_at=tipoff - pd.Timedelta(minutes=335))
    add(r=361, prediction_value=241)  # Adjacent input bucket, same trained horizon.
    add(r=1100, prediction_value=242)  # T−18h+.
    add(
        workflow=2,
        input_mode=1,
        predicted_at=tipoff + pd.Timedelta(hours=2),
        time_to_tipoff_seconds=-7200,
    )
    replay = add(
        workflow=2,
        input_mode=1,
        predicted_at=tipoff + pd.Timedelta(hours=3),
        time_to_tipoff_seconds=-10800,
        prediction_value=246,
    )
    add(r=300, workflow=2, input_mode=1, prediction_value=999)  # Not own horizon.
    add(
        predicted_at=tipoff - pd.Timedelta(minutes=5),
        finished_at=tipoff + pd.Timedelta(minutes=1),
        prediction_value=999,
    )  # Not available pre-game.
    tab = add(
        r=480,
        kind=3,
        workflow=3,
        input_mode=1,
        model_horizon_minutes=480,
        prediction_value=247,
    )
    add(
        kind=3,
        workflow=3,
        input_mode=1,
        predicted_at=tipoff - pd.Timedelta(minutes=370),
        prediction_value=999,
    )  # Future own snapshot.
    add(
        r=360,
        dataset_type=1,
        model_horizon_minutes=0,
        input_mode=0,
        snapshot_horizon_minutes=None,
    )
    closing_final = add(
        r=240,
        dataset_type=1,
        model_horizon_minutes=0,
        input_mode=0,
        snapshot_horizon_minutes=None,
        prediction_value=249,
    )
    add(schema_version="2_6", prediction_value=999)
    add(target=3, prediction_value=999)
    add(season_type=1, prediction_value=999)
    add(
        variant="alternative", prediction_value=250
    )  # A different slot remains separate.
    prefix = """WITH fixture_predictions AS (
        SELECT * FROM jsonb_to_recordset(%s) AS rows(COLUMNS)
    ), fixture_runs AS (
        SELECT * FROM jsonb_to_recordset(%s) AS rows(run_id integer, finished_at timestamptz)
    ), base AS""".replace("COLUMNS", FIXTURE_COLUMNS)
    query = (
        analysis.ANALYSIS_QUERY.replace("WITH base AS", prefix)
        .replace("prediction.predictions_full", "fixture_predictions")
        .replace("prediction.runs", "fixture_runs")
    )

    # ISO strings are explicit JSON values, not interpolated into SQL.
    def clean(rows):
        return [
            {k: v.isoformat() if hasattr(v, "isoformat") else v for k, v in row.items()}
            for row in rows
        ]

    params = (
        Jsonb(clean(inputs)),
        Jsonb(clean(runs)),
        tipoff.date(),
        tipoff.date(),
        2,
        [2],
        "2_5",
        NOW.to_pydatetime(),
        NOW.to_pydatetime(),
        200001,
    )
    return query, params, (refit, replay, tab, closing_final)


def assert_synthetic_selection(cursor):
    query, params, selected = synthetic_query()
    cursor.execute(query, params)
    frame = pd.DataFrame(
        cursor.fetchall(), columns=[column.name for column in cursor.description]
    )
    assert len(frame) == 10
    assert not frame["prediction_value"].eq(999).any()
    main = frame.loc[frame["variant"].eq("main")]
    live = main.loc[
        main["source"].eq("Live predictions")
        & main["kind"].eq(1)
        & main["dataset_type"].eq(2)
    ]
    assert dict(zip(live["bucket"], live["prediction_value"], strict=True)) == {
        360: selected[0]["prediction_value"],
        420: 241,
        1081: 242,
    }
    replay = main.loc[main["source"].eq("Snapshot replay") & main["kind"].eq(1)]
    assert replay["prediction_value"].tolist() == [selected[1]["prediction_value"]]
    tab = main.loc[main["kind"].eq(3)]
    assert set(tab["source"]) == {"Live predictions", "Snapshot replay"}
    assert tab["prediction_value"].eq(selected[2]["prediction_value"]).all()
    final = frame.loc[frame["bucket"].eq(0)]
    assert final["prediction_value"].tolist() == [selected[3]["prediction_value"]]
    return len(frame)


def assert_synthetic_closing_replay_selection(cursor):
    query, params, _ = synthetic_query()
    inputs = [dict(row) for row in params[0].obj]
    runs = [dict(row) for row in params[1].obj]
    tipoff = pd.Timestamp(inputs[0]["tipoff_at"])

    def add(**changes):
        row = dict(inputs[0])
        run_id = len(inputs) + 1
        predicted_at = tipoff + pd.Timedelta(hours=2, minutes=run_id)
        row.update(
            dataset_type=1,
            model_horizon_minutes=0,
            snapshot_horizon_minutes=240,
            workflow=2,
            input_mode=3,
            predicted_at=predicted_at.isoformat(),
            time_to_tipoff_seconds=-7200,
            run_id=run_id,
            input_id=run_id,
            model_build_id=run_id,
        )
        row.update(changes)
        inputs.append(row)
        runs.append(
            dict(
                run_id=run_id,
                finished_at=(predicted_at + pd.Timedelta(minutes=1)).isoformat(),
            )
        )

    add(prediction_value=250)
    add(workflow=4, input_mode=0, prediction_value=254)  # Newest refit wins once.
    add(snapshot_horizon_minutes=0, prediction_value=255)
    add(kind=3, workflow=4, input_mode=0, prediction_value=256)
    add(workflow=4, input_mode=3, prediction_value=999)  # Wrong backfill input mode.
    add(snapshot_horizon_minutes=-30, prediction_value=999)  # Post-tip input.
    add(workflow=3, input_mode=0, prediction_value=999)  # Late live closing.
    cursor.execute(query, (Jsonb(inputs), Jsonb(runs), *params[2:]))
    frame = pd.DataFrame(
        cursor.fetchall(), columns=[column.name for column in cursor.description]
    )
    replay = frame.loc[
        frame["source"].eq("Snapshot replay") & frame["dataset_type"].eq(1)
    ]
    assert len(replay) == 3
    assert {
        (row.kind, row.bucket): row.prediction_value for row in replay.itertuples()
    } == {(1, 240): 254, (1, 0): 255, (3, 240): 256}
    assert not frame["prediction_value"].eq(999).any()
    live_final = frame.loc[
        frame["source"].eq("Live predictions") & frame["bucket"].eq(0)
    ]
    assert live_final["prediction_value"].tolist() == [249]


@pytest.mark.postgres
@pytest.mark.skipif(
    not os.getenv("NBA_OU_TEST_PG_DSN"), reason="NBA_OU_TEST_PG_DSN not set"
)
def test_postgres_reduces_refits_buckets_and_sources_without_future_information():
    with psycopg.connect(os.environ["NBA_OU_TEST_PG_DSN"]) as conn:
        conn.read_only = True
        with conn.cursor() as cursor:
            assert_synthetic_selection(cursor)
            assert_synthetic_closing_replay_selection(cursor)
