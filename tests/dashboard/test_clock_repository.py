"""Bounded hourly reads and production SQL exercised with synthetic PostgreSQL data."""

import os
from datetime import timedelta
from unittest.mock import MagicMock

import pandas as pd
import psycopg
import pytest
from nba_ou.dashboard.repositories import clock
from nba_ou.dashboard.repositories.predictions import DashboardDataError
from psycopg.types.json import Jsonb

from .test_clock import AS_OF, DAY, SOURCES
from .test_historical_repository import FIXTURE_COLUMNS


def read(**kwargs):
    return clock.read_clock_history(DAY, DAY, 1, None, **kwargs)


def test_clock_read_is_bounded_read_only_and_binds_all_filters(monkeypatch):
    conn = MagicMock()
    conn.__enter__.return_value = conn
    monkeypatch.setattr(clock, "open_connection", MagicMock(return_value=conn))
    reader = MagicMock(side_effect=[pd.DataFrame()] * 2)
    monkeypatch.setattr(clock, "_frame", reader)
    data = read()
    assert (
        conn.read_only
        and conn.isolation_level == psycopg.IsolationLevel.REPEATABLE_READ
    )
    assert reader.call_count == 2
    expected = (
        DAY,
        DAY,
        1,
        [1, 2, 3, 4, 5],
        "2_5",
        data.fetched_at,
        data.fetched_at,
        ["_thinking_high"],
        clock.MAX_CLOCK_ROWS + 1,
    )
    assert reader.call_args_list[1].args[2] == expected
    monkeypatch.setattr(clock, "MAX_CLOCK_ROWS", 1)
    reader.side_effect = [pd.DataFrame(), pd.DataFrame({"game_id": [1, 2]})]
    with pytest.raises(DashboardDataError, match="no partial results"):
        read()


@pytest.mark.parametrize(
    "error,phrase",
    [
        (psycopg.errors.QueryCanceled("sensitive"), "timed out"),
        (psycopg.errors.UndefinedColumn("sensitive"), "schema needs updating"),
        (psycopg.errors.InsufficientPrivilege("sensitive"), "read access"),
        (psycopg.OperationalError("sensitive"), "Aiven"),
        (ValueError("sensitive"), "credentials"),
    ],
)
def test_clock_read_sanitizes_errors(monkeypatch, error, phrase):
    monkeypatch.setattr(clock, "open_connection", MagicMock(side_effect=error))
    with pytest.raises(DashboardDataError) as caught:
        read()
    assert phrase in str(caught.value) and "sensitive" not in str(caught.value)


def test_clock_invalid_filters_never_connect(monkeypatch):
    opener = MagicMock()
    monkeypatch.setattr(clock, "open_connection", opener)
    for start, end, target, season in [
        (DAY, DAY - timedelta(days=1), 1, 2),
        (DAY, DAY, 99, 2),
        (DAY, DAY, 1, 99),
    ]:
        with pytest.raises(ValueError):
            clock.read_clock_history(start, end, target, season)
    opener.assert_not_called()


def clock_sql_fixture():
    predictions, runs, inputs, builds = [], [], [], []
    tipoff = pd.Timestamp("2026-11-03T03:00Z")

    def add(
        *,
        at="2026-11-02T17:00Z",
        finish="2026-11-02T17:20Z",
        horizon=360,
        value=232,
        replay=False,
        train_date=DAY - timedelta(days=1),
        **changes,
    ):
        run_id = len(predictions) + 1
        instant = pd.Timestamp(at)
        predicted = pd.Timestamp(finish) - pd.Timedelta(minutes=1)
        minutes = int((tipoff - instant).total_seconds() / 60)
        row = dict(
            game_id=1,
            game_date=DAY,
            season_type=2,
            kind=1,
            schema_version="2_5",
            target=2,
            dataset_type=1 if horizon == 0 else 2,
            model_horizon_minutes=horizon,
            variant="main",
            workflow=4 if replay else 1,
            input_mode=(
                (0 if horizon == 0 else 1) if replay else (0 if horizon == 0 else 2)
            ),
            snapshot_horizon_minutes=minutes if replay or horizon else None,
            prediction_value=value,
            model_line=230.0,
            bet_line_x2=460,
            bet_price_over_home=-110,
            bet_price_under_away=-105,
            home_points=118,
            away_points=115,
            status=1,
            predicted_at=predicted,
            tipoff_at=tipoff,
            tipoff_at_prediction=tipoff,
            time_to_tipoff_seconds=int((tipoff - predicted).total_seconds()),
            run_id=run_id,
            input_id=run_id,
            model_build_id=run_id,
            run_started_at=instant,
        )
        row.update(changes)
        predictions.append(row)
        runs.append(dict(run_id=run_id, finished_at=pd.Timestamp(finish)))
        inputs.append(
            dict(
                input_id=run_id, total_line_age_minutes=2.0, spread_line_age_minutes=2.0
            )
        )
        builds.append(dict(model_build_id=run_id, train_date_max=train_date))

    add(horizon=0, value=231)
    add(horizon=0, at="2026-11-02T17:40Z", finish="2026-11-02T17:55Z", value=234)
    add(horizon=0, at="2026-11-02T17:50Z", finish="2026-11-02T18:01Z", value=999)
    add(value=226)
    add(at="2026-11-02T17:30Z", finish="2026-11-02T17:50Z", value=236)
    add(horizon=480, value=240)
    add(
        kind=3,
        horizon=555,
        at="2026-11-02T17:45Z",
        finish="2026-11-02T17:50Z",
        value=238,
        bet_line_x2=484,
    )
    add(kind=3, horizon=555, finish="2026-11-02T17:59Z", value=999)
    add(kind=3, variant="tabpfn_thinking_high", value=999)
    add(
        kind=3,
        horizon=120,
        at="2026-11-02T18:01Z",
        finish="2026-11-02T18:05Z",
        value=999,
    )
    add(horizon=0, replay=True, finish="2026-11-03T06:00Z", value=250)
    add(
        horizon=0,
        replay=True,
        at="2026-11-02T18:00Z",
        finish="2026-11-03T06:10Z",
        value=252,
    )
    add(horizon=600, replay=True, workflow=2, finish="2026-11-03T06:00Z", value=242)
    add(
        horizon=600,
        replay=True,
        workflow=2,
        finish="2026-11-03T06:30Z",
        train_date=DAY,
        value=999,
    )
    add(schema_version="2_6", value=999)
    # A previous prediction-time tip-off must also be after the cutoff.
    add(horizon=30, tipoff_at_prediction=pd.Timestamp("2026-11-02T17:30Z"), value=999)
    game = dict(
        game_id=1,
        game_date=DAY,
        season_type=2,
        tipoff_at=tipoff,
        home_team="Boston",
        away_team="New York",
        status=1,
        home_points=118,
        away_points=115,
    )
    prefix = """WITH fixture_predictions AS (
        SELECT * FROM jsonb_to_recordset(%s) AS rows(COLUMNS)
    ), fixture_runs AS (
        SELECT * FROM jsonb_to_recordset(%s) AS rows(run_id integer, finished_at timestamptz, request_id integer)
    ), fixture_games AS (
        SELECT * FROM jsonb_to_recordset(%s) AS rows(game_id integer, game_date date,
            season_type integer, tipoff_at timestamptz, home_team text, away_team text,
            status integer, home_points integer, away_points integer)
    ), fixture_inputs AS (
        SELECT * FROM jsonb_to_recordset(%s) AS rows(input_id integer,
            total_line_age_minutes real, spread_line_age_minutes real)
    ), fixture_builds AS (
        SELECT * FROM jsonb_to_recordset(%s) AS rows(model_build_id integer, train_date_max date)
    ), base AS""".replace(
        "COLUMNS", FIXTURE_COLUMNS
    )

    def localized(query):
        query = query.replace("WITH base AS", prefix)
        for table, fixture in (
            ("predictions_full", "predictions"),
            ("runs", "runs"),
            ("games", "games"),
            ("inputs", "inputs"),
            ("model_builds", "builds"),
        ):
            query = query.replace(f"prediction.{table}", f"fixture_{fixture}")
        return query

    def json_rows(rows):
        return Jsonb(
            [
                {
                    key: value.isoformat() if hasattr(value, "isoformat") else value
                    for key, value in row.items()
                }
                for row in rows
            ]
        )

    shared = (
        json_rows(predictions),
        json_rows(runs),
        json_rows([game]),
        json_rows(inputs),
        json_rows(builds),
        DAY,
        DAY,
        2,
        [2],
        "2_5",
        AS_OF,
        AS_OF,
    )
    return localized, shared


def run(query, params):
    with psycopg.connect(os.environ["NBA_OU_TEST_PG_DSN"]) as conn:
        conn.read_only = True
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            return pd.DataFrame(
                cursor.fetchall(), columns=[c.name for c in cursor.description]
            )


@pytest.mark.postgres
@pytest.mark.skipif(
    not os.getenv("NBA_OU_TEST_PG_DSN"), reason="NBA_OU_TEST_PG_DSN not set"
)
def test_postgres_reads_every_visible_model_with_its_own_line():
    localized, shared = clock_sql_fixture()
    params = (*shared, ["_thinking_high"], 200001)
    models = run(localized(clock.CLOCK_MODELS_QUERY), params)
    live = models.loc[models["source"].eq(SOURCES[0]), "prediction_value"]
    assert sorted(live) == [226, 231, 234, 236, 238, 240, 999, 999, 999, 999]
    replay = models.loc[models["source"].eq(SOURCES[1]), "prediction_value"]
    assert sorted(replay) == [242, 250, 252]
    assert not models["variant"].str.endswith("_thinking_high").any()
    assert set(models["bet_line_x2"]) == {460, 484}
    assert set(models["bet_price_under_away"]) == {-105}
