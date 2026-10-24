"""Write prediction runs and game results to the Aiven prediction store.

``write_run`` stores one run in one transaction: games and model builds are
upserted, then the run, its inputs and its predictions inserted. The run's
UUID is unique, so storing a run twice writes nothing the second time; a run
that fails part-way stores nothing. ``update_game_results`` is the only
writer of scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import psycopg
from psycopg import sql

from .schema import GAME_STATUS_FINAL, GAME_STATUS_SCHEDULED, SCHEMA

if TYPE_CHECKING:
    from nba_ou.prediction.prediction_store import RunTables

GAME_SCHEDULE_COLUMNS = (
    "game_id",
    "tipoff_at",
    "home_team_id",
    "away_team_id",
    "game_date",
    "season_year",
    "season_type",
    "home_team",
    "away_team",
)
MODEL_BUILD_COLUMNS = (
    "kind",
    "schema_version",
    "dataset_type",
    "target",
    "horizon_minutes",
    "variant",
    "fit_id",
    "spec_id",
    "train_date_min",
    "train_date_max",
    "training_code_tag",
    "anchor_book",
    "s3_build_prefix",
)
BUILD_KEY = MODEL_BUILD_COLUMNS[:7]
INPUT_COLUMNS = (
    "game_id",
    "input_mode",
    "snapshot_horizon_minutes",
    "tipoff_at",
    "total_model_line",
    "spread_model_line",
    "total_bet_line_x2",
    "total_price_over",
    "total_price_under",
    "total_line_age_minutes",
    "spread_bet_line_x2",
    "spread_price_home",
    "spread_price_away",
    "spread_line_age_minutes",
    "line_source",
)
INPUT_KEY = INPUT_COLUMNS[:3]
PREDICTION_COLUMNS = (
    "input_id",
    "model_build_id",
    "time_to_tipoff_seconds",
    "prediction_value",
    "pick_side",
    "na_feature_count",
    "input_limited_count",
)


def _records(frame: pd.DataFrame, columns) -> list[tuple]:
    """Rows of ``frame[columns]`` as tuples, with every missing value as None."""
    values = frame[list(columns)].astype(object)
    values = values.where(values.notna(), None)
    return [
        tuple(value.item() if hasattr(value, "item") else value for value in row)
        for row in values.itertuples(index=False, name=None)
    ]


def _table(name: str) -> sql.Composed:
    return sql.SQL("{}.{}").format(sql.Identifier(SCHEMA), sql.Identifier(name))


def _columns(columns) -> sql.Composed:
    return sql.SQL(", ").join(map(sql.Identifier, columns))


def _placeholders(columns) -> sql.Composed:
    return sql.SQL(", ").join(sql.Placeholder() * len(columns))


def _upsert_games(cur: psycopg.Cursor, games: pd.DataFrame) -> None:
    """Schedule fields only: a later run may move a tip-off, never a score."""
    updates = sql.SQL(", ").join(
        sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(column))
        for column in GAME_SCHEDULE_COLUMNS[1:]
    )
    cur.executemany(
        sql.SQL(
            "INSERT INTO {} ({}) VALUES ({}) ON CONFLICT (game_id) DO UPDATE SET {}"
        ).format(
            _table("games"),
            _columns(GAME_SCHEDULE_COLUMNS),
            _placeholders(GAME_SCHEDULE_COLUMNS),
            updates,
        ),
        _records(games, GAME_SCHEDULE_COLUMNS),
    )


def _model_build_ids(cur: psycopg.Cursor, builds: pd.DataFrame) -> pd.DataFrame:
    """``builds`` with their ``model_build_id``, inserting the unseen ones."""
    cur.executemany(
        sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING").format(
            _table("model_builds"),
            _columns(MODEL_BUILD_COLUMNS),
            _placeholders(MODEL_BUILD_COLUMNS),
        ),
        _records(builds, MODEL_BUILD_COLUMNS),
    )
    match = sql.SQL(" AND ").join(
        sql.SQL("{0} IS NOT DISTINCT FROM %s").format(sql.Identifier(column))
        for column in BUILD_KEY
    )
    query = sql.SQL("SELECT model_build_id FROM {} WHERE {}").format(
        _table("model_builds"), match
    )
    ids = []
    for key in _records(builds, BUILD_KEY):
        cur.execute(query, key)
        ids.append(cur.fetchone()[0])
    return builds.assign(model_build_id=ids)


def write_run(conn: psycopg.Connection, tables: RunTables) -> bool:
    """Store one run in one transaction. False when it was already stored."""
    run = tables.run
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "INSERT INTO {} (run_uuid, started_at, finished_at, workflow, "
                    "status, git_sha) VALUES (%s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (run_uuid) DO NOTHING RETURNING run_id"
                ).format(_table("runs")),
                (
                    run["run_uuid"],
                    run["started_at"],
                    run["finished_at"],
                    run["workflow"],
                    run["status"],
                    run["git_sha"],
                ),
            )
            inserted = cur.fetchone()
            if inserted is None:
                conn.rollback()
                return False
            run_id = inserted[0]

            if tables.predictions.empty:
                # A run that predicted nothing: its runs row is the record.
                conn.commit()
                return True
            _upsert_games(cur, tables.games)
            builds = _model_build_ids(cur, tables.model_builds)

            inputs = tables.inputs.assign(run_id=run_id)
            input_columns = ("run_id", *INPUT_COLUMNS)
            cur.executemany(
                sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                    _table("inputs"),
                    _columns(input_columns),
                    _placeholders(input_columns),
                ),
                _records(inputs, input_columns),
            )
            cur.execute(
                sql.SQL("SELECT input_id, {} FROM {} WHERE run_id = %s").format(
                    _columns(INPUT_KEY), _table("inputs")
                ),
                (run_id,),
            )
            input_ids = pd.DataFrame(cur.fetchall(), columns=["input_id", *INPUT_KEY])
            input_ids["snapshot_horizon_minutes"] = input_ids[
                "snapshot_horizon_minutes"
            ].astype("Int64")

            predictions = tables.predictions.merge(
                input_ids, on=list(INPUT_KEY), how="left", validate="many_to_one"
            ).merge(
                builds[[*BUILD_KEY, "model_build_id"]],
                on=list(BUILD_KEY),
                how="left",
                validate="many_to_one",
            )
            if predictions[["input_id", "model_build_id"]].isna().any().any():
                raise ValueError("a prediction matched no input row or model build")
            with cur.copy(
                sql.SQL("COPY {} ({}) FROM STDIN").format(
                    _table("predictions"), _columns(PREDICTION_COLUMNS)
                )
            ) as copy:
                for record in _records(predictions, PREDICTION_COLUMNS):
                    copy.write_row(record)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return True


def pending_games(conn: psycopg.Connection) -> pd.DataFrame:
    """Scheduled games whose tip-off has passed: the ones awaiting a result."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT game_id, game_date, season_year, tipoff_at FROM {} "
                "WHERE status = %s AND tipoff_at < now() ORDER BY game_id"
            ).format(_table("games")),
            (GAME_STATUS_SCHEDULED,),
        )
        return pd.DataFrame(
            cur.fetchall(), columns=["game_id", "game_date", "season_year", "tipoff_at"]
        )


def games_by_id(conn: psycopg.Connection, game_ids: list[int]) -> pd.DataFrame:
    """The stored games among ``game_ids``, whatever their status."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT game_id, game_date, season_year, tipoff_at FROM {} "
                "WHERE game_id = ANY(%s) ORDER BY game_id"
            ).format(_table("games")),
            ([int(game) for game in game_ids],),
        )
        return pd.DataFrame(
            cur.fetchall(), columns=["game_id", "game_date", "season_year", "tipoff_at"]
        )


def stale_pending_games(conn: psycopg.Connection, days: int) -> pd.DataFrame:
    """Games still scheduled ``days`` days after their stored tip-off."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT game_id, game_date, tipoff_at, home_team, away_team FROM {} "
                "WHERE status = %s AND tipoff_at < now() - make_interval(days => %s) "
                "ORDER BY tipoff_at"
            ).format(_table("games")),
            (GAME_STATUS_SCHEDULED, int(days)),
        )
        return pd.DataFrame(
            cur.fetchall(),
            columns=["game_id", "game_date", "tipoff_at", "home_team", "away_team"],
        )


def recently_final_games(conn: psycopg.Connection, days: int) -> pd.DataFrame:
    """Games given a final score in the last ``days`` days, to re-check."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT game_id, game_date, season_year, tipoff_at FROM {} "
                "WHERE status = %s "
                "AND result_updated_at > now() - make_interval(days => %s) "
                "ORDER BY game_id"
            ).format(_table("games")),
            (GAME_STATUS_FINAL, int(days)),
        )
        return pd.DataFrame(
            cur.fetchall(), columns=["game_id", "game_date", "season_year", "tipoff_at"]
        )


def update_game_results(conn: psycopg.Connection, results: pd.DataFrame) -> int:
    """Write final scores (``game_id``, ``home_points``, ``away_points``).

    Rows already holding the same final score are left alone, so a re-run
    changes nothing; a corrected score overwrites. Returns the rows changed.
    """
    if results.empty:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            sql.SQL(
                "UPDATE {} SET home_points = %s, away_points = %s, status = %s, "
                "result_updated_at = now() WHERE game_id = %s AND ("
                "status <> %s OR home_points IS DISTINCT FROM %s "
                "OR away_points IS DISTINCT FROM %s)"
            ).format(_table("games")),
            [
                (home, away, GAME_STATUS_FINAL, game_id, GAME_STATUS_FINAL, home, away)
                for game_id, home, away in _records(
                    results, ("game_id", "home_points", "away_points")
                )
            ],
            returning=False,
        )
        changed = cur.rowcount
    conn.commit()
    return max(changed, 0)


def retarget_build_prefixes(conn: psycopg.Connection, moved: dict[str, str]) -> int:
    """Point stored builds at their new S3 prefix after an archive move.

    ``moved`` maps each old build prefix to its new one
    (``scripts/archive_registry_builds.py``). Returns the rows changed.
    """
    if not moved:
        return 0
    with conn.cursor() as cur:
        cur.executemany(
            sql.SQL(
                "UPDATE {} SET s3_build_prefix = %s WHERE s3_build_prefix = %s"
            ).format(_table("model_builds")),
            [(new, old) for old, new in moved.items()],
        )
        changed = cur.rowcount
    conn.commit()
    return max(changed, 0)
