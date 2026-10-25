"""Schema for the Aiven prediction store (``docs/serving/prediction_storage.md``).

Sized for the 1 GB instance it shares with the line history. The one large
table, ``predictions``, carries only what differs per model -- 22 bytes of
data per row, no B-tree key. Everything a row's models share (lines, prices,
timing) lives once in ``inputs``; game, run and model identity in three small
tables. ``predictions_full`` joins them back and grades, so analyses read one
view and changing a grading rule never rewrites rows.

Codes (``SMALLINT``):

* ``dataset_type``: 1 closing_line, 2 intermediate_line.
* ``target``: 1 line_error, 2 total_points, 3 spread_error.
* ``input_mode``: 0 closing (the closing frame), 1 own (a model's own
  snapshot), 2 latest (the game's latest snapshot, read by every model).
* ``pick_side``: +1 OVER/HOME, -1 UNDER/AWAY, 0 no pick.
* ``games.status``: 0 scheduled, 1 final, 2 postponed/void.
* ``runs.workflow``: 1 daily predictor, 2 past-snapshot run, 3 TabPFN run.
* ``runs.status``: 1 ok, 2 partial (some slot failed), 3 S3 upload failed.
* ``model_builds.kind``: 1 registry model, 2 baseline, 3 TabPFN (fitted per
  run on a registry slot's spec; ``spec_id`` is that spec).
* ``inputs.line_source``: where the row's lines came from -- 1 SBR, 2 Covers
  (the fallback when SBR is down or frozen, ``prediction.line_source``);
  NULL on rows stored before the column existed.
"""

from __future__ import annotations

import psycopg
from psycopg import sql

SCHEMA = "prediction"

DATASET_TYPE_CODES: dict[str, int] = {"closing_line": 1, "intermediate_line": 2}
TARGET_CODES: dict[str, int] = {"line_error": 1, "total_points": 2, "spread_error": 3}
INPUT_MODE_CLOSING = 0
INPUT_MODE_OWN = 1
INPUT_MODE_LATEST = 2
INPUT_MODE_CODES: dict[str, int] = {
    "closing": INPUT_MODE_CLOSING,
    "own": INPUT_MODE_OWN,
    "latest": INPUT_MODE_LATEST,
}
WORKFLOW_DAILY = 1
WORKFLOW_PAST_SNAPSHOTS = 2
WORKFLOW_TABPFN = 3
#: The workflow's name in the run's S3 archive folder.
WORKFLOW_NAMES: dict[int, str] = {
    WORKFLOW_DAILY: "daily",
    WORKFLOW_PAST_SNAPSHOTS: "past_snapshots",
    WORKFLOW_TABPFN: "tabpfn",
}
RUN_STATUS_OK = 1
RUN_STATUS_PARTIAL = 2
RUN_STATUS_S3_FAILED = 3
KIND_REGISTRY = 1
KIND_BASELINE = 2
KIND_TABPFN = 3
GAME_STATUS_SCHEDULED = 0
GAME_STATUS_FINAL = 1
LINE_SOURCE_SBR = 1
LINE_SOURCE_COVERS = 2
LINE_SOURCE_CODES: dict[str, int] = {
    "sbr": LINE_SOURCE_SBR,
    "covers": LINE_SOURCE_COVERS,
}

DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS {schema}.games (
        game_id           INTEGER PRIMARY KEY,
        tipoff_at         TIMESTAMPTZ NOT NULL,
        result_updated_at TIMESTAMPTZ,
        home_team_id      INTEGER     NOT NULL,
        away_team_id      INTEGER     NOT NULL,
        game_date         DATE        NOT NULL,
        season_year       SMALLINT    NOT NULL,
        season_type       SMALLINT    NOT NULL,
        home_points       SMALLINT,
        away_points       SMALLINT,
        status            SMALLINT    NOT NULL DEFAULT 0,
        home_team         TEXT        NOT NULL,
        away_team         TEXT        NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {schema}.runs (
        run_id      INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        run_uuid    UUID        NOT NULL UNIQUE,
        started_at  TIMESTAMPTZ NOT NULL,
        finished_at TIMESTAMPTZ NOT NULL,
        workflow    SMALLINT    NOT NULL,
        status      SMALLINT    NOT NULL,
        git_sha     TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {schema}.model_builds (
        model_build_id    INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        train_date_min    DATE,
        train_date_max    DATE,
        kind              SMALLINT NOT NULL,
        dataset_type      SMALLINT NOT NULL,
        target            SMALLINT NOT NULL,
        horizon_minutes   SMALLINT,
        schema_version    TEXT     NOT NULL,
        variant           TEXT     NOT NULL,
        spec_id           TEXT     NOT NULL,
        fit_id            TEXT     NOT NULL,
        training_code_tag TEXT,
        anchor_book       TEXT     NOT NULL,
        s3_build_prefix   TEXT,
        UNIQUE NULLS NOT DISTINCT
            (kind, schema_version, dataset_type, target, horizon_minutes,
             variant, fit_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {schema}.inputs (
        input_id                 INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        run_id                   INTEGER     NOT NULL REFERENCES {schema}.runs,
        game_id                  INTEGER     NOT NULL REFERENCES {schema}.games,
        tipoff_at                TIMESTAMPTZ NOT NULL,
        total_model_line         REAL,
        spread_model_line        REAL,
        total_line_age_minutes   REAL,
        spread_line_age_minutes  REAL,
        input_mode               SMALLINT    NOT NULL,
        snapshot_horizon_minutes SMALLINT,
        total_bet_line_x2        SMALLINT,
        total_price_over         SMALLINT,
        total_price_under        SMALLINT,
        spread_bet_line_x2       SMALLINT,
        spread_price_home        SMALLINT,
        spread_price_away        SMALLINT,
        line_source              SMALLINT,
        UNIQUE NULLS NOT DISTINCT
            (run_id, game_id, input_mode, snapshot_horizon_minutes)
    )
    """,
    # Added 2026-10-24 to a table created without it.
    """
    ALTER TABLE {schema}.inputs ADD COLUMN IF NOT EXISTS line_source SMALLINT
    """,
    # No key: a B-tree on (input_id, model_build_id) would weigh about as much
    # as the table. The run transaction prevents duplicates, and rows arrive in
    # input_id order, so a BRIN index finds a run's or a date's rows.
    """
    CREATE TABLE IF NOT EXISTS {schema}.predictions (
        input_id               INTEGER  NOT NULL REFERENCES {schema}.inputs,
        model_build_id         INTEGER  NOT NULL REFERENCES {schema}.model_builds,
        time_to_tipoff_seconds INTEGER  NOT NULL,
        prediction_value       REAL     NOT NULL,
        pick_side              SMALLINT NOT NULL,
        na_feature_count       SMALLINT NOT NULL,
        input_limited_count    SMALLINT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS predictions_input_brin
        ON {schema}.predictions USING brin (input_id)
    """,
    # Dropped and recreated, not replaced: CREATE OR REPLACE VIEW can only add
    # columns at the end, and a new column of ``base`` lands in the middle.
    """
    DROP VIEW IF EXISTS {schema}.predictions_full
    """,
    """
    CREATE VIEW {schema}.predictions_full AS
    WITH base AS (
        SELECT p.*,
               i.run_id, i.game_id, i.input_mode, i.snapshot_horizon_minutes,
               i.line_source,
               i.tipoff_at - make_interval(secs => p.time_to_tipoff_seconds)
                                                         AS predicted_at,
               i.tipoff_at                               AS tipoff_at_prediction,
               r.run_uuid, r.workflow, r.status          AS run_status,
               r.started_at                              AS run_started_at,
               m.kind, m.schema_version, m.dataset_type, m.target,
               m.horizon_minutes                         AS model_horizon_minutes,
               m.variant, m.fit_id, m.training_code_tag,
               g.game_date, g.season_year, g.season_type, g.tipoff_at,
               g.home_team, g.away_team, g.status,
               g.home_points, g.away_points,
               CASE WHEN m.target = 3 THEN i.spread_model_line
                    ELSE i.total_model_line END          AS model_line,
               CASE WHEN m.target = 3 THEN i.spread_bet_line_x2
                    ELSE i.total_bet_line_x2 END         AS bet_line_x2,
               CASE WHEN m.target = 3 THEN i.spread_price_home
                    ELSE i.total_price_over END          AS bet_price_over_home,
               CASE WHEN m.target = 3 THEN i.spread_price_away
                    ELSE i.total_price_under END         AS bet_price_under_away,
               CASE WHEN m.target = 3 THEN i.spread_line_age_minutes
                    ELSE i.total_line_age_minutes END    AS bet_line_age_minutes,
               CASE WHEN m.target = 3 THEN g.home_points - g.away_points
                    ELSE g.home_points + g.away_points END AS actual_value
        FROM {schema}.predictions p
        JOIN {schema}.inputs i       USING (input_id)
        JOIN {schema}.model_builds m USING (model_build_id)
        JOIN {schema}.games g        USING (game_id)
        JOIN {schema}.runs r         ON r.run_id = i.run_id
    ), implied AS (
        SELECT base.*,
               -- The model's implied total or home margin.
               CASE WHEN target = 2 THEN prediction_value
                    ELSE model_line + prediction_value END AS predicted_value
        FROM base
    )
    SELECT implied.*,
           predicted_value - model_line                        AS model_edge,
           predicted_value - bet_line_x2 / 2.0                 AS bet_edge,
           sign(2 * predicted_value - bet_line_x2)::smallint   AS bet_side,
           (model_horizon_minutes = snapshot_horizon_minutes)  AS on_horizon,
           CASE
             WHEN status <> 1 OR pick_side = 0 OR bet_line_x2 IS NULL THEN NULL
             WHEN 2 * actual_value = bet_line_x2 THEN 0
             WHEN sign(2 * actual_value - bet_line_x2) = pick_side THEN 1
             ELSE -1
           END::smallint                                       AS pick_result
    FROM implied
    """,
)


def create_schema(conn: psycopg.Connection) -> None:
    """Create the schema, its five tables, the BRIN index and the view."""
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(SCHEMA))
        )
        for statement in DDL_STATEMENTS:
            cur.execute(sql.SQL(statement.format(schema=SCHEMA)))
    conn.commit()


def ddl_script() -> str:
    """Every statement ``create_schema`` runs, as one SQL script for review."""
    statements = [f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"]
    statements += [
        "\n".join(line[4:] for line in s.format(schema=SCHEMA).splitlines()).strip()
        for s in DDL_STATEMENTS
    ]
    return ";\n\n".join(statements) + ";\n"
