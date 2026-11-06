"""Schema ``app`` on Aiven: dashboard users, manual prediction requests, bets.

See ``docs/serving/manual_predictions.md``. It lives next to the prediction
store (schema ``prediction``) and references it: a bet names the game and,
optionally, the prediction it followed; a manual run's ``prediction.runs``
row names the request it answers (``runs.request_id``, foreign key added
here, so the prediction store can be created without this schema).

Codes (``SMALLINT``):

* ``users.status``: 0 pending (signed in, not approved), 1 active, 2 disabled.
* ``prediction_requests.kind``: 1 XGBoost (registry models), 3 TabPFN -- the
  ``prediction.model_builds.kind`` codes.
* ``prediction_requests.dataset_type``: 1 closing, 2 intermediate.
* ``prediction_requests.horizon_minutes``: intermediate only; NULL means each
  game's nearest horizon, a number one fixed horizon for every game.
* ``prediction_requests.status``: 0 queued, 1 running, 2 done, 3 partial
  (some model failed), 4 failed.
* ``bets.market``: 1 totals, 2 spread.
* ``bets.side``: +1 OVER/HOME, -1 UNDER/AWAY (``prediction.predictions.pick_side``).
* ``bets.line``: totals, the total; spread, the HOME handicap as books print
  it (home -5.5 is ``-5.5``), not the implied home margin of the
  prediction store.

``bets_full`` grades every bet against ``prediction.games``: ``result`` is
``pending``, ``win``, ``loss``, ``push`` or ``void`` and ``profit`` the net
return in the stake's currency (NULL while pending).
"""

from __future__ import annotations

import psycopg
from psycopg import sql

SCHEMA = "app"
PREDICTION_SCHEMA = "prediction"
#: The dashboard's own database role (``create_app_store.py --role``).
DASHBOARD_ROLE = "dashboard_app"

USER_PENDING = 0
USER_ACTIVE = 1
USER_DISABLED = 2

REQUEST_QUEUED = 0
REQUEST_RUNNING = 1
REQUEST_DONE = 2
REQUEST_PARTIAL = 3
REQUEST_FAILED = 4
REQUEST_STATUS_NAMES: dict[int, str] = {
    REQUEST_QUEUED: "queued",
    REQUEST_RUNNING: "running",
    REQUEST_DONE: "done",
    REQUEST_PARTIAL: "partial",
    REQUEST_FAILED: "failed",
}
REQUEST_STATUS_CODES: dict[str, int] = {
    name: code for code, name in REQUEST_STATUS_NAMES.items()
}

MARKET_TOTALS = 1
MARKET_SPREAD = 2

DDL_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS {schema}.users (
        user_id       SMALLINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        email         TEXT        NOT NULL UNIQUE,
        display_name  TEXT,
        status        SMALLINT    NOT NULL DEFAULT 0,
        is_admin      BOOLEAN     NOT NULL DEFAULT false,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        last_login_at TIMESTAMPTZ
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS {schema}.prediction_requests (
        request_id      INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        user_id         SMALLINT    NOT NULL REFERENCES {schema}.users,
        kind            SMALLINT    NOT NULL,
        dataset_type    SMALLINT    NOT NULL,
        horizon_minutes SMALLINT,
        game_date       DATE        NOT NULL,
        status          SMALLINT    NOT NULL DEFAULT 0,
        requested_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
        started_at      TIMESTAMPTZ,
        finished_at     TIMESTAMPTZ,
        error           TEXT,
        CHECK (kind IN (1, 3)),
        CHECK (dataset_type IN (1, 2)),
        CHECK (dataset_type = 2 OR horizon_minutes IS NULL)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS prediction_requests_user_date
        ON {schema}.prediction_requests (user_id, game_date)
    """,
    """
    CREATE TABLE IF NOT EXISTS {schema}.bets (
        bet_id         INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        user_id        SMALLINT      NOT NULL REFERENCES {schema}.users,
        game_id        INTEGER       NOT NULL REFERENCES {prediction}.games,
        market         SMALLINT      NOT NULL,
        side           SMALLINT      NOT NULL,
        line           NUMERIC(5, 1) NOT NULL,
        odds_decimal   NUMERIC(6, 3) NOT NULL,
        stake          NUMERIC(10, 2) NOT NULL,
        book           TEXT          NOT NULL DEFAULT 'bet365',
        model_build_id INTEGER REFERENCES {prediction}.model_builds,
        input_id       INTEGER REFERENCES {prediction}.inputs,
        request_id     INTEGER REFERENCES {schema}.prediction_requests,
        placed_at      TIMESTAMPTZ   NOT NULL DEFAULT now(),
        voided_at      TIMESTAMPTZ,
        CHECK (market IN (1, 2)),
        CHECK (side IN (-1, 1)),
        CHECK (odds_decimal > 1),
        CHECK (stake > 0)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS bets_user_placed
        ON {schema}.bets (user_id, placed_at)
    """,
    # The prediction store creates the column; the key needs this schema.
    """
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname = 'runs_request_id_fkey'
              AND conrelid = '{prediction}.runs'::regclass
        ) THEN
            ALTER TABLE {prediction}.runs
                ADD CONSTRAINT runs_request_id_fkey
                FOREIGN KEY (request_id) REFERENCES {schema}.prediction_requests;
        END IF;
    END
    $$
    """,
    """
    CREATE INDEX IF NOT EXISTS runs_request_id
        ON {prediction}.runs (request_id) WHERE request_id IS NOT NULL
    """,
    """
    DROP VIEW IF EXISTS {schema}.bets_full
    """,
    """
    CREATE VIEW {schema}.bets_full AS
    WITH graded AS (
        SELECT b.*,
               u.email, u.display_name,
               g.game_date, g.tipoff_at, g.home_team, g.away_team,
               g.home_points, g.away_points, g.status AS game_status,
               CASE WHEN b.market = 2
                    THEN g.home_points - g.away_points + b.line
                    ELSE g.home_points + g.away_points - b.line
               END AS cover_margin
        FROM {schema}.bets b
        JOIN {schema}.users u        USING (user_id)
        JOIN {prediction}.games g    USING (game_id)
    ), resulted AS (
        SELECT graded.*,
               CASE
                 WHEN voided_at IS NOT NULL OR game_status = 2 THEN 'void'
                 WHEN game_status <> 1 OR cover_margin IS NULL THEN 'pending'
                 WHEN cover_margin = 0 THEN 'push'
                 WHEN sign(cover_margin) = side THEN 'win'
                 ELSE 'loss'
               END AS result
        FROM graded
    )
    SELECT resulted.*,
           CASE result
             WHEN 'win'  THEN round(stake * (odds_decimal - 1), 2)
             WHEN 'loss' THEN -stake
             WHEN 'pending' THEN NULL
             ELSE 0
           END AS profit
    FROM resulted
    """,
)


def _render(statement: str) -> str:
    return statement.format(schema=SCHEMA, prediction=PREDICTION_SCHEMA)


def create_schema(conn: psycopg.Connection) -> None:
    """Create schema ``app``: three tables, their indexes and ``bets_full``.

    The prediction store (``predictions_aiven.schema.create_schema``) must
    exist first: bets and the ``runs.request_id`` key reference it.
    """
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(SCHEMA))
        )
        for statement in DDL_STATEMENTS:
            cur.execute(sql.SQL(_render(statement)))
    conn.commit()


def grant_dashboard_role(conn: psycopg.Connection, role: str = DASHBOARD_ROLE) -> None:
    """Give ``role`` what the dashboard needs and nothing more.

    It reads the prediction store and reads and writes ``app`` (no DELETE:
    a bet is voided, a user disabled). The role itself is created by the
    caller, with its password.
    """
    statements = (
        "GRANT USAGE ON SCHEMA {prediction} TO {role}",
        "GRANT SELECT ON ALL TABLES IN SCHEMA {prediction} TO {role}",
        "GRANT USAGE ON SCHEMA {schema} TO {role}",
        "GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA {schema} TO {role}",
        "GRANT USAGE ON ALL SEQUENCES IN SCHEMA {schema} TO {role}",
    )
    with conn.cursor() as cur:
        for statement in statements:
            cur.execute(
                sql.SQL(statement).format(
                    prediction=sql.Identifier(PREDICTION_SCHEMA),
                    schema=sql.Identifier(SCHEMA),
                    role=sql.Identifier(role),
                )
            )
    conn.commit()


def ddl_script() -> str:
    """Every statement ``create_schema`` runs, as one SQL script for review."""
    statements = [f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"]
    statements += [
        "\n".join(line[4:] for line in _render(s).splitlines()).strip()
        for s in DDL_STATEMENTS
    ]
    return ";\n\n".join(statements) + ";\n"
