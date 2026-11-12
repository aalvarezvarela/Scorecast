"""Schema ``app`` reads and writes: users, manual requests and bets.

Every function takes an open connection of the dashboard role
(``open_app_connection``) and commits its own write. Values are always bound
parameters. See ``nba_ou.postgre_db.app_aiven.schema`` for the codes.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import psycopg

from nba_ou.dashboard.domain.access import AppUser, normalize_email
from nba_ou.postgre_db.app_aiven.requests import mark_request
from nba_ou.postgre_db.app_aiven.schema import (
    REQUEST_FAILED,
    USER_ACTIVE,
    USER_PENDING,
)

_USER_COLUMNS = "user_id, email, display_name, status, is_admin"

ENSURE_USER = f"""
    INSERT INTO app.users (email, display_name, status, is_admin, last_login_at)
    VALUES (%(email)s, %(name)s, %(status)s, %(admin)s, now())
    ON CONFLICT (email) DO UPDATE SET
        last_login_at = now(),
        display_name = coalesce(EXCLUDED.display_name, app.users.display_name),
        status = CASE WHEN EXCLUDED.is_admin THEN {USER_ACTIVE}
                      ELSE app.users.status END,
        is_admin = app.users.is_admin OR EXCLUDED.is_admin
    RETURNING {_USER_COLUMNS}
"""

READ_USER = f"SELECT {_USER_COLUMNS} FROM app.users WHERE email = %s"

LIST_USERS = """
    SELECT user_id, email, display_name, status, is_admin, created_at, last_login_at
    FROM app.users ORDER BY status, created_at
"""

CREATE_REQUEST = """
    INSERT INTO app.prediction_requests
        (user_id, kind, dataset_type, horizon_minutes, game_date)
    VALUES (%s, %s, %s, %s, %s)
    RETURNING request_id
"""

USER_REQUESTS = """
    SELECT request_id, kind, dataset_type, horizon_minutes, game_date, status,
           requested_at, started_at, finished_at, error
    FROM app.prediction_requests
    WHERE user_id = %s AND game_date = %s
    ORDER BY requested_at DESC
"""

#: The user's manual predictions of the slate, with the request that made them.
MANUAL_PREDICTIONS = """
    SELECT v.*, q.horizon_minutes AS request_horizon_minutes,
           q.requested_at
    FROM prediction.predictions_full v
    JOIN app.prediction_requests q ON q.request_id = v.request_id
    WHERE q.user_id = %s AND v.game_date = %s AND v.kind IN (1, 3)
    ORDER BY v.predicted_at, v.run_id, v.input_id, v.model_build_id
"""

LAST_STAKE = """
    SELECT stake FROM app.bets
    WHERE user_id = %s AND voided_at IS NULL
    ORDER BY placed_at DESC LIMIT 1
"""

INSERT_BET = f"""
    INSERT INTO app.bets (user_id, game_id, market, side, line, odds_decimal,
                          stake, book, model_build_id, input_id, request_id)
    SELECT u.user_id, %(game_id)s::integer, %(market)s::smallint,
           %(side)s::smallint, %(line)s::numeric, %(odds_decimal)s::numeric,
           %(stake)s::numeric, %(book)s::text, %(model_build_id)s::integer,
           %(input_id)s::integer, %(request_id)s::integer
    FROM app.users u
    WHERE u.user_id = %(user_id)s AND u.status = {USER_ACTIVE}
    RETURNING bet_id
"""

USER_BETS = """
    SELECT bet_id, game_id, game_date, tipoff_at, home_team, away_team, market,
           side, line, odds_decimal, stake, book, placed_at, voided_at,
           home_points, away_points, game_status, result, profit,
           model_build_id, request_id
    FROM app.bets_full
    WHERE user_id = %s
    ORDER BY placed_at DESC
    LIMIT %s
"""

#: The Bets page: the user's live bets on one slate. A voided bet is a
#: deleted one there; it stays in ``app.bets`` for the history.
SLATE_BETS = """
    SELECT bet_id, game_id, game_date, tipoff_at, home_team, away_team, market,
           side, line, odds_decimal, stake, book, placed_at, voided_at,
           home_points, away_points, game_status, result, profit,
           model_build_id, request_id
    FROM app.bets_full
    WHERE user_id = %s AND game_date = %s AND voided_at IS NULL
    ORDER BY tipoff_at, game_id, placed_at
"""

#: Line, odds and stake can be corrected until the game tips off.
UPDATE_BET = """
    UPDATE app.bets b
    SET line = %(line)s::numeric, odds_decimal = %(odds_decimal)s::numeric,
        stake = %(stake)s::numeric
    FROM prediction.games g
    WHERE b.bet_id = %(bet_id)s AND b.user_id = %(user_id)s
      AND b.voided_at IS NULL
      AND g.game_id = b.game_id AND g.tipoff_at > now()
"""

#: A bet can be voided until its game tips off.
VOID_BET = """
    UPDATE app.bets b SET voided_at = now()
    FROM prediction.games g
    WHERE b.bet_id = %s AND b.user_id = %s AND b.voided_at IS NULL
      AND g.game_id = b.game_id AND g.tipoff_at > now()
"""


def _frame(cur: psycopg.Cursor, query: str, params) -> pd.DataFrame:
    cur.execute(query, params)
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def _user(row) -> AppUser | None:
    if row is None:
        return None
    user_id, email, name, status, is_admin = row
    return AppUser(int(user_id), email, name, int(status), bool(is_admin))


def ensure_user(
    conn: psycopg.Connection, email: str, name: str | None, *, admins: frozenset[str]
) -> AppUser:
    """The signed-in account's row, created pending on its first sign-in
    (active and admin for a configured admin email)."""
    email = normalize_email(email)
    if not email:
        raise ValueError("The identity provider gave no email")
    admin = email in admins
    with conn.cursor() as cur:
        cur.execute(
            ENSURE_USER,
            {
                "email": email,
                "name": name or None,
                "status": USER_ACTIVE if admin else USER_PENDING,
                "admin": admin,
            },
        )
        user = _user(cur.fetchone())
    conn.commit()
    return user


def read_user(conn: psycopg.Connection, email: str) -> AppUser | None:
    with conn.cursor() as cur:
        cur.execute(READ_USER, (normalize_email(email),))
        return _user(cur.fetchone())


def list_users(conn: psycopg.Connection) -> pd.DataFrame:
    with conn.cursor() as cur:
        return _frame(cur, LIST_USERS, ())


def set_user_status(
    conn: psycopg.Connection, user_id: int, status: int, *, by: AppUser
) -> bool:
    """An admin approves or disables another account (never their own)."""
    if not by.is_admin:
        raise PermissionError("Only an admin changes accounts")
    if int(user_id) == by.user_id:
        raise ValueError("An admin cannot change their own account")
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE app.users SET status = %s WHERE user_id = %s",
            (int(status), int(user_id)),
        )
        updated = cur.rowcount > 0
    conn.commit()
    return updated


def create_request(
    conn: psycopg.Connection,
    *,
    user_id: int,
    kind: int,
    dataset_type: int,
    horizon_minutes: int | None,
    game_date: date,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            CREATE_REQUEST, (user_id, kind, dataset_type, horizon_minutes, game_date)
        )
        request_id = int(cur.fetchone()[0])
    conn.commit()
    return request_id


def fail_request(conn: psycopg.Connection, request_id: int, error: str) -> None:
    mark_request(conn, request_id, REQUEST_FAILED, error=error)


def user_requests(
    conn: psycopg.Connection, user_id: int, game_date: date
) -> pd.DataFrame:
    with conn.cursor() as cur:
        return _frame(cur, USER_REQUESTS, (user_id, game_date))


def manual_predictions(
    conn: psycopg.Connection, user_id: int, game_date: date
) -> pd.DataFrame:
    with conn.cursor() as cur:
        return _frame(cur, MANUAL_PREDICTIONS, (user_id, game_date))


def last_stake(conn: psycopg.Connection, user_id: int) -> float | None:
    with conn.cursor() as cur:
        cur.execute(LAST_STAKE, (user_id,))
        row = cur.fetchone()
    return None if row is None else float(row[0])


def insert_bet(conn: psycopg.Connection, bet: dict) -> int | None:
    """``bet``: user_id, game_id, market, side, line, odds_decimal, stake,
    book, and the optional model_build_id, input_id, request_id. None, and
    nothing stored, when the account is not active."""
    values = {
        "model_build_id": None,
        "input_id": None,
        "request_id": None,
        **bet,
    }
    with conn.cursor() as cur:
        cur.execute(INSERT_BET, values)
        row = cur.fetchone()
    conn.commit()
    return None if row is None else int(row[0])


def user_bets(conn: psycopg.Connection, user_id: int, limit: int = 500) -> pd.DataFrame:
    with conn.cursor() as cur:
        return _frame(cur, USER_BETS, (user_id, limit))


def slate_bets(conn: psycopg.Connection, user_id: int, game_date: date) -> pd.DataFrame:
    """The user's bets on the games of ``game_date``, voided ones left out."""
    with conn.cursor() as cur:
        return _frame(cur, SLATE_BETS, (user_id, game_date))


def update_bet(
    conn: psycopg.Connection,
    bet_id: int,
    user_id: int,
    *,
    line: float,
    odds_decimal: float,
    stake: float,
) -> bool:
    """False when the bet is not the user's, is void or its game started."""
    values = {
        "bet_id": bet_id,
        "user_id": user_id,
        "line": line,
        "odds_decimal": odds_decimal,
        "stake": stake,
    }
    with conn.cursor() as cur:
        cur.execute(UPDATE_BET, values)
        updated = cur.rowcount > 0
    conn.commit()
    return updated


def void_bet(conn: psycopg.Connection, bet_id: int, user_id: int) -> bool:
    """False when the bet is not the user's, already void or its game started."""
    with conn.cursor() as cur:
        cur.execute(VOID_BET, (bet_id, user_id))
        voided = cur.rowcount > 0
    conn.commit()
    return voided
