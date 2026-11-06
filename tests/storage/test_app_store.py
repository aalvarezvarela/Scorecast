"""Schema ``app``: accounts, manual requests and graded bets, on a real PostgreSQL.

The grading in ``app.bets_full`` must agree with ``dashboard.domain.bets``.
"""

import os
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from nba_ou.dashboard.domain import bets as bet_rules
from nba_ou.dashboard.domain.access import Access, AppUser, access_of
from nba_ou.dashboard.repositories import app_store
from nba_ou.postgre_db.app_aiven import schema as app_schema
from nba_ou.postgre_db.app_aiven.requests import final_status, mark_request
from nba_ou.postgre_db.predictions_aiven import schema as store_schema

DSN = os.getenv("NBA_OU_TEST_PG_DSN")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not DSN, reason="NBA_OU_TEST_PG_DSN not set"),
]
ADMINS = frozenset({"admin@example.com"})
GAME = 22600001


@pytest.fixture
def conn():
    import psycopg

    with psycopg.connect(DSN) as connection:
        with connection.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS app CASCADE")
            cur.execute("DROP SCHEMA IF EXISTS prediction CASCADE")
        connection.commit()
        store_schema.create_schema(connection)
        app_schema.create_schema(connection)
        with connection.cursor() as cur:
            cur.execute(
                "INSERT INTO prediction.games (game_id, tipoff_at, home_team_id, "
                "away_team_id, game_date, season_year, season_type, home_team, "
                "away_team) VALUES (%s, %s, 1, 2, %s, 2026, 2, 'Home', 'Away')",
                (GAME, datetime.now(UTC) + timedelta(hours=3), date(2026, 11, 6)),
            )
        connection.commit()
        yield connection
        connection.rollback()
        with connection.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS app CASCADE")
            cur.execute("DROP SCHEMA IF EXISTS prediction CASCADE")
        connection.commit()


def _bet(user: AppUser, **changes) -> dict:
    return {
        "user_id": user.user_id,
        "game_id": GAME,
        "market": 1,
        "side": 1,
        "line": 224.5,
        "odds_decimal": 1.909,
        "stake": 10,
        "book": "bet365",
        **changes,
    }


def _final(conn, home: int, away: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE prediction.games SET status = 1, home_points = %s, "
            "away_points = %s, tipoff_at = now() - interval '3 hours'",
            (home, away),
        )
    conn.commit()


def test_schema_is_idempotent(conn):
    store_schema.create_schema(conn)
    app_schema.create_schema(conn)
    app_schema.create_schema(conn)


def test_first_sign_in_is_pending_and_admins_start_active(conn):
    fan = app_store.ensure_user(conn, "Fan@Example.com", "Fan", admins=ADMINS)
    admin = app_store.ensure_user(conn, "admin@example.com", None, admins=ADMINS)
    assert access_of(fan) == Access.PENDING
    assert fan.email == "fan@example.com"
    assert access_of(admin) == Access.ADMIN
    # A second sign-in keeps the row and its status.
    assert app_store.ensure_user(conn, "fan@example.com", None, admins=ADMINS) == fan


def test_admin_approves_and_disables_others_only(conn):
    fan = app_store.ensure_user(conn, "fan@example.com", "Fan", admins=ADMINS)
    admin = app_store.ensure_user(conn, "admin@example.com", None, admins=ADMINS)
    with pytest.raises(PermissionError):
        app_store.set_user_status(conn, admin.user_id, 2, by=fan)
    with pytest.raises(ValueError):
        app_store.set_user_status(conn, admin.user_id, 2, by=admin)
    assert app_store.set_user_status(conn, fan.user_id, 1, by=admin)
    assert access_of(app_store.read_user(conn, "fan@example.com")) == Access.ACTIVE
    assert app_store.set_user_status(conn, fan.user_id, 2, by=admin)
    assert access_of(app_store.read_user(conn, "fan@example.com")) == Access.DISABLED


def test_request_lifecycle(conn):
    fan = app_store.ensure_user(conn, "fan@example.com", "Fan", admins=ADMINS)
    request_id = app_store.create_request(
        conn,
        user_id=fan.user_id,
        kind=3,
        dataset_type=2,
        horizon_minutes=None,
        game_date=date(2026, 11, 6),
    )
    assert mark_request(conn, request_id, app_schema.REQUEST_RUNNING)
    assert mark_request(conn, request_id, final_status(["one slot"], 3))
    # A finished request is never moved again (the workflow's failure step).
    assert not mark_request(conn, request_id, app_schema.REQUEST_FAILED, error="x")
    requests = app_store.user_requests(conn, fan.user_id, date(2026, 11, 6))
    assert requests["status"].tolist() == [app_schema.REQUEST_PARTIAL]
    assert (
        requests["started_at"].notna().all() and requests["finished_at"].notna().all()
    )


def test_closing_request_cannot_carry_a_horizon(conn):
    import psycopg

    fan = app_store.ensure_user(conn, "fan@example.com", "Fan", admins=ADMINS)
    with pytest.raises(psycopg.errors.CheckViolation):
        app_store.create_request(
            conn,
            user_id=fan.user_id,
            kind=1,
            dataset_type=1,
            horizon_minutes=360,
            game_date=date(2026, 11, 6),
        )
    conn.rollback()


def test_a_manual_run_must_name_an_existing_request(conn):
    import psycopg

    with conn.cursor() as cur, pytest.raises(psycopg.errors.ForeignKeyViolation):
        cur.execute(
            "INSERT INTO prediction.runs (run_uuid, started_at, finished_at, "
            "workflow, status, request_id) VALUES "
            "(gen_random_uuid(), now(), now(), 1, 1, 999)"
        )
    conn.rollback()


def test_last_stake_and_void_before_tip(conn):
    fan = app_store.ensure_user(conn, "fan@example.com", "Fan", admins=ADMINS)
    assert app_store.last_stake(conn, fan.user_id) is None
    first = app_store.insert_bet(conn, _bet(fan, stake=10))
    app_store.insert_bet(conn, _bet(fan, stake=25))
    assert app_store.last_stake(conn, fan.user_id) == 25
    assert app_store.void_bet(conn, first, fan.user_id)
    assert not app_store.void_bet(conn, first, fan.user_id)
    other = app_store.ensure_user(conn, "other@example.com", None, admins=ADMINS)
    assert not app_store.void_bet(conn, first + 1, other.user_id)
    bets = app_store.user_bets(conn, fan.user_id)
    assert dict(zip(bets["bet_id"], bets["result"], strict=True))[first] == "void"


@pytest.mark.parametrize(
    ("market", "side", "line", "home", "away"),
    [
        (1, 1, 224.5, 120, 110),  # over wins
        (1, -1, 224.5, 120, 110),  # under loses
        (1, 1, 230.0, 120, 110),  # push
        (2, 1, -5.5, 120, 110),  # home -5.5 covers by 10
        (2, -1, -5.5, 120, 110),  # away +5.5 loses
        (2, 1, -10.0, 120, 110),  # push
        (2, -1, 3.5, 100, 102),  # home +3.5, lost by 2: home covers, away loses
    ],
)
def test_view_grades_as_the_domain_rules(conn, market, side, line, home, away):
    fan = app_store.ensure_user(conn, "fan@example.com", "Fan", admins=ADMINS)
    app_store.insert_bet(
        conn, _bet(fan, market=market, side=side, line=line, odds_decimal=2.1)
    )
    _final(conn, home, away)
    row = app_store.user_bets(conn, fan.user_id).iloc[0]
    expected = bet_rules.grade(
        market=market,
        side=side,
        line=line,
        home_points=home,
        away_points=away,
        game_status=1,
    )
    assert row["result"] == expected
    assert float(row["profit"]) == bet_rules.profit(expected, 10, 2.1)


def test_pending_bets_have_no_profit(conn):
    fan = app_store.ensure_user(conn, "fan@example.com", "Fan", admins=ADMINS)
    app_store.insert_bet(conn, _bet(fan))
    row = app_store.user_bets(conn, fan.user_id).iloc[0]
    assert row["result"] == "pending" and pd.isna(row["profit"])


def test_dashboard_role_grants(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_roles WHERE rolname = 'dashboard_app_test'")
        if not cur.fetchone():
            cur.execute("CREATE ROLE dashboard_app_test")
    conn.commit()
    app_schema.grant_dashboard_role(conn, "dashboard_app_test")
    with conn.cursor() as cur:
        cur.execute(
            "SELECT has_table_privilege('dashboard_app_test', 'prediction.runs', "
            "'SELECT'), has_table_privilege('dashboard_app_test', "
            "'prediction.runs', 'INSERT'), has_table_privilege("
            "'dashboard_app_test', 'app.bets', 'INSERT'), has_table_privilege("
            "'dashboard_app_test', 'app.bets', 'DELETE')"
        )
        assert cur.fetchone() == (True, False, True, False)
        cur.execute("REVOKE ALL ON ALL TABLES IN SCHEMA app FROM dashboard_app_test")
        cur.execute(
            "REVOKE ALL ON ALL TABLES IN SCHEMA prediction FROM dashboard_app_test"
        )
        cur.execute("REVOKE ALL ON ALL SEQUENCES IN SCHEMA app FROM dashboard_app_test")
        cur.execute("REVOKE ALL ON SCHEMA app, prediction FROM dashboard_app_test")
    conn.commit()
