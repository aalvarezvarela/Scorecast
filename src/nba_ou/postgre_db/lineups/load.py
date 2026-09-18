"""Atomic loading of a validated game's segments and status."""

from __future__ import annotations

from datetime import date

import pandas as pd
import psycopg
from psycopg import sql

from .schema import COUNT_FIELDS, SCHEMA, ensure_partition


def _lineup_id(conn: psycopg.Connection, team_id: str, players: tuple[str, ...]) -> int:
    if len(players) != 5 or len(set(players)) != 5:
        raise ValueError("A lineup must contain five distinct players")
    values = (int(team_id), *(int(pid) for pid in sorted(players, key=int)))
    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            INSERT INTO {}.lu_lineup (team_id, p1, p2, p3, p4, p5)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (team_id, p1, p2, p3, p4, p5) DO NOTHING
        """).format(sql.Identifier(SCHEMA)), values)
        cur.execute(sql.SQL("""
            SELECT lineup_id FROM {}.lu_lineup
            WHERE team_id=%s AND p1=%s AND p2=%s AND p3=%s AND p4=%s AND p5=%s
        """).format(sql.Identifier(SCHEMA)), values)
        return int(cur.fetchone()[0])


def load_game(
    conn: psycopg.Connection,
    stints: pd.DataFrame,
    *,
    game_id: str,
    season_year: int,
    game_date: date,
    tipoff_utc=None,
) -> None:
    """Replace this game's validated stints in one transaction."""
    if stints.empty or not stints.game_id.eq(game_id).all():
        raise ValueError("Stints must be nonempty and belong to one game")
    ensure_partition(conn, season_year)
    ids = {}
    for side in ("home", "away"):
        for team_id, players in stints[
            [f"{side}_team_id", f"{side}_lineup"]
        ].itertuples(index=False, name=None):
            lineup = tuple(str(pid) for pid in players)
            ids[(str(team_id), lineup)] = _lineup_id(conn, str(team_id), lineup)
    fields = [
        "game_id", "seg_idx", "season_year", "period", "start_ds", "end_ds",
        "home_lineup_id", "away_lineup_id", "home_pts", "away_pts",
        *(f"{side}_{stat}" for side in ("home", "away") for stat in COUNT_FIELDS),
    ]
    with conn.cursor() as cur:
        cur.execute(sql.SQL("DELETE FROM {}.lu_stint WHERE game_id=%s AND season_year=%s").format(
            sql.Identifier(SCHEMA)
        ), (game_id, season_year))
        query = sql.SQL("INSERT INTO {}.lu_stint ({}) VALUES ({})").format(
            sql.Identifier(SCHEMA),
            sql.SQL(", ").join(map(sql.Identifier, fields)),
            sql.SQL(", ").join(sql.Placeholder() for _ in fields),
        )
        records = []
        for row in stints.itertuples(index=False):
            values = [
                game_id, int(row.seg_idx), season_year, int(row.period),
                int(row.start_ds), int(row.end_ds),
                ids[(str(row.home_team_id), tuple(str(p) for p in row.home_lineup))],
                ids[(str(row.away_team_id), tuple(str(p) for p in row.away_lineup))],
                int(row.home_pts), int(row.away_pts),
                *(int(getattr(row, f"{side}_{stat}"))
                  for side in ("home", "away") for stat in COUNT_FIELDS),
            ]
            records.append(values)
        cur.executemany(query, records)
        cur.execute(sql.SQL("""
            INSERT INTO {}.lu_game_status
                (game_id, season_year, game_date, tipoff_utc, status, reason)
            VALUES (%s, %s, %s, %s, 'ok', NULL)
            ON CONFLICT (game_id) DO UPDATE SET status='ok', reason=NULL,
                built_at=now(), tipoff_utc=EXCLUDED.tipoff_utc
        """).format(sql.Identifier(SCHEMA)),
        (game_id, season_year, game_date, tipoff_utc))


def mark_failed(
    conn: psycopg.Connection,
    *,
    game_id: str,
    season_year: int,
    game_date: date,
    reason: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("""
            INSERT INTO {}.lu_game_status
                (game_id, season_year, game_date, status, reason)
            VALUES (%s, %s, %s, 'failed', %s)
            ON CONFLICT (game_id) DO UPDATE SET status='failed',
                reason=EXCLUDED.reason, built_at=now()
        """).format(sql.Identifier(SCHEMA)),
        (game_id, season_year, game_date, reason))
