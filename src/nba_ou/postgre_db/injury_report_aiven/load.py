"""Write dimensions and spans into Aiven.

Dimension upserts return id maps so the fact rows can be built entirely in
pandas and inserted once. Everything is ``ON CONFLICT DO NOTHING`` on the
primary key, so re-running an overlapping window is safe -- which matters
because the natural failure mode of a long backfill is being interrupted and
resumed.
"""

from __future__ import annotations

import pandas as pd
import psycopg
from psycopg import sql

from .schema import SCHEMA
from .spans import FILING_COLUMNS, SPAN_COLUMNS


def _q(statement: str) -> sql.Composed:
    return sql.SQL(statement.format(schema=SCHEMA))


def _to_native(value):
    """pandas NA/NaT -> None, numpy scalars -> Python scalars."""
    if value is None or value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def upsert_teams(conn: psycopg.Connection, teams: pd.DataFrame) -> dict[str, int]:
    """Register clubs; returns ``report_name -> team_id``."""
    with conn.cursor() as cur:
        for row in teams.itertuples(index=False):
            cur.execute(
                _q(
                    "INSERT INTO {schema}.ir_team (nba_team_id, tricode, report_name) "
                    "VALUES (%s, %s, %s) ON CONFLICT (report_name) DO NOTHING"
                ),
                (int(row.nba_team_id), row.tricode, row.report_name),
            )
        cur.execute(_q("SELECT report_name, team_id FROM {schema}.ir_team"))
        mapping = {name: team_id for name, team_id in cur.fetchall()}
    conn.commit()
    return mapping


def upsert_games(conn: psycopg.Connection, games: pd.DataFrame) -> int:
    """Copy the game dimension across so every read can join locally."""
    written = 0
    with conn.cursor() as cur:
        for row in games.itertuples(index=False):
            cur.execute(
                _q(
                    "INSERT INTO {schema}.ir_game "
                    "(game_id, game_date, season_year, tipoff_utc, team_home, "
                    "team_away, tipoff_source) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (game_id) DO UPDATE SET "
                    "tipoff_utc = EXCLUDED.tipoff_utc, "
                    "tipoff_source = EXCLUDED.tipoff_source"
                ),
                (
                    row.game_id,
                    _to_native(row.game_date),
                    int(row.season_year),
                    _to_native(row.tipoff_utc),
                    row.team_home,
                    row.team_away,
                    getattr(row, "tipoff_source", "schedule"),
                ),
            )
            written += 1
    conn.commit()
    return written


def upsert_reasons(
    conn: psycopg.Connection, pairs: list[tuple[str, str, str]]
) -> dict[tuple[str, str], int]:
    """Register ``(category_code, category_label, detail)``; returns id map.

    Unseen categories are inserted rather than rejected -- unlike statuses, new
    ones are normal (Health and Safety Protocols in 2020, Ineligible To Play in
    2022).
    """
    with conn.cursor() as cur:
        for code, label, _ in pairs:
            cur.execute(
                _q(
                    "INSERT INTO {schema}.ir_reason_category (code, label) "
                    "VALUES (%s, %s) ON CONFLICT (code) DO NOTHING"
                ),
                (code, label),
            )
        cur.execute(_q("SELECT code, category_id FROM {schema}.ir_reason_category"))
        categories = {code: cid for code, cid in cur.fetchall()}

        for code, _, detail in pairs:
            cur.execute(
                _q(
                    "INSERT INTO {schema}.ir_reason (category_id, detail) "
                    "VALUES (%s, %s) ON CONFLICT (category_id, detail) DO NOTHING"
                ),
                (categories[code], detail),
            )
        cur.execute(
            _q(
                "SELECT c.code, r.detail, r.reason_id "
                "FROM {schema}.ir_reason r "
                "JOIN {schema}.ir_reason_category c USING (category_id)"
            )
        )
        return {(code, detail): rid for code, detail, rid in cur.fetchall()}


def upsert_reports(
    conn: psycopg.Connection, reports: pd.DataFrame
) -> dict[pd.Timestamp, int]:
    """Register report coverage; returns ``observed_at -> report_id``."""
    with conn.cursor() as cur:
        for row in reports.itertuples(index=False):
            cur.execute(
                _q(
                    "INSERT INTO {schema}.ir_report "
                    "(observed_at, season_year, era, n_rows, parse_ok) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (observed_at) DO NOTHING"
                ),
                (
                    _to_native(row.observed_at),
                    int(row.season_year),
                    int(row.era),
                    int(row.n_rows),
                    bool(row.parse_ok),
                ),
            )
        cur.execute(_q("SELECT observed_at, report_id FROM {schema}.ir_report"))
        mapping = {ts: rid for ts, rid in cur.fetchall()}
    conn.commit()
    return mapping


def insert_spans(conn: psycopg.Connection, spans: pd.DataFrame) -> int:
    """Insert status spans; idempotent on the primary key."""
    if spans.empty:
        return 0
    columns = sql.SQL(", ").join(sql.Identifier(c) for c in SPAN_COLUMNS)
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in SPAN_COLUMNS)
    statement = sql.SQL(
        "INSERT INTO {}.{} ({}) VALUES ({}) ON CONFLICT DO NOTHING"
    ).format(
        sql.Identifier(SCHEMA),
        sql.Identifier("ir_status_span"),
        columns,
        placeholders,
    )
    rows = [
        tuple(_to_native(v) for v in row)
        for row in spans[SPAN_COLUMNS].itertuples(index=False)
    ]
    with conn.cursor() as cur:
        cur.executemany(statement, rows)
    conn.commit()
    return len(rows)


def insert_filing_spans(conn: psycopg.Connection, spans: pd.DataFrame) -> int:
    if spans.empty:
        return 0
    columns = sql.SQL(", ").join(sql.Identifier(c) for c in FILING_COLUMNS)
    placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in FILING_COLUMNS)
    statement = sql.SQL(
        "INSERT INTO {}.{} ({}) VALUES ({}) ON CONFLICT DO NOTHING"
    ).format(
        sql.Identifier(SCHEMA),
        sql.Identifier("ir_filing_span"),
        columns,
        placeholders,
    )
    rows = [
        tuple(_to_native(v) for v in row)
        for row in spans[FILING_COLUMNS].itertuples(index=False)
    ]
    with conn.cursor() as cur:
        cur.executemany(statement, rows)
    conn.commit()
    return len(rows)


def record_unresolved(conn: psycopg.Connection, unresolved: list[dict]) -> int:
    """Persist every name that did not resolve, with an occurrence count."""
    if not unresolved:
        return 0
    counts: dict[tuple[str, str, int], int] = {}
    for item in unresolved:
        season = item.get("season_year")
        key = (
            str(item["raw_name"]),
            str(item["raw_team"]),
            int(season) if season is not None and not pd.isna(season) else 0,
        )
        counts[key] = counts.get(key, 0) + 1
    with conn.cursor() as cur:
        for (name, team, season), n in counts.items():
            cur.execute(
                _q(
                    "INSERT INTO {schema}.ir_unresolved "
                    "(raw_name, raw_team, season_year, occurrences) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (raw_name, raw_team, season_year) "
                    "DO UPDATE SET occurrences = {schema}.ir_unresolved.occurrences "
                    "+ EXCLUDED.occurrences"
                ),
                (name, team, season, n),
            )
    conn.commit()
    return len(counts)


def record_aliases(conn: psycopg.Connection, resolved: pd.DataFrame) -> int:
    """Persist name -> player_id decisions so they are reviewable and stable."""
    if resolved.empty:
        return 0
    frame = resolved[
        ["raw_name", "team_id", "season_year", "player_id", "method"]
    ].drop_duplicates()
    with conn.cursor() as cur:
        for row in frame.itertuples(index=False):
            cur.execute(
                _q(
                    "INSERT INTO {schema}.ir_player_alias "
                    "(raw_name, team_id, season_year, player_id, method) "
                    "VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (raw_name, team_id, season_year) DO NOTHING"
                ),
                (
                    row.raw_name,
                    int(row.team_id),
                    int(row.season_year),
                    int(row.player_id),
                    row.method,
                ),
            )
    conn.commit()
    return len(frame)
