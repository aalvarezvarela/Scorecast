"""Read a single NBA slate from the prediction store in one DB snapshot."""

from configparser import Error as ConfigurationError
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd
import psycopg

from nba_ou.dashboard.config import (
    YESTERDAY_INTERMEDIATE_CORE,
    YESTERDAY_INTERMEDIATE_EXTRA,
)
from nba_ou.dashboard.repositories.connection import open_connection

GAMES_QUERY = """
    SELECT game_id, game_date, tipoff_at, home_team, away_team,
           home_team_id, away_team_id, status,
           home_points, away_points, result_updated_at
    FROM prediction.games
    WHERE game_date = %s
    ORDER BY tipoff_at, game_id
"""

# Per game, whether each team's injury report is filed at this moment (the
# pipeline's own rule: a team-game without a filing span has not filed), and
# the newest parsed report the store holds. Read apart from the slate: an
# account without access to the injury-report store still gets its games.
REPORT_FILINGS_QUERY = """
    SELECT g.game_id,
           COALESCE(bool_or(f.submitted)
                    FILTER (WHERE t.nba_team_id = g.home_team_id), FALSE)
               AS home_report_filed,
           COALESCE(bool_or(f.submitted)
                    FILTER (WHERE t.nba_team_id = g.away_team_id), FALSE)
               AS away_report_filed,
           (SELECT max(observed_at) FROM injury_report.ir_report WHERE parse_ok)
               AS reports_read_at
    FROM prediction.games g
    LEFT JOIN injury_report.ir_filing_span f
           ON f.game_id = lpad(g.game_id::text, 10, '0')
          AND f.valid_from < LEAST(%s, g.tipoff_at)
          AND f.valid_to >= LEAST(%s, g.tipoff_at)
    LEFT JOIN injury_report.ir_team t ON t.team_id = f.team_id
    WHERE g.game_date = %s
    GROUP BY g.game_id
"""
REPORT_COLUMNS = ["home_report_filed", "away_report_filed", "reports_read_at"]

PREDICTIONS_SELECT = """
    SELECT v.*, m.anchor_book, r.finished_at AS run_finished_at,
           CASE WHEN v.input_mode = 0 AND v.snapshot_horizon_minutes IS NULL
                    THEN v.run_started_at
                ELSE v.tipoff_at_prediction
                     - make_interval(mins => v.snapshot_horizon_minutes)
           END AS input_at
    FROM prediction.predictions_full v
    JOIN prediction.model_builds m USING (model_build_id)
    JOIN prediction.runs r ON r.run_id = v.run_id
"""

PREDICTIONS_QUERY = (
    PREDICTIONS_SELECT
    + """
    WHERE v.game_date = %s
      AND v.workflow IN (1, 3)
      AND v.kind IN (1, 3)
      AND v.time_to_tipoff_seconds > 0
      AND v.predicted_at <= %s
      AND r.finished_at <= %s
    ORDER BY v.predicted_at, v.run_id, v.input_id, v.model_build_id
"""
)

# Integer constants, not user input: safe to inline in the SQL text.
_HISTORY_HORIZONS = ", ".join(
    str(int(minutes))
    for minutes in sorted(YESTERDAY_INTERMEDIATE_CORE + YESTERDAY_INTERMEDIATE_EXTRA)
)
HISTORY_QUERY = (
    PREDICTIONS_SELECT
    + f"""
    WHERE v.game_date = %s
      AND v.workflow IN (1, 2, 3, 4)
      AND v.kind IN (1, 3)
      AND (v.dataset_type = 1
           OR v.kind = 3
           OR v.model_horizon_minutes IN ({_HISTORY_HORIZONS}))
      -- Manual (dashboard-requested) runs stay out of the evaluated series.
      AND r.request_id IS NULL
      AND v.predicted_at <= %s
      AND r.finished_at <= %s
    ORDER BY v.predicted_at, v.run_id, v.input_id, v.model_build_id
"""
)


@dataclass(frozen=True)
class TodayData:
    games: pd.DataFrame
    predictions: pd.DataFrame
    fetched_at: datetime


class DashboardDataError(RuntimeError):
    """A safe, actionable message; raw DB exceptions may contain credentials."""


def _frame(cursor, query: str, params: tuple) -> pd.DataFrame:
    cursor.execute(query, params)
    return pd.DataFrame(cursor.fetchall(), columns=[c.name for c in cursor.description])


def read_today(game_date: date, dsn_override: str = "") -> TodayData:
    """No DDL, writes, external APIs, model loads or Supabase fallback.

    The games also say whether each team's injury report is filed
    (``REPORT_COLUMNS``), when the injury-report store can be read.
    """
    return _read_slate(game_date, dsn_override, PREDICTIONS_QUERY, reports=True)


def _with_report_filings(
    conn, cursor, games: pd.DataFrame, game_date, now
) -> pd.DataFrame:
    """``games`` with ``REPORT_COLUMNS``; unchanged when the injury-report
    store cannot be read (the cards then show no report flag)."""
    if games.empty:
        return games
    try:
        # A savepoint: a refused read must not end the slate's transaction.
        with conn.transaction():
            filings = _frame(cursor, REPORT_FILINGS_QUERY, (now, now, game_date))
    except psycopg.Error:
        return games
    return games.merge(filings, on="game_id", how="left")


def read_yesterday(game_date: date, dsn_override: str = "") -> TodayData:
    """Include stored replay outputs and final scores in the same Aiven snapshot."""
    return _read_slate(game_date, dsn_override, HISTORY_QUERY)


def _read_slate(
    game_date: date, dsn_override: str, query: str, *, reports: bool = False
) -> TodayData:
    fetched_at = datetime.now(UTC)
    try:
        with open_connection(dsn_override) as conn:
            conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
            conn.read_only = True
            with conn.cursor() as cursor:
                games = _frame(cursor, GAMES_QUERY, (game_date,))
                predictions = _frame(cursor, query, (game_date, fetched_at, fetched_at))
                if reports:
                    games = _with_report_filings(
                        conn, cursor, games, game_date, fetched_at
                    )
        return TodayData(games, predictions, fetched_at)
    except psycopg.errors.UndefinedTable:
        raise DashboardDataError(
            "The prediction store is missing. Check that the prediction schema "
            "and predictions_full view are available in Aiven."
        ) from None
    except psycopg.errors.UndefinedColumn:
        raise DashboardDataError(
            "The Aiven prediction schema needs updating. The dashboard requires "
            "the market mean, median and bookmaker-count columns."
        ) from None
    except psycopg.errors.InsufficientPrivilege:
        raise DashboardDataError(
            "The Aiven account needs read access to the prediction schema."
        ) from None
    except psycopg.Error:
        raise DashboardDataError(
            "Could not read Aiven. Check the connection settings and network, "
            "then select Refresh data."
        ) from None
    except (ValueError, KeyError, ConfigurationError, FileNotFoundError):
        raise DashboardDataError(
            "Aiven credentials are missing or incomplete. Configure AIVEN_DB_URL "
            "in the environment or Streamlit secrets."
        ) from None
