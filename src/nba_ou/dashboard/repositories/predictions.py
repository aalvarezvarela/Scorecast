"""Read a single NBA slate from the prediction store in one DB snapshot."""

from configparser import Error as ConfigurationError
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd
import psycopg

from nba_ou.dashboard.repositories.connection import open_connection

GAMES_QUERY = """
    SELECT game_id, game_date, tipoff_at, home_team, away_team,
           home_team_id, away_team_id, status,
           home_points, away_points, result_updated_at
    FROM prediction.games
    WHERE game_date = %s
    ORDER BY tipoff_at, game_id
"""

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

HISTORY_QUERY = (
    PREDICTIONS_SELECT
    + """
    WHERE v.game_date = %s
      AND v.workflow IN (1, 2, 3, 4)
      AND v.kind IN (1, 3)
      AND (v.dataset_type = 1 OR v.model_horizon_minutes IN (240, 360, 480))
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
    """No DDL, writes, external APIs, model loads or Supabase fallback."""
    return _read_slate(game_date, dsn_override, PREDICTIONS_QUERY)


def read_yesterday(game_date: date, dsn_override: str = "") -> TodayData:
    """Include stored replay outputs and final scores in the same Aiven snapshot."""
    return _read_slate(game_date, dsn_override, HISTORY_QUERY)


def _read_slate(game_date: date, dsn_override: str, query: str) -> TodayData:
    fetched_at = datetime.now(UTC)
    try:
        with open_connection(dsn_override) as conn:
            conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
            conn.read_only = True
            with conn.cursor() as cursor:
                games = _frame(cursor, GAMES_QUERY, (game_date,))
                predictions = _frame(cursor, query, (game_date, fetched_at, fetched_at))
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
