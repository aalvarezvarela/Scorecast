"""Bounded read of stored predictions for the Madrid hourly view."""

from configparser import Error as ConfigurationError
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd
import psycopg

from nba_ou.dashboard.config import DISPLAY_SCHEMA_VERSION, HIDDEN_VARIANT_SUFFIXES
from nba_ou.dashboard.repositories.analysis import (
    ANALYSIS_TIMEOUT_MS,
    HISTORY_SOURCE_CTES,
)
from nba_ou.dashboard.repositories.connection import open_connection
from nba_ou.dashboard.repositories.predictions import DashboardDataError, _frame

# Noon on the slate date through the following midnight, in Madrid.
CLOCK_HOURS = tuple(range(12, 25))
MAX_CLOCK_ROWS = 200_000
CLOCK_GAMES_QUERY = """
    SELECT game_id, game_date, tipoff_at, home_team, away_team,
           status, home_points, away_points
    FROM prediction.games
    WHERE game_date BETWEEN %s AND %s AND season_type = ANY(%s::smallint[])
    ORDER BY game_date, game_id
"""

# Each output leaves PostgreSQL with its own line and prices, and the
# hour-by-hour selection runs locally. Crossing every output with thirteen
# cutoffs in SQL ran for minutes on a few weeks of games.
_HIDDEN_AND_TRAINED = """
      NOT EXISTS (
          SELECT 1 FROM unnest(%s::text[]) AS hidden(suffix)
          WHERE right(s.variant, length(suffix)) = suffix
      )
      AND (s.source = 'Live predictions' OR m.train_date_max < g.game_date)
"""

CLOCK_MODELS_QUERY = (
    HISTORY_SOURCE_CTES
    + """
    SELECT s.game_id, s.source, s.kind, s.schema_version, s.target, s.dataset_type,
           s.model_horizon_minutes, s.variant, s.input_at, s.available_at,
           s.predicted_at, s.prediction_value, s.model_line, s.bet_line_x2,
           s.bet_price_over_home, s.bet_price_under_away, s.tipoff_at_prediction,
           s.run_id, s.input_id, s.model_build_id
    FROM sources s
    JOIN prediction.games g USING (game_id)
    JOIN prediction.model_builds m USING (model_build_id)
    WHERE"""
    + _HIDDEN_AND_TRAINED
    + """
    ORDER BY s.game_id, s.input_at, s.available_at
    LIMIT %s
"""
)


@dataclass(frozen=True)
class ClockData:
    games: pd.DataFrame
    predictions: pd.DataFrame
    fetched_at: datetime


def read_clock_history(
    start: date,
    end: date,
    target: int,
    season_type: int | None = 2,
    dsn_override: str = "",
) -> ClockData:
    """Every eligible output with its own line and prices, in both sources."""
    if start > end or target not in (1, 2, 3):
        raise ValueError("Invalid date range or prediction target")
    if season_type is not None and season_type not in (1, 2, 3, 4, 5):
        raise ValueError("Unknown season type")
    types = [season_type] if season_type is not None else [1, 2, 3, 4, 5]
    fetched_at = datetime.now(UTC)
    shared = (start, end, target, types, DISPLAY_SCHEMA_VERSION, fetched_at, fetched_at)
    last = (list(HIDDEN_VARIANT_SUFFIXES), MAX_CLOCK_ROWS + 1)
    try:
        with open_connection(dsn_override) as conn:
            conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
            conn.read_only = True
            with conn.cursor() as cursor:
                cursor.execute(f"SET LOCAL statement_timeout = {ANALYSIS_TIMEOUT_MS}")
                games = _frame(cursor, CLOCK_GAMES_QUERY, (start, end, types))
                predictions = _frame(cursor, CLOCK_MODELS_QUERY, (*shared, *last))
                if len(predictions) > MAX_CLOCK_ROWS:
                    raise DashboardDataError(
                        "Too many hourly predictions. Choose a shorter date range; "
                        "no partial results have been displayed."
                    )
        return ClockData(games, predictions, fetched_at)
    except psycopg.errors.QueryCanceled:
        raise DashboardDataError(
            "The hourly read timed out. Choose a shorter date range."
        ) from None
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
        raise DashboardDataError(
            "The Aiven prediction schema needs updating for hourly analysis."
        ) from None
    except psycopg.errors.InsufficientPrivilege:
        raise DashboardDataError(
            "The Aiven account needs read access to the prediction schema."
        ) from None
    except psycopg.Error:
        raise DashboardDataError(
            "Could not read hourly predictions from Aiven. Select Refresh to retry."
        ) from None
    except (ValueError, KeyError, ConfigurationError, FileNotFoundError):
        raise DashboardDataError(
            "Aiven credentials are missing or incomplete."
        ) from None
