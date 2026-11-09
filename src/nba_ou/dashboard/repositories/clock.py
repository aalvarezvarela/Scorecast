"""Select the last eligible output per slot and Madrid clock cutoff in SQL."""

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

CLOCK_QUERY = (
    HISTORY_SOURCE_CTES
    + """
    , clock_base AS (
        SELECT s.*, g.game_date, g.home_team, g.away_team,
               CASE WHEN s.target = 3 THEN i.spread_line_age_minutes
                    ELSE i.total_line_age_minutes END AS quote_age_at_input,
               CASE WHEN s.source = 'Live predictions' THEN s.available_at
                    ELSE s.input_at END AS selection_at
        FROM sources s
        JOIN prediction.games g USING (game_id)
        JOIN prediction.inputs i USING (input_id)
        JOIN prediction.model_builds m USING (model_build_id)
        WHERE NOT EXISTS (
            SELECT 1 FROM unnest(%s::text[]) AS hidden(suffix)
            WHERE right(s.variant, length(suffix)) = suffix
        )
          AND (s.source = 'Live predictions' OR m.train_date_max < g.game_date)
    ), cutoffs AS (
        SELECT g.game_id, hour,
               ((g.game_date + (hour / 24)) + make_time(hour %% 24, 0, 0))
                   AT TIME ZONE 'Europe/Madrid' AS cutoff_at
        FROM prediction.games g
        CROSS JOIN unnest(%s::integer[]) AS hours(hour)
        WHERE g.game_date BETWEEN %s AND %s AND g.season_type = ANY(%s::smallint[])
    ), candidates AS (
        SELECT b.*, c.hour, c.cutoff_at
        FROM clock_base b JOIN cutoffs c USING (game_id)
        WHERE b.selection_at <= c.cutoff_at AND b.input_at <= c.cutoff_at
          AND c.cutoff_at < b.tipoff_at AND c.cutoff_at < b.tipoff_at_prediction
          AND c.cutoff_at <= %s
    ), ranked AS (
        SELECT candidates.*, ROW_NUMBER() OVER (
            PARTITION BY game_id, source, hour, kind, schema_version, target,
                         dataset_type, model_horizon_minutes, variant
            ORDER BY CASE WHEN kind = 3 THEN input_at ELSE selection_at END DESC,
                     input_at DESC, available_at DESC,
                     predicted_at DESC, run_id DESC, input_id DESC, model_build_id DESC
        ) AS priority FROM candidates
    ), quotes AS (
        SELECT DISTINCT ON (game_id, source, hour)
               game_id, source, hour, bet_line_x2 AS clock_bet_line_x2,
               bet_price_over_home AS clock_price_over_home,
               bet_price_under_away AS clock_price_under_away,
               input_at AS quote_input_at, quote_age_at_input
        FROM candidates
        WHERE bet_line_x2 IS NOT NULL
        ORDER BY game_id, source, hour, input_at DESC, available_at DESC,
                 predicted_at DESC, run_id DESC, input_id DESC, model_build_id DESC
    )
    SELECT r.game_id, r.game_date, r.home_team, r.away_team, r.tipoff_at,
           r.status, r.home_points, r.away_points, r.kind, r.schema_version,
           r.target, r.dataset_type, r.model_horizon_minutes, r.variant,
           r.source, r.hour, r.cutoff_at, r.input_at, r.available_at,
           r.selection_at, r.prediction_value, r.model_line,
           r.run_id, r.input_id, r.model_build_id,
           q.clock_bet_line_x2 AS bet_line_x2,
           q.clock_price_over_home AS bet_price_over_home,
           q.clock_price_under_away AS bet_price_under_away,
           q.quote_input_at,
           EXTRACT(EPOCH FROM (r.cutoff_at - q.quote_input_at)) / 60.0
               + q.quote_age_at_input AS quote_age_minutes
    FROM ranked r LEFT JOIN quotes q USING (game_id, source, hour)
    WHERE priority = 1
    ORDER BY r.game_date, r.game_id, r.source, r.hour, r.kind,
             r.dataset_type, r.model_horizon_minutes, r.variant
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
    if start > end or target not in (1, 2, 3):
        raise ValueError("Invalid date range or prediction target")
    if season_type is not None and season_type not in (1, 2, 3, 4, 5):
        raise ValueError("Unknown season type")
    types = [season_type] if season_type is not None else [1, 2, 3, 4, 5]
    fetched_at = datetime.now(UTC)
    try:
        with open_connection(dsn_override) as conn:
            conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
            conn.read_only = True
            with conn.cursor() as cursor:
                cursor.execute(f"SET LOCAL statement_timeout = {ANALYSIS_TIMEOUT_MS}")
                games = _frame(cursor, CLOCK_GAMES_QUERY, (start, end, types))
                predictions = _frame(
                    cursor,
                    CLOCK_QUERY,
                    (
                        start,
                        end,
                        target,
                        types,
                        DISPLAY_SCHEMA_VERSION,
                        fetched_at,
                        fetched_at,
                        list(HIDDEN_VARIANT_SUFFIXES),
                        list(CLOCK_HOURS),
                        start,
                        end,
                        types,
                        fetched_at,
                        MAX_CLOCK_ROWS + 1,
                    ),
                )
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
