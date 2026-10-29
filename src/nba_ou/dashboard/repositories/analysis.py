"""Bounded, SQL-reduced historical reads from Aiven's prediction schema only."""

from configparser import Error as ConfigurationError
from dataclasses import dataclass
from datetime import UTC, date, datetime

import pandas as pd
import psycopg

from nba_ou.dashboard.config import DISPLAY_SCHEMA_VERSION
from nba_ou.dashboard.domain.analysis import EARLY_BUCKET, TIME_GRID
from nba_ou.dashboard.repositories.connection import open_connection
from nba_ou.dashboard.repositories.predictions import DashboardDataError, _frame

MAX_ANALYSIS_ROWS = 200_000
ANALYSIS_TIMEOUT_MS = 45_000

# CASE fragments contain only module-owned integer constants. All user filters
# are bound parameters. Keep raw outputs/build identity off the network until
# refits and clustered runs have been reduced to one row per cell in PostgreSQL.
_BUCKET_CASE = (
    "CASE "
    + " ".join(f"WHEN input_minutes <= {h} THEN {h}" for h in TIME_GRID)
    + f" ELSE {EARLY_BUCKET} END"
)
_SLOT = "game_id, kind, schema_version, target, dataset_type, model_horizon_minutes, variant, source"

ANALYSIS_GAMES_QUERY = """
    SELECT game_id, game_date, status, home_points, away_points
    FROM prediction.games
    WHERE game_date BETWEEN %s AND %s AND season_type = ANY(%s::smallint[])
    ORDER BY game_date, game_id
"""

ANALYSIS_QUERY = """
    WITH base AS (
        SELECT v.game_id, v.kind, v.schema_version, v.target, v.dataset_type,
               v.model_horizon_minutes, v.variant, v.workflow, v.input_mode,
               v.snapshot_horizon_minutes, v.prediction_value, v.model_line,
               v.bet_line_x2, v.bet_price_over_home, v.bet_price_under_away,
               v.home_points, v.away_points, v.status, v.predicted_at,
               v.tipoff_at, v.tipoff_at_prediction, v.time_to_tipoff_seconds,
               v.run_id, v.input_id, v.model_build_id,
               CASE WHEN v.input_mode = 0 AND v.snapshot_horizon_minutes IS NULL
                    THEN v.run_started_at
                    ELSE v.tipoff_at_prediction
                         - make_interval(mins => v.snapshot_horizon_minutes)
               END AS input_at,
               GREATEST(v.predicted_at, r.finished_at) AS available_at,
               CASE WHEN v.dataset_type = 1 AND v.snapshot_horizon_minutes IS NULL
                    THEN EXTRACT(EPOCH FROM (v.tipoff_at_prediction - v.run_started_at)) / 60.0
                    ELSE v.snapshot_horizon_minutes END AS input_minutes
        FROM prediction.predictions_full v
        JOIN prediction.runs r ON r.run_id = v.run_id
        WHERE v.game_date BETWEEN %s AND %s
          AND v.target = %s AND v.season_type = ANY(%s::smallint[])
          AND ((v.kind = 1 AND v.schema_version = %s) OR v.kind = 3)
          AND v.workflow IN (1, 2, 3, 4)
          AND (v.dataset_type = 1 AND v.model_horizon_minutes = 0
               OR v.dataset_type = 2 AND v.model_horizon_minutes > 0)
          AND v.predicted_at <= %s AND r.finished_at <= %s
    ), sources AS (
        SELECT base.*, sources.source
        FROM base
        CROSS JOIN LATERAL (VALUES
            ('Live predictions',
                workflow IN (1, 3)
                AND available_at < tipoff_at AND available_at < tipoff_at_prediction
                AND time_to_tipoff_seconds > 0
                AND (dataset_type = 1 AND input_mode = 0
                     OR dataset_type = 2 AND input_mode IN (1, 2))),
            ('Snapshot replay',
                (dataset_type = 2 AND input_mode = 1
                 AND snapshot_horizon_minutes = model_horizon_minutes
                 AND (workflow IN (2, 4) OR kind = 3 AND workflow = 3)
                 OR dataset_type = 1 AND input_mode = 0 AND workflow = 4
                    AND snapshot_horizon_minutes > 0))
        ) AS sources(source, eligible)
        WHERE sources.eligible AND input_minutes > 0 AND input_at <= predicted_at
    ), bucketed AS (
        SELECT sources.*, BUCKET_CASE AS bucket FROM sources
    ), ranked AS (
        SELECT bucketed.*, ROW_NUMBER() OVER (
            PARTITION BY SLOT, bucket
            ORDER BY CASE WHEN source = 'Snapshot replay' THEN 0
                          ELSE ABS(LEAST(bucket, 1080) - input_minutes) END,
                     available_at DESC, predicted_at DESC, run_id DESC,
                     input_id DESC, model_build_id DESC
        ) AS priority FROM bucketed
    ), final AS (
        SELECT bucketed.*, ROW_NUMBER() OVER (
            PARTITION BY SLOT
            ORDER BY available_at DESC, predicted_at DESC, run_id DESC,
                     input_id DESC, model_build_id DESC
        ) AS priority
        FROM bucketed WHERE dataset_type = 1 AND source = 'Live predictions'
    ), reduced AS (
        SELECT game_id, kind, schema_version, target, dataset_type,
               model_horizon_minutes, variant, source, bucket,
               prediction_value, model_line, bet_line_x2,
               bet_price_over_home, bet_price_under_away,
               home_points, away_points, status, input_at, available_at,
               predicted_at, input_minutes
        FROM ranked WHERE priority = 1
        UNION ALL
        SELECT game_id, kind, schema_version, target, dataset_type,
               model_horizon_minutes, variant, source, 0 AS bucket,
               prediction_value, model_line, bet_line_x2,
               bet_price_over_home, bet_price_under_away,
               home_points, away_points, status, input_at, available_at,
               predicted_at, input_minutes
        FROM final WHERE priority = 1
    )
    SELECT * FROM reduced LIMIT %s
""".replace(
    "BUCKET_CASE", _BUCKET_CASE
).replace(
    "SLOT", _SLOT
)


@dataclass(frozen=True)
class AnalysisData:
    games: pd.DataFrame
    predictions: pd.DataFrame
    fetched_at: datetime


def read_history_range(
    start: date,
    end: date,
    target: int,
    season_type: int | None = 2,
    dsn_override: str = "",
) -> AnalysisData:
    if start > end:
        raise ValueError("Start date must not be after end date")
    if target not in (1, 2, 3):
        raise ValueError("Unknown prediction target")
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
                games = _frame(cursor, ANALYSIS_GAMES_QUERY, (start, end, types))
                predictions = _frame(
                    cursor,
                    ANALYSIS_QUERY,
                    (
                        start,
                        end,
                        target,
                        types,
                        DISPLAY_SCHEMA_VERSION,
                        fetched_at,
                        fetched_at,
                        MAX_ANALYSIS_ROWS + 1,
                    ),
                )
        if len(predictions) > MAX_ANALYSIS_ROWS:
            raise DashboardDataError(
                "This date range contains too many historical cells. Choose a shorter "
                "range; no partial results have been displayed."
            )
        return AnalysisData(games, predictions, fetched_at)
    except psycopg.errors.QueryCanceled:
        raise DashboardDataError(
            "The historical read timed out. Choose a shorter date range and try again."
        ) from None
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
        raise DashboardDataError(
            "The Aiven prediction schema needs updating. Historical analysis requires "
            "predictions_full, run completion times, stored prices and final scores."
        ) from None
    except psycopg.errors.InsufficientPrivilege:
        raise DashboardDataError(
            "The Aiven account needs read access to the prediction schema."
        ) from None
    except psycopg.Error:
        raise DashboardDataError(
            "Could not read historical predictions from Aiven. Check the connection and select Refresh."
        ) from None
    except (ValueError, KeyError, ConfigurationError, FileNotFoundError):
        raise DashboardDataError(
            "Aiven credentials are missing or incomplete. Configure AIVEN_DB_URL in the environment or Streamlit secrets."
        ) from None
