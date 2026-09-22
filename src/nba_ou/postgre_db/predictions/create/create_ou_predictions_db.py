import argparse
from typing import TYPE_CHECKING

import psycopg
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_predictions,
)
from psycopg import sql

if TYPE_CHECKING:
    import pandas as pd


def get_predictions_schema_and_table() -> tuple[str, str]:
    """Return the canonical predictions schema/table names."""
    schema = get_schema_name_predictions().strip()
    table = schema
    return schema, table


def schema_exists(schema_name: str = None) -> bool:
    """Check if the predictions schema exists in the database."""
    try:
        if schema_name is None:
            schema_name, _ = get_predictions_schema_and_table()

        conn = connect_nba_db()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
                (schema_name,),
            )
            exists = cur.fetchone()
        conn.close()
        return exists is not None
    except Exception as e:
        print(f"Error checking schema existence: {e}")
        return False


def create_prediction_schema_if_not_exists(
    conn: psycopg.Connection, schema: str
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
    conn.commit()


def get_existing_columns(conn: psycopg.Connection, schema: str, table: str) -> set[str]:
    """Get set of existing column names for a table."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema, table),
        )
        return {row[0] for row in cur.fetchall()}


def table_exists(conn: psycopg.Connection, schema: str, table: str) -> bool:
    """Check if a table exists in the target schema."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = %s AND table_name = %s
            LIMIT 1
            """,
            (schema, table),
        )
        return cur.fetchone() is not None


def _create_predictions_table_base(
    conn: psycopg.Connection, schema: str, table: str, *, drop: bool = False
) -> None:
    """Create the predictions table and indexes without running migrations."""
    create_prediction_schema_if_not_exists(conn, schema)

    with conn.cursor() as cur:
        if drop:
            cur.execute(
                sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
            )
            print(f"Dropped existing table '{schema}.{table}'")

        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.{} (
                    id SERIAL PRIMARY KEY,
                    game_id TEXT NOT NULL,
                    season_type TEXT,
                    game_date DATE,
                    game_time TEXT,
                    team_name_team_home TEXT,
                    team_name_team_away TEXT,
                    prediction_value_type TEXT NOT NULL,
                    total_over_under_line NUMERIC,
                    total_bet365_line_at_prediction NUMERIC,
                    pred_line_error NUMERIC,
                    pred_total_points NUMERIC,
                    pred_pick TEXT,
                    model_name TEXT NOT NULL,
                    model_type TEXT,
                    model_version TEXT,
                    prediction_date TEXT,
                    prediction_datetime TIMESTAMP NOT NULL,
                    time_to_match_minutes INTEGER,
                    na_columns_count INTEGER,
                    na_columns_names TEXT,
                    shap_base_value NUMERIC,
                    shap_top_positive_features TEXT,
                    shap_top_negative_features TEXT,
                    shap_directional_confidence NUMERIC,
                    shap_support_ratio NUMERIC,
                    shap_top_k_agreement NUMERIC,
                    shap_confidence_score NUMERIC,
                    prediction_source TEXT NOT NULL DEFAULT 'unknown',
                    source_run_id TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    total_scored_points NUMERIC,
                    home_pts NUMERIC,
                    away_pts NUMERIC,
                    training_code_tag TEXT NOT NULL DEFAULT '0.0',
                    train_date_min DATE,
                    train_date_max DATE,
                    pred_spread_error NUMERIC,
                    pred_home_margin NUMERIC,
                    spread_line_at_prediction NUMERIC,
                    model_target TEXT,
                    -- Minutes before tip the MODEL WAS TRAINED FOR. Distinct
                    -- from time_to_match_minutes, which is when the prediction
                    -- ran. Both are needed to ask whether the T-360 model
                    -- actually beat the T-0 model at T-360.
                    model_horizon_minutes INTEGER,
                    model_schema_version TEXT,
                    model_variant TEXT,
                    spec_id TEXT,
                    fit_id TEXT,
                    CONSTRAINT chk_prediction_value_type
                        CHECK (
                            prediction_value_type IN (
                                'TOTAL_POINTS', 'DIFF_FROM_LINE', 'SPREAD_ERROR'
                            )
                        ),
                    CONSTRAINT chk_prediction_target_present
                        CHECK (
                            pred_line_error IS NOT NULL
                            OR pred_total_points IS NOT NULL
                            OR pred_spread_error IS NOT NULL
                        ),
                    CONSTRAINT chk_pred_pick
                        CHECK (
                            pred_pick IS NULL
                            OR pred_pick IN (
                                'OVER', 'UNDER', 'PUSH', 'HOME', 'AWAY'
                            )
                        ),
                    CONSTRAINT unique_game_prediction
                        UNIQUE (
                            game_id,
                            model_name,
                            prediction_datetime,
                            prediction_value_type
                        )
                )
                """
            ).format(sql.Identifier(schema), sql.Identifier(table))
        )

        index_specs = [
            (f"idx_{table}_game_id", "game_id"),
            (f"idx_{table}_game_date", "game_date"),
            (f"idx_{table}_prediction_datetime", "prediction_datetime"),
            (f"idx_{table}_model_name", "model_name"),
        ]
        for index_name, column_name in index_specs:
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} ({})").format(
                    sql.Identifier(index_name),
                    sql.Identifier(schema),
                    sql.Identifier(table),
                    sql.Identifier(column_name),
                )
            )

        cur.execute(
            sql.SQL(
                """
                CREATE INDEX IF NOT EXISTS {} ON {}.{} (game_id)
                WHERE total_scored_points IS NULL OR home_pts IS NULL OR away_pts IS NULL
                """
            ).format(
                sql.Identifier(f"idx_{table}_pending_results"),
                sql.Identifier(schema),
                sql.Identifier(table),
            )
        )

    conn.commit()


def ensure_predictions_table_exists() -> None:
    """Create the predictions table only when it does not already exist."""
    schema, table = get_predictions_schema_and_table()
    conn = connect_nba_db()

    try:
        if table_exists(conn, schema, table):
            return

        _create_predictions_table_base(conn, schema, table, drop=False)
        print(f"Table '{schema}.{table}' created.")
    finally:
        conn.close()


def has_prediction_upsert_constraint(
    conn: psycopg.Connection, schema: str, table: str
) -> bool:
    """Check whether the upload conflict target is backed by a unique constraint/index."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = %s
              AND tablename = %s
              AND indexdef ILIKE %s
            LIMIT 1
            """,
            (
                schema,
                table,
                "%(game_id, model_name, prediction_datetime, prediction_value_type)%",
            ),
        )
        return cur.fetchone() is not None


def create_predictions_table(drop: bool = False):
    """
    Create predictions table inside the configured predictions schema.

    Note: This function only creates new tables or drops/recreates existing ones.
    For migrating existing tables with data preservation, use migrate_predictions_table().

    Args:
        drop (bool): If True, drops the existing table before creating it. Default: False
    """
    schema, table = get_predictions_schema_and_table()
    conn = connect_nba_db()

    try:
        _create_predictions_table_base(conn, schema, table, drop=drop)
        print(f"Table '{schema}.{table}' is ready.")
    except Exception as e:
        print(f"Error in create_predictions_table: {e}")
        raise
    finally:
        conn.close()


def upload_predictions_to_postgre(df: "pd.DataFrame"):
    """
    Insert the summary DataFrame into the configured predictions schema/table.
    Upserts on (game_id, model_name, prediction_datetime, prediction_value_type).
    """
    import pandas as pd

    ensure_predictions_table_exists()

    schema, table = get_predictions_schema_and_table()

    conn = connect_nba_db()

    try:
        # Normalize column names to match DB (lowercase in DB)
        df = df.rename(
            columns={
                "GAME_ID": "game_id",
                "SEASON_TYPE": "season_type",
                "GAME_DATE": "game_date",
                "GAME_TIME": "game_time",
                "TEAM_NAME_TEAM_HOME": "team_name_team_home",
                "TEAM_NAME_TEAM_AWAY": "team_name_team_away",
                "PREDICTION_VALUE_TYPE": "prediction_value_type",
                "TOTAL_OVER_UNDER_LINE": "total_over_under_line",
                "TOTAL_BET365_LINE_AT_PREDICTION": "total_bet365_line_at_prediction",
                "PRED_LINE_ERROR": "pred_line_error",
                "PRED_TOTAL_POINTS": "pred_total_points",
                "PRED_PICK": "pred_pick",
                "MODEL_NAME": "model_name",
                "MODEL_TYPE": "model_type",
                "MODEL_VERSION": "model_version",
                "PREDICTION_DATE": "prediction_date",
                "PREDICTION_DATETIME": "prediction_datetime",
                "TIME_TO_MATCH_MINUTES": "time_to_match_minutes",
                "NA_COLUMNS_COUNT": "na_columns_count",
                "NA_COLUMNS_NAMES": "na_columns_names",
                "SHAP_BASE_VALUE": "shap_base_value",
                "SHAP_TOP_POSITIVE_FEATURES": "shap_top_positive_features",
                "SHAP_TOP_NEGATIVE_FEATURES": "shap_top_negative_features",
                "SHAP_DIRECTIONAL_CONFIDENCE": "shap_directional_confidence",
                "SHAP_SUPPORT_RATIO": "shap_support_ratio",
                "SHAP_TOP_K_AGREEMENT": "shap_top_k_agreement",
                "SHAP_CONFIDENCE_SCORE": "shap_confidence_score",
                "PREDICTION_SOURCE": "prediction_source",
                "SOURCE_RUN_ID": "source_run_id",
                "HOME_PTS": "home_pts",
                "AWAY_PTS": "away_pts",
                "TRAINING_CODE_TAG": "training_code_tag",
                "TRAIN_DATE_MIN": "train_date_min",
                "TRAIN_DATE_MAX": "train_date_max",
                "PRED_SPREAD_ERROR": "pred_spread_error",
                "PRED_HOME_MARGIN": "pred_home_margin",
                "SPREAD_LINE_AT_PREDICTION": "spread_line_at_prediction",
                "MODEL_TARGET": "model_target",
                "MODEL_HORIZON_MINUTES": "model_horizon_minutes",
                "MODEL_SCHEMA_VERSION": "model_schema_version",
                "MODEL_VARIANT": "model_variant",
                "SPEC_ID": "spec_id",
                "FIT_ID": "fit_id",
            }
        )

        # If both uppercase and lowercase versions exist, keep the first non-null value.
        duplicate_cols = df.columns[df.columns.duplicated()].unique().tolist()
        for col in duplicate_cols:
            dup_values = df.loc[:, df.columns == col]
            df = df.loc[:, df.columns != col]
            df[col] = dup_values.bfill(axis=1).iloc[:, 0]

        # Ensure optional enrichment columns exist after normalization.
        for col in (
            "prediction_value_type",
            "total_over_under_line",
            "total_bet365_line_at_prediction",
            "pred_line_error",
            "pred_total_points",
            "pred_pick",
            "na_columns_count",
            "na_columns_names",
            "shap_base_value",
            "shap_top_positive_features",
            "shap_top_negative_features",
            "shap_directional_confidence",
            "shap_support_ratio",
            "shap_top_k_agreement",
            "shap_confidence_score",
            "prediction_source",
            "source_run_id",
            "total_scored_points",
            "home_pts",
            "away_pts",
            "training_code_tag",
            "train_date_min",
            "train_date_max",
            "pred_spread_error",
            "pred_home_margin",
            "spread_line_at_prediction",
            "model_target",
            "model_horizon_minutes",
            "model_schema_version",
            "model_variant",
            "spec_id",
            "fit_id",
        ):
            if col not in df.columns:
                df[col] = None

        df["prediction_value_type"] = df["prediction_value_type"].fillna(
            "DIFF_FROM_LINE"
        )
        df["prediction_source"] = (
            df["prediction_source"]
            .fillna(df["model_type"])
            .fillna("unknown")
            .astype(str)
            .str.strip()
        )
        df.loc[df["prediction_source"] == "", "prediction_source"] = "unknown"
        source_run_id_fallback = (
            df["model_name"].astype(str).fillna("unknown")
            + "|"
            + df["prediction_datetime"].astype(str).fillna("unknown")
            + "|"
            + df["prediction_value_type"].astype(str).fillna("unknown")
        )
        df["source_run_id"] = df["source_run_id"].fillna(source_run_id_fallback)
        df["training_code_tag"] = df["training_code_tag"].fillna("0.0")

        # Keep only DB columns (excluding id which is SERIAL)
        columns = [
            "game_id",
            "season_type",
            "game_date",
            "game_time",
            "team_name_team_home",
            "team_name_team_away",
            "prediction_value_type",
            "total_over_under_line",
            "total_bet365_line_at_prediction",
            "pred_line_error",
            "pred_total_points",
            "pred_pick",
            "model_name",
            "model_type",
            "model_version",
            "prediction_date",
            "prediction_datetime",
            "time_to_match_minutes",
            "na_columns_count",
            "na_columns_names",
            "shap_base_value",
            "shap_top_positive_features",
            "shap_top_negative_features",
            "shap_directional_confidence",
            "shap_support_ratio",
            "shap_top_k_agreement",
            "shap_confidence_score",
            "prediction_source",
            "source_run_id",
            "total_scored_points",
            "home_pts",
            "away_pts",
            "training_code_tag",
            "train_date_min",
            "train_date_max",
            "pred_spread_error",
            "pred_home_margin",
            "spread_line_at_prediction",
            "model_target",
            "model_horizon_minutes",
            "model_schema_version",
            "model_variant",
            "spec_id",
            "fit_id",
        ]

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for upload: {missing_columns}")

        existing_table_columns = get_existing_columns(conn, schema, table)
        missing_table_columns = [
            col for col in columns if col not in existing_table_columns
        ]
        if missing_table_columns:
            raise ValueError(
                "Predictions table exists but is missing required columns for upload: "
                f"{missing_table_columns}. Run create_predictions_table() once to "
                "migrate the table before uploading."
            )

        if not has_prediction_upsert_constraint(conn, schema, table):
            raise ValueError(
                "Predictions table exists but is missing the unique constraint/index "
                "required for upserts on "
                "(game_id, model_name, prediction_datetime, prediction_value_type). "
                "Run create_predictions_table() once to migrate the table before "
                "uploading."
            )

        # Convert pandas NA to None
        df = df.where(pd.notna(df), None)

        values = [tuple(row) for row in df[columns].itertuples(index=False, name=None)]
        if not values:
            return

        conflict_columns = (
            "game_id",
            "model_name",
            "prediction_datetime",
            "prediction_value_type",
        )
        update_assignments = [
            sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(col), sql.Identifier(col))
            for col in columns
            if col not in conflict_columns
        ]
        update_assignments.append(sql.SQL("updated_at = NOW()"))

        insert_query = sql.SQL(
            """
            INSERT INTO {}.{} ({})
            VALUES ({})
            ON CONFLICT ({}) DO UPDATE SET {}
            """
        ).format(
            sql.Identifier(schema),
            sql.Identifier(table),
            sql.SQL(", ").join(map(sql.Identifier, columns)),
            sql.SQL(", ").join(sql.Placeholder() for _ in columns),
            sql.SQL(", ").join(map(sql.Identifier, conflict_columns)),
            sql.SQL(", ").join(update_assignments),
        )

        with conn.cursor() as cur:
            cur.executemany(insert_query, values)
        conn.commit()

    finally:
        conn.close()


def add_training_metadata_columns() -> None:
    """
    Idempotently bring an existing predictions table up to the current schema.

    Adds the training-metadata columns, the spread target's columns, and the
    slot-identity columns (model_target, model_horizon_minutes,
    model_schema_version, model_variant, spec_id, fit_id), then widens the
    CHECK constraints that would otherwise reject every spread row.

    model_horizon_minutes is the one worth understanding: it is what the model
    was TRAINED for, while time_to_match_minutes is when the prediction RAN.
    Without both, a T-720 model serving every game at T-60 leaves no trace.

    Safe to run repeatedly. Existing rows get NULL in the new columns.
    """
    schema, table = get_predictions_schema_and_table()
    conn = connect_nba_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {}.{} "
                    "ADD COLUMN IF NOT EXISTS training_code_tag TEXT NOT NULL DEFAULT '0.0', "
                    "ADD COLUMN IF NOT EXISTS train_date_min DATE, "
                    "ADD COLUMN IF NOT EXISTS train_date_max DATE, "
                    "ADD COLUMN IF NOT EXISTS pred_spread_error NUMERIC, "
                    "ADD COLUMN IF NOT EXISTS pred_home_margin NUMERIC, "
                    "ADD COLUMN IF NOT EXISTS spread_line_at_prediction NUMERIC, "
                    "ADD COLUMN IF NOT EXISTS model_target TEXT, "
                    "ADD COLUMN IF NOT EXISTS model_horizon_minutes INTEGER, "
                    "ADD COLUMN IF NOT EXISTS model_schema_version TEXT, "
                    "ADD COLUMN IF NOT EXISTS model_variant TEXT, "
                    "ADD COLUMN IF NOT EXISTS spec_id TEXT, "
                    "ADD COLUMN IF NOT EXISTS fit_id TEXT"
                ).format(sql.Identifier(schema), sql.Identifier(table))
            )
            # The value-type and target-present CHECKs predate the spread
            # target, so an existing table rejects every spread row until they
            # are widened. Dropped and recreated because Postgres has no
            # "alter constraint" for CHECKs.
            for constraint, definition in (
                (
                    "chk_prediction_value_type",
                    "CHECK (prediction_value_type IN "
                    "('TOTAL_POINTS', 'DIFF_FROM_LINE', 'SPREAD_ERROR'))",
                ),
                (
                    "chk_prediction_target_present",
                    "CHECK (pred_line_error IS NOT NULL "
                    "OR pred_total_points IS NOT NULL "
                    "OR pred_spread_error IS NOT NULL)",
                ),
                (
                    "chk_pred_pick",
                    "CHECK (pred_pick IS NULL OR pred_pick IN "
                    "('OVER', 'UNDER', 'PUSH', 'HOME', 'AWAY'))",
                ),
            ):
                cur.execute(
                    sql.SQL("ALTER TABLE {}.{} DROP CONSTRAINT IF EXISTS {}").format(
                        sql.Identifier(schema),
                        sql.Identifier(table),
                        sql.Identifier(constraint),
                    )
                )
                cur.execute(
                    sql.SQL("ALTER TABLE {}.{} ADD CONSTRAINT {} " + definition).format(
                        sql.Identifier(schema),
                        sql.Identifier(table),
                        sql.Identifier(constraint),
                    )
                )
        conn.commit()
        print(
            f"Training-metadata, spread and slot-identity columns added "
            f"(or already present) on '{schema}.{table}', and the "
            f"prediction_value_type / target-present / pred_pick constraints "
            f"widened for the spread target."
        )
    except Exception as e:
        print(f"Error adding training metadata columns: {e}")
        raise
    finally:
        conn.close()


def recreate_predictions_table() -> None:
    """Drop and recreate predictions table with the latest schema."""
    create_predictions_table(drop=True)


def migrate_predictions_table() -> None:
    """
    Safely migrate the predictions table to the latest schema by:
    1. Reading all existing rows into memory
    2. Dropping and recreating the table with the full schema
    3. Re-inserting the old rows (missing columns filled with NULL)

    This avoids ALTER TABLE lock contention on hosted databases like Supabase.
    """
    import pandas as pd

    schema, table = get_predictions_schema_and_table()
    conn = connect_nba_db()

    try:
        if not table_exists(conn, schema, table):
            conn.close()
            print(f"Table '{schema}.{table}' does not exist. Creating from scratch.")
            _create_predictions_table_base(conn, schema, table, drop=False)
            print(f"Table '{schema}.{table}' created.")
            return

        # 1. Read all existing data
        print(f"Reading existing data from '{schema}.{table}'...")
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT * FROM {}.{}").format(
                    sql.Identifier(schema), sql.Identifier(table)
                )
            )
            rows = cur.fetchall()
            col_names = [desc[0] for desc in cur.description]

        df_backup = pd.DataFrame(rows, columns=col_names)
        print(f"  Backed up {len(df_backup)} rows.")

        # 2. Drop and recreate with full schema
        print("  Recreating table with latest schema...")
        _create_predictions_table_base(conn, schema, table, drop=True)
        print(f"  Table '{schema}.{table}' recreated.")

        if df_backup.empty:
            print("  No rows to restore.")
            return

        # 3. Re-insert old rows (skip 'id' — it's SERIAL)
        insert_cols = [c for c in col_names if c != "id"]

        # Get the new table's columns to know which ones exist
        new_cols = get_existing_columns(conn, schema, table)
        # Only insert columns that exist in the new table
        insert_cols = [c for c in insert_cols if c in new_cols]

        insert_values = []
        for _, row in df_backup.iterrows():
            vals = []
            for c in insert_cols:
                v = row[c] if c in row.index else None
                if pd.isna(v):
                    v = None
                vals.append(v)
            insert_values.append(tuple(vals))

        insert_query = sql.SQL("INSERT INTO {}.{} ({}) VALUES ({})").format(
            sql.Identifier(schema),
            sql.Identifier(table),
            sql.SQL(", ").join(map(sql.Identifier, insert_cols)),
            sql.SQL(", ").join(sql.Placeholder() for _ in insert_cols),
        )

        with conn.cursor() as cur:
            cur.executemany(insert_query, insert_values)
        conn.commit()
        print(f"  Restored {len(insert_values)} rows with new schema.")

    except Exception as e:
        print(f"Error during migration: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create or recreate predictions table in configured schema"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate table from scratch (loses data)",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help="Migrate existing table: backup rows, recreate schema, restore data",
    )
    parser.add_argument(
        "--add-columns",
        action="store_true",
        help="Idempotently add the training-metadata, spread and slot-identity columns to an existing table",
    )
    args = parser.parse_args()

    if args.migrate:
        migrate_predictions_table()
    elif args.recreate:
        recreate_predictions_table()
    elif args.add_columns:
        add_training_metadata_columns()
    else:
        create_predictions_table(False)

    schema, table = get_predictions_schema_and_table()
    print(f"{schema}.{table} table is ready.")
