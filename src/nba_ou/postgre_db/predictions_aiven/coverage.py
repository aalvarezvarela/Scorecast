"""Read existing own-horizon predictions before a historical backfill."""

from __future__ import annotations

import pandas as pd

from nba_ou.postgre_db.config.db_config import connect_nba_db


def historical_coverage(start_date: str, end_date: str) -> pd.DataFrame:
    """Only predictions whose model was trained before the evaluated date.

    A different spec does not cover the requested model. Invalid retrospective
    runs (a production model trained on the evaluated game or later) never
    prevent a correct reconstruction. No table creation or updates here.
    """
    with connect_nba_db("aiven") as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT i.game_id, m.spec_id, m.schema_version,
                       m.target, m.horizon_minutes, m.variant, m.train_date_max, m.kind,
                       m.dataset_type, i.snapshot_horizon_minutes AS input_horizon_minutes
                FROM prediction.predictions p
                JOIN prediction.inputs i USING (input_id)
                JOIN prediction.games g USING (game_id)
                JOIN prediction.model_builds m USING (model_build_id)
                WHERE g.game_date BETWEEN %s AND %s
                  AND m.kind IN (1, 3)
                  AND ((m.dataset_type = 2 AND i.input_mode = 1
                        AND i.snapshot_horizon_minutes = m.horizon_minutes)
                       OR (m.dataset_type = 1 AND i.input_mode = 0
                           AND m.horizon_minutes = 0 AND i.snapshot_horizon_minutes > 0))
                  AND m.train_date_max < g.game_date
                """,
                (start_date, end_date),
            )
            return pd.DataFrame(
                cur.fetchall(), columns=[column.name for column in cur.description]
            )
