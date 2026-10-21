"""The training frames the daily refit fitted on, kept in S3 for other jobs.

``scripts/retrain_prediction_models.py`` builds the whole closing-line history
every day to refit the registry slots. The same frame is what any other model
fitted on those slots' rows needs -- TabPFN, which has no stored model and is
fitted on every run (``nba_ou.prediction.prediction_tabpfn_client``) -- so the
refit keeps it instead of discarding it::

    train_data/2_5/daily/closing_line_data_2_5_20261021.parquet

The date is the refit's cutoff. Only the newest ``KEEP_DAILY_FRAMES`` of each
(dataset type, schema version) are kept: a frame is a rebuild of the one
before plus a day of games, so older ones hold nothing the newest lacks.
"""

from __future__ import annotations

import io
from datetime import date, datetime

import pandas as pd
from nba_ou.create_training_data.train_data_store import (
    daily_frame_key,
    daily_frames_prefix,
    parse_daily_frame_key,
)
from nba_ou.utils.s3_models import (
    list_s3_objects,
    load_parquet_from_bytes,
    read_s3_object_bytes,
    upload_bytes_to_s3,
)

KEEP_DAILY_FRAMES = 7


def _daily_frames(
    *, s3_client, bucket: str, dataset_type: str, schema_version: str
) -> list[tuple[str, str]]:
    """``(limit_date, key)`` of every daily frame of this kind, oldest first."""
    found = []
    prefix = daily_frames_prefix(schema_version=schema_version)
    for obj in list_s3_objects(s3_client=s3_client, bucket=bucket, prefix=prefix):
        parsed = parse_daily_frame_key(obj["Key"])
        if parsed and parsed[:2] == (dataset_type, schema_version):
            found.append((parsed[2], obj["Key"]))
    return sorted(found)


def upload_daily_frame(
    df: pd.DataFrame,
    *,
    s3_client,
    bucket: str,
    dataset_type: str,
    schema_version: str,
    limit_date: str,
    keep: int = KEEP_DAILY_FRAMES,
) -> str:
    """Store ``df`` as the day's frame and prune the older ones; returns its key.

    ``limit_date`` is ``YYYY-MM-DD`` or ``YYYYMMDD``.
    """
    stamp = pd.Timestamp(limit_date).strftime("%Y%m%d")
    key = daily_frame_key(
        dataset_type=dataset_type, schema_version=schema_version, limit_date=stamp
    )
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, compression="zstd")
    upload_bytes_to_s3(
        s3_client=s3_client, bucket=bucket, key=key, data=buffer.getvalue()
    )
    stored = _daily_frames(
        s3_client=s3_client,
        bucket=bucket,
        dataset_type=dataset_type,
        schema_version=schema_version,
    )
    for _, old_key in stored[: max(len(stored) - keep, 0)]:
        s3_client.delete_object(Bucket=bucket, Key=old_key)
    return key


def latest_daily_frame(
    *, s3_client, bucket: str, dataset_type: str, schema_version: str
) -> tuple[str, date] | None:
    """The newest daily frame's key and cutoff date, or None when there is none."""
    stored = _daily_frames(
        s3_client=s3_client,
        bucket=bucket,
        dataset_type=dataset_type,
        schema_version=schema_version,
    )
    if not stored:
        return None
    limit_date, key = stored[-1]
    return key, datetime.strptime(limit_date, "%Y%m%d").date()


def load_daily_frame(*, s3_client, bucket: str, key: str) -> pd.DataFrame:
    return load_parquet_from_bytes(
        read_s3_object_bytes(s3_client=s3_client, bucket=bucket, key=key)
    )
