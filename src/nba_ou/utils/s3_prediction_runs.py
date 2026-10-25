"""Archive one prediction run to S3 (``docs/serving/prediction_storage.md``).

Folder layout
─────────────
prediction_snapshots/v2/game_date={game_date}/{started_at}_{workflow}_{run_uuid}/
                               ← started_at in UTC (20261025T190012Z); workflow
                                 daily, past_snapshots or tabpfn
  manifest.json                ← run metadata, per-slot outcomes, file status
  inputs_closing_line.parquet  ← the closing rows the models read
  inputs_intermediate_line.parquet
                               ← the snapshot rows the models read
  predictions.parquet          ← every output of the run, long format

Each input row is written once per run, not once per model: with ~3,000
columns a Parquet file pays 1-2 MB of footer whatever its row count, and
duplicates are only shared within a file. The pandas schema metadata (another
~1.4 MB per file) and the column statistics are left out. In ``latest`` mode a
model read its row through ``input_limits.limit_to_training_support``; that is
recomputed from the stored row, the model's horizon and the support table
(``support_table_version`` in the manifest).

Every file is uploaded on its own and the manifest is written last, whatever
happened to the others: it records each file's status, so a missing object is
never mistaken for one that was never meant to exist.
"""

from __future__ import annotations

import hashlib
import io
import json
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nba_ou.postgre_db.predictions_aiven.schema import WORKFLOW_NAMES
from nba_ou.utils.s3_models import upload_bytes_to_s3

if TYPE_CHECKING:
    from nba_ou.prediction.prediction_store import PredictionRunRecorder, RunTables

DEFAULT_RUN_PREFIX = "prediction_snapshots/v2"


def run_prefix(
    game_date: str,
    started_at,
    workflow: int,
    run_uuid,
    *,
    prefix: str = DEFAULT_RUN_PREFIX,
) -> str:
    """``prediction_snapshots/v2/game_date=2026-10-21/20261021T190012Z_daily_<uuid>/``.

    The UTC start first, so a date's runs list in the order they ran; the
    workflow tells them apart without opening a manifest; the UUID is
    ``runs.run_uuid``. ``game_date=`` lets Parquet readers treat the archive as
    one dataset partitioned by date.
    """
    stamp = pd.Timestamp(started_at)
    stamp = (
        stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    )
    return (
        f"{prefix}/game_date={game_date}/"
        f"{stamp:%Y%m%dT%H%M%SZ}_{WORKFLOW_NAMES[workflow]}_{run_uuid}/"
    )


def compact_parquet_bytes(frame: pd.DataFrame) -> bytes:
    """``frame`` as zstd Parquet without pandas metadata or column statistics."""
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata(None)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression="zstd", write_statistics=False)
    return buffer.getvalue()


def support_table_version() -> str:
    """A short hash of the training-support table the latest-input limits read."""
    from nba_ou.prediction.input_support_table import UNSUPPORTED_NEWS_COLUMNS

    payload = json.dumps(sorted(UNSUPPORTED_NEWS_COLUMNS.items())).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def predictions_long(tables: RunTables) -> pd.DataFrame:
    """The run's predictions with their input row's lines and their game."""
    return (
        tables.predictions.merge(
            tables.inputs,
            on=["game_id", "input_mode", "snapshot_horizon_minutes"],
            how="left",
        )
        .merge(
            tables.games[["game_id", "game_date", "home_team", "away_team"]],
            on="game_id",
            how="left",
        )
        .assign(run_uuid=str(tables.run["run_uuid"]))
    )


def upload_prediction_run(
    *,
    s3_client,
    bucket: str,
    recorder: PredictionRunRecorder,
    tables: RunTables,
    prefix: str = DEFAULT_RUN_PREFIX,
) -> tuple[str, list[str]]:
    """Write the run's inputs, predictions and manifest.

    Returns the folder key and the files that failed to upload. The manifest
    itself failing raises.
    """
    base = run_prefix(
        recorder.game_date,
        tables.run["started_at"],
        tables.run["workflow"],
        recorder.run_uuid,
        prefix=prefix,
    )
    files: dict[str, dict] = {}
    failures: list[str] = []

    def upload(name: str, frame: pd.DataFrame) -> None:
        key = f"{base}{name}"
        try:
            upload_bytes_to_s3(
                s3_client=s3_client,
                bucket=bucket,
                key=key,
                data=compact_parquet_bytes(frame),
            )
        except Exception as exc:  # noqa: BLE001 -- recorded in the manifest
            files[name] = {"rows": len(frame), "uploaded": False, "error": str(exc)}
            failures.append(f"{name}: {exc}")
        else:
            files[name] = {"rows": len(frame), "uploaded": True}

    for dataset_type, rows in recorder.input_frames().items():
        upload(f"inputs_{dataset_type}.parquet", rows)
    upload("predictions.parquet", predictions_long(tables))

    run = tables.run
    manifest = {
        "run_uuid": str(run["run_uuid"]),
        "game_date": recorder.game_date,
        "workflow": run["workflow"],
        "status": run["status"],
        "started_at": run["started_at"].isoformat(),
        "finished_at": run["finished_at"].isoformat(),
        "git_sha": run["git_sha"],
        "main_book": recorder.book,
        "frame_versions": recorder.frame_versions,
        "support_table_version": support_table_version(),
        "counts": {
            "games": len(tables.games),
            "inputs": len(tables.inputs),
            "predictions": len(tables.predictions),
            "model_builds": len(tables.model_builds),
        },
        "setup_errors": recorder.setup_errors,
        "slots": recorder.slots,
        "files": files,
    }
    upload_bytes_to_s3(
        s3_client=s3_client,
        bucket=bucket,
        key=f"{base}manifest.json",
        data=json.dumps(manifest, indent=2, default=str).encode(),
    )
    return base, failures
