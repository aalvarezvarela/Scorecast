"""Game-text embeddings in S3: one Parquet file per season, plus a manifest.

::

    train_data/text_embeddings/v1/source=report/model=qwen3-embedding-8b/
        season=2025-26.parquet
        ...
        manifest.json

One row per embedded input (``embedding_inputs.VIEWS``: three per report).
The vectors are the API's values, bit for bit. Each is stored either as
bfloat16 codes and one float32 divisor (``vector_codec``, half the size) when
that reproduces it exactly, or as float32 otherwise; Parquet's zstd adds only
lossless compression. A write re-reads what it serialized and refuses to upload
anything that does not decode to the same bits. The manifest records how the
vectors were made (model, tokenizer, instructions, rewrite options) and a
sha256 of every season file, checked on every read; a write made with other
settings than the manifest's is refused, so vectors of different inputs are
never mixed in one store.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from nba_ou.data_processing.game_texts import vector_codec
from nba_ou.utils.s3_models import (
    read_s3_object_bytes,
    upload_bytes_to_s3,
)

STORE_VERSION = 1
DEFAULT_PREFIX = "train_data/text_embeddings/v1/source=report/model=qwen3-embedding-8b"
#: Metadata columns, in stored order; ``KEY`` identifies a row.
ROW_COLUMNS = (
    "game_id",
    "season",
    "game_date",
    "view",
    "side",
    "team_id",
    "published_at",
    "published_at_utc",
    "report_sha256",
    "input_sha256",
    "n_players",
    "n_tokens",
    "api_norm",
    "embedded_at_utc",
)
KEY = ("game_id", "view", "side")
#: Manifest fields that must match between a write and the stored manifest.
SETTINGS_FIELDS = (
    "model_id",
    "endpoint",
    "tokenizer",
    "instructions",
    "rewrite_options",
    "views",
    "vector_dimension",
)
ROW_TYPES = {
    "game_id": pa.string(),
    "season": pa.string(),
    "game_date": pa.string(),
    "view": pa.string(),
    "side": pa.string(),
    "team_id": pa.int64(),
    "published_at": pa.string(),
    "published_at_utc": pa.string(),
    "report_sha256": pa.string(),
    "input_sha256": pa.string(),
    "n_players": pa.int64(),
    "n_tokens": pa.int64(),
    "api_norm": pa.float64(),
    "embedded_at_utc": pa.string(),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance only
        return None


@dataclass
class EmbeddingTable:
    """Row metadata and the matching vectors (float32, the API's values)."""

    rows: pd.DataFrame
    vectors: np.ndarray

    def __post_init__(self):
        self.rows = self.rows.reset_index(drop=True)
        self.vectors = np.ascontiguousarray(self.vectors, np.float32)
        if self.vectors.ndim != 2 or len(self.rows) != len(self.vectors):
            raise ValueError(
                f"{len(self.rows)} rows but vectors of shape {self.vectors.shape}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    @classmethod
    def empty(cls, dimension: int) -> EmbeddingTable:
        return cls(pd.DataFrame(columns=list(ROW_COLUMNS)), np.empty((0, dimension)))

    def take(self, mask) -> EmbeddingTable:
        mask = np.asarray(mask, bool)
        return EmbeddingTable(self.rows[mask], self.vectors[mask])

    def sorted(self) -> EmbeddingTable:
        """By game, then raw before the rewrites, home before away."""
        order = (
            pd.DataFrame(
                {
                    "game_id": self.rows["game_id"].astype(str).to_numpy(),
                    "view": (self.rows["view"] != "raw").to_numpy(),
                    "side": self.rows["side"].map({"home": 0, "away": 1}).to_numpy(),
                }
            )
            .sort_values(
                ["game_id", "view", "side"], na_position="first", kind="stable"
            )
            .index.to_numpy()
        )
        return EmbeddingTable(self.rows.iloc[order], self.vectors[order])

    @staticmethod
    def concat(tables: list[EmbeddingTable]) -> EmbeddingTable:
        tables = [table for table in tables if len(table)] or tables[:1]
        return EmbeddingTable(
            pd.concat([table.rows for table in tables], ignore_index=True),
            np.concatenate([table.vectors for table in tables]),
        )


def _fixed_lists(flat: np.ndarray, dimension: int) -> pa.Array:
    return pa.FixedSizeListArray.from_arrays(pa.array(flat.reshape(-1)), dimension)


def serialize(table: EmbeddingTable) -> tuple[bytes, dict]:
    """Parquet bytes of ``table`` and counts of each vector encoding.

    ``encoding`` says where each vector is: ``bf16`` (``vector_bf16`` codes over
    ``vector_divisor``) or ``f32`` (``vector_f32``). The unused column of a row
    holds zeros, which zstd stores almost for free (Parquet cannot read back
    null fixed-size lists)."""
    count, dimension = table.vectors.shape
    codes = np.zeros((count, dimension), np.uint16)
    divisors = np.zeros(count, np.float32)
    compact = np.zeros(count, bool)
    for index, vector in enumerate(table.vectors):
        encoded = vector_codec.encode(vector)
        if encoded is not None:
            codes[index], divisors[index] = encoded
            compact[index] = True
    wide = np.where(compact[:, None], np.float32(0), table.vectors)
    columns = {
        name: pa.array(
            table.rows[name].astype(object).where(table.rows[name].notna(), None),
            type=ROW_TYPES[name],
        )
        for name in ROW_COLUMNS
    }
    columns["encoding"] = pa.array(np.where(compact, "bf16", "f32"))
    columns["vector_bf16"] = _fixed_lists(codes, dimension)
    columns["vector_divisor"] = pa.array(divisors)
    columns["vector_f32"] = _fixed_lists(wide, dimension)
    buffer = io.BytesIO()
    pq.write_table(pa.table(columns), buffer, compression="zstd", compression_level=9)
    data = buffer.getvalue()
    if not vector_codec.same_bits(deserialize(data).vectors, table.vectors):
        raise RuntimeError("Serialized embeddings do not decode to the same bits")
    return data, {"bf16_rows": int(compact.sum()), "f32_rows": int((~compact).sum())}


def deserialize(data: bytes) -> EmbeddingTable:
    arrow = pq.read_table(io.BytesIO(data))
    rows = arrow.select(list(ROW_COLUMNS)).to_pandas()
    rows["team_id"] = rows["team_id"].astype("Int64")
    count = arrow.num_rows
    dimension = arrow.schema.field("vector_f32").type.list_size

    def matrix(name: str) -> np.ndarray:
        values = arrow[name].combine_chunks().flatten()
        return values.to_numpy().reshape(count, dimension)

    compact = arrow["encoding"].to_numpy(zero_copy_only=False) == "bf16"
    vectors = matrix("vector_f32").astype(np.float32, copy=True)
    if compact.any():
        codes = matrix("vector_bf16")[compact]
        divisors = arrow["vector_divisor"].to_numpy()[compact].astype(np.float32)
        vectors[compact] = vector_codec.bf16_values(codes) / divisors[:, None]
    return EmbeddingTable(rows, vectors)


class EmbeddingStore:
    def __init__(self, s3_client, bucket: str, prefix: str = DEFAULT_PREFIX):
        self.s3_client = s3_client
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")

    @property
    def manifest_key(self) -> str:
        return f"{self.prefix}/manifest.json"

    def season_key(self, season: str) -> str:
        return f"{self.prefix}/season={season}.parquet"

    def manifest(self) -> dict | None:
        try:
            data = read_s3_object_bytes(
                s3_client=self.s3_client, bucket=self.bucket, key=self.manifest_key
            )
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise
        return json.loads(data)

    def seasons(self) -> list[str]:
        return sorted((self.manifest() or {}).get("seasons", {}))

    def read_season(self, season: str) -> EmbeddingTable | None:
        entry = (self.manifest() or {}).get("seasons", {}).get(season)
        if entry is None:
            return None
        data = read_s3_object_bytes(
            s3_client=self.s3_client, bucket=self.bucket, key=entry["key"]
        )
        if sha256_bytes(data) != entry["sha256"]:
            raise RuntimeError(
                f"s3://{self.bucket}/{entry['key']} does not match its manifest sha256"
            )
        return deserialize(data)

    def write_season(self, season: str, table: EmbeddingTable, settings: dict) -> dict:
        """Replace one season's file, then its manifest entry."""
        manifest = self.manifest()
        if manifest is None:
            manifest = {"store_version": STORE_VERSION, **settings, "seasons": {}}
        differing = [
            field
            for field in SETTINGS_FIELDS
            if manifest.get(field) != settings.get(field)
        ]
        if differing:
            raise RuntimeError(
                f"Embedding settings differ from s3://{self.bucket}/"
                f"{self.manifest_key} in {differing}; use another prefix"
            )
        table = table.sorted()
        duplicated = table.rows.duplicated(list(KEY))
        if duplicated.any():
            raise ValueError(f"{int(duplicated.sum())} duplicated rows in {season}")
        data, encodings = serialize(table)
        key = self.season_key(season)
        upload_bytes_to_s3(
            s3_client=self.s3_client, bucket=self.bucket, key=key, data=data
        )
        entry = {
            "key": key,
            "sha256": sha256_bytes(data),
            "bytes": len(data),
            "rows": len(table),
            "games": int(table.rows["game_id"].nunique()),
            **encodings,
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": git_commit(),
        }
        manifest["seasons"][season] = entry
        upload_bytes_to_s3(
            s3_client=self.s3_client,
            bucket=self.bucket,
            key=self.manifest_key,
            data=json.dumps(manifest, indent=2, sort_keys=True).encode(),
        )
        return entry
