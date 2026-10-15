"""Past seasons of database tables, stored in S3 as Parquet.

Layout, under the model registry bucket::

    s3://<bucket>/train_data/input_cache/v1/<dataset>/<code key>/season=YYYY.parquet
    s3://<bucket>/train_data/input_cache/v1/<dataset>/<code key>/season=YYYY.json

The JSON beside each season is its manifest: the server fingerprint the
season was fetched (or last validated) under, when that was, the row count and
the SHA-256 of the Parquet bytes. The refresher writes both; the prediction
job reads them (``season_tables``).

A frame is stored only if Parquet gives it back exactly: same columns,
dtypes and index, floats bit for bit, and the same Python type for every
object value (a ``date`` must not come back as a ``Timestamp``). A frame that
would not is refused rather than stored approximately.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Under ``train_data/``: these are the inputs training data is built from, and
#: the prefix is writable by the CLI user and readable by the CI role, which
#: ``input_cache/`` at the root is not (see docs/injury_report_archive_plan.md
#: 17.2). ``organize_s3_train_data.py`` only moves objects at the root of
#: ``train_data/``, so this subfolder is left alone.
CACHE_PREFIX = "train_data/input_cache/v1"


class NotExactInParquet(ValueError):
    """A frame Parquet would not give back exactly."""


def frame_differences(actual: pd.DataFrame, expected: pd.DataFrame) -> list[str]:
    """Why two frames are not identical; empty when they are.

    Stricter than ``assert_frame_equal(check_exact=True)``: floats are compared
    bit for bit (``-0.0`` is not ``0.0``) and object values by type as well.
    """
    problems = []
    if list(actual.columns) != list(expected.columns):
        return [
            f"columns {list(actual.columns)[:5]}... != {list(expected.columns)[:5]}..."
        ]
    if not actual.index.equals(expected.index) or type(actual.index) is not type(
        expected.index
    ):
        problems.append("index differs")
    for column in expected.columns:
        left, right = actual[column], expected[column]
        if left.dtype != right.dtype:
            problems.append(f"{column}: dtype {left.dtype} != {right.dtype}")
            continue
        if right.dtype.kind == "f":
            a, b = left.to_numpy(), right.to_numpy()
            nan = np.isnan(b)
            if not np.array_equal(np.isnan(a), nan) or not np.array_equal(
                a[~nan].view(np.int64), b[~nan].view(np.int64)
            ):
                problems.append(f"{column}: float values differ")
        elif right.dtype == object:
            if [_typed(v) for v in left.tolist()] != [
                _typed(v) for v in right.tolist()
            ]:
                problems.append(f"{column}: object values or their types differ")
        elif not left.equals(right):
            problems.append(f"{column}: values differ")
    return problems


def _typed(value):
    if isinstance(value, float) and value != value:
        return ("float", "nan")
    return (type(value).__name__, value)


def to_exact_parquet(frame: pd.DataFrame) -> bytes:
    """Parquet bytes of ``frame``, verified to read back identically."""
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    payload = buffer.getvalue()
    back = from_parquet(payload)
    problems = frame_differences(back, frame.reset_index(drop=True))
    if problems:
        raise NotExactInParquet("; ".join(problems[:5]))
    return payload


def from_parquet(payload: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(payload))


@dataclass(frozen=True)
class SeasonEntry:
    fingerprint: list
    validated_at: pd.Timestamp
    rows: int
    sha256: str


class SeasonStore:
    """Season pieces of datasets in one S3 bucket."""

    def __init__(self, s3_client, bucket: str, prefix: str = CACHE_PREFIX) -> None:
        self.s3 = s3_client
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")

    def _key(self, dataset: str, code_key: str, season: int, suffix: str) -> str:
        return f"{self.prefix}/{dataset}/{code_key}/season={int(season)}.{suffix}"

    def _get(self, key: str) -> bytes | None:
        try:
            return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except Exception as exc:  # botocore ClientError / NoSuchKey
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404", "NotFound"):
                return None
            raise

    def _put(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data)

    def entry(self, dataset: str, code_key: str, season: int) -> SeasonEntry | None:
        raw = self._get(self._key(dataset, code_key, season, "json"))
        if raw is None:
            return None
        meta = json.loads(raw)
        return SeasonEntry(
            fingerprint=meta["fingerprint"],
            validated_at=pd.Timestamp(meta["validated_at"]),
            rows=int(meta["rows"]),
            sha256=meta["sha256"],
        )

    def read(self, dataset: str, code_key: str, season: int, entry: SeasonEntry):
        """The season's frame, or None if missing or not the bytes ``entry`` vouches for."""
        payload = self._get(self._key(dataset, code_key, season, "parquet"))
        if payload is None or hashlib.sha256(payload).hexdigest() != entry.sha256:
            return None
        return from_parquet(payload)

    def write(
        self,
        dataset: str,
        code_key: str,
        season: int,
        frame: pd.DataFrame,
        fingerprint: list,
        validated_at: pd.Timestamp,
    ) -> SeasonEntry:
        payload = to_exact_parquet(frame)
        entry = SeasonEntry(
            fingerprint=fingerprint,
            validated_at=validated_at,
            rows=len(frame),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        # Data first, then the manifest that vouches for it: a reader never
        # pairs a manifest with a piece it does not describe (the hash check
        # would refuse one anyway).
        self._put(self._key(dataset, code_key, season, "parquet"), payload)
        self._write_entry(dataset, code_key, season, entry)
        return entry

    def touch(
        self, dataset: str, code_key: str, season: int, entry: SeasonEntry, validated_at
    ):
        """Record that the season was re-validated unchanged."""
        updated = SeasonEntry(entry.fingerprint, validated_at, entry.rows, entry.sha256)
        self._write_entry(dataset, code_key, season, updated)
        return updated

    def _write_entry(self, dataset, code_key, season, entry: SeasonEntry) -> None:
        meta = {
            "fingerprint": entry.fingerprint,
            "validated_at": pd.Timestamp(entry.validated_at).isoformat(),
            "rows": entry.rows,
            "sha256": entry.sha256,
        }
        self._put(
            self._key(dataset, code_key, season, "json"), json.dumps(meta).encode()
        )


def default_store() -> SeasonStore | None:
    """The store in the configured bucket, or None when S3 is not configured."""
    from nba_ou.config.settings import SETTINGS
    from nba_ou.utils.s3_models import make_s3_client

    if not SETTINGS.s3_bucket:
        return None
    client = make_s3_client(
        profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region
    )
    return SeasonStore(client, SETTINGS.s3_bucket)
