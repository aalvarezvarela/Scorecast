"""Date-scoped S3 referee snapshots, with a direct pointer to the latest crew."""

from __future__ import annotations

import hashlib
import io
import json

import pandas as pd
from nba_ou.fetch_data.referees.fetch_refs_data import ASSIGNMENT_COLUMNS

PREFIX = "train_data/live_inputs/v1/referees"


def normalize_assignments(frame: pd.DataFrame) -> pd.DataFrame:
    """Stable nullable strings and row order for storage and content comparison."""
    required = ASSIGNMENT_COLUMNS[:-1]
    if any(column not in frame for column in required):
        raise ValueError("Referee assignments are missing required columns")
    out = frame.reindex(columns=ASSIGNMENT_COLUMNS).astype("string")
    for column in out:
        out[column] = (
            out[column]
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .replace("", pd.NA)
        )
    if out.Game.isna().any() or not out.Game.str.fullmatch(r"[^@]+@[^@]+").all():
        raise ValueError("Invalid referee assignment matchup")
    if out.Game.duplicated().any():
        raise ValueError("Duplicate referee assignment matchups")
    return out.sort_values("Game").reset_index(drop=True)


class AssignmentStore:
    def __init__(self, s3_client, bucket: str):
        self.s3 = s3_client
        self.bucket = bucket

    def _base(self, game_date: str) -> str:
        day = pd.Timestamp(game_date).date().isoformat()
        return f"{PREFIX}/game_date={day}"

    def _get(self, key: str) -> bytes | None:
        try:
            body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"]
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404", "NotFound"):
                return None
            raise
        try:
            return body.read()
        finally:
            close = getattr(body, "close", None)
            if close:
                close()

    def _manifest(self, game_date: str) -> dict | None:
        payload = self._get(f"{self._base(game_date)}/latest.json")
        if payload is None:
            return None
        meta = json.loads(payload)
        required = {
            "version",
            "assignment_date",
            "fetched_at_utc",
            "key",
            "rows",
            "content_sha256",
            "parquet_sha256",
        }
        if not isinstance(meta, dict) or not required.issubset(meta):
            raise ValueError("Incomplete referee snapshot manifest")
        fetched_at = pd.Timestamp(meta["fetched_at_utc"])
        if (
            meta["version"] != 1
            or meta["assignment_date"] != game_date
            or fetched_at.tzinfo is None
            or fetched_at.tz_convert("America/New_York").date().isoformat() != game_date
        ):
            raise ValueError("Invalid referee snapshot date or version")
        stamp = fetched_at.tz_convert("UTC").strftime("%Y-%m-%dT%H-%M-%S.%fZ")
        if (
            meta["key"]
            != f"{self._base(game_date)}/fetched_at={stamp}/assignments.parquet"
        ):
            raise ValueError("Invalid referee snapshot key")
        return meta

    def read(self, game_date: str) -> tuple[pd.DataFrame, dict] | None:
        meta = self._manifest(game_date)
        if meta is None:
            return None
        payload = self._get(meta["key"])
        if payload is None:
            return None
        if hashlib.sha256(payload).hexdigest() != meta["parquet_sha256"]:
            raise ValueError("Referee snapshot checksum mismatch")
        frame = normalize_assignments(pd.read_parquet(io.BytesIO(payload)))
        if len(frame) != meta["rows"] or self._hash(frame) != meta["content_sha256"]:
            raise ValueError("Referee snapshot content mismatch")
        return frame, meta

    @staticmethod
    def _hash(frame: pd.DataFrame) -> str:
        return hashlib.sha256(
            frame.to_json(orient="split", index=False).encode()
        ).hexdigest()

    def save(
        self, game_date: str, frame: pd.DataFrame, fetched_at: pd.Timestamp
    ) -> bool:
        """Write changed nonempty data first, then publish its pointer; skip duplicates."""
        frame = normalize_assignments(frame)
        if frame.empty:
            return False
        fetched_at = pd.Timestamp(fetched_at).tz_convert("UTC")
        if fetched_at.tz_convert("America/New_York").date().isoformat() != game_date:
            raise ValueError("Cannot save live assignments under a different game date")
        digest = self._hash(frame)
        try:
            existing = self._manifest(game_date)
        except (ValueError, KeyError, TypeError):
            existing = None  # Replace malformed metadata after a successful live read.
        if existing and (
            existing["content_sha256"] == digest
            or pd.Timestamp(existing["fetched_at_utc"]) >= fetched_at
        ):
            return False
        stamp = fetched_at.strftime("%Y-%m-%dT%H-%M-%S.%fZ")
        key = f"{self._base(game_date)}/fetched_at={stamp}/assignments.parquet"
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False, compression="zstd", engine="pyarrow")
        payload = buffer.getvalue()
        meta = {
            "version": 1,
            "assignment_date": game_date,
            "fetched_at_utc": fetched_at.isoformat(),
            "source": "https://official.nba.com/referee-assignments/",
            "key": key,
            "rows": len(frame),
            "content_sha256": digest,
            "parquet_sha256": hashlib.sha256(payload).hexdigest(),
        }
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=payload)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self._base(game_date)}/latest.json",
            Body=json.dumps(meta).encode(),
            ContentType="application/json",
        )
        return True
