"""Raw-response archive; local and S3 use the same object keys."""

from __future__ import annotations

import gzip
from pathlib import Path

from nba_ou.fetch_data.injury_reports.archive.storage import LocalStorage, Storage
from nba_ou.utils.s3_models import make_s3_client


class S3JsonStorage:
    """S3 implementation of Storage with JSON rather than PDF metadata."""

    def __init__(self, bucket: str, *, profile: str | None, region: str) -> None:
        self.bucket = bucket
        self.client = make_s3_client(profile=profile, region=region)

    def exists(self, key: str) -> int | None:
        from botocore.exceptions import ClientError

        try:
            return int(self.client.head_object(Bucket=self.bucket, Key=key)["ContentLength"])
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"404", "NoSuchKey"}:
                return None
            raise

    def get(self, key: str) -> bytes | None:
        from botocore.exceptions import ClientError

        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"404", "NoSuchKey"}:
                return None
            raise

    def put(self, key: str, data: bytes, metadata: dict[str, str]) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=metadata.get("content_type", "application/octet-stream"),
            ContentEncoding=metadata.get("encoding", "identity"),
        )

    def describe(self) -> str:
        return f"s3://{self.bucket}"


def key_for(endpoint: str, season_year: int, game_id: str) -> str:
    if endpoint not in {"gamerotation", "playbyplayv3"}:
        raise ValueError(endpoint)
    if not game_id.isdigit() or len(game_id) != 10:
        raise ValueError(f"Invalid NBA game ID: {game_id}")
    return f"nba_api_raw/{endpoint}/{season_year}/{game_id}.json.gz"


class RawArchive:
    def __init__(self, storage: Storage | None = None, *, root: Path = Path("data")):
        self.storage = storage or LocalStorage(root)

    def put(self, endpoint: str, season_year: int, game_id: str, raw: bytes) -> int:
        compressed = gzip.compress(raw, mtime=0)
        self.storage.put(
            key_for(endpoint, season_year, game_id),
            compressed,
            {"content_type": "application/json", "encoding": "gzip"},
        )
        return len(compressed)

    def get(self, endpoint: str, season_year: int, game_id: str) -> bytes | None:
        compressed = self.storage.get(key_for(endpoint, season_year, game_id))
        return gzip.decompress(compressed) if compressed is not None else None

    def exists(self, endpoint: str, season_year: int, game_id: str) -> bool:
        return self.storage.exists(key_for(endpoint, season_year, game_id)) is not None
