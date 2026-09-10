"""Where the bytes land: S3, or a local mirror of the same layout.

The local mirror is not a toy -- it is what makes the pipeline runnable and
testable without AWS credentials, the same way ``data/sbr_line_history`` works
for the odds scraper. Both backends take the identical key, so switching is a
flag and never a code path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from nba_ou.utils.s3_models import make_s3_client


class Storage(Protocol):
    def exists(self, key: str) -> int | None:
        """Size in bytes if the object is present, else ``None``."""

    def put(self, key: str, data: bytes, metadata: dict[str, str]) -> None: ...

    def describe(self) -> str: ...


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key

    def exists(self, key: str) -> int | None:
        p = self._path(key)
        return p.stat().st_size if p.exists() else None

    def put(self, key: str, data: bytes, metadata: dict[str, str]) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)

    def describe(self) -> str:
        return f"local:{self.root}"


class S3Storage:
    def __init__(self, *, bucket: str, profile: str | None, region: str) -> None:
        self.bucket = bucket
        self.client = make_s3_client(profile=profile, region=region)

    def exists(self, key: str) -> int | None:
        try:
            head = self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception:
            return None
        return int(head["ContentLength"])

    def put(self, key: str, data: bytes, metadata: dict[str, str]) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType="application/pdf",
            Metadata={k: str(v)[:1024] for k, v in metadata.items()},
        )

    def describe(self) -> str:
        return f"s3://{self.bucket}"


class ManifestSync:
    """Keep the per-season manifests mirrored in S3 alongside the PDFs.

    The manifest is written locally first -- it is rewritten on every checkpoint
    and needs cheap atomic replaces -- then mirrored up at the end of each
    season. Pulling it back down on start is what lets a backfill resume from a
    different machine, or survive losing the working directory.
    """

    def __init__(self, *, storage: S3Storage, prefix: str = "injury_reports") -> None:
        self.storage = storage
        self.prefix = prefix

    def key_for(self, season: str) -> str:
        return f"{self.prefix}/manifest/season={season}/manifest.parquet"

    def pull(self, season: str, local_path: Path) -> bool:
        """Fetch the remote manifest if we have no local copy. Returns True if
        something was restored."""
        if local_path.exists():
            return False
        key = self.key_for(season)
        try:
            obj = self.storage.client.get_object(Bucket=self.storage.bucket, Key=key)
        except Exception:
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(obj["Body"].read())
        return True

    def push(self, season: str, local_path: Path) -> str | None:
        if not local_path.exists():
            return None
        key = self.key_for(season)
        self.storage.client.put_object(
            Bucket=self.storage.bucket,
            Key=key,
            Body=local_path.read_bytes(),
            ContentType="application/octet-stream",
        )
        return key
