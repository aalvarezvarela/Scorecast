"""Immutable raw MoviOdds response bundles, separate from normalized DB ticks."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from uuid import uuid4

PREFIX = "train_data/odds_sources/moviodds/raw/v1"


def bundle_key(day, label: str, fetched_at: datetime) -> str:
    stamp = fetched_at.astimezone(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{PREFIX}/{day.isoformat()}/{label}/{stamp}_{uuid4().hex}.json.gz"


class MoviOddsArchive:
    def __init__(self, client, bucket: str):
        self.client, self.bucket = client, bucket

    def put(self, key: str, payload: dict) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=gzip.compress(
                json.dumps(payload, separators=(",", ":"), default=str).encode()
            ),
            ContentType="application/json",
            ContentEncoding="gzip",
        )


def make_archive() -> MoviOddsArchive:
    from nba_ou.config.settings import SETTINGS
    from nba_ou.utils.s3_models import make_s3_client

    if not SETTINGS.s3_bucket:
        raise ValueError("S3 bucket is not configured")
    return MoviOddsArchive(
        make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region),
        SETTINGS.s3_bucket,
    )
