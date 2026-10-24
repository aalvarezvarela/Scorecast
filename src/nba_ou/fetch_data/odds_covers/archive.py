"""Raw Covers answers in S3, so the parser can be redone without the site.

::

    train_data/odds_sources/covers/raw/v1/<YYYY-MM-DD>/
      matchups/<fetched_at>.html.gz            the date's matchups page
      <covers_id>/<fetched_at>.html.gz         one game's line-history answer

Insert-only by construction: each fetch writes a new key named by its UTC
fetch time, so a re-fetch of a game (a book that was missing) never replaces
the earlier answer. The HTML is stored as served, gzipped (about a tenth of
its size).
"""

from __future__ import annotations

import gzip
from datetime import UTC, date, datetime

PREFIX = "train_data/odds_sources/covers"
RAW_PREFIX = f"{PREFIX}/raw/v1"


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def matchups_key(game_date: date, fetched_at: datetime) -> str:
    return f"{RAW_PREFIX}/{game_date.isoformat()}/matchups/{_stamp(fetched_at)}.html.gz"


def game_key(game_date: date, covers_id: int, fetched_at: datetime) -> str:
    return (
        f"{RAW_PREFIX}/{game_date.isoformat()}/{int(covers_id)}/"
        f"{_stamp(fetched_at)}.html.gz"
    )


def make_archive():
    """A :class:`CoversArchive` on the configured bucket and credentials."""
    from nba_ou.config.settings import SETTINGS
    from nba_ou.utils.s3_models import make_s3_client

    if not SETTINGS.s3_bucket:
        raise ValueError("S3 bucket is not configured ([S3] BUCKET)")
    client = make_s3_client(
        profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region
    )
    return CoversArchive(client, SETTINGS.s3_bucket)


class CoversArchive:
    def __init__(self, s3_client, bucket: str) -> None:
        self.s3 = s3_client
        self.bucket = bucket

    def put(self, key: str, html: str) -> None:
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=gzip.compress(html.encode("utf-8")),
            ContentType="text/html; charset=utf-8",
            ContentEncoding="gzip",
        )

    def get(self, key: str) -> str:
        body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return gzip.decompress(body).decode("utf-8")
