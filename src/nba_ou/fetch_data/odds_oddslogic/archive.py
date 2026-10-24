"""Raw OddsLogic answers in S3, so the parser can be redone without the widget.

::

    train_data/odds_sources/oddslogic/raw/v1/<YYYY-MM-DD>/
      schedule/<fetched_at>.json.gz            the date's schedule answer
      <event_id>/<fetched_at>.json.gz          one game: every answer fetched

Insert-only by construction: each fetch writes a new key named by its UTC
fetch time, so a re-fetch of a game (a book that was missing) never replaces
the earlier bundle. A bundle holds the history answers per book and request
type (``None`` where the answer stayed empty).
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from .parse import OddsLogicGame

PREFIX = "train_data/odds_sources/oddslogic"
RAW_PREFIX = f"{PREFIX}/raw/v1"

#: Version of the bundle layout below.
BUNDLE_FORMAT = 1


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")


def schedule_key(game_date, fetched_at: datetime) -> str:
    return f"{RAW_PREFIX}/{game_date.isoformat()}/schedule/{_stamp(fetched_at)}.json.gz"


def game_key(game: OddsLogicGame, fetched_at: datetime) -> str:
    return (
        f"{RAW_PREFIX}/{game.game_date.isoformat()}/{game.event_id}/"
        f"{_stamp(fetched_at)}.json.gz"
    )


def game_bundle(
    game: OddsLogicGame,
    *,
    fetched_at: datetime,
    history: dict[int, dict[str, str | None]],
) -> dict[str, Any]:
    """``history``: book id -> request type -> answer text (or ``None``)."""
    game_fields = asdict(game)
    game_fields["game_date"] = game.game_date.isoformat()
    game_fields["tipoff_utc"] = game.tipoff_utc.isoformat()
    return {
        "format": BUNDLE_FORMAT,
        "fetched_at": fetched_at.astimezone(UTC).isoformat(),
        "game": game_fields,
        "history": {str(book): answers for book, answers in history.items()},
    }


def make_archive():
    """An :class:`OddsLogicArchive` on the configured bucket and credentials."""
    from nba_ou.config.settings import SETTINGS
    from nba_ou.utils.s3_models import make_s3_client

    if not SETTINGS.s3_bucket:
        raise ValueError("S3 bucket is not configured ([S3] BUCKET)")
    client = make_s3_client(
        profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region
    )
    return OddsLogicArchive(client, SETTINGS.s3_bucket)


class OddsLogicArchive:
    def __init__(self, s3_client, bucket: str) -> None:
        self.s3 = s3_client
        self.bucket = bucket

    def put(self, key: str, payload: Any) -> None:
        body = gzip.compress(json.dumps(payload, separators=(",", ":")).encode())
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            ContentEncoding="gzip",
        )

    def get(self, key: str) -> Any:
        body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return json.loads(gzip.decompress(body))
