"""S3 layout of the OddsTrader archive.

::

    train_data/odds_sources/oddstrader/
      raw/v1/<YYYY-MM-DD>/events.json.gz     the day's eventsByDateNew + currentLines
      raw/v1/<YYYY-MM-DD>/<eid>.json.gz      one game: every lineHistory answer
      line_history/                          = backups/db/<date>/line_history/
        lh_book.parquet, lh_market.parquet, lh_game.parquet,
        lh_line_<season_year>.parquet, lh_load_meta.parquet, manifest.json

``raw/`` is the source of truth and insert-only: an object that exists is
never overwritten, so the parser can be redone without the API (which may
close at any time). ``line_history/`` is derived from it and rebuilt whole.

The tables have the columns and encodings of the Aiven store
(``line_history_aiven.schema``) -- doubled-half-point SMALLINT lines, nulled
price sentinel, NOT NULL ``mins_to_tip``/``is_pregame`` -- with two additions:
``lh_book.source_book_id`` (the OddsTrader ``paid``) and ``lh_game.event_id``
holding the OddsTrader ``eid``. ``lh_load_meta`` records timezone ``UTC``,
because the source's timestamps are epoch ms.
"""

from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterable
from datetime import date
from typing import Any

import pandas as pd

from nba_ou.postgre_db.line_history_aiven.fetch import TICK_COLUMNS, _decode_ticks

PREFIX = "train_data/odds_sources/oddstrader"
RAW_PREFIX = f"{PREFIX}/raw/v1"
TABLE_PREFIX = f"{PREFIX}/line_history"
EVENTS_NAME = "events.json.gz"
MANIFEST_NAME = "manifest.json"


def make_store():
    """An :class:`OddsTraderStore` on the configured bucket and credentials."""
    from nba_ou.config.settings import SETTINGS
    from nba_ou.utils.s3_models import make_s3_client

    if not SETTINGS.s3_bucket:
        raise ValueError("S3 bucket is not configured ([S3] BUCKET)")
    client = make_s3_client(
        profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region
    )
    return OddsTraderStore(client, SETTINGS.s3_bucket)


def raw_day_prefix(day: date) -> str:
    return f"{RAW_PREFIX}/{day.isoformat()}"


def events_key(day: date) -> str:
    return f"{raw_day_prefix(day)}/{EVENTS_NAME}"


def game_key(day: date, event_id: int) -> str:
    return f"{raw_day_prefix(day)}/{int(event_id)}.json.gz"


def line_table_name(season_year: int) -> str:
    return f"lh_line_{int(season_year)}"


def _is_missing(exc: Exception) -> bool:
    code = getattr(exc, "response", {}).get("Error", {}).get("Code")
    return code in ("NoSuchKey", "404", "NotFound")


class OddsTraderStore:
    def __init__(self, s3_client, bucket: str) -> None:
        self.s3 = s3_client
        self.bucket = bucket

    # -- raw -----------------------------------------------------------------

    def list_raw_keys(self) -> set[str]:
        """Every raw object key, for resuming without one HEAD per game."""
        keys: set[str] = set()
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{RAW_PREFIX}/"):
            keys.update(item["Key"] for item in page.get("Contents") or [])
        return keys

    def exists(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if _is_missing(exc):
                return False
            raise
        return True

    def put_raw(self, key: str, payload: Any) -> bool:
        """Write gzipped JSON unless the key exists. True if written."""
        if self.exists(key):
            return False
        body = gzip.compress(json.dumps(payload, separators=(",", ":")).encode())
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            ContentEncoding="gzip",
        )
        return True

    def get_raw(self, key: str) -> Any:
        body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return json.loads(gzip.decompress(body))

    # -- tables --------------------------------------------------------------

    def put_table(self, name: str, frame: pd.DataFrame) -> tuple[str, int]:
        """Write one ``lh_*`` table as Parquet; returns (key, bytes)."""
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        body = buffer.getvalue()
        key = f"{TABLE_PREFIX}/{name}.parquet"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=body)
        return key, len(body)

    def get_table(self, name: str) -> pd.DataFrame:
        key = f"{TABLE_PREFIX}/{name}.parquet"
        body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        return pd.read_parquet(io.BytesIO(body))

    def put_manifest(self, manifest: dict) -> str:
        key = f"{TABLE_PREFIX}/{MANIFEST_NAME}"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(manifest, indent=2, default=str).encode(),
            ContentType="application/json",
        )
        return key

    def get_manifest(self) -> dict:
        key = f"{TABLE_PREFIX}/{MANIFEST_NAME}"
        return json.loads(
            self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        )


def load_ticks(
    season_years: Iterable[int],
    *,
    store: OddsTraderStore | None = None,
    books: Iterable[str] | None = None,
    pregame_only: bool = True,
    min_minutes_before_tip: int = 5,
) -> pd.DataFrame:
    """Decoded ticks shaped like ``line_history_aiven.fetch.fetch_pregame_ticks``.

    The seam for using this archive in the pipeline later: same columns, same
    decoding (lines halved, positive ``minutes_before_tip``), same blanket
    safety margin around tip-off.
    """
    store = store or make_store()
    book_dim = store.get_table("lh_book")[["book_id", "slug"]]
    market_dim = store.get_table("lh_market")[["market_id", "code"]]
    wanted = None if books is None else set(books)

    frames = []
    for season in season_years:
        rows = store.get_table(line_table_name(season))
        if pregame_only:
            rows = rows[
                rows["is_pregame"]
                & (rows["mins_to_tip"] <= -int(min_minutes_before_tip))
            ]
        rows = rows.merge(book_dim, on="book_id").merge(market_dim, on="market_id")
        if wanted is not None:
            rows = rows[rows["slug"].isin(wanted)]
        frames.append(
            rows.rename(
                columns={
                    "code": "market",
                    "slug": "book",
                    "mins_to_tip": "minutes_before_tip",
                }
            )[TICK_COLUMNS]
        )
    if not frames:
        return pd.DataFrame(columns=TICK_COLUMNS)
    ticks = pd.concat(frames, ignore_index=True).sort_values(
        ["game_id", "market", "book", "line_ts"]
    )
    return _decode_ticks(ticks)
