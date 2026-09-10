"""The manifest: one row per probed candidate URL, and the only resume state.

Stored as Parquet, one file per season, so a checkpoint rewrites a bounded file
and a season can be backfilled independently.

The column that carries the correctness burden is ``nba_available``. It is
**three-valued** -- ``true`` / ``false`` / ``unknown`` -- because the CDN answers
"this file does not exist" and "you are asking too fast" with the same
``403 AccessDenied``. A bare 403 is recorded as ``unknown`` and retried later; it
may only become ``false`` when a canary confirms we were not being throttled at
the time. Collapsing that to a boolean is how real reports get silently recorded
as missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

AVAILABLE_TRUE = "true"
AVAILABLE_FALSE = "false"
AVAILABLE_UNKNOWN = "unknown"

DOWNLOAD_PENDING = "pending"
DOWNLOAD_STORED = "stored"
DOWNLOAD_FAILED = "failed"
DOWNLOAD_INVALID = "invalid"

COLUMNS: dict[str, str] = {
    "report_key": "string",
    "season": "string",
    "season_year": "Int32",
    "report_date_et": "string",
    "report_time_label": "string",
    "source_era": "string",
    "report_datetime_et": "string",
    "report_datetime_utc": "datetime64[ns, UTC]",
    "original_url": "string",
    "original_filename": "string",
    "nba_available": "string",
    "nba_http_status": "Int16",
    "content_type": "string",
    "content_length": "Int64",
    "etag": "string",
    "last_modified": "string",
    "discovery_timestamp": "datetime64[ns, UTC]",
    "discovery_attempts": "Int16",
    "download_status": "string",
    "s3_key": "string",
    "sha256": "string",
    "bytes_stored": "Int64",
    "download_timestamp": "datetime64[ns, UTC]",
    "pdf_page_count": "Int32",
    "pdf_header_datetime_et": "string",
    "validation_status": "string",
    "notes": "string",
}

#: A verdict we never re-probe. Anything else is the retry queue.
TERMINAL_AVAILABILITY = {AVAILABLE_TRUE, AVAILABLE_FALSE}


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in COLUMNS.items()})


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    for col, dtype in COLUMNS.items():
        if col not in df.columns:
            df[col] = pd.Series([pd.NA] * len(df), dtype=dtype)
        else:
            try:
                df[col] = df[col].astype(dtype)
            except (TypeError, ValueError):
                pass
    return df[list(COLUMNS)]


@dataclass
class ManifestStore:
    """Per-season Parquet manifests under ``root``."""

    root: Path
    _cache: dict[str, pd.DataFrame] = field(default_factory=dict)

    def path_for(self, season: str) -> Path:
        return self.root / f"season={season}" / "manifest.parquet"

    def load(self, season: str) -> pd.DataFrame:
        if season in self._cache:
            return self._cache[season]
        path = self.path_for(season)
        df = _coerce(pd.read_parquet(path)) if path.exists() else empty_frame()
        self._cache[season] = df
        return df

    def save(self, season: str) -> Path:
        """Atomic-ish write: temp file then replace, so an interrupt cannot
        truncate a manifest that already held good rows."""
        df = _coerce(self._cache[season].copy())
        path = self.path_for(season)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False, engine="pyarrow")
        tmp.replace(path)
        return path

    def save_all(self) -> list[Path]:
        return [self.save(s) for s in self._cache]

    def upsert(self, season: str, rows: list[dict]) -> int:
        """Merge rows on ``report_key``, **field by field**.

        Deliberately not a row replace. The download phase writes partial rows
        (a status, a checksum) and a wholesale replace would blank the columns
        discovery had already filled -- including ``nba_available``, which would
        push a settled row back onto the retry queue and re-probe URLs forever.
        Incoming non-null values win; everything else is kept.
        """
        if not rows:
            return 0
        current = self.load(season)
        incoming = _coerce(pd.DataFrame(rows))

        if not len(current):
            merged = incoming
        else:
            cur = current.set_index("report_key")
            inc = incoming.set_index("report_key")
            # Collapse duplicate keys inside this batch before aligning.
            inc = inc[~inc.index.duplicated(keep="last")]
            index = cur.index.union(inc.index)
            merged = inc.reindex(index).combine_first(cur.reindex(index))
            merged = merged.reset_index()

        merged = _coerce(merged)
        merged = merged.sort_values("report_datetime_utc", ignore_index=True)
        self._cache[season] = merged
        return len(incoming)

    def load_seasons(self, seasons: list[str]) -> pd.DataFrame:
        frames = [self.load(s) for s in seasons]
        frames = [f for f in frames if len(f)]
        return pd.concat(frames, ignore_index=True) if frames else empty_frame()

    def known_seasons(self) -> list[str]:
        return sorted(
            p.parent.name.split("=", 1)[1]
            for p in self.root.glob("season=*/manifest.parquet")
        )

    def resolved_keys(self, season: str) -> set[str]:
        """Keys with a terminal availability verdict -- never probed again."""
        df = self.load(season)
        if not len(df):
            return set()
        done = df[df.nba_available.isin(TERMINAL_AVAILABILITY)]
        return set(done.report_key.dropna())

    def pending_downloads(self, season: str) -> pd.DataFrame:
        df = self.load(season)
        if not len(df):
            return df
        return df[
            (df.nba_available == AVAILABLE_TRUE)
            & (df.download_status != DOWNLOAD_STORED)
        ]
