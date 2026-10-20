"""Fetch the injury reports published since the last run, before a prediction.

Discovery (``discovery.discover``) records a candidate it finds missing for
good, which is right for the past and wrong for today: a slot probed before
the CDN publishes it would never be fetched. Here only the slots already due
are probed, starting after the latest report the manifest holds -- an earlier
slot still missing was never published -- and found ones are downloaded into
the archive (S3) like any other, so later runs and the daily ingest reuse them.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from . import manifest as mf
from . import urls as U
from .client import ArchiveClient
from .discovery import _row
from .download import download_rows
from .manifest import ManifestStore
from .storage import ManifestSync, S3Storage


def refresh_recent_reports(
    game_date: date,
    as_of: pd.Timestamp,
    *,
    storage: S3Storage,
    manifest_root: str | Path = "data/injury_reports/manifest",
    s3_prefix: str = "injury_reports",
    client: ArchiveClient | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """The season's manifest, with every report due by ``as_of`` that lists
    ``game_date``'s games (from the day before) fetched into ``storage``."""
    store = ManifestStore(Path(manifest_root))
    sync = ManifestSync(storage=storage, prefix=s3_prefix)
    season = U.season_label(game_date)
    sync.pull(season, store.path_for(season))
    manifest = store.load(season)
    found = manifest.loc[manifest["nba_available"].eq(mf.AVAILABLE_TRUE)]
    latest = (
        pd.to_datetime(found["report_datetime_utc"], utc=True).max()
        if not found.empty
        else pd.NaT
    )
    as_of = pd.Timestamp(as_of)
    due = [
        candidate
        for day in (game_date - timedelta(days=1), game_date)
        for candidate in U.candidates_for_date(day)
        if pd.Timestamp(candidate.report_datetime_utc) <= as_of
        and (pd.isna(latest) or pd.Timestamp(candidate.report_datetime_utc) > latest)
    ]
    if not due:
        return manifest
    client = client or ArchiveClient(verbose=verbose)
    rows = [
        _row(candidate, client.probe(candidate.url), 1, s3_prefix) for candidate in due
    ]
    store.upsert(season, rows)
    store.save(season)
    published = [row for row in rows if row["nba_available"] == mf.AVAILABLE_TRUE]
    if published:
        download_rows(
            rows=pd.DataFrame(published),
            store=store,
            client=client,
            storage=storage,
            verbose=verbose,
        )
    sync.push(season, store.path_for(season))
    return store.load(season)
