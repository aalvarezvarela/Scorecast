"""Tonight's injury reports: what is fetched, and in what order it is read."""

from __future__ import annotations

from datetime import date

import pandas as pd
from nba_ou.fetch_data.injury_reports.archive import live as archive_live
from nba_ou.fetch_data.injury_reports.archive import manifest as mf
from nba_ou.fetch_data.injury_reports.archive import urls as U
from nba_ou.fetch_data.injury_reports.archive.client import Probe, Verdict
from nba_ou.fetch_data.injury_reports.archive.discovery import _row
from nba_ou.postgre_db.injury_report_aiven import live
from nba_ou.postgre_db.row_order import canonical_row_order

GAME_DAY = date(2026, 4, 10)


class _Client:
    def __init__(self):
        self.probed: list[str] = []

    def probe(self, url):
        self.probed.append(url)
        return Probe(Verdict.EXISTS, status=200, content_length=10)


def test_only_due_slots_after_the_latest_report_are_probed(monkeypatch, tmp_path):
    candidates = [
        c for day in (date(2026, 4, 9), GAME_DAY) for c in U.candidates_for_date(day)
    ]
    stored = [
        c for c in candidates if c.report_datetime_et.hour < 12 and c.day == GAME_DAY
    ]
    latest = max(pd.Timestamp(c.report_datetime_utc) for c in stored)
    store = mf.ManifestStore(tmp_path)
    season = U.season_label(GAME_DAY)
    store.load(season)
    store.upsert(
        season,
        [
            _row(c, Probe(Verdict.EXISTS, status=200), 1, "injury_reports")
            for c in stored
        ],
    )
    store.save(season)
    monkeypatch.setattr(
        archive_live, "ManifestStore", lambda root: mf.ManifestStore(tmp_path)
    )
    monkeypatch.setattr(
        archive_live,
        "ManifestSync",
        lambda **_: type(
            "Sync", (), {"pull": lambda *a: False, "push": lambda *a: None}
        )(),
    )
    downloaded = {}
    monkeypatch.setattr(
        archive_live,
        "download_rows",
        lambda *, rows, **_: downloaded.setdefault(
            "urls", rows["original_url"].tolist()
        ),
    )
    as_of = latest + pd.Timedelta(hours=2)
    client = _Client()
    archive_live.refresh_recent_reports(GAME_DAY, as_of, storage=None, client=client)
    due = {
        c.url
        for c in candidates
        if latest < pd.Timestamp(c.report_datetime_utc) <= as_of
    }
    assert set(client.probed) == due
    assert len(due) > 0
    assert set(downloaded["urls"]) == due


def test_stored_and_live_reads_merge_into_one_reads_order():
    stored = pd.DataFrame(
        {"game_id": ["2", "1", "1"], "snapshot_minutes": [30, 60, 30], "v": [1, 2, 3]}
    )
    tonight = pd.DataFrame(
        {"game_id": ["3", "0"], "snapshot_minutes": [60, 30], "v": [4, 5]}
    )
    merged = live._merged(canonical_row_order(stored), canonical_row_order(tonight))
    expected = canonical_row_order(pd.concat([stored, tonight], ignore_index=True))
    pd.testing.assert_frame_equal(merged, expected)
