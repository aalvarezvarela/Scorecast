"""No local storage, no repeat NBA downloads, durable retries after failures."""

from contextlib import nullcontext
from types import SimpleNamespace

import pandas as pd
import pytest
from nba_ou.fetch_data.injury_reports import sync
from nba_ou.fetch_data.injury_reports.archive import urls as U
from nba_ou.postgre_db.injury_report_aiven.parse import ParsedReport

from tests.support.fake_s3 import FakeS3Client

DAY = "2026-10-16"
AS_OF = pd.Timestamp("2026-10-16T18:30:00Z")
CANDIDATES = [
    c
    for c in U.candidates_for_date(pd.Timestamp(DAY).date())
    if c.label in {"02_00PM", "02_15PM"}
]


@pytest.fixture
def harness(monkeypatch):
    receipt = {}
    s3 = FakeS3Client()
    downloads, parses, ingests = [], [], []

    class Conn:
        def commit(self):
            pass

        def rollback(self):
            pass

    conn = Conn()
    monkeypatch.setattr(sync, "connect_nba_db", lambda *a: nullcontext(conn))
    monkeypatch.setattr(sync, "live_candidates", lambda *a: CANDIDATES)
    monkeypatch.setattr(sync, "make_s3_client", lambda **kw: s3)
    monkeypatch.setattr(sync.documents, "ensure_document_table", lambda c: None)

    def register(c, candidates, *, prefix):
        for candidate in candidates:
            receipt.setdefault(
                candidate.url,
                dict(
                    source_url=candidate.url,
                    expected_at=candidate.report_datetime_utc,
                    s3_key=U.s3_key(candidate.report_datetime_et, prefix=prefix),
                    ingested_at=None,
                    archived_at=None,
                ),
            )

    def archive(c, url, observed):
        receipt[url].update(archived_at=AS_OF, observed_at=observed)

    monkeypatch.setattr(sync.documents, "register_documents", register)
    monkeypatch.setattr(
        sync.documents,
        "pending_documents",
        lambda *a: [r.copy() for r in receipt.values() if not r["ingested_at"]],
    )
    monkeypatch.setattr(sync.documents, "mark_archived", archive)
    monkeypatch.setattr(
        sync.documents,
        "mark_failed",
        lambda c, url, error: receipt[url].update(last_error=error),
    )

    def get(url, **kwargs):
        downloads.append(url)
        return SimpleNamespace(
            status_code=200, content=url.encode(), raise_for_status=lambda: None
        )

    session = SimpleNamespace(get=get)
    monkeypatch.setattr(sync, "create_robust_session", lambda: nullcontext(session))
    by_url = {c.url: c for c in CANDIDATES}
    monkeypatch.setattr(
        sync,
        "validate",
        lambda data, *a, **kw: SimpleNamespace(
            ok=True,
            header_datetime_et=by_url[data.decode()].report_datetime_et.replace(
                tzinfo=None
            ),
        ),
    )

    def parse(data, at):
        parses.append(data)
        return ParsedReport(at, pd.DataFrame(), pd.DataFrame())

    def ingest(c, reports, urls, **kwargs):
        ingests.append(urls)
        for url in urls:
            receipt[url]["ingested_at"] = AS_OF

    monkeypatch.setattr(sync, "parse_report", parse)
    monkeypatch.setattr(sync, "ingest_live_reports", ingest)
    monkeypatch.setattr(sync.R, "load_team_dimension", lambda c: pd.DataFrame())
    monkeypatch.setattr(sync.R, "load_game_dimension", lambda *a: pd.DataFrame())
    monkeypatch.setattr(sync.R, "load_rosters", lambda *a: pd.DataFrame())
    return SimpleNamespace(
        receipt=receipt,
        s3=s3,
        downloads=downloads,
        parses=parses,
        ingests=ingests,
        session=session,
    )


def run():
    return sync.synchronize_injury_reports(DAY, as_of=AS_OF, delay_s=0)


def test_new_reports_are_archived_once_and_completed_reports_are_skipped(harness):
    result = run()
    assert result.ingested == 2
    assert len(harness.s3.store) == len(harness.parses) == 2
    assert all(key.endswith(".pdf") for key in harness.s3.store)
    assert all(
        key.startswith("train_data/injury_reports/raw/season=2026-27/date=2026-10-16/")
        for key in harness.s3.store
    )
    before = list(harness.s3.put_order)
    again = run()
    assert again.pending == 0
    assert len(harness.downloads) == len(harness.parses) == 2
    assert harness.s3.put_order == before


def test_uploaded_pdf_is_reused_when_parsing_failed(harness, monkeypatch):
    parse = sync.parse_report
    monkeypatch.setattr(
        sync,
        "parse_report",
        lambda *a: (_ for _ in ()).throw(ValueError("parser failed")),
    )
    result = run()
    assert result.failed == 2
    assert all(
        r["archived_at"] and r["ingested_at"] is None for r in harness.receipt.values()
    )
    monkeypatch.setattr(sync, "parse_report", parse)
    assert run().ingested == 2
    assert len(harness.downloads) == len(harness.s3.put_order) == 2


def test_upload_without_a_receipt_is_recovered_from_its_s3_key(harness, monkeypatch):
    archive = sync.documents.mark_archived
    monkeypatch.setattr(
        sync.documents,
        "mark_archived",
        lambda *a: (_ for _ in ()).throw(OSError("database acknowledgement failed")),
    )
    assert run().failed == 2
    assert all(r["archived_at"] is None for r in harness.receipt.values())
    monkeypatch.setattr(sync.documents, "mark_archived", archive)
    assert run().ingested == 2
    assert len(harness.downloads) == len(harness.s3.put_order) == 2


def test_a_hole_is_retried_even_when_a_newer_report_is_complete(harness, monkeypatch):
    parse = sync.parse_report
    early = CANDIDATES[0].url

    def fail_early(data, at):
        if data.decode() == early:
            raise ValueError("temporary parse error")
        return parse(data, at)

    monkeypatch.setattr(sync, "parse_report", fail_early)
    first = run()
    assert first.failed == first.ingested == 1
    assert harness.receipt[CANDIDATES[1].url]["ingested_at"]
    monkeypatch.setattr(sync, "parse_report", parse)
    assert run().ingested == 1
    assert len(harness.downloads) == 2


def test_unpublished_pdf_is_not_marked_complete_and_is_retried(harness):
    original_get = harness.session.get
    harness.session.get = lambda *a, **kw: SimpleNamespace(status_code=403)
    result = run()
    assert result.unavailable == 2
    assert not harness.s3.store
    assert all(r["ingested_at"] is None for r in harness.receipt.values())
    harness.session.get = original_get
    assert run().ingested == 2


def test_s3_upload_failure_does_not_produce_ingested_data(harness, monkeypatch):
    monkeypatch.setattr(
        harness.s3,
        "put_object",
        lambda **kw: (_ for _ in ()).throw(OSError("S3 unavailable")),
    )
    assert run().failed == 2
    assert not harness.parses
    assert not harness.ingests


def test_failed_database_update_keeps_documents_pending(harness, monkeypatch):
    original = sync.ingest_live_reports
    monkeypatch.setattr(
        sync,
        "ingest_live_reports",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("write failed")),
    )
    assert run().failed == 2
    assert all(r["ingested_at"] is None for r in harness.receipt.values())
    monkeypatch.setattr(sync, "ingest_live_reports", original)
    assert run().ingested == 2
    assert len(harness.downloads) == 2


def test_403_and_access_errors_from_our_s3_are_not_cache_misses():
    from botocore.exceptions import ClientError

    class Broken:
        def get_object(self, **kwargs):
            raise ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")

    with pytest.raises(ClientError):
        sync.PDFStore(Broken(), "test").get("missing.pdf")


def test_candidates_cover_previous_day_and_exclude_future_slots():
    candidates = sync.live_candidates(DAY, AS_OF)
    assert {c.day.isoformat() for c in candidates} == {"2026-10-15", "2026-10-16"}
    assert all(pd.Timestamp(c.report_datetime_utc) < AS_OF for c in candidates)
    assert candidates[-1].label == "02_15PM"


def test_scoreboard_can_supply_tonight_without_a_box_score():
    scheduled = pd.DataFrame(
        {
            "GAME_ID": ["0022600001"],
            "GAMECODE": ["20261016/NYKBOS"],
            "GAME_DATE_EST": [DAY],
            "GAME_TIME": [pd.Timestamp("2026-10-16T19:00:00-04:00")],
        }
    )
    games = sync.scoreboard_dimension(scheduled)
    assert games.iloc[0].game_id == "0022600001"
    assert games.iloc[0].team_home == "BOS"
    assert games.iloc[0].team_away == "NYK"
    assert games.iloc[0].tipoff_utc == pd.Timestamp("2026-10-16T23:00Z")
