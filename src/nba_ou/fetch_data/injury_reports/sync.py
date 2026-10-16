"""Before prediction: immutable PDFs -> S3 -> atomic Aiven injury timelines.

No local files. Aiven receipts decide which URLs are complete. Deterministic
S3 keys recover uploads whose database acknowledgement failed. Gaps are retried
individually, never skipped because a newer report was already ingested.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd
from botocore.config import Config
from nba_ou.config.settings import SETTINGS
from nba_ou.fetch_data.injury_reports.archive import urls as U
from nba_ou.fetch_data.injury_reports.archive.validation import validate
from nba_ou.fetch_data.injury_reports.get_latest_injury_report import (
    create_robust_session,
)
from nba_ou.postgre_db.config.db_config import connect_nba_db
from nba_ou.postgre_db.injury_report_aiven import documents
from nba_ou.postgre_db.injury_report_aiven import resolve as R
from nba_ou.postgre_db.injury_report_aiven.ingest import _Read
from nba_ou.postgre_db.injury_report_aiven.live_ingest import ingest_live_reports
from nba_ou.postgre_db.injury_report_aiven.parse import parse_report
from nba_ou.postgre_db.injury_report_aiven.schema import ERAS
from nba_ou.utils.s3_models import make_s3_client

log = logging.getLogger(__name__)
S3_PREFIX = "train_data/injury_reports"


@dataclass
class SyncResult:
    pending: int = 0
    ingested: int = 0
    unavailable: int = 0
    failed: int = 0


class PDFStore:
    def __init__(self, client, bucket: str):
        self.client, self.bucket = client, bucket

    def get(self, key: str) -> bytes | None:
        try:
            body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404", "NotFound"):
                return None
            raise
        try:
            return body.read()
        finally:
            if hasattr(body, "close"):
                body.close()

    def put(self, key: str, data: bytes, url: str) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType="application/pdf",
            Metadata={"source-url": url},
        )


def live_candidates(game_date, as_of) -> list[U.Candidate]:
    """Catch up yesterday and today using the existing era-aware URL schedule.

    The season landing page is not a durable index of old reports. Enumerating
    eligible publication slots also finds PDFs that disappeared from that page.
    Future slots are never fetched. Missing CDN slots remain retryable.
    """
    day = pd.Timestamp(game_date).date()
    end_day = min(day, as_of.tz_convert(U.ET).date())
    return sorted(
        [
            c
            for d in U.date_range(day - timedelta(days=1), end_day)
            for c in U.candidates_for_date(d)
            if pd.Timestamp(c.report_datetime_utc) < as_of
        ],
        key=lambda c: c.report_datetime_utc,
    )


def _read_pdf(
    document: dict, candidate: U.Candidate, *, storage, session, as_of, delay_s
):
    data = storage.get(document["s3_key"])
    downloaded = data is None
    if data is None:
        time.sleep(delay_s)  # Stay below the CDN's measured throttling threshold.
        response = session.get(document["source_url"], timeout=20)
        # The NBA CDN can return 403 for an unpublished slot. Never record it
        # as permanently missing: both publication delays and throttling retry.
        if response.status_code in (403, 404):
            return None
        response.raise_for_status()
        data = response.content
    validation = validate(
        data,
        candidate.report_datetime_et,
        max_offset_minutes=U.validation_tolerance_minutes(candidate.era),
    )
    if not validation.ok or validation.header_datetime_et is None:
        raise ValueError(f"Invalid injury PDF: {validation.status} {validation.detail}")
    published = pd.Timestamp(
        validation.header_datetime_et.replace(tzinfo=U.ET, fold=0)
    ).tz_convert("UTC")
    if published >= as_of:
        return None
    # Existing immutable objects are not uploaded again.
    if downloaded:
        storage.put(document["s3_key"], data, document["source_url"])
    return data, published


def synchronize_injury_reports(
    game_date,
    *,
    scheduled_games: pd.DataFrame | None = None,
    as_of=None,
    delay_s: float = 0.6,
) -> SyncResult:
    as_of = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        raise ValueError("Injury synchronization requires a timezone-aware as_of")
    as_of = as_of.tz_convert("UTC")
    candidates = live_candidates(game_date, as_of)
    result = SyncResult()
    if not candidates:
        return result
    by_url = {c.url: c for c in candidates}
    # Aiven remains mandatory for predictions; network/S3 failures can only
    # leave it at its previous known state, never substitute an in-memory feed.
    with connect_nba_db("aiven") as conn:
        documents.ensure_document_table(conn)
        documents.register_documents(conn, candidates, prefix=S3_PREFIX)
        pending = documents.pending_documents(
            conn, candidates[0].report_datetime_utc, as_of
        )
        pending = [row for row in pending if row["source_url"] in by_url]
        result.pending = len(pending)
        if not pending:
            return result
        storage = PDFStore(
            make_s3_client(
                profile=SETTINGS.s3_aws_profile,
                region=SETTINGS.s3_aws_region,
                config=Config(
                    connect_timeout=3,
                    read_timeout=10,
                    retries={"total_max_attempts": 2},
                ),
            ),
            SETTINGS.s3_bucket,
        )
        parsed, urls = [], []
        with create_robust_session() as session:
            for document in pending:
                url = document["source_url"]
                candidate = by_url[url]
                try:
                    fetched = _read_pdf(
                        document,
                        candidate,
                        storage=storage,
                        session=session,
                        as_of=as_of,
                        delay_s=delay_s,
                    )
                    if fetched is None:
                        result.unavailable += 1
                        documents.mark_failed(
                            conn, url, "Publication unavailable; will retry"
                        )
                        continue
                    data, published = fetched
                    documents.mark_archived(conn, url, published)
                    report = parse_report(data, published.to_pydatetime())
                    parsed.append(
                        _Read(report, ERAS[candidate.era], int(candidate.season[:4]))
                    )
                    urls.append(url)
                except Exception as exc:
                    result.failed += 1
                    documents.mark_failed(conn, url, str(exc))
                    log.warning("Injury report processing failed for %s: %s", url, exc)
        if parsed:
            try:
                seasons = sorted({p.season_year for p in parsed})
                with connect_nba_db() as source:
                    teams = R.load_team_dimension(source)
                    games = R.load_game_dimension(seasons, source)
                    rosters = R.load_rosters(source, seasons)
                if scheduled_games is not None:
                    current = scoreboard_dimension(scheduled_games)
                    games = pd.concat(
                        [games, current], ignore_index=True
                    ).drop_duplicates("game_id", keep="last")
                conn.commit()  # Explicit root transaction for the atomic writer.
                ingest_live_reports(
                    conn, parsed, urls, teams=teams, games=games, rosters=rosters
                )
                result.ingested = len(parsed)
            except Exception as exc:
                conn.rollback()
                for url in urls:
                    documents.mark_failed(conn, url, str(exc))
                result.failed += len(parsed)
                log.warning(
                    "Injury timeline update failed; retaining previous Aiven state: %s",
                    exc,
                )
    log.info(
        "Injury report sync: %s ingested, %s unavailable, %s failed",
        result.ingested,
        result.unavailable,
        result.failed,
    )
    return result


def scoreboard_dimension(scheduled_games: pd.DataFrame) -> pd.DataFrame:
    """Use the current scoreboard even if the season schedule feed is behind."""
    columns = [
        "game_id",
        "game_date",
        "season_year",
        "tipoff_utc",
        "team_home",
        "team_away",
        "tipoff_source",
    ]
    rows = []
    for row in scheduled_games.itertuples():
        code = str(row.GAMECODE).split("/")[-1]
        tip = pd.Timestamp(row.GAME_TIME)
        if len(code) != 6 or pd.isna(tip) or tip.tzinfo is None:
            continue
        day = pd.Timestamp(row.GAME_DATE_EST).date()
        rows.append(
            dict(
                game_id=str(row.GAME_ID),
                game_date=day,
                season_year=int(U.season_label(day)[:4]),
                tipoff_utc=tip.tz_convert("UTC"),
                team_home=code[3:],
                team_away=code[:3],
                tipoff_source="schedule",
            )
        )
    return pd.DataFrame(rows, columns=columns)
