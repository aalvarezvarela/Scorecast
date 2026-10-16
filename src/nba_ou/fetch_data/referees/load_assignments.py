"""Prefer live NBA crews; use the same date's S3 snapshot only on failure."""

import logging

import pandas as pd
from botocore.config import Config
from nba_ou.config.settings import SETTINGS
from nba_ou.fetch_data.referees.assignment_store import (
    AssignmentStore,
    normalize_assignments,
)
from nba_ou.fetch_data.referees.fetch_refs_data import (
    ASSIGNMENT_COLUMNS,
    fetch_nba_referee_assignments_today,
)
from nba_ou.utils.s3_models import make_s3_client

log = logging.getLogger(__name__)


def _store() -> AssignmentStore:
    if not SETTINGS.s3_bucket:
        raise ValueError("S3 bucket is not configured")
    client = make_s3_client(
        profile=SETTINGS.s3_aws_profile,
        region=SETTINGS.s3_aws_region,
        config=Config(
            connect_timeout=3, read_timeout=5, retries={"total_max_attempts": 2}
        ),
    )
    return AssignmentStore(client, SETTINGS.s3_bucket)


def load_referee_assignments(game_date: str) -> pd.DataFrame:
    game_date = pd.Timestamp(game_date).date().isoformat()
    try:
        frame = normalize_assignments(
            fetch_nba_referee_assignments_today(
                SETTINGS.nba_official_assignments_url,
                expected_assignment_date=game_date,
            )
        )
    except Exception as exc:  # A failed source must not stop either prediction path.
        log.warning(
            "Live referee assignments failed for %s: %s; trying S3", game_date, exc
        )
        try:
            cached = _store().read(game_date)
            if cached is not None:
                frame, meta = cached
                log.warning(
                    "Using S3 referee assignments for %s fetched at %s (%s)",
                    game_date,
                    meta["fetched_at_utc"],
                    meta["key"],
                )
                return frame
        except Exception as cache_exc:
            log.warning("S3 referee fallback failed for %s: %s", game_date, cache_exc)
        log.warning(
            "No usable referee snapshot for %s; continuing with NaN features", game_date
        )
        return pd.DataFrame(columns=ASSIGNMENT_COLUMNS)

    fetched_at = pd.Timestamp.now(tz="UTC")
    if not frame.empty:
        try:
            saved = _store().save(game_date, frame, fetched_at)
            log.info(
                "Live referee assignments for %s: S3 %s",
                game_date,
                "saved" if saved else "unchanged",
            )
        except Exception as exc:
            log.warning(
                "Could not save referee assignments to S3 for %s: %s", game_date, exc
            )
    return frame
