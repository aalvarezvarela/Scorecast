"""Read-only MoviOdds REST client; keep every original page for the archive."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime

import pandas as pd
import requests

BASE_URL = "https://moviodds.com"
MAIN_GROUP = "1453"
HISTORY_LIMIT = 50_000


class MoviOddsError(RuntimeError):
    """A request or a complete historical download could not be obtained."""


@dataclass(frozen=True)
class History:
    points: list[dict]
    fixture_ids: tuple[str, ...]
    pages: list[dict]


def utc_timestamp(value) -> pd.Timestamp:
    """Refuse naive/malformed source times instead of assuming the host's zone."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"Not an aware timestamp: {value!r}")
    return stamp.tz_convert("UTC")


class MoviOddsClient:
    def __init__(self, *, timeout: float = 30, retries: int = 3, interval: float = 1):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "NBA-line-history/1.0"})
        self.timeout, self.retries, self.interval = timeout, retries, interval
        self.requests_made = 0
        self.records: list[dict] = []
        self._last_start = float("-inf")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.session.close()

    def _retry_after(self, value: str | None) -> float:
        if not value:
            return 0
        try:
            return max(0, float(value))
        except ValueError:
            try:
                return max(
                    0,
                    (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds(),
                )
            except (ValueError, TypeError):
                return 0

    def get_json(self, path: str, params: dict) -> dict:
        error = None
        for attempt in range(self.retries):
            time.sleep(max(0, self.interval - (time.monotonic() - self._last_start)))
            self._last_start = time.monotonic()
            self.requests_made += 1
            wait = 0
            try:
                response = self.session.get(
                    BASE_URL + path, params=params, timeout=self.timeout
                )
                wait = self._retry_after(response.headers.get("Retry-After"))
                try:
                    payload = response.json()
                except ValueError:
                    payload = {"unparsed_body": response.text}
                self.records.append(
                    {
                        "path": path,
                        "params": dict(params),
                        "status": response.status_code,
                        "fetched_at": datetime.now(UTC).isoformat(),
                        "response": payload,
                    }
                )
                if 400 <= response.status_code < 500 and response.status_code not in (
                    408,
                    429,
                ):
                    raise MoviOddsError(f"{path}: HTTP {response.status_code}")
                response.raise_for_status()
                if (
                    not isinstance(payload, dict)
                    or "unparsed_body" in payload
                    or payload.get("error")
                ):
                    raise ValueError("Expected a JSON object without an API error")
                return payload
            except (requests.RequestException, ValueError) as exc:
                error = exc
                if attempt + 1 < self.retries:
                    time.sleep(max(wait, 5 * (attempt + 1)))
        raise MoviOddsError(f"{path} failed after {self.retries} attempts: {error}")

    def results(self, day: date) -> dict:
        payload = self.get_json(
            "/results", {"sport": "basketball", "date": day.isoformat()}
        )
        if not isinstance(payload.get("results"), list):
            raise MoviOddsError("Malformed results response")
        return payload

    def history(self, fixture_id: str) -> History:
        """Page backwards inclusively so timestamp-boundary quotes are not lost.

        The API keeps the NEWEST rows when capped. Re-fetch the boundary time,
        deduplicate exact overlaps, and refuse a non-progressing boundary (e.g.
        more than 50k points at one timestamp) rather than store a partial game.
        """
        params = {"fi": str(fixture_id), "group": MAIN_GROUP, "limit": HISTORY_LIMIT}
        pages, points, seen = [], [], set()
        ids = {str(fixture_id)}
        previous = None
        for _ in range(1000):
            payload = self.get_json("/history", params)
            pages.append(payload)
            rows = payload.get("points")
            if not isinstance(rows, list) or not isinstance(
                payload.get("truncated"), bool
            ):
                raise MoviOddsError(
                    "Malformed history response or missing truncation flag"
                )
            aliases = payload.get("fixture_ids", [str(fixture_id)])
            if not isinstance(aliases, list):
                raise MoviOddsError("Malformed fixture aliases")
            ids.update(str(i) for i in aliases)
            try:
                stamps = [utc_timestamp(q["ts"]) for q in rows]
            except (KeyError, TypeError, ValueError) as exc:
                raise MoviOddsError(f"Invalid history timestamp: {exc}") from exc
            if previous is not None and any(t > previous for t in stamps):
                raise MoviOddsError("History endpoint ignored its upper time bound")
            for q in rows:
                key = json.dumps(q, sort_keys=True, separators=(",", ":"))
                if key not in seen:
                    seen.add(key)
                    points.append(q)
            if not payload["truncated"]:
                return History(
                    sorted(points, key=lambda q: utc_timestamp(q["ts"])),
                    tuple(sorted(ids)),
                    pages,
                )
            if not stamps or (previous is not None and min(stamps) >= previous):
                raise MoviOddsError("Truncated history made no paging progress")
            previous = min(stamps)
            params = {**params, "to": previous.isoformat()}
        raise MoviOddsError("History exceeded the pagination safety bound")
