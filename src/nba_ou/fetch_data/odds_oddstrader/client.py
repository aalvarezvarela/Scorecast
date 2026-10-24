"""HTTP client for OddsTrader's GraphQL odds service.

The site itself sits behind Cloudflare, but the service it renders from does
not, and answers without auth. Three properties of it shape this module:

* It accepts a POST whose body is a *list* of queries and answers with a list
  of results (Apollo-style batching). ``lineHistory`` takes a single ``paid``
  and a single ``mtid``, so one game is ~80 queries; batching makes that one
  HTTP request instead of 80.
* Field aliases break its proxy ("Undefined response from proxy"), so a batch
  is the only way to ask the same field twice.
* Lines come back only when ``paid`` is passed explicitly.

Queries are plain strings; every argument is an int, so they are formatted in
rather than sent as variables.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests

SERVICE_URL = "https://ms.virginia.us-east-1.oddstrader.com/odds-v2/odds-v2-service"

EASTERN_TZ = ZoneInfo("America/New_York")

#: NBA league id.
NBA_LID = 5

#: OddsTrader market type id -> the market code used by the SBR store
#: (``line_history_aiven.schema.MARKETS``). All three include overtime.
MARKET_CODES: dict[int, str] = {
    402: "totals",
    401: "point_spread",
    83: "money_line",
}

#: Totals participants: OVER and UNDER. Spread and moneyline use the teams.
TOTALS_OVER_PARTID = 15143
TOTALS_UNDER_PARTID = 15144

#: ``currentLines`` is asked for every ``paid`` in this range. A book without a
#: current line on a game has no history on it either (checked on 600 absent
#: (market, paid) pairs of a 2024 game), so this is also the history index.
PAID_SCAN = tuple(range(1, 401))

#: Polite pause between requests, in seconds.
SLEEP_MIN_S = 1.0
SLEEP_MAX_S = 1.6

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_RETRIES = 4

#: Queries per POST. A game needs ~80; larger batches were not tested.
MAX_BATCH = 120

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Content-Type": "application/json",
}


class OddsTraderError(RuntimeError):
    """A request failed after its retries, or came back with GraphQL errors."""


def _ints(values: Iterable[int]) -> str:
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


def day_start_ms(day: date) -> int:
    """Midnight Eastern of ``day`` as epoch ms -- the slate's start."""
    start = datetime(day.year, day.month, day.day, tzinfo=EASTERN_TZ)
    return int(start.timestamp() * 1000)


def events_query(day: date) -> str:
    return (
        f"{{ eventsByDateNew(lid: [{NBA_LID}], startDate: {day_start_ms(day)}, "
        "hoursRange: 24, mtid: [402], showEmptyEvents: true) { events { eid dt "
        "des es seid scores { partid val pn } participants { partid ih rot "
        "source { ... on Team { tmid nam nn abbr } } } } } }"
    )


def current_lines_query(event_ids: Sequence[int]) -> str:
    return (
        f"{{ currentLines(eid: {_ints(event_ids)}, mtid: {_ints(MARKET_CODES)}, "
        f"paid: {_ints(PAID_SCAN)}) }}"
    )


def line_history_query(
    event_id: int, mtid: int, paid: int, partids: Sequence[int]
) -> str:
    return (
        f"{{ lineHistory(eid: {int(event_id)}, mtid: {int(mtid)}, "
        f"paid: {int(paid)}, partid: {_ints(partids)}) {{ tim lines }} }}"
    )


class OddsTraderClient:
    """A session that paces itself and retries with linear backoff."""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        retries: int = DEFAULT_RETRIES,
        sleep: tuple[float, float] = (SLEEP_MIN_S, SLEEP_MAX_S),
    ) -> None:
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        self.timeout = timeout
        self.retries = retries
        self.sleep = sleep
        self.requests_made = 0
        self._last_request = 0.0

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> OddsTraderClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _pace(self) -> None:
        wait = random.uniform(*self.sleep) - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)

    def _post(self, body: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            self._pace()
            try:
                response = self.session.post(
                    SERVICE_URL, data=json.dumps(body), timeout=self.timeout
                )
                self._last_request = time.monotonic()
                self.requests_made += 1
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                self._last_request = time.monotonic()
                last_error = exc
                time.sleep(5.0 * attempt)
        raise OddsTraderError(
            f"request failed after {self.retries} attempts: {last_error}"
        )

    def query(self, query: str) -> dict[str, Any]:
        """One query -> its ``data``. GraphQL errors raise."""
        result = self._post({"query": query})
        if result.get("errors"):
            raise OddsTraderError(f"GraphQL errors: {result['errors']}")
        return result.get("data") or {}

    def batch(self, queries: Sequence[str]) -> list[dict[str, Any]]:
        """Several queries in as few POSTs as ``MAX_BATCH`` allows.

        Results come back in query order, each as the raw ``{"data": ...}`` /
        ``{"errors": ...}`` object, so the caller decides what an error means.
        """
        results: list[dict[str, Any]] = []
        for start in range(0, len(queries), MAX_BATCH):
            chunk = queries[start : start + MAX_BATCH]
            answer = self._post([{"query": q} for q in chunk])
            if not isinstance(answer, list) or len(answer) != len(chunk):
                raise OddsTraderError(
                    f"batch of {len(chunk)} came back as {type(answer).__name__}"
                    f" of length {len(answer) if isinstance(answer, list) else '-'}"
                )
            results.extend(answer)
        return results
