"""HTTP client for the Covers matchups page and its line-history fragment.

Two plain GETs, no auth, server-rendered HTML:

* ``/sports/nba/matchups?selectedDate=YYYY-MM-DD`` -- that date's games, each
  an ``<article id="nba-<covers id>">`` naming its two teams.
* ``/sport/MatchupOdds/LineHistoryBrick?gameId=<covers id>&...`` -- the "Line
  Movement" modal of one game: every book's full history of moneyline, spread
  and total in one answer (about 1.8 MB). The books depend on
  ``countryCode``/``stateProvince``; any US state gives the same nine. The
  lower-case path answers with a 302 to this one, so this one is used directly.

A third, on ``contests.covers.com``, is read by the prediction's live fallback
and stored by the daily update (``lh_consensus``):
``/consensus/matchupconsensusdetails?gameId=<covers id>``, the "Covers
Consensus" pick counts of one game, which is what SBR's consensus pick
percentages are (``parse.parse_consensus``).
"""

from __future__ import annotations

import random
import time
from datetime import date
from urllib.parse import urlencode

import requests

BASE_URL = "https://www.covers.com"

#: The region whose books the line history lists.
COUNTRY_CODE = "us"
STATE_PROVINCE = "nj"

#: Polite pause between requests, in seconds.
SLEEP_MIN_S = 1.2
SLEEP_MAX_S = 2.0

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_RETRIES = 3

#: Every line-history answer carries these tabs, empty or not.
HISTORY_MARKER = 'id="lh-tab-total"'

CONSENSUS_URL = "https://contests.covers.com/consensus/matchupconsensusdetails"
#: Every consensus answer carries this section, with or without picks.
CONSENSUS_MARKER = 'id="consensus_analysis_content"'

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
}


class CoversError(RuntimeError):
    """A request failed after its retries."""


def matchups_url(day: date) -> str:
    return (
        f"{BASE_URL}/sports/nba/matchups?{urlencode({'selectedDate': day.isoformat()})}"
    )


def history_url(covers_id: int) -> str:
    query = urlencode(
        {
            "gameId": int(covers_id),
            "location": "odds",
            "countryCode": COUNTRY_CODE,
            "stateProvince": STATE_PROVINCE,
            "oddsFormat": "american",
            # Every tab comes back whatever this says; it only picks the one shown.
            "betType": "total",
        }
    )
    return f"{BASE_URL}/sport/MatchupOdds/LineHistoryBrick?{query}"


def consensus_url(covers_id: int) -> str:
    return f"{CONSENSUS_URL}?{urlencode({'gameId': int(covers_id)})}"


class CoversClient:
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

    def __enter__(self) -> CoversClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _pace(self) -> None:
        wait = random.uniform(*self.sleep) - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)

    def get_text(self, url: str) -> str:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            self._pace()
            try:
                response = self.session.get(url, timeout=self.timeout)
                self._last_request = time.monotonic()
                self.requests_made += 1
                response.raise_for_status()
                return response.text
            except requests.RequestException as exc:
                self._last_request = time.monotonic()
                last_error = exc
                time.sleep(5.0 * attempt)
        raise CoversError(f"GET failed after {self.retries} attempts: {last_error}")

    def matchups(self, day: date) -> str:
        return self.get_text(matchups_url(day))

    def line_history(self, covers_id: int) -> str:
        """One game's line-history answer. An answer without the history tabs
        (an error page served with 200) is a failed request."""
        text = self.get_text(history_url(covers_id))
        if HISTORY_MARKER not in text:
            raise CoversError(f"no line-history tabs in the answer for {covers_id}")
        return text

    def consensus(self, covers_id: int) -> str:
        """One game's Covers Consensus page (pick counts per side)."""
        text = self.get_text(consensus_url(covers_id))
        if CONSENSUS_MARKER not in text:
            raise CoversError(f"no consensus section in the answer for {covers_id}")
        return text
