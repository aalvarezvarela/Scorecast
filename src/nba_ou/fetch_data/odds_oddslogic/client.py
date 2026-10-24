"""HTTP client for the OddsLogic free-odds widget (the odds shown on WagerTalk).

The widget renders from two plain GET endpoints that need no auth:

* ``get.php?free_odds_lite=1&data=all.txt&date=YYYY-MM-DD`` -- that date's
  schedule (every league), with each game's rotation number and event id.
* ``line_history_free_odds.php`` -- one book's full history of one game.
  ``s`` is the book id and must be a single id. ``t=SPREAD`` answers with the
  SPREADS and TOTALS histories, ``t=ML`` with MONEYLINES and TOTALS, so a
  game's three markets take two requests per book. ``z=0000`` returns times in
  UTC.

Archived dates go back to 2020-12-01 (``archive_available.php``). The
history endpoint sometimes answers "No line history found" for a game and book
it served a minute earlier, so an empty answer is retried before being taken
as a real absence.
"""

from __future__ import annotations

import random
import time
from datetime import date
from urllib.parse import urlencode

import requests

BASE_URL = "https://freeodds.oddslogic.com/ol-lite-stage/sources/php"
REFERER = "https://freeodds.oddslogic.com/ol-lite-stage/"

#: The history request types and the market sections each one answers with.
REQUEST_TYPES: dict[str, tuple[str, ...]] = {
    "SPREAD": ("SPREADS", "TOTALS"),
    "ML": ("MONEYLINES", "TOTALS"),
}

#: The body of an empty answer, real or transient.
NO_HISTORY = "No line history found"

#: Polite pause between requests, in seconds. Each answer itself takes 1-3 s.
SLEEP_MIN_S = 0.8
SLEEP_MAX_S = 1.2

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_RETRIES = 3
#: Extra attempts at an empty history answer, ``EMPTY_RETRY_PAUSE_S * n`` apart.
DEFAULT_EMPTY_RETRIES = 2
EMPTY_RETRY_PAUSE_S = 5.0

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": REFERER,
}


class OddsLogicError(RuntimeError):
    """A request failed after its retries."""


def schedule_url(day: date) -> str:
    query = urlencode({"free_odds_lite": 1, "data": "all.txt", "date": day.isoformat()})
    return f"{BASE_URL}/get.php?{query}"


def history_url(
    day: date,
    *,
    rotation: int,
    book_id: int,
    request_type: str,
    team_away: str = "",
    team_home: str = "",
    away_abbr: str = "",
    home_abbr: str = "",
) -> str:
    """One book's history of one game. The team fields are what the widget
    sends; the server keys on ``d``, ``n``, ``s`` and ``t``."""
    if request_type not in REQUEST_TYPES:
        raise ValueError(f"unknown request type {request_type!r}")
    query = urlencode(
        {
            "t": request_type,
            "d": day.isoformat(),
            "n": int(rotation),
            "s": int(book_id),
            "p": 0,
            "z": "0000",
            "c": 1,
            "sn": "",
            "an": team_away,
            "hn": team_home,
            "aabr": away_abbr,
            "habr": home_abbr,
        }
    )
    return f"{BASE_URL}/line_history_free_odds.php?{query}"


def is_empty_history(text: str) -> bool:
    return not any(f"{section}{{{{" in text for section in _ALL_SECTIONS)


_ALL_SECTIONS = ("SPREADS", "TOTALS", "MONEYLINES", "DATA")


class OddsLogicClient:
    """A session that paces itself and retries with linear backoff."""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        retries: int = DEFAULT_RETRIES,
        empty_retries: int = DEFAULT_EMPTY_RETRIES,
        sleep: tuple[float, float] = (SLEEP_MIN_S, SLEEP_MAX_S),
        empty_pause: float = EMPTY_RETRY_PAUSE_S,
    ) -> None:
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        self.timeout = timeout
        self.retries = retries
        self.empty_retries = empty_retries
        self.sleep = sleep
        self.empty_pause = empty_pause
        self.requests_made = 0
        self.empty_answers = 0
        self._last_request = 0.0

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> OddsLogicClient:
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
        raise OddsLogicError(f"GET failed after {self.retries} attempts: {last_error}")

    def schedule(self, day: date) -> str:
        return self.get_text(schedule_url(day))

    def history(self, url: str) -> str | None:
        """A history answer, or ``None`` if it stayed empty through the retries."""
        for attempt in range(self.empty_retries + 1):
            text = self.get_text(url)
            if not is_empty_history(text):
                return text
            self.empty_answers += 1
            if attempt < self.empty_retries:
                time.sleep(self.empty_pause * (attempt + 1))
        return None
