"""Async HTTP client for Yahoo Sports pages.

Plain GETs, no browser: every page Yahoo serves holds its data in the
server-rendered payload (``parse.flight_payload``). Yahoo answers 200 to some
requests with a page whose server render failed, and now and then 404 to a
page that exists, so a request is only done when its page parses; anything
else is retried with a growing pause.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from typing import TypeVar

import httpx
from nba_ou.fetch_data.odds_yahoo.parse import flight_payload, render_failed

BASE_URL = "https://sports.yahoo.com"

#: Requests in flight at once. Bursts of 30-40 were answered, with a few
#: failed renders that a retry fixes; the pages are large (about 1.3 MB) and
#: a day is ten-odd games, so there is no need to push.
DEFAULT_CONCURRENCY = 4
DEFAULT_RETRIES = 4
DEFAULT_TIMEOUT_S = 30.0

#: Pause before each request, in seconds, and the base of the retry backoff.
SLEEP_MIN_S = 0.3
SLEEP_MAX_S = 0.8
BACKOFF_S = 3.0

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

T = TypeVar("T")


class YahooError(RuntimeError):
    """A page could not be read after its retries."""


class YahooClient:
    """An ``httpx.AsyncClient`` that bounds concurrency and retries pages.

    Use as ``async with YahooClient() as client``.
    """

    def __init__(
        self,
        *,
        concurrency: int = DEFAULT_CONCURRENCY,
        retries: int = DEFAULT_RETRIES,
        timeout: float = DEFAULT_TIMEOUT_S,
        sleep: tuple[float, float] = (SLEEP_MIN_S, SLEEP_MAX_S),
        backoff: float = BACKOFF_S,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._semaphore = asyncio.Semaphore(concurrency)
        self.retries = retries
        self.sleep = sleep
        self.backoff = backoff
        self.requests_made = 0
        self._http = httpx.AsyncClient(
            base_url=BASE_URL,
            headers=_HEADERS,
            timeout=timeout,
            follow_redirects=True,
            transport=transport,
        )

    async def __aenter__(self) -> YahooClient:
        return self

    async def __aexit__(self, *exc) -> None:
        await self._http.aclose()

    async def fetch(
        self,
        path: str,
        params: dict[str, str] | None,
        parse: Callable[[str, str], T | None],
    ) -> T:
        """GET ``path`` and return ``parse(html, payload)``.

        A transport error, an answer other than 200, a failed server render
        or a ``parse`` that returns None is a failed attempt.
        """
        problem = "no attempt made"
        for attempt in range(1, self.retries + 1):
            async with self._semaphore:
                await asyncio.sleep(random.uniform(*self.sleep))
                try:
                    response = await self._http.get(path, params=params)
                except httpx.HTTPError as exc:
                    problem = repr(exc)
                else:
                    self.requests_made += 1
                    problem = self._read(response, parse)
                    if not isinstance(problem, str):
                        return problem[0]
            if attempt < self.retries:
                await asyncio.sleep(self.backoff * attempt)
        raise YahooError(f"{path} failed after {self.retries} attempts: {problem}")

    @staticmethod
    def _read(
        response: httpx.Response, parse: Callable[[str, str], T | None]
    ) -> tuple[T] | str:
        """``(value,)`` for a page that parses, else why it does not."""
        if response.status_code != 200:
            return f"HTTP {response.status_code}"
        html = response.text
        payload = flight_payload(html)
        if render_failed(payload):
            return "the server render failed"
        value = parse(html, payload)
        if value is None:
            return "the page lacks the expected data"
        return (value,)
