"""Fetch SBR NBA line history from the page's embedded JSON.

The rendered line-history table is drawn client-side in the *browser's* local
timezone and carries no offset. That is why the CSV-era scrape silently
depended on the machine it ran from, and why ``Europe/Madrid`` had to be
recovered after the fact from DST steps (``docs/odds/line_history_phase0_findings.md``).

The same page ships a ``__NEXT_DATA__`` payload whose ``oddsDate`` values are
explicit UTC, so this module reads that instead and the timezone question never
arises again -- the result is identical whether it runs from Madrid, a UTC CI
runner, or anywhere else.

Two further consequences of using the payload rather than the DOM:

* It is server-rendered, so a plain HTTP GET is enough. No browser, no cookie
  banner, and no clicking through every book/market combination -- one request
  returns all sportsbooks across totals, spread and moneyline.
* It carries the game's own ``startDate``, so ``minutes_to_tip`` is computed
  here, against the same payload the ticks came from, instead of being joined
  on from an external schedule afterwards.

Timestamps are truncated to the minute. SBR polls the books on a fixed cadence
and stamps every tick with the same seconds value (``:20`` at the time of
writing), so the seconds carry no information -- and dropping them makes a
re-scrape reproduce the keys already in the store exactly, which is what keeps
loads idempotent.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import (
    BASE_URL,
    BOOK_SLUG_ALIASES,
    SLEEP_MAX_S,
    SLEEP_MIN_S,
    season_year_for_date,
)

MARKET_TOTALS = "totals"
MARKET_SPREAD = "point_spread"
MARKET_MONEYLINE = "money_line"
ALL_MARKETS: tuple[str, ...] = (MARKET_TOTALS, MARKET_SPREAD, MARKET_MONEYLINE)

#: Payload key -> market code. Each holds one ascending list of ticks per book.
HISTORY_KEYS: dict[str, str] = {
    "totalHistory": MARKET_TOTALS,
    "spreadHistory": MARKET_SPREAD,
    "moneyLineHistory": MARKET_MONEYLINE,
}

#: SBR lists a game under the date its tipoff falls on in *Eastern* time -- a
#: 00:30 UTC tipoff belongs to the previous day's slate. ``nba_games.game_date``
#: uses the same convention, which is what makes the two joinable.
EASTERN_TZ = ZoneInfo("America/New_York")

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json"[^>]*>(.*?)</script>',
    re.DOTALL,
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_RETRIES = 3


class LineHistoryFetchError(RuntimeError):
    """A page could not be fetched, or did not carry a usable payload."""


@dataclass(frozen=True)
class LineTick:
    """One book's quote for one market at one minute.

    ``left``/``right`` follow the column convention already in the store:
    left is the *away* side (or OVER on totals), right is the *home* side (or
    UNDER). Moneyline carries prices only, so both line fields are ``None``.
    """

    market: str
    book_slug: str
    book_name: str
    line_ts: datetime
    minutes_to_tip: int
    is_opener: bool
    left_line: float | None
    left_price: int | None
    right_line: float | None
    right_price: int | None

    @property
    def is_pregame(self) -> bool:
        return self.minutes_to_tip < 0


@dataclass(frozen=True)
class ScrapedGame:
    """A game's full line history: every book, every market, one fetch."""

    event_id: int
    game_date: date
    season_year: int
    tipoff_utc: datetime
    team_away: str
    team_home: str
    status_text: str
    away_score: int | None
    home_score: int | None
    ticks: tuple[LineTick, ...]

    @property
    def books(self) -> tuple[str, ...]:
        return tuple(sorted({tick.book_slug for tick in self.ticks}))

    def ticks_for(self, market: str) -> tuple[LineTick, ...]:
        return tuple(tick for tick in self.ticks if tick.market == market)


@dataclass(frozen=True)
class GameSummary:
    """A game as listed on a day's odds page -- enough to decide whether to fetch it."""

    event_id: int
    game_date: date
    tipoff_utc: datetime
    team_away: str
    team_home: str
    status_text: str


def build_line_history_url(event_id: int | str) -> str:
    return f"{BASE_URL}/betting-odds/nba-basketball/line-history/{event_id}/"


def build_daily_odds_url(day: date) -> str:
    return (
        f"{BASE_URL}/betting-odds/nba-basketball/totals/full-game/"
        f"?date={day.isoformat()}"
    )


def new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(_HEADERS)
    return session


def slugify_bookmaker(name: str) -> str:
    """Book display name -> the slug used by ``lh_book``.

    Kept byte-compatible with the CSV-era slugs so new rows land on the books
    already in the store ("Fanatics Sportsbook" -> ``fanatics_sportsbook``)
    rather than creating near-duplicate entries; a renamed book keeps its old
    slug (``BOOK_SLUG_ALIASES``).
    """
    out: list[str] = []
    previous_underscore = False
    for char in name.strip().lower():
        if char.isalnum():
            out.append(char)
            previous_underscore = False
        elif not previous_underscore:
            out.append("_")
            previous_underscore = True
    slug = "".join(out).strip("_")
    return BOOK_SLUG_ALIASES.get(slug, slug)


def _sleep_politely() -> None:
    time.sleep(random.uniform(SLEEP_MIN_S, SLEEP_MAX_S))


def extract_next_data(html: str) -> dict[str, Any]:
    match = _NEXT_DATA_RE.search(html)
    if not match:
        raise LineHistoryFetchError("No __NEXT_DATA__ payload in response")
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise LineHistoryFetchError(f"Malformed __NEXT_DATA__ payload: {exc}") from exc


def fetch_next_data(
    session: requests.Session,
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
) -> dict[str, Any]:
    """GET ``url`` and return its embedded payload, retrying on transient failures."""
    last_error: Exception | None = None
    for attempt in range(retries):
        if attempt:
            # Back off before retrying; SBR rate-limits bursts.
            time.sleep(2.0 * attempt)
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return extract_next_data(response.text)
        except (requests.RequestException, LineHistoryFetchError) as exc:
            last_error = exc
    raise LineHistoryFetchError(f"{url}: {last_error}")


def _parse_utc(value: Any) -> datetime | None:
    """ISO-8601 with an explicit offset -> tz-aware UTC, truncated to the minute."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # The payload always carries an offset; a naive value would reintroduce
        # exactly the ambiguity this module exists to avoid, so refuse it.
        return None
    return parsed.astimezone(UTC).replace(second=0, microsecond=0)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _team_full_name(team: dict[str, Any] | None) -> str:
    if not team:
        return ""
    return str(team.get("fullName") or team.get("name") or "").strip()


def _sides(
    entry: dict[str, Any], market: str
) -> tuple[float | None, int | None, float | None, int | None]:
    """One payload entry -> (left_line, left_price, right_line, right_price)."""
    if market == MARKET_TOTALS:
        total = _as_float(entry.get("total"))
        return (
            total,
            _as_int(entry.get("overOdds")),
            total,
            _as_int(entry.get("underOdds")),
        )
    if market == MARKET_SPREAD:
        return (
            _as_float(entry.get("awaySpread")),
            _as_int(entry.get("awayOdds")),
            _as_float(entry.get("homeSpread")),
            _as_int(entry.get("homeOdds")),
        )
    return None, _as_int(entry.get("awayOdds")), None, _as_int(entry.get("homeOdds"))


def _ticks_for_book(
    odds_view: dict[str, Any],
    *,
    book_slug: str,
    book_name: str,
    tipoff_utc: datetime,
    markets: Iterable[str],
) -> list[LineTick]:
    wanted = set(markets)
    ticks: list[LineTick] = []

    for payload_key, market in HISTORY_KEYS.items():
        if market not in wanted:
            continue

        # Collapse to one tick per minute, keeping the latest quote in that
        # minute, so truncation can never produce two rows on the same key.
        by_minute: dict[
            datetime, tuple[float | None, int | None, float | None, int | None]
        ] = {}
        for entry in odds_view.get(payload_key) or []:
            if not isinstance(entry, dict):
                continue
            line_ts = _parse_utc(entry.get("oddsDate"))
            if line_ts is None:
                continue
            by_minute[line_ts] = _sides(entry, market)

        for index, line_ts in enumerate(sorted(by_minute)):
            left_line, left_price, right_line, right_price = by_minute[line_ts]
            if (
                left_line is None
                and right_line is None
                and left_price is None
                and right_price is None
            ):
                continue
            ticks.append(
                LineTick(
                    market=market,
                    book_slug=book_slug,
                    book_name=book_name,
                    line_ts=line_ts,
                    minutes_to_tip=round((line_ts - tipoff_utc).total_seconds() / 60.0),
                    # The earliest surviving quote is the "Opener" row SBR
                    # renders as its own section above the history table.
                    is_opener=index == 0,
                    left_line=left_line,
                    left_price=left_price,
                    right_line=right_line,
                    right_price=right_price,
                )
            )

    return ticks


def parse_line_history_payload(
    payload: dict[str, Any],
    *,
    markets: Iterable[str] = ALL_MARKETS,
) -> ScrapedGame:
    """``__NEXT_DATA__`` from a line-history page -> a :class:`ScrapedGame`."""
    page_props = payload.get("props", {}).get("pageProps", {})
    model = (page_props.get("lineHistoryModel") or {}).get("lineHistory") or {}
    game_view = model.get("gameView") or {}

    event_id = _as_int(game_view.get("gameId"))
    if event_id is None:
        raise LineHistoryFetchError("Payload carries no gameId")

    tipoff_utc = _parse_utc(game_view.get("startDate"))
    if tipoff_utc is None:
        raise LineHistoryFetchError(f"Event {event_id}: no usable startDate")

    book_names = {
        str(book.get("machineName") or ""): str(book.get("name") or "")
        for book in (page_props.get("lineHistoryModel") or {}).get("sportsbooks") or []
    }

    ticks: list[LineTick] = []
    for odds_view in model.get("oddsViews") or []:
        if not isinstance(odds_view, dict):
            continue
        machine_name = str(odds_view.get("sportsbook") or "").strip()
        if not machine_name:
            continue
        book_name = book_names.get(machine_name) or machine_name
        ticks.extend(
            _ticks_for_book(
                odds_view,
                book_slug=slugify_bookmaker(book_name),
                book_name=book_name,
                tipoff_utc=tipoff_utc,
                markets=markets,
            )
        )

    game_date = tipoff_utc.astimezone(EASTERN_TZ).date()
    return ScrapedGame(
        event_id=event_id,
        game_date=game_date,
        season_year=season_year_for_date(game_date),
        tipoff_utc=tipoff_utc,
        team_away=_team_full_name(game_view.get("awayTeam")),
        team_home=_team_full_name(game_view.get("homeTeam")),
        status_text=str(game_view.get("gameStatusText") or ""),
        away_score=_as_int(game_view.get("awayTeamScore")),
        home_score=_as_int(game_view.get("homeTeamScore")),
        ticks=tuple(sorted(ticks, key=lambda t: (t.market, t.book_slug, t.line_ts))),
    )


def parse_daily_payload(payload: dict[str, Any]) -> list[GameSummary]:
    """``__NEXT_DATA__`` from a day's odds page -> the games it lists."""
    tables = payload.get("props", {}).get("pageProps", {}).get("oddsTables") or []
    summaries: list[GameSummary] = []
    seen: set[int] = set()

    for table in tables:
        for row in (table.get("oddsTableModel") or {}).get("gameRows") or []:
            game_view = (row or {}).get("gameView") or {}
            event_id = _as_int(game_view.get("gameId"))
            tipoff_utc = _parse_utc(game_view.get("startDate"))
            if event_id is None or tipoff_utc is None or event_id in seen:
                continue
            seen.add(event_id)
            summaries.append(
                GameSummary(
                    event_id=event_id,
                    game_date=tipoff_utc.astimezone(EASTERN_TZ).date(),
                    tipoff_utc=tipoff_utc,
                    team_away=_team_full_name(game_view.get("awayTeam")),
                    team_home=_team_full_name(game_view.get("homeTeam")),
                    status_text=str(game_view.get("gameStatusText") or ""),
                )
            )

    return summaries


def fetch_game_line_history(
    session: requests.Session,
    event_id: int | str,
    *,
    markets: Iterable[str] = ALL_MARKETS,
    timeout: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
) -> ScrapedGame:
    payload = fetch_next_data(
        session,
        build_line_history_url(event_id),
        timeout=timeout,
        retries=retries,
    )
    return parse_line_history_payload(payload, markets=markets)


def discover_games_for_date(
    session: requests.Session,
    day: date,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
) -> list[GameSummary]:
    """The games SBR lists for ``day``. An empty list means no slate that day."""
    payload = fetch_next_data(
        session,
        build_daily_odds_url(day),
        timeout=timeout,
        retries=retries,
    )
    return parse_daily_payload(payload)


def scrape_events(
    event_ids: Iterable[int | str],
    *,
    session: requests.Session | None = None,
    markets: Iterable[str] = ALL_MARKETS,
    on_error: str = "warn",
) -> Iterator[ScrapedGame]:
    """Yield a :class:`ScrapedGame` per event id, pausing politely between fetches.

    ``on_error='warn'`` keeps a long backfill going when one game 404s or comes
    back without a payload; ``'raise'`` is the strict mode for tests and for
    small, targeted runs.
    """
    if on_error not in {"warn", "raise"}:
        raise ValueError("on_error must be 'warn' or 'raise'")

    owned = session is None
    session = session or new_session()
    try:
        for index, event_id in enumerate(event_ids):
            if index:
                _sleep_politely()
            try:
                yield fetch_game_line_history(session, event_id, markets=markets)
            except LineHistoryFetchError as exc:
                if on_error == "raise":
                    raise
                print(f"  ! event {event_id}: {exc}")
    finally:
        if owned:
            session.close()


def scrape_dates(
    days: Iterable[date],
    *,
    session: requests.Session | None = None,
    markets: Iterable[str] = ALL_MARKETS,
    on_error: str = "warn",
) -> Iterator[ScrapedGame]:
    """Discover every game listed on each day, then scrape each one's history."""
    owned = session is None
    session = session or new_session()
    try:
        for day in days:
            try:
                summaries = discover_games_for_date(session, day)
            except LineHistoryFetchError as exc:
                if on_error == "raise":
                    raise
                print(f"  ! {day}: {exc}")
                continue

            if not summaries:
                continue
            _sleep_politely()
            yield from scrape_events(
                [summary.event_id for summary in summaries],
                session=session,
                markets=markets,
                on_error=on_error,
            )
    finally:
        if owned:
            session.close()
