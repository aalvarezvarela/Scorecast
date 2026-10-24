"""OddsLogic widget answers -> :class:`ScrapedGame`, the SBR scraper's shape.

The output is the dataclass ``scrape_sportsbook_line_history`` produces, so
game_id resolution, the load-time repairs and the SMALLINT encodings are reused
as is (``line_history_aiven.ingest.build_frames``).

A history answer is a brace-delimited text table, newest change first::

    {SPREADS{{10/20,10:45:46pm,4&frac12;-112,-4&frac12;-108,"NY 57%, 40%"}...}}
    {TOTALS{{10/20,10:45:46pm,231&frac12;o-108,231&frac12;u-112,"o54%, o54%"}...}}

How a row becomes a tick:

* The date has no year. It gets the latest year that does not put it more
  than 30 days after the game, so a December row of a January game lands in
  the previous year and a row just past midnight of a game's night stays in
  the game's.
* **A run of rows sharing a timestamp keeps only its oldest row.** The widget
  records later polls under the time of the last change it saw, so the newer
  rows of such a run carry an old time over values often quoted after tip-off
  (FanDuel, PHI@NY 2026-10-20: -1.5 -132 stamped 15:35 UTC, against -4.5 at
  every other book and a 23:00 tip). Their real time is unknown, so they are
  dropped -- before the per-minute collapse, which would otherwise let them
  win their minute.
* Then SBR's rule: seconds truncated, the latest quote in a minute wins, and
  the earliest surviving tick is the opener.
* ``left``/``right`` follow the store: OVER/UNDER on totals, AWAY/HOME on
  spread and moneyline. Spread lines are per-team handicaps (home favourite
  negative), as in SBR. Moneyline lines are ``None``.

The trailing quoted field is the public-betting split. It is not decoded (see
``books.PUBLIC_FEEDS``); the raw answers are archived with it.
"""

from __future__ import annotations

import csv
import html
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import season_year_for_date
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    LineTick,
    ScrapedGame,
)

from .books import book_name, book_slug

#: History section -> store market code.
SECTION_MARKETS: dict[str, str] = {
    "SPREADS": "point_spread",
    "TOTALS": "totals",
    "MONEYLINES": "money_line",
}

#: A game is only emitted once this long after tip-off, so a game in progress
#: is never stored (the SBR ingest's ``is_finished`` gate).
SETTLED_AFTER = timedelta(hours=4)

_HEADER_RE = re.compile(r'\{\{header\d+,[^{}]*?"([^"]*)"')
_GAME_RE = re.compile(
    r"\{(\d+),,(\d+),,[^{}]*?,(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),[^{]*"
    r"\{([^,{}]*),([^,{}]*),[^,{}]*,[^,{}]*,[^,{}]*,([^,{}]*),([^,{}]*),"
)
_SECTION_RE = re.compile(r"(SPREADS|TOTALS|MONEYLINES)\{\{(.*?)\}\}", re.S)
_ROW_RE = re.compile(r"\{([^{}]*)\}")
_QUOTE_RE = re.compile(
    r"^(?P<line>[+-]?(?:\d+(?:\.\d+)?|\.\d+|pk))?(?P<side>[ou])?(?P<price>[+-]\d+|ev)?$"
)


@dataclass(frozen=True)
class OddsLogicGame:
    """A game as listed on the widget's schedule for one date."""

    game_date: date
    rotation: int
    event_id: int
    tipoff_utc: datetime
    team_away: str
    away_abbr: str
    team_home: str
    home_abbr: str


@dataclass(frozen=True)
class HistoryRow:
    line_ts: datetime
    left: str
    right: str


def _clean(text: str) -> str:
    return html.unescape(text).replace("\u00bd", ".5")


def _is_nba_header(label: str) -> bool:
    upper = label.upper()
    return (
        upper.startswith("NBA ")
        and "PRESEASON" not in upper
        and "SUMMER" not in upper
        and "SERIES PRICE" not in upper
        and "FUTURES" not in upper
    )


def parse_schedule(text: str, game_date: date) -> list[OddsLogicGame]:
    """The NBA games of one date's schedule answer, in listed order."""
    text = _clean(text)
    headers = list(_HEADER_RE.finditer(text))
    games: list[OddsLogicGame] = []
    for index, header in enumerate(headers):
        if not _is_nba_header(header.group(1)):
            continue
        end = headers[index + 1].start() if index + 1 < len(headers) else len(text)
        for match in _GAME_RE.finditer(text, header.end(), end):
            rotation, event_id, tip, away, away_abbr, home, home_abbr = match.groups()
            games.append(
                OddsLogicGame(
                    game_date=game_date,
                    rotation=int(rotation),
                    event_id=int(event_id),
                    tipoff_utc=datetime.strptime(tip, "%Y-%m-%d %H:%M:%S").replace(
                        tzinfo=UTC
                    ),
                    team_away=away.strip(),
                    away_abbr=away_abbr.strip(),
                    team_home=home.strip(),
                    home_abbr=home_abbr.strip(),
                )
            )
    return games


#: How far after its game a history row can be. Rows run up to the end of the
#: game (and a little past it); anything later belongs to the previous year.
_MAX_ROW_AFTER_GAME = timedelta(days=30)


def _row_timestamp(month_day: str, clock: str, near: date) -> datetime:
    """``"10/20"``, ``"03:35:21pm"`` -> UTC datetime: the latest year that puts
    the row no later than ``_MAX_ROW_AFTER_GAME`` after the game on ``near``."""
    month, day = (int(part) for part in month_day.split("/"))
    hours, minutes, seconds = (int(part) for part in clock[:-2].split(":"))
    hours = hours % 12 + (12 if clock[-2:].lower() == "pm" else 0)
    latest = datetime(near.year, near.month, near.day) + _MAX_ROW_AFTER_GAME
    candidates = []
    for year in (near.year - 1, near.year, near.year + 1):
        try:
            moment = datetime(year, month, day, hours, minutes, seconds)
        except ValueError:  # 02/29 outside a leap year
            continue
        if moment <= latest:
            candidates.append(moment)
    return max(candidates).replace(tzinfo=UTC)


def parse_history_sections(text: str, near: date) -> dict[str, list[HistoryRow]]:
    """Section name -> rows, newest first, as the answer lists them."""
    sections: dict[str, list[HistoryRow]] = {}
    for name, body in _SECTION_RE.findall(_clean(text)):
        rows: list[HistoryRow] = []
        for raw in _ROW_RE.findall("{" + body + "}"):
            fields = next(csv.reader([raw]))
            if len(fields) < 4:
                continue
            rows.append(
                HistoryRow(
                    line_ts=_row_timestamp(fields[0], fields[1], near),
                    left=fields[2].strip(),
                    right=fields[3].strip(),
                )
            )
        sections[name] = rows
    return sections


def drop_stale_repeats(rows: list[HistoryRow]) -> list[HistoryRow]:
    """Keep the oldest row of every run of consecutive rows sharing a timestamp.

    ``rows`` are newest first, so the oldest row of a run is its last one.
    """
    kept: list[HistoryRow] = []
    for index, row in enumerate(rows):
        following = rows[index + 1] if index + 1 < len(rows) else None
        if following is not None and following.line_ts == row.line_ts:
            continue
        kept.append(row)
    return kept


def parse_quote(text: str) -> tuple[float | None, int | None, str | None]:
    """``"231.5o-110"`` -> (231.5, -110, "o"); ``"-4.5-108"`` -> (-4.5, -108, None).

    ``pk`` is a zero line and ``ev`` an even (+100) price. An unparseable or
    empty quote gives ``(None, None, None)``.
    """
    match = _QUOTE_RE.match(text.replace(" ", ""))
    if not match or not text.strip():
        return None, None, None
    line_text, side, price_text = match.group("line", "side", "price")
    if line_text is None:
        line = None
    elif line_text.lstrip("+-") == "pk":
        line = 0.0
    else:
        line = float(line_text)
    if price_text is None:
        price = None
    elif price_text == "ev":
        price = 100
    else:
        price = int(price_text)
    return line, price, side


def _sides(
    market: str, row: HistoryRow
) -> tuple[float | None, int | None, float | None, int | None]:
    """(left_line, left_price, right_line, right_price) of one history row."""
    if market == "money_line":
        # Moneyline quotes are prices only ("+145", "-175"); the parser reads
        # a bare number as a line, so it is taken back as the price.
        left = parse_quote(row.left)
        right = parse_quote(row.right)
        left_price = left[1] if left[1] is not None else _as_price(left[0])
        right_price = right[1] if right[1] is not None else _as_price(right[0])
        return None, left_price, None, right_price

    left_line, left_price, left_side = parse_quote(row.left)
    right_line, right_price, right_side = parse_quote(row.right)
    if market == "totals" and left_side == "u" and right_side == "o":
        # Totals carry their side; never trust the column order alone.
        return right_line, right_price, left_line, left_price
    return left_line, left_price, right_line, right_price


def _as_price(value: float | None) -> int | None:
    if value is None or -100 < value < 100:
        return None
    return int(value)


def _row_time(row: HistoryRow) -> datetime:
    return row.line_ts


def ticks_from_rows(
    rows: Iterable[HistoryRow],
    *,
    market: str,
    book_id: int,
    tipoff_utc: datetime,
) -> list[LineTick]:
    """One (market, book)'s rows -> one tick per minute, oldest first."""
    by_minute: dict[datetime, tuple] = {}
    # Walking the rows in time order lets the latest quote of a minute
    # overwrite the earlier ones. The listing is only roughly newest first:
    # about 3% of consecutive rows are swapped, usually a second apart, so the
    # rows are sorted by their full timestamp rather than reversed.
    for row in sorted(reversed(drop_stale_repeats(list(rows))), key=_row_time):
        minute = row.line_ts.replace(second=0, microsecond=0)
        by_minute[minute] = _sides(market, row)

    ticks: list[LineTick] = []
    for line_ts in sorted(by_minute):
        left_line, left_price, right_line, right_price = by_minute[line_ts]
        if all(v is None for v in (left_line, left_price, right_line, right_price)):
            continue
        ticks.append(
            LineTick(
                market=market,
                book_slug=book_slug(book_id),
                book_name=book_name(book_id),
                line_ts=line_ts,
                minutes_to_tip=round((line_ts - tipoff_utc).total_seconds() / 60.0),
                is_opener=not ticks,
                left_line=left_line,
                left_price=left_price,
                right_line=right_line,
                right_price=right_price,
            )
        )
    return ticks


def book_ticks(
    answers: Iterable[str],
    *,
    book_id: int,
    game_date: date,
    tipoff_utc: datetime,
) -> list[LineTick]:
    """A book's answers for one game (``t=SPREAD`` and ``t=ML``) -> its ticks.

    TOTALS comes back in both answers; the copies are the same history, so the
    longer one is used rather than merging two runs of the stale-row filter.
    """
    sections: dict[str, list[HistoryRow]] = {}
    for answer in answers:
        for name, rows in parse_history_sections(answer, game_date).items():
            if len(rows) > len(sections.get(name, [])):
                sections[name] = rows
    ticks: list[LineTick] = []
    for name, rows in sections.items():
        ticks.extend(
            ticks_from_rows(
                rows,
                market=SECTION_MARKETS[name],
                book_id=book_id,
                tipoff_utc=tipoff_utc,
            )
        )
    return ticks


def scraped_game(
    game: OddsLogicGame,
    *,
    answers: dict[int, list[str]],
    tipoff_utc: datetime,
) -> ScrapedGame:
    """Every book's answers for one game -> a :class:`ScrapedGame`.

    ``tipoff_utc`` is the schedule feed's, not the widget's: the widget lists
    games about ten minutes after the scheduled tip, which would shift every
    ``mins_to_tip``. ``status_text`` is ``"Final"`` because callers only build
    settled games (``is_settled``).
    """
    ticks: list[LineTick] = []
    for book_id, book_answers in sorted(answers.items()):
        ticks.extend(
            book_ticks(
                book_answers,
                book_id=book_id,
                game_date=game.game_date,
                tipoff_utc=tipoff_utc,
            )
        )
    return ScrapedGame(
        event_id=game.event_id,
        game_date=game.game_date,
        season_year=season_year_for_date(game.game_date),
        tipoff_utc=tipoff_utc,
        team_away=game.team_away,
        team_home=game.team_home,
        status_text="Final",
        away_score=None,
        home_score=None,
        ticks=tuple(ticks),
    )


def is_settled(tipoff_utc: datetime, now: datetime) -> bool:
    return now - tipoff_utc >= SETTLED_AFTER
