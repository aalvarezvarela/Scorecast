"""Covers HTML -> :class:`ScrapedGame`, the SBR scraper's shape.

The output is the dataclass ``scrape_sportsbook_line_history`` produces, so
game_id resolution, the load-time repairs and the SMALLINT encodings are reused
as is (``line_history_aiven.ingest.build_frames``).

The line-history answer has one tab per market (``lh-tab-moneyline``,
``lh-tab-spread``, ``lh-tab-total``) and, in each, two tables per book
(``<table data-book-id=N>``): a one-row ``OPEN`` table and the history, newest
change first. A history row is::

    <td> "Jan 15" "13:57" </td>  <td> away team / home team </td>  <td/>
    <td> one span per side and odds format </td>

How a row becomes a tick:

* Only the visible spans count (``display: inline-flex``); the hidden ones are
  the same quote in decimal and fractional odds.
* The time is Eastern, to the minute. The date label is ``"Oct 23"``, or
  ``"Today"``/``"Yesterday"`` relative to the fetch. It has no year: it gets
  the latest year that does not put it more than 30 days after the game (the
  OddsLogic rule), so a December row of a January game lands in the previous
  year.
* The ``OPEN`` table is skipped: it repeats the oldest history row with only
  a clock time.
* Then SBR's rule: the latest quote in a minute wins, and the earliest tick is
  the opener.
* ``left``/``right`` follow the store: OVER/UNDER on totals, AWAY/HOME on
  spread and moneyline. Spread lines are per-team handicaps (home favourite
  negative), as in SBR. Moneyline lines are ``None``.

``consensus_opener_ticks`` rebuilds SBR's opener column: BetMGM's opener.
"""

from __future__ import annotations

import html
import re
import string
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import season_year_for_date
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    EASTERN_TZ,
    LineTick,
    ScrapedGame,
)

from .books import (
    CONSENSUS_OPENER_ID,
    CONSENSUS_SOURCE_ID,
    COVERS_BOOKS,
    book_name,
    book_slug,
)

#: History tab -> store market code.
TAB_MARKETS: dict[str, str] = {
    "total": "totals",
    "spread": "point_spread",
    "moneyline": "money_line",
}

#: A game is only emitted once this long after tip-off, so a game in progress
#: is never stored (the SBR ingest's ``is_finished`` gate).
SETTLED_AFTER = timedelta(hours=4)

#: How far after its game a history row can be. Rows run up to the end of the
#: game (and a little past it); anything later belongs to the previous year.
_MAX_ROW_AFTER_GAME = timedelta(days=30)

_ARTICLE_RE = re.compile(r'<article id="nba-(\d+)"([^>]*)>')
_TABLE_RE = re.compile(r'<table data-book-id="(\d+)"(.*?)</table>', re.S)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S)
_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.S)
_LABEL_RE = re.compile(r"<span[^>]*>([^<]*)</span>")
#: A visible quote span. With a price it nests one span (``+5.5<span>-110</span>``),
#: so it ends at the ``</span>`` followed by the next span or the end of the cell.
_VISIBLE_RE = re.compile(
    r'<span[^>]*style="display: inline-flex;"[^>]*>(.*?)</span>\s*(?=<span|</td>)',
    re.S,
)
_LINE_PRICE_RE = re.compile(r"^\s*([^<]*)<span[^>]*>([^<]*)</span>\s*$", re.S)


@dataclass(frozen=True)
class CoversGame:
    """A game as listed on the matchups page of one date."""

    game_date: date
    covers_id: int
    team_away: str
    team_home: str


@dataclass(frozen=True)
class HistoryRow:
    line_ts: datetime
    left: tuple[float | None, int | None, str | None]
    right: tuple[float | None, int | None, str | None]


def _attribute(attrs: str, name: str) -> str | None:
    """An attribute of the ``<article>`` tag. The page leaves values with spaces
    unquoted (``data-away-team-fullname=memphis grizzlies data-...``)."""
    match = re.search(rf'{name}=(?:"([^"]*)"|(.*?))(?=\s+[a-z-]+=|\s*$)', attrs)
    if not match:
        return None
    value = match.group(1) if match.group(1) is not None else match.group(2)
    return value.strip() or None


def parse_matchups(text: str, game_date: date) -> list[CoversGame]:
    """The NBA games of one date's matchups page, in listed order."""
    games: list[CoversGame] = []
    seen: set[int] = set()
    for match in _ARTICLE_RE.finditer(text):
        covers_id = int(match.group(1))
        away = _attribute(match.group(2), "data-away-team-fullname")
        home = _attribute(match.group(2), "data-home-team-fullname")
        if covers_id in seen or not away or not home:
            continue
        seen.add(covers_id)
        games.append(
            CoversGame(
                game_date=game_date,
                covers_id=covers_id,
                # capwords, not title(): "76ers" must not become "76Ers".
                team_away=string.capwords(html.unescape(away)),
                team_home=string.capwords(html.unescape(home)),
            )
        )
    return games


def _number(text: str) -> float | None:
    text = html.unescape(text).strip().lower()
    if text in ("pk", "pick"):
        return 0.0
    if text in ("ev", "even"):
        return 100.0
    try:
        return float(text)
    except ValueError:
        return None


def parse_quote(span: str) -> tuple[float | None, int | None, str | None]:
    """One visible span -> (line, price, totals side).

    ``"+5.5<span>-110</span>"`` -> (5.5, -110, None); ``"o227.5<span>-110</span>"``
    -> (227.5, -110, "o"); ``"+180"`` (moneyline) -> (None, 180, None). ``pk`` is
    a zero line and ``ev`` an even (+100) price.
    """
    match = _LINE_PRICE_RE.match(span)
    if match is None:
        price = _number(re.sub(r"<[^>]+>", "", span))
        return None, (int(price) if price is not None else None), None
    line_text = html.unescape(match.group(1)).strip().lower()
    side = line_text[0] if line_text[:1] in ("o", "u") else None
    line = _number(line_text[1:] if side else line_text)
    price = _number(match.group(2))
    return line, (int(price) if price is not None else None), side


def _row_timestamp(
    label: str, clock: str, near: date, fetched_at: datetime
) -> datetime:
    """``"Jan 15"``/``"Today"``, ``"13:57"`` (Eastern) -> UTC datetime."""
    hours, minutes = (int(part) for part in clock.split(":"))
    lowered = label.strip().lower()
    if lowered in ("today", "yesterday"):
        day = fetched_at.astimezone(EASTERN_TZ).date()
        if lowered == "yesterday":
            day -= timedelta(days=1)
        local = datetime(day.year, day.month, day.day, hours, minutes)
    else:
        # Parsed in a leap year: strptime's default 1900 would reject "Feb 29".
        month_day = datetime.strptime(
            f"{label.strip().replace('Sept', 'Sep')} 2000", "%b %d %Y"
        )
        latest = datetime(near.year, near.month, near.day) + _MAX_ROW_AFTER_GAME
        candidates = []
        for year in (near.year - 1, near.year, near.year + 1):
            try:
                moment = datetime(year, month_day.month, month_day.day, hours, minutes)
            except ValueError:  # Feb 29 outside a leap year
                continue
            if moment <= latest:
                candidates.append(moment)
        local = max(candidates)
    return local.replace(tzinfo=EASTERN_TZ).astimezone(UTC)


def _row(cells: list[str], near: date, fetched_at: datetime) -> HistoryRow | None:
    """A history ``<tr>`` -> row, or ``None`` for the header and OPEN rows."""
    if len(cells) < 4:
        return None
    labels = [html.unescape(x).strip() for x in _LABEL_RE.findall(cells[0])]
    labels = [x for x in labels if x]
    if len(labels) < 2 or labels[0].upper() == "OPEN":
        return None
    spans = _VISIBLE_RE.findall(cells[3])
    if len(spans) < 2:
        return None
    return HistoryRow(
        line_ts=_row_timestamp(labels[0], labels[1], near, fetched_at),
        left=parse_quote(spans[0]),
        right=parse_quote(spans[1]),
    )


def _sides(
    market: str, row: HistoryRow
) -> tuple[float | None, int | None, float | None, int | None]:
    """(left_line, left_price, right_line, right_price) of one history row."""
    (left_line, left_price, left_side), (right_line, right_price, right_side) = (
        row.left,
        row.right,
    )
    if market == "money_line":
        return None, left_price, None, right_price
    if market == "totals" and left_side == "u" and right_side == "o":
        # Totals carry their side; never trust the column order alone.
        return right_line, right_price, left_line, left_price
    return left_line, left_price, right_line, right_price


def ticks_from_rows(
    rows: Iterable[HistoryRow],
    *,
    market: str,
    book_id: int,
    tipoff_utc: datetime,
) -> list[LineTick]:
    """One (market, book)'s rows, newest first -> one tick per minute, oldest first."""
    by_minute: dict[datetime, tuple] = {}
    # Oldest first, so the newest quote of a minute overwrites the others. The
    # sort is stable: rows of the same minute keep their (reversed) order.
    for row in sorted(reversed(list(rows)), key=lambda r: r.line_ts):
        by_minute[row.line_ts] = _sides(market, row)

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


def parse_line_history(
    text: str, *, near: date, fetched_at: datetime
) -> dict[tuple[str, int], list[HistoryRow]]:
    """(market, Covers book id) -> history rows, newest first, for every book
    table with at least one row (``OPEN`` tables are skipped)."""
    out: dict[tuple[str, int], list[HistoryRow]] = {}
    for tab, market in TAB_MARKETS.items():
        start = text.find(f'id="lh-tab-{tab}"')
        if start < 0:
            continue
        end = text.find('role="tabpanel"', start + 10)
        segment = text[start : end if end > 0 else len(text)]
        for book_id, body in _TABLE_RE.findall(segment):
            rows = [
                row
                for tr in _ROW_RE.findall(body)
                if (row := _row(_CELL_RE.findall(tr), near, fetched_at)) is not None
            ]
            if rows:
                out.setdefault((market, int(book_id)), []).extend(rows)
    return out


def consensus_opener_ticks(ticks: Iterable[LineTick]) -> list[LineTick]:
    """SBR's ``consensus_opener``: BetMGM's opening quote of each market.

    The opener SBR stored in ``odds_sportsbook`` is BetMGM's first quote, line
    and prices as quoted (every one of 39 games of 2024-26, all markets; 97%
    of moneyline prices). The -110 centering seen in the training data is the
    pipeline's own (``data_processing.odds.normalize_total_lines``), applied
    to every book at load time, so it is not repeated here.

    Takes any ticks and copies BetMGM's openers; without them it returns nothing.
    """
    source = book_slug(CONSENSUS_SOURCE_ID)
    return [
        LineTick(
            market=tick.market,
            book_slug=book_slug(CONSENSUS_OPENER_ID),
            book_name=book_name(CONSENSUS_OPENER_ID),
            line_ts=tick.line_ts,
            minutes_to_tip=tick.minutes_to_tip,
            is_opener=True,
            left_line=tick.left_line,
            left_price=tick.left_price,
            right_line=tick.right_line,
            right_price=tick.right_price,
        )
        for tick in ticks
        if tick.book_slug == source and tick.is_opener
    ]


@dataclass(frozen=True)
class ParsedGame:
    game: ScrapedGame
    #: Covers book ids with rows that ``COVERS_BOOKS`` does not know.
    unknown_books: frozenset[int]


def scraped_game(
    game: CoversGame,
    *,
    text: str,
    tipoff_utc: datetime,
    fetched_at: datetime,
    status_text: str = "Final",
) -> ParsedGame:
    """One game's line-history answer -> a :class:`ScrapedGame`.

    ``tipoff_utc`` is the NBA schedule feed's: Covers' finished games carry no
    tip time. ``status_text`` defaults to ``"Final"`` because the store only
    builds settled games (``is_settled``); the live fallback passes its own.
    """
    ticks: list[LineTick] = []
    unknown: set[int] = set()
    rows = parse_line_history(text, near=game.game_date, fetched_at=fetched_at)
    for (market, book_id), book_rows in sorted(rows.items()):
        if book_id not in COVERS_BOOKS or book_id == CONSENSUS_OPENER_ID:
            unknown.add(book_id)
            continue
        ticks.extend(
            ticks_from_rows(
                book_rows, market=market, book_id=book_id, tipoff_utc=tipoff_utc
            )
        )
    ticks.extend(consensus_opener_ticks(ticks))
    return ParsedGame(
        game=ScrapedGame(
            event_id=game.covers_id,
            game_date=game.game_date,
            season_year=season_year_for_date(game.game_date),
            tipoff_utc=tipoff_utc,
            team_away=game.team_away,
            team_home=game.team_home,
            status_text=status_text,
            away_score=None,
            home_score=None,
            ticks=tuple(ticks),
        ),
        unknown_books=frozenset(unknown),
    )


@dataclass(frozen=True)
class PickConsensus:
    """One game's Covers Consensus pick counts, per side.

    SBR's consensus pick percentages are these counts' shares: its game ids
    are Covers', and on past games the two agree within a point (SBR's are
    read at another moment of the same counts).
    """

    spread_away: int
    spread_home: int
    total_over: int
    total_under: int

    def sbr_consensus(self) -> dict[str, float]:
        """The ``gameView.consensus`` SBR's day page would carry."""

        def share(part: int, other: int) -> float:
            return 100.0 * part / (part + other) if part + other else 0.0

        return {
            # SBR serves no moneyline consensus: zeros, rendered empty.
            "homeMoneyLinePickPercent": 0,
            "awayMoneyLinePickPercent": 0,
            "homeSpreadPickPercent": share(self.spread_home, self.spread_away),
            "awaySpreadPickPercent": share(self.spread_away, self.spread_home),
            "overPickPercent": share(self.total_over, self.total_under),
            "underPickPercent": share(self.total_under, self.total_over),
        }


_FINAL_PICKS_RE = re.compile(
    r'finalWagersleft">\s*(\d+)\s*<.*?finalWagersRight">\s*(\d+)\s*<', re.S
)


def parse_consensus(text: str) -> PickConsensus | None:
    """The consensus page's two "Total Picks" rows: ATS (away, home), then O/U
    (over, under). Only the public consensus section is read; the experts'
    section after it is not. None when the page has not both rows."""
    start = text.find('id="consensus_analysis_content"')
    if start < 0:
        return None
    end = text.find('id="experts_analysis_content"', start)
    rows = _FINAL_PICKS_RE.findall(text[start : end if end > 0 else len(text)])
    if len(rows) < 2:
        return None
    (away, home), (over, under) = rows[:2]
    return PickConsensus(int(away), int(home), int(over), int(under))


def is_settled(tipoff_utc: datetime, now: datetime) -> bool:
    return now - tipoff_utc >= SETTLED_AFTER
