"""Raw OddsTrader game bundle -> :class:`ScrapedGame`, the SBR scraper's shape.

The output is the same dataclass ``scrape_sportsbook_line_history`` produces,
so everything downstream of it -- game_id resolution, the two load-time
repairs, the SMALLINT encodings, the per-reason counters -- is reused as is
(``line_history_aiven.ingest.build_frames``).

How the payload maps onto a tick:

* ``lineHistory`` groups changes into minute buckets, but a bucket can hold
  several sub-minute updates of a side, and a line's own ``tim`` (epoch ms,
  UTC) can fall in another minute than its bucket's. So the buckets are
  ignored: every line is replayed in ``(tim, sequence)`` order onto a running
  two-sided state, and each minute keeps the state after its last update. That
  is SBR's rule (truncate to the minute, latest quote wins) and gives one row
  per (market, book, minute).
* A side that has not been quoted yet stays ``None``; a side that was quoted
  keeps its last value until it moves. A bucket that re-quotes only one side
  therefore still yields a two-sided row.
* ``left``/``right`` follow the store: OVER/UNDER on totals, AWAY/HOME on
  spread and moneyline. Spread lines are per-team handicaps (home favourite
  = negative), the same sign as SBR's ``homeSpread``. Moneyline lines are
  ``None``.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook import season_year_for_date
from nba_ou.fetch_data.odds_sportsbook.scrape_sportsbook_line_history import (
    LineTick,
    ScrapedGame,
)

from .books import book_for_paid
from .client import (
    EASTERN_TZ,
    MARKET_CODES,
    TOTALS_OVER_PARTID,
    TOTALS_UNDER_PARTID,
    OddsTraderError,
)

#: Version of the bundle layout written by ``backfill.fetch_game_bundle``.
BUNDLE_FORMAT = 1


def _utc_from_ms(value: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
    except (TypeError, ValueError, OverflowError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def american_price(line: dict[str, Any]) -> int | None:
    """The quote's American price; derived from the decimal one if missing.

    American odds live outside (-100, +100); a value inside it (0 is the usual
    one) is not a price and becomes ``None``.
    """
    price = _as_float(line.get("ap"))
    if price is None:
        decimal = _as_float(line.get("pri"))
        if decimal is None or decimal <= 1.0:
            return None
        price = (decimal - 1.0) * 100.0 if decimal >= 2.0 else -100.0 / (decimal - 1.0)
    if -100 < price < 100:
        return None
    return int(round(price))


def teams(event: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(away, home) participants of an event. Raises if it is not one of each."""
    home = [p for p in event.get("participants") or [] if p.get("ih") is True]
    away = [p for p in event.get("participants") or [] if p.get("ih") is False]
    if len(home) != 1 or len(away) != 1:
        raise OddsTraderError(
            f"event {event.get('eid')}: expected one home and one away participant"
        )
    return away[0], home[0]


def side_partids(mtid: int, event: dict[str, Any]) -> tuple[int, int]:
    """(left, right) participant ids of a market on this event."""
    if MARKET_CODES[mtid] == "totals":
        return TOTALS_OVER_PARTID, TOTALS_UNDER_PARTID
    away, home = teams(event)
    return int(away["partid"]), int(home["partid"])


def _team_name(participant: dict[str, Any]) -> str:
    source = participant.get("source") or {}
    # ``nam`` is the city form ("Cleveland", "L.A. Lakers"), which
    # TEAM_NAME_STANDARDIZATION maps; ``nn`` is the nickname.
    return str(source.get("nam") or source.get("abbr") or "").strip()


def _final_score(event: dict[str, Any], partid: int) -> int | None:
    periods = [s for s in event.get("scores") or [] if s.get("partid") == partid]
    if not periods:
        return None
    try:
        return sum(int(float(s.get("val") or 0)) for s in periods)
    except (TypeError, ValueError):
        return None


def ticks_from_history(
    buckets: Iterable[dict[str, Any]],
    *,
    market: str,
    left_partid: int,
    right_partid: int,
    book_slug: str,
    book_name: str,
    tipoff_utc: datetime,
) -> list[LineTick]:
    """One (market, book)'s ``lineHistory`` buckets -> one tick per minute."""
    lines = [
        line
        for bucket in buckets or []
        for line in (bucket or {}).get("lines") or []
        if isinstance(line, dict) and line.get("partid") in (left_partid, right_partid)
    ]
    lines.sort(
        key=lambda line: (float(line.get("tim") or 0), int(line.get("sequence") or 0))
    )

    has_line = market != "money_line"
    state: dict[int, tuple[float | None, int | None]] = {}
    by_minute: dict[
        datetime, tuple[float | None, int | None, float | None, int | None]
    ] = {}
    for line in lines:
        stamp = _utc_from_ms(line.get("tim"))
        if stamp is None:
            continue
        value = _as_float(line.get("adj")) if has_line else None
        state[int(line["partid"])] = (value, american_price(line))
        left = state.get(left_partid, (None, None))
        right = state.get(right_partid, (None, None))
        by_minute[stamp.replace(second=0, microsecond=0)] = (*left, *right)

    ticks: list[LineTick] = []
    for index, line_ts in enumerate(sorted(by_minute)):
        left_line, left_price, right_line, right_price = by_minute[line_ts]
        ticks.append(
            LineTick(
                market=market,
                book_slug=book_slug,
                book_name=book_name,
                line_ts=line_ts,
                minutes_to_tip=round((line_ts - tipoff_utc).total_seconds() / 60.0),
                is_opener=index == 0,
                left_line=left_line,
                left_price=left_price,
                right_line=right_line,
                right_price=right_price,
            )
        )
    return ticks


def parse_bundle(bundle: dict[str, Any]) -> ScrapedGame:
    """A raw game bundle -> :class:`ScrapedGame`. Malformed bundles raise."""
    if bundle.get("format") != BUNDLE_FORMAT:
        raise OddsTraderError(f"unknown bundle format {bundle.get('format')!r}")
    event = bundle.get("event") or {}
    event_id = int(event["eid"])
    tipoff_utc = _utc_from_ms(event.get("dt"))
    if tipoff_utc is None:
        raise OddsTraderError(f"event {event_id}: no tip-off time")
    away, home = teams(event)

    ticks: list[LineTick] = []
    for entry in bundle.get("history") or []:
        result = entry.get("result") or {}
        if result.get("errors"):
            raise OddsTraderError(
                f"event {event_id} mtid {entry.get('mtid')} paid {entry.get('paid')}:"
                f" stored result carries errors"
            )
        mtid = int(entry["mtid"])
        left, right = side_partids(mtid, event)
        slug, name = book_for_paid(entry["paid"])
        ticks.extend(
            ticks_from_history(
                (result.get("data") or {}).get("lineHistory") or [],
                market=MARKET_CODES[mtid],
                left_partid=left,
                right_partid=right,
                book_slug=slug,
                book_name=name,
                tipoff_utc=tipoff_utc,
            )
        )

    game_date = tipoff_utc.astimezone(EASTERN_TZ).date()
    return ScrapedGame(
        event_id=event_id,
        game_date=game_date,
        season_year=season_year_for_date(game_date),
        tipoff_utc=tipoff_utc,
        team_away=_team_name(away),
        team_home=_team_name(home),
        status_text=str(event.get("es") or ""),
        away_score=_final_score(event, int(away["partid"])),
        home_score=_final_score(event, int(home["partid"])),
        ticks=tuple(sorted(ticks, key=lambda t: (t.market, t.book_slug, t.line_ts))),
    )
