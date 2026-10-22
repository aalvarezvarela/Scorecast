"""Fetch OddsTrader NBA line history into the raw S3 archive.

Per Eastern date: one request for the slate (``eventsByDateNew``) and one for
every game's current lines (``currentLines`` over all ``paid``), stored together
as the day's ``events.json.gz``. Then per game one batched POST with a
``lineHistory`` query for each (market, book) that has a current line, stored
as ``<eid>.json.gz``. A book with no current line on a game has no history on
it either, so the current lines are the index of what to ask for.

Resumable and insert-only: a day whose file exists is read back from S3 rather
than refetched, and a game whose file exists is skipped. A game is stored only
when every one of its queries answered cleanly, so a partial failure leaves a
gap that the next run fills instead of a half-written game.

Only games that tipped off more than ``SETTLED_HOURS`` ago are stored, because
a raw object is never rewritten: a game fetched before its close would keep a
truncated history for good.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from .client import (
    MARKET_CODES,
    OddsTraderClient,
    OddsTraderError,
    current_lines_query,
    events_query,
    line_history_query,
)
from .parse import BUNDLE_FORMAT, side_partids
from .store import OddsTraderStore, events_key, game_key

#: First season with line history (2006-07; nothing on 2006-01-16).
FIRST_SEASON = 2006

#: Default fetch order, most valuable first in case the API closes mid-run:
#: the seasons with all five model books, then the rest of the SBR overlap, the
#: offshore-only seasons, the 2014-18 seasons ``nba_games`` covers, and last the
#: seasons only the raw archive will hold.
PRIORITY_SEASONS: tuple[int, ...] = (
    2021, 2022, 2023, 2024, 2019, 2020, 2025, 2026,
    2018, 2017, 2016, 2015, 2014,
    2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006,
)  # fmt: skip

#: Pause before re-asking the queries of a game that errored.
RETRY_PAUSE_S = 30.0

#: A game is stored only this long after tip-off.
SETTLED_HOURS = 8


def now_utc() -> datetime:
    return datetime.now(UTC)


def season_dates(season_year: int, *, last_day: date) -> list[date]:
    """Every date ``season_year_for_date`` assigns to the season, up to ``last_day``.

    That is October through the next September, so the 2020 bubble (July to
    October 2020) is split across 2019 and 2020 but no date is left out.
    """
    days = []
    day = date(season_year, 10, 1)
    while day <= min(last_day, date(season_year + 1, 9, 30)):
        days.append(day)
        day += timedelta(days=1)
    return days


@dataclass
class BackfillResult:
    days: int = 0
    days_fetched: int = 0
    events: int = 0
    games_written: int = 0
    games_present: int = 0
    games_unsettled: int = 0
    failed_dates: list[str] = field(default_factory=list)
    failed_events: list[str] = field(default_factory=list)
    requests: int = 0

    def summary(self) -> str:
        return (
            f"{self.days} days ({self.days_fetched} fetched), {self.events} events: "
            f"{self.games_written} games written, {self.games_present} already "
            f"stored, {self.games_unsettled} not settled; {self.requests} requests; "
            f"{len(self.failed_dates)} failed dates, {len(self.failed_events)} failed games"
        )


def fetch_day(client: OddsTraderClient, day: date) -> dict[str, Any]:
    """The day's slate and the current lines of every game on it."""
    events = (client.query(events_query(day)).get("eventsByDateNew") or {}).get(
        "events"
    ) or []
    current: list[dict] = []
    if events:
        current = (
            client.query(current_lines_query([int(e["eid"]) for e in events])).get(
                "currentLines"
            )
            or []
        )
    return {
        "format": BUNDLE_FORMAT,
        "day": day.isoformat(),
        "fetched_at": now_utc().isoformat(),
        "events": events,
        "current_lines": current,
    }


def history_pairs(current_lines: Iterable[dict]) -> list[tuple[int, int]]:
    """(mtid, paid) pairs with a current line, in a stable order."""
    return sorted(
        {
            (int(line["mtid"]), int(line["paid"]))
            for line in current_lines
            if line.get("mtid") in MARKET_CODES and line.get("paid") is not None
        }
    )


def fetch_game_bundle(
    client: OddsTraderClient, event: dict[str, Any], current_lines: list[dict]
) -> dict[str, Any]:
    """Every (market, book) history of one game, in one batched request."""
    event_id = int(event["eid"])
    pairs = history_pairs(current_lines)
    partids = [side_partids(mtid, event) for mtid, _ in pairs]
    queries = [
        line_history_query(event_id, mtid, paid, sides)
        for (mtid, paid), sides in zip(pairs, partids, strict=True)
    ]
    results = client.batch(queries) if queries else []
    # Errors come in bursts (all 92 queries of one 2022 game, fine a minute
    # later), so the errored ones are asked once more before giving up.
    retry = [i for i, result in enumerate(results) if result.get("errors")]
    if retry:
        time.sleep(RETRY_PAUSE_S)
        for i, result in zip(
            retry, client.batch([queries[i] for i in retry]), strict=True
        ):
            results[i] = result
    failed = [
        pair
        for pair, result in zip(pairs, results, strict=True)
        if result.get("errors")
    ]
    if failed:
        raise OddsTraderError(f"event {event_id}: {len(failed)} queries errored")
    return {
        "format": BUNDLE_FORMAT,
        "fetched_at": now_utc().isoformat(),
        "event": event,
        "current_lines": current_lines,
        "history": [
            {"mtid": mtid, "paid": paid, "partid": list(sides), "result": result}
            for (mtid, paid), sides, result in zip(pairs, partids, results, strict=True)
        ],
    }


def _is_settled(event: dict[str, Any], now: datetime) -> bool:
    try:
        tipoff = datetime.fromtimestamp(float(event["dt"]) / 1000.0, tz=UTC)
    except (KeyError, TypeError, ValueError):
        return False
    return tipoff <= now - timedelta(hours=SETTLED_HOURS)


def backfill_days(
    days: Iterable[date],
    *,
    store: OddsTraderStore,
    client: OddsTraderClient,
    existing: set[str] | None = None,
    progress: bool = True,
) -> BackfillResult:
    """Fetch every listed day into the raw archive. Failures warn and continue."""
    existing = store.list_raw_keys() if existing is None else existing
    result = BackfillResult()
    started = time.monotonic()

    for day in days:
        result.days += 1
        key = events_key(day)
        try:
            if key in existing:
                payload = store.get_raw(key)
            else:
                payload = fetch_day(client, day)
                result.days_fetched += 1
                store.put_raw(key, payload)
                existing.add(key)
        except Exception as exc:  # noqa: BLE001 -- a backfill must not die on one date
            result.failed_dates.append(f"{day}: {exc}")
            print(f"  ! {day}: {exc}")
            continue

        events = payload.get("events") or []
        result.events += len(events)
        lines_by_event: dict[int, list[dict]] = {}
        for line in payload.get("current_lines") or []:
            lines_by_event.setdefault(int(line["eid"]), []).append(line)

        written = 0
        now = now_utc()
        for event in events:
            event_id = int(event["eid"])
            gkey = game_key(day, event_id)
            if gkey in existing:
                result.games_present += 1
                continue
            if not _is_settled(event, now):
                result.games_unsettled += 1
                continue
            try:
                bundle = fetch_game_bundle(
                    client, event, lines_by_event.get(event_id, [])
                )
                store.put_raw(gkey, bundle)
            except Exception as exc:  # noqa: BLE001
                result.failed_events.append(f"{day} {event_id}: {exc}")
                print(f"  ! {day} event {event_id}: {exc}")
                continue
            existing.add(gkey)
            written += 1
            result.games_written += 1

        result.requests = client.requests_made
        if progress and (events or result.days % 30 == 0):
            elapsed = (time.monotonic() - started) / 60.0
            print(
                f"  {day}: {len(events)} events, {written} written "
                f"(total {result.games_written} written, {result.requests} requests, "
                f"{elapsed:.0f} min)",
                flush=True,
            )

    result.requests = client.requests_made
    return result


def iter_raw_days(existing: set[str]) -> Iterator[date]:
    """Dates that have a stored ``events.json.gz``, ascending."""
    days = sorted(
        {key.rsplit("/", 2)[-2] for key in existing if key.endswith("/events.json.gz")}
    )
    for value in days:
        yield date.fromisoformat(value)
