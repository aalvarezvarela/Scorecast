"""Paced NBA Stats requests with a circuit breaker for persistent throttling."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

from nba_api.stats.endpoints import GameRotation, PlayByPlayV3
from nba_api.stats.library.http import NBAStatsHTTP
from requests.exceptions import ConnectionError, ReadTimeout

from nba_ou.fetch_data.nba_api_session import reset_nba_http_session

LOGGER = logging.getLogger(__name__)


class CircuitOpen(RuntimeError):
    """The endpoint failed repeatedly; resume the backfill later."""


class EmptyResponse(ValueError):
    """The endpoint returned no usable team/player events."""


class GameUnavailable(EmptyResponse):
    """The NBA cannot serve this game at all, however often we ask."""


class Pacer:
    def __init__(
        self,
        interval: float = 3.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = interval
        self.clock = clock
        self.sleep = sleep
        self.last_call: float | None = None

    def wait(self) -> None:
        if self.last_call is not None:
            delay = self.interval - (self.clock() - self.last_call)
            if delay > 0:
                self.sleep(delay)
        self.last_call = self.clock()


# Shared by both endpoints, including across successive client instances.
PACER = Pacer()


def _nba_request(endpoint_class):
    """Send the request the endpoint class would send, without parsing it.

    Constructing the class outright parses the body inside ``__init__``, so a
    server error surfaces as a bare JSONDecodeError with the status code
    already thrown away. Deferring the request keeps the class as the source of
    truth for the endpoint name and its parameters while handing us the raw
    response, status code included.
    """

    def call(*, game_id: str, timeout: float):
        deferred = endpoint_class(game_id=game_id, timeout=timeout, get_request=False)
        return NBAStatsHTTP().send_api_request(
            endpoint=deferred.endpoint,
            parameters=deferred.parameters,
            proxy=deferred.proxy,
            headers=deferred.headers,
            timeout=timeout,
        )

    return call


ENDPOINTS = {
    "gamerotation": _nba_request(GameRotation),
    "playbyplayv3": _nba_request(PlayByPlayV3),
}


def _status_code(response) -> int | None:
    """NBAResponse keeps the status code private and exposes no getter."""
    return getattr(response, "_status_code", None)


def _nonempty(payload: dict, endpoint: str) -> bool:
    if endpoint == "playbyplayv3":
        return bool(payload.get("game", {}).get("actions"))
    sets = payload.get("resultSets", payload.get("resultSet", []))
    if isinstance(sets, dict):
        if "name" in sets:
            sets = [sets]
        else:
            sets = [dict(value, name=name) for name, value in sets.items()]
    if not isinstance(sets, list):
        return False
    names = {str(item.get("name", "")).lower(): item for item in sets}
    required = (
        ("hometeam", "awayteam")
        if endpoint == "gamerotation"
        else ("playbyplay",)
    )
    return all(name in names and bool(names[name].get("rowSet")) for name in required)


class LineupClient:
    def __init__(
        self,
        *,
        pacer: Pacer = PACER,
        sleep: Callable[[float], None] = time.sleep,
        reset: Callable[[], None] = reset_nba_http_session,
        endpoints: dict | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.pacer = pacer
        self.sleep = sleep
        self.reset = reset
        self.endpoints = endpoints or ENDPOINTS
        # Every hung socket costs a full timeout before the session is reset,
        # so this is the main knob on how fast a flaky night recovers.
        self.timeout = timeout
        self.consecutive_blocks = 0

    def fetch(self, endpoint: str, game_id: str) -> bytes:
        """Return untouched API JSON bytes, or raise after bounded retries."""
        if endpoint not in self.endpoints:
            raise ValueError(f"Unknown endpoint: {endpoint}")
        last_error: Exception | None = None
        for block in range(4):
            for attempt in range(2):
                self.pacer.wait()
                try:
                    response = self.endpoints[endpoint](
                        game_id=game_id, timeout=self.timeout
                    )
                    status = _status_code(response)
                    # A 5xx with an empty body is the NBA failing on this game,
                    # not a throttle: neighbouring game IDs answer 200 in 0.2 s
                    # while this one returns 500 on every attempt. Treating it
                    # as a block costs 570 s per game, and four such games in a
                    # row would open the circuit and end the run.
                    if status is not None and 500 <= status < 600:
                        raise GameUnavailable(
                            f"{endpoint} returned HTTP {status} for {game_id}"
                        )
                    raw = response.get_json()
                    payload = json.loads(raw)
                    if not _nonempty(payload, endpoint):
                        raise EmptyResponse(f"{endpoint} returned no rows for {game_id}")
                    self.consecutive_blocks = 0
                    return raw.encode("utf-8") if isinstance(raw, str) else raw
                except EmptyResponse:
                    raise
                except (ReadTimeout, ConnectionError, json.JSONDecodeError) as exc:
                    last_error = exc
                    if attempt == 0:
                        # A stale keep-alive connection in nba_api's cached
                        # session fails exactly like a throttle, but closing the
                        # session cures it on the very next request. Reset
                        # before the cheap retry instead of after both attempts
                        # have been spent, so a dead socket costs 5 s and not
                        # the whole 90/180/300 s block ladder.
                        self.reset()
                        LOGGER.warning(
                            "%s %s failed (%s); new session, retrying in 5 s",
                            endpoint, game_id, type(exc).__name__,
                        )
                        self.sleep(5)
            self.consecutive_blocks += 1
            if self.consecutive_blocks >= 4:
                raise CircuitOpen(
                    f"NBA API blocked four consecutive attempts; resume at {game_id}"
                ) from last_error
            self.reset()
            LOGGER.warning(
                "%s %s blocked (%d/4); waiting %d s",
                endpoint, game_id, self.consecutive_blocks,
                (90, 180, 300)[block],
            )
            self.sleep((90, 180, 300)[block])
        raise AssertionError("unreachable")
