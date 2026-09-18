"""Paced NBA Stats requests with a circuit breaker for persistent throttling."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable

from nba_api.stats.endpoints import GameRotation, PlayByPlayV3
from requests.exceptions import ConnectionError, ReadTimeout

from nba_ou.fetch_data.nba_api_session import reset_nba_http_session

LOGGER = logging.getLogger(__name__)


class CircuitOpen(RuntimeError):
    """The endpoint failed repeatedly; resume the backfill later."""


class EmptyResponse(ValueError):
    """The endpoint returned no usable team/player events."""


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
ENDPOINTS = {"gamerotation": GameRotation, "playbyplayv3": PlayByPlayV3}


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
    ) -> None:
        self.pacer = pacer
        self.sleep = sleep
        self.reset = reset
        self.endpoints = endpoints or ENDPOINTS
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
                    response = self.endpoints[endpoint](game_id=game_id, timeout=30)
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
                        LOGGER.warning(
                            "%s %s failed (%s); retrying in 5 s",
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
