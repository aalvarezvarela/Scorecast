"""The raw fetcher paces both endpoints and stops on persistent throttling."""

import json

import pytest
from nba_ou.fetch_data.nba_lineups.client import (
    CircuitOpen,
    EmptyResponse,
    LineupClient,
    Pacer,
)
from requests.exceptions import ReadTimeout


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.now += duration


class Response:
    def __init__(self, payload):
        self.payload = payload

    def get_json(self):
        return json.dumps(self.payload)


ROTATION = {"resultSets": [
    {"name": "HomeTeam", "rowSet": [[1]]},
    {"name": "AwayTeam", "rowSet": [[2]]},
]}
PBP = {"game": {"actions": [{"actionNumber": 1}]}}


def test_pacer_shared_between_endpoints():
    fake = FakeTime()
    calls = []

    def endpoint(payload):
        def call(**kwargs):
            calls.append(fake.now)
            return Response(payload)
        return call

    client = LineupClient(
        pacer=Pacer(clock=fake.clock, sleep=fake.sleep),
        sleep=fake.sleep,
        endpoints={"gamerotation": endpoint(ROTATION), "playbyplayv3": endpoint(PBP)},
    )
    client.fetch("gamerotation", "0022400001")
    client.fetch("playbyplayv3", "0022400001")
    assert calls == [0, 3]


def test_retry_then_backoff_then_circuit_open():
    fake = FakeTime()
    resets = []
    attempts = []

    def fail(**kwargs):
        attempts.append(fake.now)
        raise ReadTimeout()

    client = LineupClient(
        pacer=Pacer(clock=fake.clock, sleep=fake.sleep),
        sleep=fake.sleep,
        reset=lambda: resets.append(True),
        endpoints={"gamerotation": fail},
    )
    with pytest.raises(CircuitOpen):
        client.fetch("gamerotation", "0022400001")
    assert len(attempts) == 8
    assert len(resets) == 3
    assert fake.sleeps.count(5) == 4
    assert all(delay in fake.sleeps for delay in (90, 180, 300))


def test_empty_response_is_not_a_success():
    client = LineupClient(
        pacer=Pacer(interval=0),
        endpoints={"playbyplayv3": lambda **kw: Response({"game": {"actions": []}})},
    )
    with pytest.raises(EmptyResponse):
        client.fetch("playbyplayv3", "0022400001")
