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
    # One reset before each cheap retry, plus one before each escalating wait.
    assert len(resets) == 7
    assert fake.sleeps.count(5) == 4
    assert all(delay in fake.sleeps for delay in (90, 180, 300))


def test_a_dead_session_recovers_on_the_retry_without_a_long_wait():
    """A stale keep-alive socket must cost 5 s, not the whole block ladder."""
    fake = FakeTime()
    resets = []
    calls = []

    def flaky(**kwargs):
        calls.append(len(resets))
        if not resets:
            raise ReadTimeout()
        return Response(ROTATION)

    client = LineupClient(
        pacer=Pacer(clock=fake.clock, sleep=fake.sleep),
        sleep=fake.sleep,
        reset=lambda: resets.append(True),
        endpoints={"gamerotation": flaky},
    )
    client.fetch("gamerotation", "0022400001")
    assert len(calls) == 2
    assert len(resets) == 1
    assert fake.sleeps == [5]
    assert client.consecutive_blocks == 0


def test_empty_response_is_not_a_success():
    client = LineupClient(
        pacer=Pacer(interval=0),
        endpoints={"playbyplayv3": lambda **kw: Response({"game": {"actions": []}})},
    )
    with pytest.raises(EmptyResponse):
        client.fetch("playbyplayv3", "0022400001")


class ServerError:
    """What send_api_request hands back for a 500: a status and an empty body."""

    def __init__(self, status=500):
        self._status_code = status

    def get_json(self):
        raise json.JSONDecodeError("Expecting value", "", 0)


def test_a_server_error_on_one_game_is_not_treated_as_a_block():
    from nba_ou.fetch_data.nba_lineups.client import GameUnavailable

    fake = FakeTime()
    client = LineupClient(
        pacer=Pacer(clock=fake.clock, sleep=fake.sleep),
        sleep=fake.sleep,
        reset=lambda: None,
        endpoints={"gamerotation": lambda **kwargs: ServerError()},
    )
    with pytest.raises(GameUnavailable):
        client.fetch("gamerotation", "0021800445")
    # No 90/180/300 s ladder, and nothing counted towards the circuit breaker.
    assert fake.sleeps == []
    assert client.consecutive_blocks == 0


def test_unavailable_games_never_open_the_circuit():
    from nba_ou.fetch_data.nba_lineups.client import GameUnavailable

    fake = FakeTime()
    client = LineupClient(
        pacer=Pacer(clock=fake.clock, sleep=fake.sleep),
        sleep=fake.sleep,
        reset=lambda: None,
        endpoints={"gamerotation": lambda **kwargs: ServerError()},
    )
    for game_id in ["0021800445", "0021800446", "0021800447", "0021800448"]:
        with pytest.raises(GameUnavailable):
            client.fetch("gamerotation", game_id)
    assert client.consecutive_blocks == 0


def test_a_healthy_response_still_reports_its_status():
    class Ok(Response):
        _status_code = 200

    fake = FakeTime()
    client = LineupClient(
        pacer=Pacer(clock=fake.clock, sleep=fake.sleep),
        sleep=fake.sleep,
        endpoints={"gamerotation": lambda **kwargs: Ok(ROTATION)},
    )
    assert json.loads(client.fetch("gamerotation", "0021800444")) == ROTATION
