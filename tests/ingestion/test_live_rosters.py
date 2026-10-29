"""Live rosters are scoped, recoverable, and never projected into the past."""

import hashlib
import io
from types import SimpleNamespace

import pandas as pd
import pytest
from botocore.exceptions import ClientError
from nba_ou.fetch_data import live_rosters as live

TEAM = "1610612744"
DAY = "2026-10-16"
SEASON = "2026-27"
EARLY = pd.Timestamp("2026-10-16T12:00Z")
LATE = pd.Timestamp("2026-10-16T18:00Z")


class S3:
    def __init__(self):
        self.objects = {}
        self.writes = []

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {
            "Body": io.BytesIO(self.objects[Key]),
            "ETag": hashlib.sha256(self.objects[Key]).hexdigest(),
        }

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        prior = self.objects.get(Key)
        etag = None if prior is None else hashlib.sha256(prior).hexdigest()
        if (kwargs.get("IfNoneMatch") == "*" and prior is not None) or (
            "IfMatch" in kwargs and kwargs["IfMatch"] != etag
        ):
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body
        self.writes.append(Key)
        return {"ETag": hashlib.sha256(Body).hexdigest()}


def players(ids=("1",)):
    return live.normalize_players(
        pd.DataFrame(
            {
                "PLAYER_ID": ids,
                "PLAYER": [f"Player {p}" for p in ids],
                "POSITION": "G",
            }
        )
    )


def schedule(day=DAY):
    return pd.DataFrame(
        {
            "GAME_ID": ["0022600001"],
            "GAME_DATE_EST": [day],
            "HOME_TEAM_ID": [TEAM],
            "VISITOR_TEAM_ID": [TEAM],
        }
    )


def test_cache_keeps_changes_once_and_supports_as_of():
    client = S3()
    store = live.RosterStore(client, "bucket")
    store.save(DAY, SEASON, TEAM, players(), EARLY, [])
    entries = store.entries(DAY, SEASON, TEAM)
    assert len(client.writes) == 2
    store.save(DAY, SEASON, TEAM, players(), LATE, entries)
    assert len(client.writes) == 2
    store.save(DAY, SEASON, TEAM, players(("2",)), LATE, entries)
    entries = store.entries(DAY, SEASON, TEAM)
    assert len(entries) == 2
    snapshots = store.read(entries, SEASON, TEAM, LATE)
    early = live.roster_overrides(schedule(), snapshots, as_of=EARLY)
    late = live.roster_overrides(schedule(), snapshots, as_of=LATE)
    assert early[("0022600001", TEAM)] == {"1": "Player 1"}
    assert late[("0022600001", TEAM)] == {"2": "Player 2"}
    assert not live.roster_overrides(
        schedule(), snapshots, as_of=EARLY - pd.Timedelta(seconds=1)
    )
    assert "fetched_at=2026-10-16T12-00-00.000000Z/players.parquet" in client.writes[0]


def test_replay_reads_s3_without_hitting_the_endpoint(monkeypatch):
    client = S3()
    store = live.RosterStore(client, "bucket")
    store.save(DAY, SEASON, TEAM, players(), EARLY, [])
    monkeypatch.setattr(live, "_store", lambda: store)
    monkeypatch.setattr(
        live, "CommonTeamRoster", lambda **kw: pytest.fail("live lookup in a replay")
    )
    snapshot = live.load_live_rosters(schedule(), as_of=LATE)
    assert snapshot.PLAYER_ID.tolist() == ["1"]
    assert snapshot.SOURCE.tolist() == ["s3"]


def _league_offline(monkeypatch):
    monkeypatch.setattr(
        live, "PlayerIndex", lambda **kw: (_ for _ in ()).throw(TimeoutError())
    )


def player_index(teams_players, **kwargs):
    """A PlayerIndex stub: ``{team_id: [player_id, ...]}``."""
    rows = [
        {
            "PERSON_ID": int(pid),
            "PLAYER_FIRST_NAME": "Player",
            "PLAYER_LAST_NAME": pid,
            "TEAM_ID": int(team),
            "POSITION": "G",
            "ROSTER_STATUS": 1.0,
        }
        for team, pids in teams_players.items()
        for pid in pids
    ]
    frame = pd.DataFrame(rows)
    return SimpleNamespace(get_data_frames=lambda: [frame])


def test_live_fetch_is_once_per_team_and_valid_without_s3(monkeypatch):
    """The per-team fallback, when the league-wide request fails."""
    day = pd.Timestamp.now(tz="America/New_York").date().isoformat()
    calls = []
    _league_offline(monkeypatch)

    def fetch(**kwargs):
        calls.append(kwargs)
        frame = (
            players()
            .rename(columns={"PLAYER_NAME": "PLAYER"})
            .assign(TeamID=TEAM, SEASON=kwargs["season"])
        )
        return SimpleNamespace(
            common_team_roster=SimpleNamespace(get_data_frame=lambda: frame)
        )

    monkeypatch.setattr(live, "CommonTeamRoster", fetch)
    monkeypatch.setattr(
        live, "_store", lambda: (_ for _ in ()).throw(OSError("S3 offline"))
    )
    snapshots = live.load_live_rosters(pd.concat([schedule(day)] * 2))
    assert len(calls) == 1
    assert calls[0]["team_id"] == int(TEAM)
    assert snapshots.PLAYER_ID.tolist() == ["1"]
    assert snapshots.SOURCE.tolist() == ["nba"]


def test_failed_live_endpoint_uses_cached_roster(monkeypatch):
    now = pd.Timestamp.now(tz="UTC")
    day = now.tz_convert("America/New_York").date().isoformat()
    games = schedule(day)
    season = live.season_for_game(next(games.itertuples()))
    store = live.RosterStore(S3(), "bucket")
    store.save(day, season, TEAM, players(), now, [])
    monkeypatch.setattr(live, "_store", lambda: store)
    _league_offline(monkeypatch)
    monkeypatch.setattr(
        live, "CommonTeamRoster", lambda **kw: (_ for _ in ()).throw(TimeoutError())
    )
    snapshots = live.load_live_rosters(games)
    assert snapshots.PLAYER_ID.tolist() == ["1"]
    assert snapshots.SOURCE.tolist() == ["s3"]


@pytest.mark.parametrize("ids", [[], ["bad"], ["1", "1"], [None], ["1.5"]])
def test_bad_rosters_are_rejected(ids):
    with pytest.raises(ValueError):
        players(ids)


def test_different_game_day_cannot_reuse_a_cached_roster():
    snapshot = players().assign(
        TEAM_ID=TEAM, SEASON=SEASON, FETCHED_AT_UTC=EARLY, SOURCE="s3"
    )
    assert not live.roster_overrides(
        schedule("2026-10-17"), snapshot, as_of=LATE + pd.Timedelta(days=1)
    )


def test_cache_corruption_is_detected():
    store = live.RosterStore(S3(), "bucket")
    store.save(DAY, SEASON, TEAM, players(), EARLY, [])
    entries = store.entries(DAY, SEASON, TEAM)
    store.client.objects[entries[0]["key"]] = b"corrupt"
    with pytest.raises(ValueError, match="corrupt"):
        store.read(entries, SEASON, TEAM, LATE)


def test_concurrent_writer_cannot_replace_newer_manifest():
    client = S3()
    first, second = (
        live.RosterStore(client, "bucket"),
        live.RosterStore(client, "bucket"),
    )
    assert first.entries(DAY, SEASON, TEAM) == second.entries(DAY, SEASON, TEAM) == []
    second.save(DAY, SEASON, TEAM, players(("2",)), LATE, [])
    first.save(DAY, SEASON, TEAM, players(), EARLY, [])
    entries = first.entries(DAY, SEASON, TEAM)
    assert len(entries) == 1
    assert entries[0]["fetched_at_utc"] == LATE.isoformat()


OTHER = "1610612738"


def test_one_league_request_serves_every_team(monkeypatch):
    day = pd.Timestamp.now(tz="America/New_York").date().isoformat()
    games = schedule(day).assign(VISITOR_TEAM_ID=OTHER)
    calls = []

    def index(**kwargs):
        calls.append(kwargs)
        return player_index({TEAM: ["1", "2"], OTHER: ["3"], "1610612765": ["4"]})

    monkeypatch.setattr(live, "PlayerIndex", index)
    monkeypatch.setattr(
        live, "CommonTeamRoster", lambda **kw: pytest.fail("per-team request")
    )
    monkeypatch.setattr(
        live, "_store", lambda: (_ for _ in ()).throw(OSError("S3 offline"))
    )
    snapshots = live.load_live_rosters(games)
    assert len(calls) == 1
    assert calls[0]["season"] == live.season_for_game(next(games.itertuples()))
    by_team = snapshots.groupby("TEAM_ID")["PLAYER_ID"].apply(sorted).to_dict()
    assert by_team == {TEAM: ["1", "2"], OTHER: ["3"]}
    assert set(snapshots.PLAYER_NAME) == {"Player 1", "Player 2", "Player 3"}
    assert set(snapshots.SOURCE) == {"nba"}


def test_a_team_missing_from_the_league_request_is_fetched_alone(monkeypatch):
    day = pd.Timestamp.now(tz="America/New_York").date().isoformat()
    games = schedule(day).assign(VISITOR_TEAM_ID=OTHER)
    asked = []

    def roster(**kwargs):
        asked.append(str(kwargs["team_id"]))
        frame = (
            players(("9",))
            .rename(columns={"PLAYER_NAME": "PLAYER"})
            .assign(TeamID=str(kwargs["team_id"]), SEASON=kwargs["season"])
        )
        return SimpleNamespace(
            common_team_roster=SimpleNamespace(get_data_frame=lambda: frame)
        )

    monkeypatch.setattr(live, "PlayerIndex", lambda **kw: player_index({TEAM: ["1"]}))
    monkeypatch.setattr(live, "CommonTeamRoster", roster)
    monkeypatch.setattr(
        live, "_store", lambda: (_ for _ in ()).throw(OSError("S3 offline"))
    )
    snapshots = live.load_live_rosters(games)
    assert asked == [OTHER]
    by_team = snapshots.groupby("TEAM_ID")["PLAYER_ID"].apply(list).to_dict()
    assert by_team == {TEAM: ["1"], OTHER: ["9"]}


def test_unrostered_players_are_not_on_a_league_roster(monkeypatch):
    frame = pd.DataFrame(
        {
            "PERSON_ID": [1, 2, 3],
            "PLAYER_FIRST_NAME": ["A", "B", "C"],
            "PLAYER_LAST_NAME": ["One", "Two", "Three"],
            "TEAM_ID": [int(TEAM), int(TEAM), 0],
            "POSITION": ["G", "F", "C"],
            "ROSTER_STATUS": [1.0, None, None],
        }
    )
    monkeypatch.setattr(
        live,
        "PlayerIndex",
        lambda **kw: SimpleNamespace(get_data_frames=lambda: [frame]),
    )
    rosters = live.league_rosters(SEASON)
    assert list(rosters) == [TEAM]
    assert rosters[TEAM].to_dict("records") == [
        {"PLAYER_ID": "1", "PLAYER_NAME": "A One", "POSITION": "G"}
    ]


def test_replay_roster_cutoff_is_applied_per_game():
    games = pd.concat(
        [schedule(), schedule().assign(GAME_ID="0022600002")], ignore_index=True
    )
    snapshots = pd.concat(
        [
            players().assign(TEAM_ID=TEAM, SEASON=SEASON, FETCHED_AT_UTC=EARLY),
            players(("2",)).assign(TEAM_ID=TEAM, SEASON=SEASON, FETCHED_AT_UTC=LATE),
        ],
        ignore_index=True,
    )
    result = live.roster_overrides(
        games, snapshots, as_of={"0022600001": EARLY, "0022600002": LATE}
    )
    assert result[("0022600001", TEAM)] == {"1": "Player 1"}
    assert result[("0022600002", TEAM)] == {"2": "Player 2"}
