"""Step 3 of the coach presser archive: resolving the team channels."""

from __future__ import annotations

from lab.coach_pressers import channels
from lab.game_reports.season_reports import RequestPacer

FAKE = {
    "https://www.youtube.com/@celtics": {
        "channel": "Boston Celtics",
        "channel_id": "UC-celtics",
        "channel_follower_count": 1_000_000,
    },
    "https://www.youtube.com/@ATLHawks": {
        "channel": "Some Fan Page",
        "channel_id": "UCx",
    },
    "https://www.youtube.com/@hawks": {
        "channel": "Atlanta Hawks",
        "channel_id": "UC-hawks",
        "channel_follower_count": 109_000,
    },
    # An impostor with the exact team name, and the real channel found by search.
    "https://www.youtube.com/@mavericks": {
        "channel": "Dallas Mavericks",
        "channel_id": "UC-fake",
        "channel_follower_count": 117,
    },
    "https://www.youtube.com/channel/UC-mavs": {
        "channel": "Dallas Mavericks",
        "channel_id": "UC-mavs",
        "uploader_id": "@realmavs",
        "channel_follower_count": 226_000,
    },
}


def extract(url):
    if url not in FAKE:
        raise RuntimeError("404")
    return FAKE[url]


def test_resolution_checks_the_channel_name(tmp_path):
    path = tmp_path / "channels.csv"
    rows, failures = channels.resolve_all(
        path, extract, RequestPacer(0), teams=["BOS", "ATL", "CHI"]
    )
    by_team = {r["team"]: r for r in rows}
    assert by_team["BOS"]["channel_id"] == "UC-celtics"
    assert by_team["ATL"]["handle"] == "@hawks"  # the first handle is not the team
    assert by_team["BOS"]["verified_by_human"] is False
    assert set(failures) == {"CHI"}
    reread = {r["team"]: r for r in channels.read_channels(path)}
    assert (
        reread["ATL"]["priority"] == 1 and reread["ATL"]["verified_by_human"] is False
    )


def test_resolved_teams_are_not_resolved_again(tmp_path):
    path = tmp_path / "channels.csv"
    channels.resolve_all(path, extract, RequestPacer(0), teams=["BOS"])
    calls = []

    def counting(url):
        calls.append(url)
        return extract(url)

    channels.resolve_all(path, counting, RequestPacer(0), teams=["BOS"])
    assert calls == []


def test_team_channels_orders_by_priority(tmp_path):
    rows = [
        {"team": "BOS", "channel_id": "b", "priority": 2},
        {"team": "BOS", "channel_id": "a", "priority": 1},
        {"team": "ATL", "channel_id": "c", "priority": 1},
        {"team": "BOS", "channel_id": "", "priority": 1},
    ]
    assert [r["channel_id"] for r in channels.team_channels(rows, "BOS")] == ["a", "b"]


def test_small_namesake_is_rejected_and_search_finds_the_team(tmp_path):
    def search(query):
        assert query == "Dallas Mavericks postgame press conference"
        return [
            {"channel": "Dallas Mavericks", "channel_id": "UC-mavs"},
            {"channel": "Dallas Mavericks", "channel_id": "UC-mavs"},
            {"channel": "Dallas Sports Journal", "channel_id": "UC-journal"},
        ]

    rows, failures = channels.resolve_all(
        tmp_path / "c.csv", extract, RequestPacer(0), teams=["DAL"], search=search
    )
    assert failures == {}
    assert (rows[0]["channel_id"], rows[0]["handle"]) == ("UC-mavs", "@realmavs")


def test_refresh_replaces_unverified_rows_only(tmp_path):
    path = tmp_path / "c.csv"
    channels.write_channels(
        path,
        [
            {
                "team": "BOS",
                "channel_id": "UC-old",
                "handle": "@x",
                "name": "Boston Celtics",
                "kind": "team",
                "priority": 1,
                "verified_by_human": False,
            },
            {
                "team": "ATL",
                "channel_id": "UC-kept",
                "handle": "@y",
                "name": "Atlanta Hawks",
                "kind": "team",
                "priority": 1,
                "verified_by_human": True,
            },
        ],
    )
    rows, _ = channels.resolve_all(
        path, extract, RequestPacer(0), teams=["BOS", "ATL"], refresh=True
    )
    assert {r["team"]: r["channel_id"] for r in rows} == {
        "BOS": "UC-celtics",
        "ATL": "UC-kept",
    }
