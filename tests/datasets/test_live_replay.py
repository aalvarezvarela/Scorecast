"""Closing replay must keep historical cutoffs and archived roster reads."""

from types import SimpleNamespace

import pandas as pd
from nba_ou.create_training_data import live_replay as replay
from nba_ou.data_processing.scheduled_games import manage_injury_data
from nba_ou.fetch_data import live_rosters
from nba_ou.fetch_data.injury_reports import get_latest_injury_report
from nba_ou.fetch_data.injury_reports.archive import manifest, storage
from nba_ou.postgre_db.odds_sportsbook.fetch_data_from_db import (
    fetch_data_from_odds_sportsbook_db as sportsbook,
)
from nba_ou.postgre_db.odds_yahoo.fetch_data_from_db import (
    fetch_data_from_odds_yahoo_db as yahoo,
)


def test_closing_replay_passes_cutoff_and_archived_rosters(monkeypatch):
    as_of = pd.Timestamp("2026-04-10T22:30Z")
    games = pd.DataFrame({"GAME_ID": ["0022500001"]})
    rosters = pd.DataFrame({"PLAYER_ID": ["archived-player"]})
    reports = pd.DataFrame(
        {"nba_available": [manifest.AVAILABLE_TRUE], "s3_key": ["archived.pdf"]}
    )
    monkeypatch.setattr(replay, "replay_scheduled_games", lambda day: games)
    monkeypatch.setattr(replay, "stored_crews", lambda *a: pd.DataFrame())
    monkeypatch.setattr(
        manifest,
        "ManifestStore",
        lambda root: SimpleNamespace(load=lambda season: reports),
    )
    monkeypatch.setattr(
        manifest,
        "published_utc",
        lambda frame: pd.Series([as_of - pd.Timedelta(minutes=15)]),
    )
    monkeypatch.setattr(
        storage, "S3Storage", lambda **kw: SimpleNamespace(get=lambda key: b"PDF")
    )
    monkeypatch.setattr(
        get_latest_injury_report, "read_injury_report", lambda raw: pd.DataFrame()
    )
    monkeypatch.setattr(
        manage_injury_data, "process_injury_data", lambda *a: ({}, [])
    )
    monkeypatch.setattr(yahoo, "load_odds_yahoo_from_db", lambda *a: pd.DataFrame())
    monkeypatch.setattr(
        sportsbook, "load_odds_sportsbook_from_db", lambda *a: pd.DataFrame()
    )

    def archived_rosters(frame, **kwargs):
        assert frame is games
        assert kwargs == {"as_of": as_of}
        return rosters

    monkeypatch.setattr(live_rosters, "load_live_rosters", archived_rosters)
    result = replay.replay_scheduled_data("2026-04-10", as_of)
    assert result["injury_as_of_utc"] == as_of
    assert result["roster_snapshots"] is rosters
