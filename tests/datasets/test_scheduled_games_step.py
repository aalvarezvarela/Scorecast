"""The prediction run's step 2: the odds are fetched beside the injury reports."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pandas as pd
import pytest
from nba_ou.create_training_data import get_all_info_for_scheduled_games as info

GAME = "0022600104"


@pytest.fixture
def step(monkeypatch):
    odds_started = threading.Event()
    calls = []
    scheduled = pd.DataFrame(
        {
            "GAME_ID": [GAME],
            "GAME_DATE_EST": ["2026-10-24"],
            "HOME_TEAM_ID": ["1610612748"],
            "VISITOR_TEAM_ID": ["1610612766"],
        }
    )
    monkeypatch.setattr(info, "get_schedule_games", lambda day: scheduled)
    monkeypatch.setattr(
        info, "process_scheduled_referee_assignments", lambda games: pd.DataFrame()
    )
    monkeypatch.setattr(info, "load_live_rosters", lambda games: pd.DataFrame())

    def synchronize(*_, **__):
        # Started first, the odds are under way before the reports sync.
        assert odds_started.wait(5), "odds not fetched beside the injury reports"
        calls.append("sync")
        return SimpleNamespace(pending=1, ingested=1, unavailable=0, failed=0)

    monkeypatch.setattr(info, "synchronize_injury_reports", synchronize)
    state = SimpleNamespace(covered={(GAME, "1610612748"), (GAME, "1610612766")})
    monkeypatch.setattr(info, "load_prediction_report_state", lambda *a: state)
    monkeypatch.setattr(info, "report_out_overrides", lambda state: {})
    yahoo, sportsbook = pd.DataFrame({"y": [1]}), pd.DataFrame({"s": [1]})

    def fetch_yahoo(*_):
        odds_started.set()
        calls.append("yahoo")
        return yahoo

    monkeypatch.setattr(info, "get_yahoo_prediction_data", fetch_yahoo)
    monkeypatch.setattr(
        info, "get_sportsbook_prediction_data", lambda *a, **k: sportsbook
    )
    return SimpleNamespace(calls=calls, yahoo=yahoo, sportsbook=sportsbook)


def run():
    return info.get_all_info_for_scheduled_games(
        "2026-10-24", nba_injury_reports_url="", headless=True
    )


def test_odds_are_fetched_beside_the_injury_reports(step, capsys):
    data = run()
    assert sorted(step.calls) == ["sync", "yahoo"]
    assert data["df_odds_yahoo_scheduled"] is step.yahoo
    assert data["df_odds_sportsbook_scheduled"] is step.sportsbook
    assert data["games_not_updated"] == []
    out = capsys.readouterr().out
    for part in ("odds", "referees", "live rosters", "injury report sync"):
        assert f"[time] step 2 {part}" in out
    assert "Injury report sync: 1 pending, 1 ingested" in out


def test_a_failed_odds_fetch_still_fails_the_step(step, monkeypatch):
    def fail(*_, **__):
        raise ValueError("No Sportsbook odds data found")

    monkeypatch.setattr(info, "get_sportsbook_prediction_data", fail)
    with pytest.raises(ValueError, match="No Sportsbook odds"):
        run()
