"""Current intermediate runs fetch rosters before ingesting injury names."""

import pandas as pd
import pytest
from nba_ou.create_training_data import live_game_day as live
from nba_ou.postgre_db.line_history_aiven.live import LiveLineHistory


@pytest.mark.parametrize("replay", [False, True])
def test_intermediate_bundle_shares_rosters_and_replay_never_refreshes_injuries(
    monkeypatch, replay
):
    as_of = pd.Timestamp("2026-10-16T18:00Z")
    games = pd.DataFrame({"GAME_ID": ["0022600001"]})
    rosters = pd.DataFrame({"PLAYER_ID": ["new_rookie"]})
    calls = []

    def fetch_rosters(frame, **kwargs):
        assert frame is games
        assert kwargs["as_of"] == (as_of if replay else None)
        calls.append("rosters")
        return rosters

    def sync(day, **kwargs):
        assert not replay
        assert kwargs["roster_snapshots"] is rosters
        calls.append("injuries")
        # An outage can use Aiven's eligible prior state for intermediates.
        raise OSError("Aiven refresh unavailable")

    def lines(*args, **kwargs):
        calls.append("lines")
        assert (
            kwargs["as_of"] == as_of if replay else kwargs["as_of"].tzinfo is not None
        )
        return LiveLineHistory(
            ticks=pd.DataFrame(), games=pd.DataFrame(), as_of=kwargs["as_of"]
        )

    monkeypatch.setattr(live, "load_live_rosters", fetch_rosters)
    monkeypatch.setattr(live, "synchronize_injury_reports", sync)
    monkeypatch.setattr(
        live, "scheduled_team_games", lambda *args, **kw: pd.DataFrame()
    )
    monkeypatch.setattr(live, "fetch_schedules", lambda *args: pd.DataFrame())
    monkeypatch.setattr(live, "fetch_live_line_history", lines)
    monkeypatch.setattr(
        live,
        "fetch_live_injury_reports",
        lambda *a, **kw: pytest.fail("current predictions must read Aiven"),
    )
    bundle = live.fetch_live_game_day(
        "2026-10-16", scheduled_games=games, as_of=as_of if replay else None
    )
    assert bundle.roster_snapshots is rosters
    assert bundle.injury_reports is None
    assert calls == (
        ["rosters", "lines"] if replay else ["rosters", "injuries", "lines"]
    )
