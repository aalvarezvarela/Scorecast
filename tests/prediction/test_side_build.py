"""A build in a spawned side process, gated with the one this process runs."""

from __future__ import annotations

import time

import pytest
from nba_ou.prediction import side_build
from nba_ou.prediction.side_build import SideBuild, SideBuildFailed
from nba_ou.utils import warm_gate

from tests.support import side_jobs


def test_returns_the_value_and_prints_the_side_log(capsys):
    side = SideBuild.start(side_jobs.add, 1, 2, c=3, current_season=2026, label="sum")
    outcome = side.result()
    assert outcome.value == 6
    assert outcome.cold_cause is None
    assert outcome.peak.endswith("GB")
    out = capsys.readouterr().out
    assert "sum (side process, " in out
    assert "[sum] adding" in out.splitlines()


def test_a_quiet_side_build_is_followed_live(capsys):
    """Lines show as the side process writes them, and a quiet stretch says
    the build is still running."""
    side = SideBuild.start(side_jobs.slow, 3.0, current_season=2026, label="slow")
    outcome = side.result(heartbeat_seconds=1.0)
    assert outcome.value == "done"
    lines = capsys.readouterr().out.splitlines()
    marked = [line for line in lines if line.startswith("[slow] ")]
    assert marked[0] == "[slow] starting"
    assert any("still running" in line for line in marked)
    assert marked[-1] == "[slow] finishing"


def test_a_raising_build_raises_here_with_its_traceback():
    side = SideBuild.start(side_jobs.fail, current_season=2026, label="failing")
    with pytest.raises(SideBuildFailed, match="ValueError: boom"):
        side.result()


def test_a_killed_build_gives_no_result():
    side = SideBuild.start(side_jobs.die, current_season=2026, label="killed")
    assert side.result() is None


def test_a_cold_side_build_waits_for_this_one():
    side = SideBuild.start(
        side_jobs.cold_past_season, 2024, current_season=2026, label="cold"
    )
    try:
        # This process's own memo misses report to the same gate meanwhile.
        assert warm_gate._GATE is side.gate
        time.sleep(3)  # spawn, import and reach the miss
        released = time.time()
    finally:
        outcome = side.result()
    assert warm_gate._GATE is None
    assert outcome.cold_cause == "player_history_rows 2024"
    assert outcome.value >= released


def test_side_by_side_needs_a_usable_memo(monkeypatch):
    monkeypatch.setattr(side_build, "history_memo_rebuild", lambda: False)
    monkeypatch.setattr(side_build, "default_backend", lambda: None)
    assert side_build.side_by_side_reason() == "no history memo"
    monkeypatch.setattr(side_build, "history_memo_rebuild", lambda: True)
    assert side_build.side_by_side_reason() == "the history memo is being rebuilt"
    monkeypatch.setattr(side_build, "history_memo_rebuild", lambda: False)
    monkeypatch.setattr(side_build, "default_backend", lambda: object())
    assert side_build.side_by_side_reason() is None
