"""The warm gate: two builds overlap until one of them turns cold."""

from __future__ import annotations

import threading
import time

from nba_ou.utils import warm_gate
from nba_ou.utils.warm_gate import WarmGate


def _gates(tmp_path):
    path = tmp_path / "warm.lock"
    return (
        WarmGate(path, current_season=2026, label="closing rows"),
        WarmGate(path, current_season=2026, label="intermediate rows"),
    )


def _in_thread(fn):
    done = threading.Event()

    def run():
        fn()
        done.set()

    threading.Thread(target=run, daemon=True).start()
    return done


def test_warm_builds_hold_the_gate_together(tmp_path):
    closing, intermediate = _gates(tmp_path)
    # A current-season miss is part of every warm run: it passes.
    assert _in_thread(lambda: closing.cold("report_states", 2026)).wait(5)
    assert closing.cold_cause is None
    closing.release()
    intermediate.release()


def test_a_cold_build_waits_for_the_other_to_finish(tmp_path):
    closing, intermediate = _gates(tmp_path)
    done = _in_thread(lambda: closing.cold("closing_line_repair", 2024))
    assert not done.wait(0.5)
    intermediate.release()
    assert done.wait(5)
    assert closing.cold_cause == "closing_line_repair 2024"
    assert closing.waited >= 0.4
    # Cold once is cold for the rest of the build: no second wait.
    assert _in_thread(lambda: closing.cold("player_history_rows", 2019)).wait(5)
    closing.release()


def test_two_builds_turning_cold_at_once_take_turns(tmp_path):
    closing, intermediate = _gates(tmp_path)
    first = _in_thread(lambda: closing.cold("closing_line_repair", 2024))
    time.sleep(0.2)
    second = _in_thread(lambda: intermediate.cold("report_states", 2025))
    # One of them goes on alone; the other waits until that one finishes.
    assert first.wait(5) != second.wait(0.5)
    (closing if first.is_set() else intermediate).release()
    assert first.wait(5) and second.wait(5)
    closing.release()
    intermediate.release()


def test_no_gate_installed_never_waits(tmp_path):
    warm_gate.install(None)
    warm_gate.cold("closing_line_repair", 2019)
    gate, other = _gates(tmp_path)
    warm_gate.install(gate)
    try:
        warm_gate.cold("closing_line_repair", None)  # an unusable season
        assert gate.cold_cause is None
    finally:
        warm_gate.install(None)
        gate.release()
        other.release()
