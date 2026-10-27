"""A job's phases print a [time] line as they end and add up in the summary."""

import pytest
from nba_ou.utils import phase_timer
from nba_ou.utils.phase_timer import PhaseTimer, duration


def test_duration_is_minutes_and_zero_padded_seconds():
    assert duration(0) == "0m 00s"
    assert duration(65.4) == "1m 05s"
    assert duration(3599.6) == "60m 00s"


def test_each_phase_prints_as_it_ends_and_the_summary_lists_them(
    monkeypatch, capsys
):
    clock = iter([0.0, 10.0, 75.0, 80.0, 200.0])
    monkeypatch.setattr(phase_timer.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(phase_timer, "peak_rss", lambda: "1.0 GB")
    timer = PhaseTimer()  # starts at 0
    with timer.phase("closing rows"):  # 10 -> 75
        pass
    timer.record("51 slots", 125.0)
    with pytest.raises(RuntimeError):
        with timer.phase("storing"):  # 80 -> 200
            raise RuntimeError("store down")
    printed = capsys.readouterr().out.splitlines()
    assert printed == [
        "[time] closing rows: 1m 05s; peak RSS 1.0 GB",
        "[time] 51 slots: 2m 05s; peak RSS 1.0 GB",
        "[time] storing: 2m 00s (did not finish); peak RSS 1.0 GB",
    ]
    clock = iter([300.0])
    assert timer.summary().splitlines() == [
        "Timings (wall clock):",
        "  closing rows  1m 05s",
        "  51 slots      2m 05s",
        "  storing       2m 00s  (did not finish)",
        "  total         5m 00s; peak RSS 1.0 GB",
    ]
