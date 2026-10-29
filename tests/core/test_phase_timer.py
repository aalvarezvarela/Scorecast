"""A job's phases print a [time] line as they end and add up in the summary."""

import pytest
from nba_ou.utils import phase_timer
from nba_ou.utils.phase_timer import PhaseTimer, duration


def test_duration_is_minutes_and_zero_padded_seconds():
    assert duration(0) == "0m 00s"
    assert duration(65.4) == "1m 05s"
    assert duration(3599.6) == "60m 00s"


def test_each_phase_prints_as_it_ends_and_the_summary_lists_them(monkeypatch, capsys):
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
        "[start] closing rows",
        "[time] closing rows: 1m 05s; peak RSS 1.0 GB",
        "[time] 51 slots: 2m 05s; peak RSS 1.0 GB",
        "[start] storing",
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


def test_interrupted_run_keeps_completed_and_failed_timings(
    monkeypatch, tmp_path, capsys
):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    timer = PhaseTimer(heartbeat_seconds=0)
    with pytest.raises(RuntimeError, match="store down"):
        with timer.report("Daily job"):
            with timer.phase("load"):
                pass
            with timer.phase("store"):
                raise RuntimeError("store down")
    assert "Timings (wall clock):" in capsys.readouterr().out
    markdown = summary.read_text()
    assert "### Daily job (interrupted)" in markdown
    assert "| load |" in markdown
    assert "| store (did not finish) |" in markdown


def test_heartbeat_reports_elapsed_time_and_stops_at_phase_end(monkeypatch):
    import threading

    reported = threading.Event()
    workers = []
    messages = []

    def capture(message, **kwargs):
        messages.append(message)
        if message.startswith("[progress]"):
            workers.append(threading.current_thread())
            reported.set()

    monkeypatch.setattr(phase_timer, "print", capture, raising=False)
    timer = PhaseTimer(heartbeat_seconds=0.01)
    with timer.phase("slow fit"):
        assert reported.wait(timeout=2), "no heartbeat while the phase was running"
    assert any("slow fit: still running," in message for message in messages)
    assert messages[-1].startswith("[time] slow fit:")
    assert workers and all(not worker.is_alive() for worker in workers)


def test_report_is_written_on_early_return(monkeypatch, tmp_path):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    def no_games():
        timer = PhaseTimer(heartbeat_seconds=0)
        with timer.report("No games today"):
            return 0

    assert no_games() == 0
    assert "### No games today\n" in summary.read_text()
    assert "interrupted" not in summary.read_text()
