"""The daily refit's phase timings: each dataset's chain summed on its own.

They are what deciding whether the closing and intermediate refits can run on
separate runners reads, so a phase that raises must still be counted, and the
exception must still reach the job.
"""

import pytest

import scripts.retrain_prediction_models as retrain


@pytest.fixture
def clock(monkeypatch):
    """A perf_counter that only moves when told to."""
    now = [0.0]
    monkeypatch.setattr(retrain.time, "perf_counter", lambda: now[0])
    return now


def test_phases_are_summed_per_dataset_type(clock, capsys):
    timings = retrain.PhaseTimings()
    with timings.phase("closing_line", "update", "closing update"):
        clock[0] += 600
    for horizon in (60, 120):
        with timings.phase("intermediate_line", "frame", f"T-{horizon} frame"):
            clock[0] += 30
        with timings.phase("intermediate_line", "fit", f"T-{horizon} fit"):
            clock[0] += 5

    printed = capsys.readouterr().out
    assert "[time] closing update: 10m 00s; peak RSS" in printed
    summary = timings.summary()
    assert "closing_line       update 1x 10m 00s | total 10m 00s" in summary
    assert (
        "intermediate_line  frame 2x 1m 00s | fit 2x 0m 10s | total 1m 10s" in summary
    )
    assert "whole run 11m 10s" in summary


def test_a_phase_that_raises_is_counted_and_reraised(clock, capsys):
    timings = retrain.PhaseTimings()
    with pytest.raises(ModuleNotFoundError):
        with timings.phase("intermediate_line", "update", "intermediate update"):
            clock[0] += 2
            raise ModuleNotFoundError("No module named 'pymupdf'")

    assert "intermediate update: 0m 02s (did not finish)" in capsys.readouterr().out
    assert "update 1x 0m 02s (1 did not finish)" in timings.summary()
