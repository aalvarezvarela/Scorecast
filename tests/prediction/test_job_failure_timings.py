"""Public job entry points publish timings outside groups on early failure."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name,entry_point,runner",
    [
        ("predict_nba_games", "predict_nba_games", "run_prediction_pipeline"),
        ("predict_tabpfn", "predict_tabpfn", "run_tabpfn"),
        ("predict_past_snapshots", "predict_past_snapshots", "_predict_past_snapshots"),
        ("retrain_prediction_models", "main", "_retrain"),
    ],
)
def test_failed_jobs_publish_partial_timings_and_close_their_group(
    module_name, entry_point, runner, monkeypatch, tmp_path, capsys
):
    module = importlib.import_module(f"scripts.{module_name}")
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr("sys.argv", [module_name])

    def fail(*args, **kwargs):
        if module_name == "retrain_prediction_models":
            _, timer, log = args
            phase = timer.phase("closing_line", "update", "load inputs")
        else:
            log = kwargs.get("log", args[0])
            timer = kwargs["timer"]
            phase = timer.phase("load inputs")
        log.step("Loading inputs")
        with phase:
            raise RuntimeError("input unavailable")

    monkeypatch.setattr(module, runner, fail)
    with pytest.raises(RuntimeError, match="input unavailable"):
        getattr(module, entry_point)()

    output = capsys.readouterr().out
    assert output.index("::endgroup::") < output.index("Timings (wall clock):")
    assert "(interrupted)" in summary.read_text()
    assert "| load inputs (did not finish) |" in summary.read_text()
