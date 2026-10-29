"""Workflow commands stream output, keep their arguments and propagate failure."""

import sys

import pytest

from scripts.run_timed import main


def test_command_arguments_are_literal_and_output_is_streamed(capfd):
    literal = "space ; $(false) `false`"
    result = main(
        [
            "Example",
            "--",
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1])",
            literal,
        ]
    )
    assert result == 0
    output = capfd.readouterr().out
    assert literal in output
    assert output.index("[start] Example") < output.index(literal)
    assert output.index(literal) < output.index("[time] Example")


def test_command_failure_keeps_exit_code_and_publishes_timings(
    monkeypatch, tmp_path, capfd
):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    result = main(["Failing step", "--", sys.executable, "-c", "raise SystemExit(7)"])
    assert result == 7
    assert "did not finish" in capfd.readouterr().out
    assert "Failing step (interrupted)" in summary.read_text()
    assert "| Failing step (did not finish) |" in summary.read_text()


def test_missing_command_is_rejected():
    with pytest.raises(SystemExit) as exc:
        main(["Missing command", "--"])
    assert exc.value.code == 2
