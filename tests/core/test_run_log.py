"""A prediction job's log: numbered steps, GitHub Actions groups, annotations
and the job summary, and the phase timer's table for it."""

from __future__ import annotations

from nba_ou.utils import run_log
from nba_ou.utils.phase_timer import PhaseTimer
from nba_ou.utils.run_log import RunLog


def test_steps_are_numbered_in_the_order_they_start(capsys):
    log = RunLog(github=False)
    log.step("Updating")
    log.step("Checking")
    out = capsys.readouterr().out
    assert "STEP 1 — Updating [" in out
    assert "STEP 2 — Checking [" in out
    assert "::group::" not in out


def test_in_github_actions_each_step_is_a_group_but_the_summary(capsys):
    log = RunLog(github=True)
    log.step("Updating")
    print("work")
    log.step("Summary", group=False)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("::group::STEP 1 — Updating [")
    assert lines[1:3] == ["work", "::endgroup::"]
    assert "STEP 2 — Summary [" in "\n".join(lines[3:])
    assert "::group::" not in "\n".join(lines[3:])


def test_closing_an_unopened_group_prints_nothing(capsys):
    log = RunLog(github=True)
    log.end_group()
    log.step("Summary", group=False)
    log.end_group()
    assert "::endgroup::" not in capsys.readouterr().out


def test_warnings_and_errors_are_annotations_in_github_actions(capsys):
    RunLog(github=True).warning("cold: 100% of\nseasons")
    RunLog(github=True).error("slot failed")
    RunLog(github=False).error("local")
    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "  ⚠ cold: 100% of",
        "seasons",
        "::warning::cold: 100%25 of%0Aseasons",
        "  ✖ slot failed",
        "::error::slot failed",
        "  ✖ local",
    ]


def test_the_job_summary_is_appended_when_github_gives_a_file(tmp_path, monkeypatch):
    path = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(path))
    RunLog(github=True).summary("### one\n")
    RunLog(github=True).summary("### two")
    assert path.read_text() == "### one\n\n### two\n\n"
    monkeypatch.delenv("GITHUB_STEP_SUMMARY")
    RunLog(github=True).summary("### three")
    assert "three" not in path.read_text()


def test_a_bar_without_a_label_is_off_and_one_with_a_label_follows_the_terminal():
    assert run_log.progress_disabled(None) is True
    assert run_log.progress_disabled("Injuries") is None


def test_the_timer_prints_another_processs_peak_and_a_markdown_table(capsys):
    timer = PhaseTimer()
    timer.record("closing rows (side process)", 125.0, peak="2.5 GB")
    timer.record("storing", 3.0, failed=True)
    assert "[time] closing rows (side process): 2m 05s; peak RSS 2.5 GB" in (
        capsys.readouterr().out
    )
    table = timer.markdown().splitlines()
    assert table[:4] == [
        "| Phase | Wall clock |",
        "| --- | ---: |",
        "| closing rows (side process) | 2m 05s |",
        "| storing (did not finish) | 0m 03s |",
    ]
    assert table[4].startswith("| **total** (peak RSS ")
