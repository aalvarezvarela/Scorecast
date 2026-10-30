"""The overnight backfill is previewed offline and booked only explicitly."""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, call

import pytest
import yaml

import scripts.cronjobs.schedule_historical_backfill as job
import scripts.cronjobs.wait_for_prediction_run as wait

NOW = datetime(2026, 10, 29, 18, tzinfo=UTC)
FINAL = datetime(2026, 10, 31, 5, tzinfo=UTC)


def test_plan_uses_requested_dates_models_and_madrid_time():
    plan = job.make_plan(now=NOW)
    assert plan["run_at_madrid"] == "2026-10-31T06:20:00+01:00"
    assert plan["run_at"] == "2026-10-31T05:20:00+00:00"
    inputs = plan["inputs"]
    assert inputs["start_date"] == "2026-10-20"
    assert inputs["end_date"] == "2026-10-25"
    assert inputs["include_tabpfn"] == inputs["include_closing"] == "true"
    assert inputs["tabpfn_only"] == "false"
    assert inputs["tabpfn_horizon_minutes"] == "240"
    assert inputs["persist"] == "true"
    assert inputs["overwrite"] == "false"
    assert "slots" not in inputs  # All enabled registry slots.
    assert wait.parse_dependency(inputs["after_prediction"])[1] == FINAL
    payload = job.job_payload(plan, "test-token")
    assert payload["schedule"] == {
        "timezone": "UTC",
        "expiresAt": 20261031052059,
        "hours": [5],
        "minutes": [20],
        "mdays": [31],
        "months": [10],
        "wdays": [-1],
    }


def test_overwrite_repeat_has_new_title_and_executes_requested_branch():
    plan = job.make_plan(
        now=NOW, overwrite=True, ref="fix/historical-backfill-overwrite"
    )
    payload = job.job_payload(plan, "test-token")
    body = json.loads(payload["extendedData"]["body"])
    assert body["ref"] == "fix/historical-backfill-overwrite"
    assert body["inputs"]["overwrite"] == "true"
    assert "2026-10-31T05:20:00+00:00" in payload["title"]
    assert payload["title"].endswith(" overwrite")


def test_unpublished_ref_cannot_book_cron(monkeypatch):
    registered = Mock(status_code=200)
    registered.json.return_value = {"state": "active"}
    monkeypatch.setattr(
        job.requests, "get", Mock(side_effect=[registered, Mock(status_code=404)])
    )
    api = Mock()
    monkeypatch.setattr(job, "cronjob_request", api)
    with pytest.raises(ValueError, match="Publish.*ref feature"):
        job.apply_plan(
            job.make_plan(now=NOW, ref="feature"), api_key="key", token="token"
        )
    api.assert_not_called()


@pytest.mark.parametrize(
    "overrides",
    [
        {"run_at": "2026-10-29T06:20:00+01:00"},
        {"run_at": "2026-10-31T06:20:00"},
        {"run_at": "2026-10-31T06:20:01+01:00"},
        {"run_at": "2026-10-31T05:59:00+01:00"},
        {"start_date": "2026-10-26", "end_date": "2026-10-20"},
        {"end_date": "2026-10-29"},
        {"after_prediction": "unknown.yml@2026-10-31T05:00:00Z"},
        {"after_prediction": "nba_predict_past_snapshots.yml@2026-10-31T05:00:00"},
    ],
)
def test_invalid_plans_cannot_be_booked(overrides):
    with pytest.raises(ValueError):
        job.make_plan(now=NOW, **overrides)


@pytest.mark.parametrize("dry_predictions", [False, True])
def test_cli_preview_is_offline_and_does_not_fit(monkeypatch, capsys, dry_predictions):
    clock = Mock(wraps=datetime)
    clock.now.return_value = NOW
    monkeypatch.setattr(job, "datetime", clock)
    forbidden = Mock(side_effect=AssertionError("Preview must remain offline"))
    for name in ("read_secret", "cronjob_request", "apply_plan"):
        monkeypatch.setattr(job, name, forbidden)
    assert job.main(["--dry-run-predictions"] if dry_predictions else []) == 0
    output = capsys.readouterr().out
    assert f'"persist": "{str(not dry_predictions).lower()}"' in output
    assert "Preview only" in output
    forbidden.assert_not_called()


@pytest.fixture
def booking(monkeypatch):
    plan = job.make_plan(now=NOW)
    monkeypatch.setattr(job, "check_workflow", Mock())
    booked = {
        "jobDetails": {
            "enabled": True,
            "nextExecution": int(datetime.fromisoformat(plan["run_at"]).timestamp()),
        }
    }
    api = Mock(side_effect=[{"jobs": []}, {"jobId": 42}, booked])
    monkeypatch.setattr(job, "cronjob_request", api)
    return plan, api, booked


def test_apply_books_once_and_verifies_time(booking):
    plan, api, _ = booking
    assert job.apply_plan(plan, api_key="key", token="test-token") == 42
    assert [(c.args[0], c.args[1]) for c in api.call_args_list] == [
        ("GET", "/jobs"),
        ("PUT", "/jobs"),
        ("GET", "/jobs/42"),
    ]
    job.check_workflow.assert_called_once_with("test-token", ref="main")


def test_apply_updates_matching_cron_and_leaves_other_jobs_alone(booking):
    plan, api, booked = booking
    title = job.job_payload(plan, "test-token")["title"]
    api.side_effect = [
        {
            "jobs": [
                {"jobId": 4, "title": "daily retrain"},
                {"jobId": 42, "title": title},
            ]
        },
        {},
        booked,
    ]
    assert job.apply_plan(plan, api_key="key", token="test-token") == 42
    assert api.call_args_list[1].args[:2] == ("PATCH", "/jobs/42")


@pytest.mark.parametrize("already_fired", [False, True])
def test_duplicates_and_already_fired_crons_are_not_rearmed(booking, already_fired):
    plan, api, _ = booking
    existing = {"jobId": 42, "title": job.job_payload(plan, "test-token")["title"]}
    if already_fired:
        existing["lastExecution"] = 1793337600
    api.side_effect = [{"jobs": [existing] if already_fired else [existing, existing]}]
    with pytest.raises(ValueError):
        job.apply_plan(plan, api_key="key", token="test-token")
    assert api.call_count == 1  # Read only.


def test_incorrect_booked_time_disables_cron(booking):
    plan, api, booked = booking
    booked["jobDetails"]["nextExecution"] += 60
    api.side_effect = [{"jobs": []}, {"jobId": 42}, booked, {}]
    with pytest.raises(ValueError, match="disabled"):
        job.apply_plan(plan, api_key="key", token="test-token")
    assert api.call_args == call(
        "PATCH", "/jobs/42", "key", {"job": {"enabled": False}}
    )


def test_unpublished_workflow_cannot_change_any_cron(monkeypatch):
    response = Mock(status_code=404)
    monkeypatch.setattr(job.requests, "get", Mock(return_value=response))
    api = Mock()
    monkeypatch.setattr(job, "cronjob_request", api)
    with pytest.raises(ValueError, match="default branch"):
        job.apply_plan(job.make_plan(now=NOW), api_key="key", token="test-token")
    api.assert_not_called()


def run(status="completed", **changes):
    return {
        "head_branch": "main",
        "event": "workflow_dispatch",
        "created_at": FINAL.isoformat(),
        "status": status,
        "conclusion": "success" if status == "completed" else None,
        "html_url": "https://example.test/run",
        **changes,
    }


@pytest.mark.parametrize(
    "status", ["queued", "in_progress", "waiting", "requested", "pending"]
)
def test_gate_waits_for_other_predictions_even_after_final_run(monkeypatch, status):
    polls = {name: 0 for name in wait.PREDICTION_WORKFLOWS}

    def read(repo, workflow, token):
        polls[workflow] += 1
        if workflow == "nba_predict_past_snapshots.yml":
            return [run()]
        # A different branch's queued TabPFN job must also finish.
        if workflow == "nba_predictor_daily_tabpfn_client.yml" and polls[workflow] == 1:
            return [run(status, head_branch="feature")]
        return []

    monkeypatch.setattr(wait, "read_runs", read)
    sleep = Mock()
    monkeypatch.setattr(wait.time, "sleep", sleep)
    result = wait.wait_for_run(
        job.LAST_PREDICTION, repo="test/repo", token="test-token"
    )
    assert result["status"] == "completed"
    sleep.assert_called_once_with(30)
    assert all(count == 2 for count in polls.values())


@pytest.mark.parametrize(
    "final_runs",
    [
        [],
        [run("in_progress")],
        [run(created_at="2026-10-31T04:59:59Z")],
        [run(head_branch="feature")],
        [run(created_at="2026-11-01T05:00:00Z")],
    ],
)
def test_missing_or_unfinished_final_run_times_out(monkeypatch, final_runs):
    monkeypatch.setattr(wait, "read_runs", lambda *args: final_runs)
    monkeypatch.setattr(wait.time, "monotonic", Mock(side_effect=[0, 0, 61]))
    monkeypatch.setattr(wait.time, "sleep", Mock())
    with pytest.raises(TimeoutError, match="no backfill was started"):
        wait.wait_for_run(
            job.LAST_PREDICTION, repo="test/repo", token="token", timeout_minutes=1
        )


def test_manual_workflow_without_dependency_needs_no_github_access(monkeypatch):
    api = Mock(side_effect=AssertionError("No API needed"))
    monkeypatch.setattr(wait, "read_runs", api)
    assert wait.main([]) == 0
    api.assert_not_called()


def test_dependency_gate_does_not_occupy_runner_or_daily_prediction_lock():
    workflow = yaml.load(
        Path(".github/workflows/nba_backfill_predictions.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    jobs = workflow["jobs"]
    assert jobs["wait_for_predictions"]["runs-on"] == "ubuntu-latest"
    assert "concurrency" not in jobs["wait_for_predictions"]
    assert workflow["concurrency"]["group"] != "daily-nba-predictor"
    assert jobs["prepare"]["needs"] == "wait_for_predictions"
    assert jobs["backfill"]["needs"] == "prepare"
    assert workflow["permissions"]["actions"] == "read"


def test_backfill_updates_scores_after_persisted_matrix_even_with_partial_failures():
    workflow = yaml.load(
        Path(".github/workflows/nba_backfill_predictions.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    results = workflow["jobs"]["update_results"]
    assert results["needs"] == "backfill"
    condition = results["if"]
    assert "!cancelled()" in condition and "inputs.persist" in condition
    assert "needs.backfill.result == 'success'" in condition
    assert "needs.backfill.result == 'failure'" in condition
    assert any(
        "scripts/prediction_store/update_game_results.py --recheck-days 0"
        in step.get("run", "")
        for step in results["steps"]
    )


def test_workflow_forwards_overwrite_only_to_backfill():
    workflow = yaml.load(
        Path(".github/workflows/nba_backfill_predictions.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert (
        workflow["on"]["workflow_dispatch"]["inputs"]["overwrite"]["default"] == "false"
    )
    backfill = workflow["jobs"]["backfill"]
    assert backfill["env"]["OVERWRITE"] == "${{ inputs.overwrite }}"
    assert any(
        "args+=(--overwrite)" in step.get("run", "") for step in backfill["steps"]
    )
    assert "OVERWRITE" not in workflow["jobs"]["prepare"]["env"]
