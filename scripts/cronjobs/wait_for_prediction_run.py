#!/usr/bin/env python3
"""Wait for the scheduled final prediction, without occupying the NBA runner."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime, timedelta
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

PREDICTION_WORKFLOWS = {
    "nba_predictor_daily.yml",
    "nba_predictor_daily_tabpfn_client.yml",
    "nba_predict_past_snapshots.yml",
}


def parse_dependency(value):
    workflow, separator, stamp = value.partition("@")
    if not separator or workflow not in PREDICTION_WORKFLOWS:
        raise ValueError("Expected prediction-workflow.yml@scheduled-UTC-timestamp")
    at = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    if at.tzinfo is None:
        raise ValueError("Scheduled prediction timestamp must include its timezone")
    return workflow, at.astimezone(UTC)


def matching_run(runs, after, *, branch="main"):
    """Ignore earlier/manual runs and predictions for the following night."""
    candidates = [
        run
        for run in runs
        if run.get("head_branch") == branch
        and run.get("event") == "workflow_dispatch"
        and after
        <= datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
        < after + timedelta(hours=4)
    ]
    return min(candidates, key=lambda run: run["created_at"]) if candidates else None


def read_runs(repo, workflow, token):
    request = Request(
        f"https://api.github.com/repos/{repo}/actions/workflows/{quote(workflow)}/runs?{urlencode({'per_page': 100, 'event': 'workflow_dispatch'})}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)["workflow_runs"]


def wait_for_run(
    dependency, *, repo, token, timeout_minutes=120, poll_seconds=30, branch="main"
):
    workflow, after = parse_dependency(dependency)
    deadline = time.monotonic() + timeout_minutes * 60
    while time.monotonic() < deadline:
        run = matching_run(read_runs(repo, workflow, token), after, branch=branch)
        if run and run["status"] == "completed":
            print(
                f"Final prediction completed ({run.get('conclusion')}): {run['html_url']}",
                flush=True,
            )
            return run
        print(
            f"Waiting for {workflow} scheduled at {after.isoformat()}: {run['status'] if run else 'not dispatched yet'}",
            flush=True,
        )
        time.sleep(poll_seconds)
    raise TimeoutError(
        "Final prediction did not finish in time; no backfill was started"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after", default="", help="workflow.yml@scheduled-timestamp")
    parser.add_argument(
        "--branch", default="main", help="Branch of the scheduled prediction"
    )
    args = parser.parse_args(argv)
    if not args.after:
        print("No scheduled prediction dependency requested")
        return 0
    wait_for_run(
        args.after,
        repo=os.environ["GITHUB_REPOSITORY"],
        token=os.environ["GITHUB_TOKEN"],
        branch=args.branch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
