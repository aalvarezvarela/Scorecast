#!/usr/bin/env python3
"""Create or update the cron-job.org jobs that dispatch this repo's workflows.

    python scripts/cronjobs/setup_cronjobs.py                 # dry run: show the jobs
    python scripts/cronjobs/setup_cronjobs.py --check-token   # can the token see each workflow?
    python scripts/cronjobs/setup_cronjobs.py --apply         # create or update them

GitHub's ``schedule:`` trigger is UTC-only, often late and sometimes skipped, so
the daily workflows are instead started by cron-job.org, which calls the
``workflow_dispatch`` endpoint at a wall-clock time in ``Europe/Madrid``
(daylight saving handled for us). Each workflow must keep ``workflow_dispatch``.

Jobs are matched by title, so ``--apply`` is idempotent: an existing job is
patched, a missing one is created. Add a workflow by appending to ``JOBS``.

The jobs expire at the end of ``SEASON_END`` (10 July) and simply stop running;
nothing needs deleting. Re-run ``--apply`` before the next season starts: it
sets the next season's expiry and re-enables the jobs.

Secrets: ``CRONJOB_API_KEY`` and ``CRONJOB_GITHUB_DISPATCH_TOKEN``, read by
``nba_ou.scheduling.cronjob_org.read_secret`` (environment, then
``config.secrets.ini``).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date

import requests
from nba_ou.scheduling.cronjob_org import (
    GITHUB_REPO,
    REQUEST_METHOD_POST,
    cronjob_request,
    dispatch_body,
    github_headers,
    read_secret,
    redacted,
)
from nba_ou.scheduling.cronjob_org import dispatch_url as workflow_dispatch_url

TIMEZONE = "Europe/Madrid"
# (month, day) the jobs stop running: after the Finals, before Summer League.
SEASON_END = (7, 10)


@dataclass(frozen=True)
class WorkflowJob:
    title: str
    workflow_file: str
    hour: int
    minute: int


JOBS = [
    WorkflowJob(
        title="Scorecast - update databases daily",
        workflow_file="update_database_daily.yml",
        hour=9,
        minute=0,
    ),
]


def season_expiry(today: date | None = None) -> int:
    """The next SEASON_END, as cron-job.org's YYYYMMDDhhmmss in TIMEZONE."""
    today = today or date.today()
    end = date(today.year, *SEASON_END)
    if today > end:
        end = date(today.year + 1, *SEASON_END)
    return int(end.strftime("%Y%m%d") + "235959")


def dispatch_url(job: WorkflowJob) -> str:
    return workflow_dispatch_url(job.workflow_file)


def job_payload(job: WorkflowJob, github_token: str) -> dict:
    return {
        "title": job.title,
        "url": dispatch_url(job),
        "enabled": True,
        "saveResponses": True,
        "requestTimeout": 30,
        "requestMethod": REQUEST_METHOD_POST,
        "schedule": {
            "timezone": TIMEZONE,
            "expiresAt": season_expiry(),
            "hours": [job.hour],
            "minutes": [job.minute],
            "mdays": [-1],
            "months": [-1],
            "wdays": [-1],
        },
        "notification": {
            "onFailure": True,
            "onFailureCount": 1,
            "onSuccess": False,
            "onDisable": True,
        },
        "extendedData": {
            "headers": github_headers(github_token),
            "body": dispatch_body(),
        },
    }


def check_token(github_token: str) -> bool:
    """Read-only check that the token sees each workflow; it dispatches nothing."""
    ok = True
    for job in JOBS:
        url = (
            f"https://api.github.com/repos/{GITHUB_REPO}"
            f"/actions/workflows/{job.workflow_file}"
        )
        resp = requests.get(url, headers=github_headers(github_token), timeout=30)
        if resp.status_code == 200:
            print(f"OK   {job.workflow_file}: state={resp.json().get('state')}")
        else:
            ok = False
            print(
                f"FAIL {job.workflow_file}: HTTP {resp.status_code} {resp.text[:200]}"
            )
    return ok


def apply_jobs(api_key: str, github_token: str) -> None:
    existing = {
        j["title"]: j["jobId"] for j in cronjob_request("GET", "/jobs", api_key)["jobs"]
    }
    for job in JOBS:
        payload = {"job": job_payload(job, github_token)}
        if job.title in existing:
            job_id = existing[job.title]
            cronjob_request("PATCH", f"/jobs/{job_id}", api_key, payload)
            print(f"Updated job {job_id}: {job.title}")
        else:
            job_id = cronjob_request("PUT", "/jobs", api_key, payload)["jobId"]
            print(f"Created job {job_id}: {job.title}")
        print(f"  {job.hour:02d}:{job.minute:02d} {TIMEZONE} -> {dispatch_url(job)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true", help="create/update the jobs")
    action.add_argument(
        "--check-token", action="store_true", help="read-only GitHub token check"
    )
    args = parser.parse_args()

    github_token = read_secret("CRONJOB_GITHUB_DISPATCH_TOKEN")
    if args.check_token:
        sys.exit(0 if check_token(github_token) else 1)
    if args.apply:
        apply_jobs(read_secret("CRONJOB_API_KEY"), github_token)
        return
    for job in JOBS:
        print(json.dumps(redacted(job_payload(job, github_token)), indent=2))
    print("\nDry run. Re-run with --apply to create/update these jobs.")


if __name__ == "__main__":
    main()
