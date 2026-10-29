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
patched, a missing one is created. Add a workflow to the jobs of
``read_schedule``; a job taken out of them goes in ``RETIRED_TITLES``, which ``--apply``
disables (a job left on cron-job.org would keep firing).

The day, on the one self-hosted runner, at the times of ``[Schedule]`` in
``config.ini`` (08:00, 09:35 and 11:11 Madrid by default):

    UPDATE_AND_RETRAIN_AT    database update, then -- on a game day -- refit,
                             smoke test and promotion
    SCHEDULE_PREDICTIONS_AT  booking of the day's runs
                             (schedule_snapshot_triggers.py): the predictor at
                             every snapshot time, TabPFN for all three targets
                             intermediate + closing at T-600/480/360 of the
                             last game and closing only at T-240,
                             and after the last game's
                             estimated end the past-snapshot predictions
                             (+30 min) and final scores (+2 h, capped at
                             FINISHED_MATCHES_LATEST_AT the following morning)
    FINISHED_MATCHES_RETRY_AT independent daily final-score retry

A changed time reaches cron-job.org only with ``--apply``. The repository
variables ``DAILY_UPDATE_ENABLED`` and ``PREDICTIONS_ENABLED`` switch the
workflows off without touching the jobs (docs/serving/snapshot_triggers.md).

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
import time
from configparser import ConfigParser
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import time as clock_time
from zoneinfo import ZoneInfo

import requests
from nba_ou.config.settings import SETTINGS
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

PACIFIC = ZoneInfo("US/Pacific")
# Pause between write calls: cron-job.org limits PUT /jobs to five a minute
# and throttles calls in quick succession (one --apply once stopped after
# creating the first job).
API_SPACING_S = 13
# (month, day) the jobs stop running: after the Finals, before Summer League.
SEASON_END = (7, 10)


@dataclass(frozen=True)
class WorkflowJob:
    title: str
    workflow_file: str
    hour: int
    minute: int
    # workflow_dispatch inputs, as strings; the ones left out take their default.
    inputs: dict[str, str] = field(default_factory=dict)


def parse_clock(text: str) -> tuple[int, int]:
    """``HH:MM`` as ``(hour, minute)``."""
    hour, sep, minute = text.strip().partition(":")
    if not (sep and hour.isdigit() and minute.isdigit() and len(minute) == 2):
        raise ValueError(f"expected HH:MM, got {text!r}")
    if not (0 <= int(hour) <= 23 and 0 <= int(minute) <= 59):
        raise ValueError(f"not a time of day: {text!r}")
    return int(hour), int(minute)


def before_pacific_midnight(
    hour: int, minute: int, timezone: str, year: int
) -> list[date]:
    """Days of ``year`` on which ``hour:minute`` in ``timezone`` still falls on
    the previous US/Pacific date. The scheduler books the slate of its run's
    Eastern date, and the predictor predicts its run's Pacific date, so a
    booking before Pacific midnight would plan the wrong night."""
    zone = ZoneInfo(timezone)
    day, late = date(year, 1, 1), []
    while day.year == year:
        at = datetime.combine(day, clock_time(hour, minute), tzinfo=zone)
        if at.astimezone(PACIFIC).date() != day:
            late.append(day)
        day += timedelta(days=1)
    return late


def read_schedule(config: ConfigParser) -> tuple[str, list[WorkflowJob]]:
    """``[Schedule]`` of ``config.ini``: the timezone and the daily jobs."""
    timezone = config.get("Schedule", "TIMEZONE", fallback="Europe/Madrid").strip()
    ZoneInfo(timezone)  # fails on a misspelt zone
    update = parse_clock(config.get("Schedule", "UPDATE_AND_RETRAIN_AT"))
    booking = parse_clock(config.get("Schedule", "SCHEDULE_PREDICTIONS_AT"))
    retry = parse_clock(
        config.get("Schedule", "FINISHED_MATCHES_RETRY_AT", fallback="11:11")
    )
    jobs = [
        # The database update runs as the retrain workflow's first job, so the
        # refit never reads a half-updated database; the refit then promotes
        # what passes the smoke test.
        WorkflowJob(
            title="Scorecast - update databases and retrain",
            workflow_file="retrain_prediction_models.yml",
            hour=update[0],
            minute=update[1],
            inputs={"update_database": "true", "promote": "true"},
        ),
        # Books the day's snapshot-time runs (one-off jobs in UTC, see
        # schedule_snapshot_triggers.py). It must run after Pacific midnight,
        # when the game date is already the predictor's (check_schedule).
        WorkflowJob(
            title="Scorecast - schedule snapshot predictions",
            workflow_file="schedule_snapshot_triggers.yml",
            hour=booking[0],
            minute=booking[1],
        ),
        WorkflowJob(
            title="Scorecast - retry final scores",
            workflow_file="update_finished_matches_daily.yml",
            hour=retry[0],
            minute=retry[1],
        ),
    ]
    return timezone, jobs


def check_schedule(jobs: list[WorkflowJob], timezone: str, today: date) -> None:
    """Exit if the booking job would run before Pacific midnight on some day of
    this year or the next (a season spans both)."""
    for job in jobs:
        if job.workflow_file != "schedule_snapshot_triggers.yml":
            continue
        late = [
            day
            for year in (today.year, today.year + 1)
            for day in before_pacific_midnight(job.hour, job.minute, timezone, year)
        ]
        if late:
            sys.exit(
                f"[Schedule] SCHEDULE_PREDICTIONS_AT {job.hour:02d}:{job.minute:02d} "
                f"{timezone} falls before US/Pacific midnight on {len(late)} day(s) "
                f"(first {late[0]}): the scheduler would book the "
                "previous night. Pick a later time."
            )


TIMEZONE, JOBS = read_schedule(SETTINGS.config)

# Jobs taken out of JOBS: --apply disables them if they still exist.
RETIRED_TITLES = [
    # Replaced by the retrain job, whose first job is the same update.
    "Scorecast - update databases daily",
]

# Workflows dispatched by the jobs schedule_snapshot_triggers.py books, which
# the token must also reach.
BOOKED_WORKFLOWS = [
    "nba_predictor_daily.yml",
    "nba_predictor_daily_tabpfn_client.yml",
    "nba_predict_past_snapshots.yml",
    "update_finished_matches_daily.yml",
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
            "body": dispatch_body(job.inputs),
        },
    }


def check_token(github_token: str) -> bool:
    """Read-only check that the token sees each workflow; it dispatches nothing."""
    ok = True
    for workflow_file in [job.workflow_file for job in JOBS] + BOOKED_WORKFLOWS:
        url = (
            f"https://api.github.com/repos/{GITHUB_REPO}"
            f"/actions/workflows/{workflow_file}"
        )
        resp = requests.get(url, headers=github_headers(github_token), timeout=30)
        if resp.status_code == 200:
            print(f"OK   {workflow_file}: state={resp.json().get('state')}")
        else:
            ok = False
            print(f"FAIL {workflow_file}: HTTP {resp.status_code} {resp.text[:200]}")
    return ok


def apply_jobs(api_key: str, github_token: str) -> None:
    existing = {
        j["title"]: j["jobId"] for j in cronjob_request("GET", "/jobs", api_key)["jobs"]
    }
    for job in JOBS:
        time.sleep(API_SPACING_S)
        payload = {"job": job_payload(job, github_token)}
        if job.title in existing:
            job_id = existing[job.title]
            cronjob_request("PATCH", f"/jobs/{job_id}", api_key, payload)
            print(f"Updated job {job_id}: {job.title}")
        else:
            job_id = cronjob_request("PUT", "/jobs", api_key, payload)["jobId"]
            print(f"Created job {job_id}: {job.title}")
        print(f"  {job.hour:02d}:{job.minute:02d} {TIMEZONE} -> {dispatch_url(job)}")
    for title in RETIRED_TITLES:
        if title in existing:
            job_id = existing[title]
            time.sleep(API_SPACING_S)
            cronjob_request(
                "PATCH", f"/jobs/{job_id}", api_key, {"job": {"enabled": False}}
            )
            print(f"Disabled retired job {job_id}: {title}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true", help="create/update the jobs")
    action.add_argument(
        "--check-token", action="store_true", help="read-only GitHub token check"
    )
    args = parser.parse_args()

    check_schedule(JOBS, TIMEZONE, date.today())
    github_token = read_secret("CRONJOB_GITHUB_DISPATCH_TOKEN")
    if args.check_token:
        sys.exit(0 if check_token(github_token) else 1)
    if args.apply:
        apply_jobs(read_secret("CRONJOB_API_KEY"), github_token)
        return
    for job in JOBS:
        print(json.dumps(redacted(job_payload(job, github_token)), indent=2))
    for title in RETIRED_TITLES:
        print(f"Would disable, if it exists: {title}")
    print("\nDry run. Re-run with --apply to create/update these jobs.")


if __name__ == "__main__":
    main()
