#!/usr/bin/env python3
"""Prepare the one-off October 20–25 backfill for October 31 at 06:20 Madrid.

Default: print an offline preview, without credentials or network access.
--apply books the cron on cron-job.org; it never dispatches a workflow now.
--overwrite replaces covered historical predictions after successful refits.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests
from nba_ou.scheduling.cronjob_org import (
    GITHUB_REF,
    GITHUB_REPO,
    REQUEST_METHOD_POST,
    cronjob_request,
    dispatch_url,
    github_headers,
    read_secret,
)

from scripts.cronjobs.wait_for_prediction_run import parse_dependency

MADRID = ZoneInfo("Europe/Madrid")
EASTERN = ZoneInfo("America/New_York")
WORKFLOW = "nba_backfill_predictions.yml"
TITLE_PREFIX = "Scorecast - historical backfill "
RUN_AT = "2026-10-31T06:20:00+01:00"
LAST_PREDICTION = "nba_predict_past_snapshots.yml@2026-10-31T05:00:00+00:00"


def make_plan(
    *,
    now,
    start_date="2026-10-20",
    end_date="2026-10-25",
    run_at=RUN_AT,
    after_prediction=LAST_PREDICTION,
    persist=True,
    overwrite=False,
    ref=GITHUB_REF,
):
    start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    if start > end or end >= now.astimezone(EASTERN).date():
        raise ValueError("Expected an ordered range of past game dates (Eastern)")
    at = datetime.fromisoformat(run_at.replace("Z", "+00:00"))
    if at.tzinfo is None or at.second or at.microsecond:
        raise ValueError("--run-at must include a timezone and specify a whole minute")
    at = at.astimezone(UTC)
    if at <= now:
        raise ValueError("--run-at must be in the future; this is a one-off cron")
    _, last_at = parse_dependency(after_prediction)
    if last_at > at:
        raise ValueError("Backfill cannot be scheduled before the final prediction")
    return {
        "run_at": at.isoformat(),
        "run_at_madrid": at.astimezone(MADRID).isoformat(),
        "workflow": WORKFLOW,
        "ref": ref,
        "inputs": {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "include_tabpfn": "true",
            "include_closing": "true",
            "tabpfn_only": "false",
            "tabpfn_horizon_minutes": "240",
            "persist": str(persist).lower(),
            "overwrite": str(overwrite).lower(),
            "after_prediction": after_prediction,
        },
    }


def job_payload(plan, token):
    at = datetime.fromisoformat(plan["run_at"])
    inputs = plan["inputs"]
    return {
        "title": (
            f"{TITLE_PREFIX}{inputs['start_date']}..{inputs['end_date']}"
            f" @ {plan['run_at']}"
            + (" overwrite" if inputs["overwrite"] == "true" else "")
        ),
        "url": dispatch_url(WORKFLOW),
        "enabled": True,
        "saveResponses": True,
        "requestTimeout": 30,
        "requestMethod": REQUEST_METHOD_POST,
        "schedule": {
            "timezone": "UTC",
            "expiresAt": int(at.strftime("%Y%m%d%H%M") + "59"),
            "hours": [at.hour],
            "minutes": [at.minute],
            "mdays": [at.day],
            "months": [at.month],
            "wdays": [-1],
        },
        "notification": {
            "onFailure": True,
            "onFailureCount": 1,
            "onSuccess": False,
            "onDisable": True,
        },
        "extendedData": {
            "headers": github_headers(token),
            "body": json.dumps({"ref": plan["ref"], "inputs": inputs}),
        },
    }


def check_workflow(token, *, ref=GITHUB_REF):
    response = requests.get(
        f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW}",
        headers=github_headers(token),
        timeout=30,
    )
    if response.status_code != 200 or response.json().get("state") != "active":
        raise ValueError(
            f"GitHub has not registered an active {WORKFLOW}. Publish the workflow "
            "on the default branch before --apply; no cron changes made"
        )
    # workflow_dispatch is registered on main, but executes the selected ref.
    response = requests.get(
        f"https://api.github.com/repos/{GITHUB_REPO}/contents/"
        f".github/workflows/{WORKFLOW}?ref={quote(ref, safe='')}",
        headers=github_headers(token),
        timeout=30,
    )
    if response.status_code != 200:
        raise ValueError(
            f"Publish {WORKFLOW} on ref {ref} before --apply; no cron changes made"
        )


def apply_plan(plan, *, api_key, token):
    check_workflow(token, ref=plan["ref"])
    payload = job_payload(plan, token)
    jobs = cronjob_request("GET", "/jobs", api_key)["jobs"]
    existing = [job for job in jobs if job.get("title") == payload["title"]]
    if len(existing) > 1:
        raise ValueError("Duplicate backfill cron titles; resolve before scheduling")
    if existing:
        job_id = existing[0]["jobId"]
        if existing[0].get("lastExecution", 0) > 0:
            raise ValueError(
                "This backfill cron has already fired; refusing to re-arm it"
            )
        cronjob_request("PATCH", f"/jobs/{job_id}", api_key, {"job": payload})
    else:
        job_id = cronjob_request("PUT", "/jobs", api_key, {"job": payload})["jobId"]
    booked = cronjob_request("GET", f"/jobs/{job_id}", api_key)
    details = booked["jobDetails"]
    expected = int(datetime.fromisoformat(plan["run_at"]).timestamp())
    if not details.get("enabled") or details.get("nextExecution") != expected:
        cronjob_request(
            "PATCH", f"/jobs/{job_id}", api_key, {"job": {"enabled": False}}
        )
        raise ValueError(
            "Cron did not confirm the expected execution time; disabled it"
        )
    return job_id


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2026-10-20")
    parser.add_argument("--end-date", default="2026-10-25")
    parser.add_argument("--run-at", default=RUN_AT, help="ISO timestamp with timezone")
    parser.add_argument("--after-prediction", default=LAST_PREDICTION)
    parser.add_argument(
        "--ref", default=GITHUB_REF, help="Published GitHub branch/tag to execute"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute and replace existing historical backfill predictions",
    )
    parser.add_argument(
        "--dry-run-predictions",
        action="store_true",
        help="Eventual workflow fits models but does not insert/archive predictions",
    )
    parser.add_argument(
        "--apply", action="store_true", help="Book the cron; default: offline preview"
    )
    args = parser.parse_args(argv)
    try:
        plan = make_plan(
            now=datetime.now(UTC),
            start_date=args.start_date,
            end_date=args.end_date,
            run_at=args.run_at,
            after_prediction=args.after_prediction,
            persist=not args.dry_run_predictions,
            overwrite=args.overwrite,
            ref=args.ref,
        )
        print(json.dumps(plan, indent=2))
        if args.apply:
            job_id = apply_plan(
                plan,
                api_key=read_secret("CRONJOB_API_KEY"),
                token=read_secret("CRONJOB_GITHUB_DISPATCH_TOKEN"),
            )
            print(f"Confirmed one-off cron {job_id}: {plan['run_at_madrid']}")
        else:
            print("Preview only. No cron booked, workflow dispatched or models fitted.")
        return 0
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
