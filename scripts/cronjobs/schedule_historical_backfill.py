#!/usr/bin/env python3
"""Audit missing season dates and book a one-off backfill after tonight's predictions.

Dry run is the default. --apply creates/updates only this backfill's cron job;
--persist asks the eventual workflow to insert predictions. No prediction or
model is fitted by this scheduler. Secrets use the existing cron-job.org client.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from nba_ou.postgre_db.config.db_config import connect_nba_db, get_schema_name_games
from nba_ou.prediction.historical_backfill import backfill_dates
from nba_ou.scheduling.cronjob_org import (
    GITHUB_REF,
    GITHUB_REPO,
    REQUEST_METHOD_POST,
    cronjob_request,
    dispatch_url,
    github_headers,
    read_secret,
)
from nba_ou.utils.general_utils import get_season_year_from_date
from psycopg import sql

from scripts.cronjobs.wait_for_prediction_run import PREDICTION_WORKFLOWS

MADRID = ZoneInfo("Europe/Madrid")
EASTERN = ZoneInfo("America/New_York")
WORKFLOW = "nba_backfill_predictions.yml"
TITLE_PREFIX = "Scorecast - historical backfill "


def season_coverage(today):
    """Compare actual regular-season games with stored registry/TabPFN predictions.

    This identifies entirely unpredicted games/dates. The backfill itself checks
    exact spec/target/horizon coverage again before every fit.
    """
    season = int(get_season_year_from_date(today.isoformat()))
    with connect_nba_db("supabase") as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = 60000")
            schema = get_schema_name_games()
            cur.execute(
                sql.SQL(
                    "SELECT DISTINCT game_date::date, game_id FROM {}.{} WHERE season_year=%s AND game_date::date<%s AND game_id LIKE %s ORDER BY 1, 2"
                ).format(sql.Identifier(schema), sql.Identifier(schema)),
                (season, today, "002%"),
            )
            games = cur.fetchall()
    with connect_nba_db("aiven") as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = 60000")
            cur.execute(
                """SELECT DISTINCT g.game_id FROM prediction.games g
                JOIN prediction.inputs i USING(game_id)
                JOIN prediction.predictions p USING(input_id)
                JOIN prediction.model_builds m USING(model_build_id)
                WHERE g.game_date >= %s AND g.game_date < %s AND m.kind IN (1,3)
                  AND m.train_date_max < g.game_date""",
                (date(season, 10, 1), today),
            )
            predicted = {str(row[0]).zfill(10) for row in cur.fetchall()}
    return coverage_rows(games, predicted)


def coverage_rows(games, predicted):
    days = {}
    for day, game in games:
        row = days.setdefault(
            day,
            {
                "game_date": str(day),
                "games": 0,
                "predicted_games": 0,
                "missing_game_ids": [],
            },
        )
        row["games"] += 1
        game = str(game).zfill(10)
        if game in predicted:
            row["predicted_games"] += 1
        else:
            row["missing_game_ids"].append(game)
    return list(days.values())


def booked_prediction_times(jobs, game_date, *, now):
    """Expand every UTC/wall-clock cron hour; nextExecution alone hides later runs."""
    start = datetime.combine(date.fromisoformat(game_date), time(), tzinfo=EASTERN)
    stop = start + timedelta(days=1, hours=9)
    found = []
    for job in jobs:
        workflow = next(
            (
                name
                for name in PREDICTION_WORKFLOWS
                if job.get("url", "").endswith(f"/{name}/dispatches")
            ),
            None,
        )
        if not workflow or not job.get("enabled"):
            continue
        schedule = job["schedule"]
        zone = ZoneInfo(schedule["timezone"])
        local_day = start.astimezone(zone).date()
        while local_day <= stop.astimezone(zone).date():
            matches = all(
                -1 in schedule[key] or value in schedule[key]
                for key, value in [
                    ("months", local_day.month),
                    ("mdays", local_day.day),
                    ("wdays", (local_day.weekday() + 1) % 7),
                ]
            )
            if matches:
                hours = range(24) if -1 in schedule["hours"] else schedule["hours"]
                minutes = (
                    range(60) if -1 in schedule["minutes"] else schedule["minutes"]
                )
                for hour in hours:
                    for minute in minutes:
                        at = datetime.combine(
                            local_day, time(hour, minute), tzinfo=zone
                        )
                        expiry = schedule.get("expiresAt", 0)
                        if expiry and int(at.strftime("%Y%m%d%H%M%S")) > int(expiry):
                            continue
                        at = at.astimezone(UTC)
                        if start <= at < stop and at > now:
                            found.append(
                                {
                                    "at": at,
                                    "workflow": workflow,
                                    "job_id": job["jobId"],
                                    "title": job["title"],
                                }
                            )
            local_day += timedelta(days=1)
    return sorted(found, key=lambda row: row["at"])


def make_plan(
    coverage,
    jobs,
    *,
    game_date,
    now,
    delay_minutes=20,
    start_date=None,
    end_date=None,
    persist=False,
    tabpfn_only=False,
    ref=GITHUB_REF,
):
    if delay_minutes < 1:
        raise ValueError("delay-minutes must be positive")
    missing = [row["game_date"] for row in coverage if row["missing_game_ids"]]
    if start_date is None:
        if not missing:
            raise ValueError(
                "Every past game has predictions; no missing dates to schedule"
            )
        start_date, end_date = min(missing), max(missing)
    dates = backfill_dates(start_date, end_date, now=now)
    predictions = booked_prediction_times(jobs, game_date, now=now)
    if not predictions:
        raise ValueError(
            "No future prediction cron jobs found for this slate; book the daily schedule first"
        )
    last = predictions[-1]
    at = last["at"] + timedelta(minutes=delay_minutes)
    dependency = f"{last['workflow']}@{last['at'].isoformat()}"
    inputs = {
        "start_date": dates[0],
        "end_date": dates[-1],
        "include_tabpfn": "true",
        "include_closing": "true",
        "tabpfn_only": str(tabpfn_only).lower(),
        "tabpfn_horizon_minutes": "240",
        "persist": str(persist).lower(),
        "after_prediction": dependency,
    }
    return {
        "game_date": game_date,
        "coverage": coverage,
        "start_date": dates[0],
        "end_date": dates[-1],
        "last_prediction": {
            **last,
            "at": last["at"].isoformat(),
            "at_madrid": last["at"].astimezone(MADRID).isoformat(),
        },
        "run_at": at.isoformat(),
        "run_at_madrid": at.astimezone(MADRID).isoformat(),
        "workflow": WORKFLOW,
        "ref": ref,
        "inputs": inputs,
    }


def job_payload(plan, token):
    at = datetime.fromisoformat(plan["run_at"]).astimezone(UTC)
    return {
        "title": f"{TITLE_PREFIX}{plan['start_date']}..{plan['end_date']}",
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
            "body": json.dumps({"ref": plan["ref"], "inputs": plan["inputs"]}),
        },
    }


def check_workflow(token):
    response = requests.get(
        f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{WORKFLOW}",
        headers=github_headers(token),
        timeout=30,
    )
    if response.status_code != 200 or response.json().get("state") != "active":
        raise ValueError(
            f"GitHub has not registered an active {WORKFLOW}. Publish the workflow on the default branch before --apply; no cron changes made"
        )


def apply_plan(plan, jobs, *, api_key, token):
    check_workflow(token)
    payload = job_payload(plan, token)
    existing = [job for job in jobs if job.get("title") == payload["title"]]
    if len(existing) > 1:
        raise ValueError(
            "Duplicate backfill cron titles; resolve them before scheduling"
        )
    if existing:
        job_id = existing[0]["jobId"]
        # Do not re-arm a one-off dispatch that has already been sent.
        if existing[0].get("lastExecution", 0) >= int(
            datetime.fromisoformat(plan["run_at"]).timestamp()
        ):
            raise ValueError(
                "This backfill cron has already fired; refusing to re-arm it"
            )
        cronjob_request("PATCH", f"/jobs/{job_id}", api_key, {"job": payload})
    else:
        job_id = cronjob_request("PUT", "/jobs", api_key, {"job": payload})["jobId"]
    booked = cronjob_request("GET", f"/jobs/{job_id}", api_key)
    details = booked.get("jobDetails", booked.get("job", booked))
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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--game-date", help="Tonight's NBA slate, Eastern; default: today"
    )
    parser.add_argument("--start-date", help="Override automatic missing-date range")
    parser.add_argument("--end-date")
    parser.add_argument("--delay-minutes", type=int, default=20)
    parser.add_argument("--ref", default=GITHUB_REF)
    parser.add_argument("--tabpfn-only", action="store_true")
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Insert the eventual backfill predictions",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Create/update the one-off cron; default: preview only",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/backfill_schedule/plan.json")
    )
    args = parser.parse_args(argv)
    if args.end_date and not args.start_date:
        parser.error("--end-date requires --start-date")
    now = datetime.now(UTC)
    game_date = args.game_date or now.astimezone(EASTERN).date().isoformat()
    try:
        api_key = read_secret("CRONJOB_API_KEY")
        jobs = cronjob_request("GET", "/jobs", api_key)["jobs"]
        plan = make_plan(
            season_coverage(now.astimezone(EASTERN).date()),
            jobs,
            game_date=game_date,
            now=now,
            delay_minutes=args.delay_minutes,
            start_date=args.start_date,
            end_date=args.end_date,
            persist=args.persist,
            tabpfn_only=args.tabpfn_only,
            ref=args.ref,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(plan, indent=2))
        print(json.dumps(plan, indent=2))
        if args.apply:
            job_id = apply_plan(
                plan,
                jobs,
                api_key=api_key,
                token=read_secret("CRONJOB_GITHUB_DISPATCH_TOKEN"),
            )
            print(f"Confirmed one-off cron {job_id}: {plan['run_at_madrid']}")
        else:
            print(
                "Preview only. --apply creates the cron; --persist enables prediction insertion."
            )
        return 0
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
