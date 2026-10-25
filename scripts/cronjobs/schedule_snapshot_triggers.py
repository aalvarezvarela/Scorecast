#!/usr/bin/env python3
"""Book today's snapshot-time runs of the daily predictor on cron-job.org.

    python scripts/cronjobs/schedule_snapshot_triggers.py            # dry run
    python scripts/cronjobs/schedule_snapshot_triggers.py --apply    # book them

Reads the date's games from the NBA scoreboard and books dispatches of
``nba_predictor_daily.yml``, each predicting every enabled slot (closing and
intermediate), at the snapshot times of every game: tip-off minus each enabled
horizon, plus ``--offset-minutes``. Snapshots at most ``GROUP_MINUTES`` (30)
apart share one run, fired at the last of them, so every snapshot of the group
has passed. The T-30 snapshot (``EXACT_HORIZONS``) always gets a run at its own
time.

Also one TabPFN run (``nba_predictor_daily_tabpfn_client.yml``, ``TABPFN``:
line_error on its T-360 intermediate slot, every game) between 6 h 30 and 5 h
before the first tip-off (``TABPFN_WINDOW_MINUTES``), at the minute of that
window farthest from every predictor run, so neither waits long for the other
on the one runner; on a tie, the minute nearest the first game's T-360. The
later games read their ticks so far. ``--no-tabpfn`` leaves it out.

And, after the last game, whose end is estimated as its tip-off plus
``GAME_LENGTH_MINUTES``: the past-snapshot predictions of the date
(``nba_predict_past_snapshots.yml``, ``game_date`` passed) 30 min after it, and
the final scores (``update_finished_matches_daily.yml``) two hours after it.

Left out: times already past, and times outside the game date in US/Pacific --
the predictor predicts the slate of its run's Pacific date, so a trigger before
Pacific midnight (03:00 ET; an early tip's T-1080, a noon game's T-720) would
predict the day before's games.

The tip-off is the scoreboard's ``gameTimeUTC``; the predictor times its
snapshots on the SBR page's tip-off, which agreed on every game of 2026-10-20
to 2026-10-22.

cron-job.org allows 100 API requests a day and five job creations a minute, so
the times are booked on a fixed pool of jobs (``Scorecast - snapshot trigger
NN``), one per UTC date and minute of the hour, each firing at that minute of
several hours. The pool is patched every day; jobs not needed that day are
disabled. Schedules are in UTC (no daylight-saving gaps) and expire at the end
of their date, so a job left over from a failed day never fires again. After
booking, each job's ``nextExecution`` is read back and checked against the plan.

Secrets (dry run needs none): ``CRONJOB_API_KEY`` and
``CRONJOB_GITHUB_DISPATCH_TOKEN``, see ``nba_ou.scheduling.cronjob_org``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from nba_ou.config.settings import SETTINGS
from nba_ou.fetch_data.nba_schedule.fetch_nba_schedule import fetch_tipoffs_for_date
from nba_ou.scheduling.cronjob_org import (
    REQUEST_METHOD_POST,
    cronjob_request,
    dispatch_body,
    dispatch_url,
    github_headers,
    read_secret,
)


@dataclass(frozen=True)
class Dispatch:
    """A workflow to dispatch and its inputs (strings; the rest take defaults)."""

    workflow_file: str
    inputs: tuple[tuple[str, str], ...] = ()


# Every enabled slot.
PREDICTOR = Dispatch("nba_predictor_daily.yml")
# TabPFN line_error on the T-360 slot; the predictor runs keep the databases
# current, so no update of its own.
TABPFN = Dispatch(
    "nba_predictor_daily_tabpfn_client.yml",
    (
        ("target", "line_error"),
        ("line", "intermediate"),
        ("horizon_minutes", "360"),
        ("update_db", "false"),
    ),
)
# Minutes before the first tip-off the TabPFN run may fall in (earliest, latest).
TABPFN_WINDOW_MINUTES = (390, 300)
# Its slot's horizon: ties go to the minute nearest the first game's snapshot.
TABPFN_HORIZON_MINUTES = 360
FINISHED_MATCHES = Dispatch("update_finished_matches_daily.yml")
PAST_SNAPSHOTS_WORKFLOW = "nba_predict_past_snapshots.yml"
# Tip-off to final buzzer: about 2 h 15 min, more with overtime.
GAME_LENGTH_MINUTES = 150
# After the last game's estimated end.
PAST_SNAPSHOTS_AFTER_END_MINUTES = 30
FINISHED_MATCHES_AFTER_END_MINUTES = 120
POOL_PREFIX = "Scorecast - snapshot trigger "
# A busy night needs about 10 jobs; more than this means something is off.
MAX_POOL = 30
EASTERN = ZoneInfo("America/New_York")
# The zone of the predictor's game date (scripts/predict_nba_games.py).
PACIFIC = ZoneInfo("US/Pacific")
MADRID = ZoneInfo("Europe/Madrid")
# The run builds its rows minutes after the dispatch, so the snapshot has
# passed by then without an offset.
DEFAULT_OFFSET_MINUTES = 0
# Snapshots at most this far apart share one run, fired at the last of them.
GROUP_MINUTES = 30
# Horizons that always get a run at their own time, never grouped.
EXACT_HORIZONS = (30,)
# A trigger closer than this may pass before cron-job.org has it.
MIN_LEAD = timedelta(minutes=1)
# PUT /jobs is limited to five a minute.
CREATE_SPACING_S = 13
# Scoreboard statuses of games without a real tip-off time.
SKIP_STATUSES = ("TBD", "PPD", "POSTPONED", "CANCEL")


@dataclass(frozen=True)
class ScheduleGroup:
    """The runs of one workflow on one UTC date that share a minute of the hour."""

    day: date
    minute: int
    hours: tuple[int, ...]
    dispatch: Dispatch = PREDICTOR

    def fire_times(self) -> list[datetime]:
        return [
            datetime(
                self.day.year,
                self.day.month,
                self.day.day,
                hour,
                self.minute,
                tzinfo=UTC,
            )
            for hour in self.hours
        ]

    def schedule(self) -> dict:
        return {
            "timezone": "UTC",
            "expiresAt": int(self.day.strftime("%Y%m%d") + "235959"),
            "hours": list(self.hours),
            "minutes": [self.minute],
            "mdays": [self.day.day],
            "months": [self.day.month],
            "wdays": [-1],
        }


@dataclass(frozen=True)
class PoolPlan:
    """One (title, jobId or None to create) per group, and the jobs to disable."""

    assign: list[tuple[str, int | None]]
    disable: list[int]


def pool_title(index: int) -> str:
    return f"{POOL_PREFIX}{index:02d}"


def pool_index(title: str) -> int | None:
    rest = title[len(POOL_PREFIX) :] if title.startswith(POOL_PREFIX) else ""
    return int(rest) if rest.isdigit() else None


def snapshot_horizons(slots: Iterable) -> list[int]:
    """The enabled slots' horizons before tip-off; closing and pooled left out."""
    return sorted({slot.horizon_minutes for slot in slots if slot.horizon_minutes})


def game_tipoffs(games: pd.DataFrame) -> list[tuple[str, datetime]]:
    """``(AWAY@HOME, tipoff UTC)`` per game with a real tip-off time."""
    tipoffs = []
    for game in games.itertuples(index=False):
        label = f"{game.team_away}@{game.team_home}"
        status = str(game.game_status or "").upper()
        if any(skip in status for skip in SKIP_STATUSES):
            print(f"! {label} skipped: status {game.game_status!r}")
            continue
        tipoffs.append((label, pd.Timestamp(game.tipoff_utc).to_pydatetime()))
    return sorted(tipoffs, key=lambda item: item[1])


def predictor_window(game_date: str) -> tuple[datetime, datetime]:
    """The UTC span in which the predictor's Pacific date is ``game_date``."""
    start = datetime.combine(date.fromisoformat(game_date), datetime.min.time())
    start = start.replace(tzinfo=PACIFIC)
    return start.astimezone(UTC), (start + timedelta(days=1)).astimezone(UTC)


def cluster_times(
    items: Iterable[tuple[datetime, str]], minutes: int = GROUP_MINUTES
) -> list[list[tuple[datetime, str]]]:
    """``(time, reason)`` items in groups spanning at most ``minutes``: a group
    starts at the earliest item left and takes every item up to
    ``start + minutes``."""
    groups: list[list[tuple[datetime, str]]] = []
    for at, reason in sorted(items):
        if groups and at - groups[-1][0][0] <= timedelta(minutes=minutes):
            groups[-1].append((at, reason))
        else:
            groups.append([(at, reason)])
    return groups


def trigger_times(
    tipoffs: Iterable[tuple[str, datetime]],
    horizons: Iterable[int],
    now: datetime,
    offset_minutes: int = DEFAULT_OFFSET_MINUTES,
    window: tuple[datetime, datetime] | None = None,
    group_minutes: int = GROUP_MINUTES,
) -> dict[datetime, tuple[str, ...]]:
    """UTC run minute -> the snapshots it stands for, for the snapshots still
    ahead and, with ``window``, within ``[start, end)``.

    Snapshots of ``EXACT_HORIZONS`` keep their own time; the others are grouped
    (``cluster_times``) and each group fires at its last snapshot. Runs at the
    same minute are one run.
    """
    exact: list[tuple[datetime, str]] = []
    grouped: list[tuple[datetime, str]] = []
    for label, tipoff in tipoffs:
        for horizon in horizons:
            at = tipoff - timedelta(minutes=horizon - offset_minutes)
            at = at.astimezone(UTC).replace(second=0, microsecond=0)
            reason = f"{label} T-{horizon}"
            if at <= now + MIN_LEAD:
                continue
            if window is not None and not window[0] <= at < window[1]:
                print(
                    f"! {reason} at {_clock(at)} skipped: outside the predictor's date"
                )
                continue
            (exact if horizon in EXACT_HORIZONS else grouped).append((at, reason))

    runs: dict[datetime, tuple[str, ...]] = defaultdict(tuple)
    for at, reason in exact:
        runs[at] += (reason,)
    for group in cluster_times(grouped, group_minutes):
        runs[group[-1][0]] += tuple(reason for _, reason in group)
    return dict(sorted(runs.items()))


def tabpfn_time(
    tipoffs: Iterable[tuple[str, datetime]],
    now: datetime,
    predictor_runs: Iterable[datetime] = (),
    window: tuple[datetime, datetime] | None = None,
) -> datetime | None:
    """The TabPFN run: the minute of ``TABPFN_WINDOW_MINUTES`` before the first
    tip-off farthest from every predictor run, on a tie the one nearest the
    first game's T-360. Minutes already past or outside ``window`` are left
    out; None when there is no game or no minute is left."""
    tips = [tipoff for _, tipoff in tipoffs]
    if not tips:
        return None
    first = min(tips).astimezone(UTC).replace(second=0, microsecond=0)
    earliest, latest = (first - timedelta(minutes=m) for m in TABPFN_WINDOW_MINUTES)
    minutes = int((latest - earliest) / timedelta(minutes=1))
    candidates = [
        at
        for at in (earliest + timedelta(minutes=i) for i in range(minutes + 1))
        if at > now + MIN_LEAD and (window is None or window[0] <= at < window[1])
    ]
    if not candidates:
        print(
            f"! TabPFN between {_clock(earliest)} and {_clock(latest)} skipped: "
            "already past or outside the predictor's date"
        )
        return None
    runs = list(predictor_runs)
    snapshot = first - timedelta(minutes=TABPFN_HORIZON_MINUTES)

    def clearance(at: datetime) -> timedelta:
        return min((abs(at - run) for run in runs), default=timedelta.max)

    # Equally far from the runs and the snapshot: the later minute, past it.
    return max(candidates, key=lambda at: (clearance(at), -abs(at - snapshot), at))


def post_game_runs(
    tipoffs: Iterable[tuple[str, datetime]], game_date: str, now: datetime
) -> list[tuple[datetime, Dispatch]]:
    """The runs after the last game's estimated end: the date's past-snapshot
    predictions, then the final scores. Empty without games; a time already
    past is left out."""
    tips = [tipoff for _, tipoff in tipoffs]
    if not tips:
        return []
    end = max(tips) + timedelta(minutes=GAME_LENGTH_MINUTES)
    runs = [
        (
            end + timedelta(minutes=PAST_SNAPSHOTS_AFTER_END_MINUTES),
            Dispatch(PAST_SNAPSHOTS_WORKFLOW, (("game_date", game_date),)),
        ),
        (end + timedelta(minutes=FINISHED_MATCHES_AFTER_END_MINUTES), FINISHED_MATCHES),
    ]
    kept = []
    for at, dispatch in runs:
        at = at.astimezone(UTC).replace(second=0, microsecond=0)
        if at <= now + MIN_LEAD:
            print(f"! {dispatch.workflow_file} at {_clock(at)} skipped: already past")
            continue
        kept.append((at, dispatch))
    return kept


def group_schedules(
    times: Iterable[datetime], dispatch: Dispatch = PREDICTOR
) -> list[ScheduleGroup]:
    """One group per (UTC date, minute): it fires at exactly ``times``."""
    hours: dict[tuple[date, int], set[int]] = defaultdict(set)
    for at in times:
        at = at.astimezone(UTC)
        hours[(at.date(), at.minute)].add(at.hour)
    groups = [
        ScheduleGroup(
            day=day, minute=minute, hours=tuple(sorted(hrs)), dispatch=dispatch
        )
        for (day, minute), hrs in hours.items()
    ]
    return sorted(groups, key=lambda group: group.fire_times()[0])


def plan_pool(existing_jobs: list[dict], n_groups: int) -> PoolPlan:
    """Patch pool jobs 01..n, create the missing ones, disable the rest."""
    pool = {
        index: job
        for job in existing_jobs
        if (index := pool_index(job.get("title", ""))) is not None
    }
    assign = [
        (pool_title(i), pool[i]["jobId"] if i in pool else None)
        for i in range(1, n_groups + 1)
    ]
    disable = [
        job["jobId"]
        for index, job in sorted(pool.items())
        if index > n_groups and job.get("enabled", True)
    ]
    return PoolPlan(assign=assign, disable=disable)


def job_payload(title: str, group: ScheduleGroup, github_token: str) -> dict:
    return {
        "title": title,
        "url": dispatch_url(group.dispatch.workflow_file),
        "enabled": True,
        "saveResponses": True,
        "requestTimeout": 30,
        "requestMethod": REQUEST_METHOD_POST,
        "schedule": group.schedule(),
        "notification": {
            "onFailure": True,
            "onFailureCount": 1,
            "onSuccess": False,
            "onDisable": False,
        },
        "extendedData": {
            "headers": github_headers(github_token),
            "body": dispatch_body(dict(group.dispatch.inputs)),
        },
    }


def apply_pool(
    plan: PoolPlan, groups: list[ScheduleGroup], api_key: str, github_token: str
) -> list[int]:
    """Book ``groups`` on the pool; the job id of each group, in order."""
    job_ids = []
    created = 0
    for (title, job_id), group in zip(plan.assign, groups, strict=True):
        payload = {"job": job_payload(title, group, github_token)}
        if job_id is not None:
            cronjob_request("PATCH", f"/jobs/{job_id}", api_key, payload)
            print(f"Updated job {job_id}: {title}")
        else:
            if created:
                time.sleep(CREATE_SPACING_S)
            job_id = cronjob_request("PUT", "/jobs", api_key, payload)["jobId"]
            created += 1
            print(f"Created job {job_id}: {title}")
        job_ids.append(job_id)
    for job_id in plan.disable:
        cronjob_request(
            "PATCH", f"/jobs/{job_id}", api_key, {"job": {"enabled": False}}
        )
        print(f"Disabled job {job_id}")
    return job_ids


def booking_mismatches(
    groups: list[ScheduleGroup], job_ids: list[int], jobs: list[dict], now: datetime
) -> list[str]:
    """Jobs whose ``nextExecution`` (from ``GET /jobs``) is not the group's next
    planned time."""
    by_id = {job["jobId"]: job for job in jobs}
    problems = []
    for group, job_id in zip(groups, job_ids, strict=True):
        expected = min(at for at in group.fire_times() if at > now)
        job = by_id.get(job_id, {})
        actual = job.get("nextExecution")
        if not job.get("enabled") or actual != int(expected.timestamp()):
            shown = datetime.fromtimestamp(actual, UTC) if actual else actual
            problems.append(
                f"job {job_id}: next run {shown}, planned {expected} "
                f"(enabled={job.get('enabled')})"
            )
        else:
            print(f"OK   job {job_id}: next run {_clock(expected)}")
    return problems


def _clock(at: datetime) -> str:
    return f"{at:%m-%d %H:%M} UTC / {at.astimezone(MADRID):%m-%d %H:%M} Madrid"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--game-date",
        default=None,
        help="YYYY-MM-DD (default: today's Eastern date)",
    )
    parser.add_argument("--apply", action="store_true", help="book the jobs")
    parser.add_argument(
        "--offset-minutes",
        type=int,
        default=DEFAULT_OFFSET_MINUTES,
        help="fire this long after each snapshot",
    )
    parser.add_argument("--no-tabpfn", action="store_true", help="book no TabPFN run")
    args = parser.parse_args()

    now = datetime.now(UTC)
    game_date = args.game_date or now.astimezone(EASTERN).date().isoformat()
    horizons = snapshot_horizons(SETTINGS.prediction_model_slots)
    print(f"Game date {game_date}; horizons {horizons}; now {_clock(now)}")

    games = fetch_tipoffs_for_date(game_date)
    if games is None:
        sys.exit(f"Scoreboard unavailable for {game_date}; the pool is left as it is")
    tipoffs = game_tipoffs(games)
    for label, tipoff in tipoffs:
        print(f"  {label:<9} tip {_clock(tipoff)}")

    window = predictor_window(game_date)
    triggers = trigger_times(tipoffs, horizons, now, args.offset_minutes, window)
    tabpfn_at = None if args.no_tabpfn else tabpfn_time(tipoffs, now, triggers, window)
    after_games = post_game_runs(tipoffs, game_date, now)
    groups = group_schedules(triggers)
    if tabpfn_at is not None:
        groups += group_schedules([tabpfn_at], TABPFN)
    for at, dispatch in after_games:
        groups += group_schedules([at], dispatch)
    groups.sort(key=lambda group: group.fire_times()[0])
    print(f"\n{len(triggers)} predictor run(s) still ahead:")
    for at, reasons in triggers.items():
        print(f"  {_clock(at)}  {', '.join(reasons)}")
    if tabpfn_at is not None:
        gap = min((abs(tabpfn_at - at) for at in triggers), default=None)
        print(
            f"TabPFN line_error T-360: {_clock(tabpfn_at)}"
            + (
                f" ({gap // timedelta(minutes=1)} min from a predictor run)"
                if gap
                else ""
            )
        )
    for at, dispatch in after_games:
        print(f"After the last game: {dispatch.workflow_file} at {_clock(at)}")
    print(f"{len(groups)} job(s)")
    if len(groups) > MAX_POOL:
        sys.exit(f"{len(groups)} jobs needed, more than MAX_POOL={MAX_POOL}")

    if not args.apply:
        for i, group in enumerate(groups, start=1):
            print(
                f"  {pool_title(i)} -> {group.dispatch.workflow_file}: "
                f"{json.dumps(group.schedule())}"
            )
        print("\nDry run. Re-run with --apply to book these jobs.")
        return

    api_key = read_secret("CRONJOB_API_KEY")
    github_token = read_secret("CRONJOB_GITHUB_DISPATCH_TOKEN")
    existing = cronjob_request("GET", "/jobs", api_key)["jobs"]
    job_ids = apply_pool(
        plan_pool(existing, len(groups)), groups, api_key, github_token
    )

    booked = cronjob_request("GET", "/jobs", api_key)["jobs"]
    problems = booking_mismatches(groups, job_ids, booked, datetime.now(UTC))
    if problems:
        sys.exit("Booked jobs disagree with the plan:\n  " + "\n  ".join(problems))


if __name__ == "__main__":
    main()
