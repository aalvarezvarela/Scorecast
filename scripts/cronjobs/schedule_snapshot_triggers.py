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

Also four TabPFN runs at T-600, T-480, T-360 and T-240 of the date's last
scheduled tip-off, with the fits of ``TABPFN_FITS``: T-600 and T-360 predict
line_error and total_points, standard only; T-480 adds spread_error and, on
the intermediate line only, thinking_high for line_error and total_points;
T-240 predicts all three targets, standard only.
T-480 and T-240 use the alternate API key. T-600, T-480 and T-360 predict intermediate
and closing in the same GitHub job; T-240 predicts closing only, always using
the currently available closing inputs for eligible pending games.
Intermediate reconstructs every game at its own snapshot of that horizon,
including games already started or finished. In joined mode these jobs run
inside their covering predictor runs; standalone dispatches fire
TABPFN_DELAY_MINUTES after the covering run (or the snapshot if none covers
it). Each job retains its snapshot cutoff. Snapshots before the daily booking
are left out. ``--no-tabpfn`` leaves these out.

After the last game, whose end is estimated as its tip-off plus
GAME_LENGTH_MINUTES: the date's past-snapshot predictions fire 30 min after
that end, and final scores two hours after it, capped at
[Schedule] FINISHED_MATCHES_LATEST_AT the following morning (07:50 by default).
setup_cronjobs.py also books an independent daily final-score retry at 11:11.

Left out: times already past, and times outside the game date in US/Pacific --
the predictor predicts the slate of its run's Pacific date, so a trigger before
Pacific midnight (03:00 ET; an early tip's T-1080, a noon game's T-720) would
predict the day before's games.

The tip-off is the scoreboard's ``gameTimeUTC``; the predictor times its
snapshots on the SBR page's tip-off, which agreed on every game of 2026-10-20
to 2026-10-22.

cron-job.org allows 100 API requests a day and five job creations a minute, so
the times are booked on a fixed pool of jobs (``Scorecast - snapshot trigger
NN``). Predictor jobs share a UTC date and minute of the hour; TabPFN jobs
have a separate payload per snapshot time and horizon. The pool is patched
every day; jobs not needed that day are
disabled. Schedules are in UTC (no daylight-saving gaps) and expire at the end
of their date, so a job left over from a failed day never fires again. After
booking, each job's ``nextExecution`` is read back and checked against the plan;
if the provider cannot predict it, the stored schedule must match exactly.

Secrets (dry run needs none): ``CRONJOB_API_KEY`` and
``CRONJOB_GITHUB_DISPATCH_TOKEN``, see ``nba_ou.scheduling.cronjob_org``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as clock_time
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
from nba_ou.scheduling.tabpfn_jobs import (
    DEFAULT_TARGETS,
    JOB_SEPARATOR,
    Fits,
    format_fits,
    format_job,
)


@dataclass(frozen=True)
class Dispatch:
    """A workflow to dispatch and its inputs (strings; the rest take defaults)."""

    workflow_file: str
    inputs: tuple[tuple[str, str], ...] = ()
    # Display only; it never becomes a workflow input.
    label: str = ""


# Every enabled slot.
PREDICTOR = Dispatch("nba_predictor_daily.yml")
TABPFN_WORKFLOW = "nba_predictor_daily_tabpfn_client.yml"
TOTALS_TARGETS = ("line_error", "total_points")


@dataclass(frozen=True)
class TabPFNFits:
    """The fits of one TabPFN run: ``(mode, targets)`` per mode on each line;
    no intermediate fits means a closing-only run."""

    intermediate: Fits
    closing: Fits


#: Lead time before the last tip-off -> that run's fits.
TABPFN_FITS = {
    600: TabPFNFits(
        intermediate=(("standard", TOTALS_TARGETS),),
        closing=(("standard", TOTALS_TARGETS),),
    ),
    480: TabPFNFits(
        intermediate=(
            ("standard", DEFAULT_TARGETS),
            ("thinking_high", TOTALS_TARGETS),
        ),
        closing=(("standard", DEFAULT_TARGETS),),
    ),
    360: TabPFNFits(
        intermediate=(("standard", TOTALS_TARGETS),),
        closing=(("standard", TOTALS_TARGETS),),
    ),
    240: TabPFNFits(intermediate=(), closing=(("standard", DEFAULT_TARGETS),)),
}
TABPFN_HORIZONS = tuple(TABPFN_FITS)
#: The runs whose fits use TABPFN_API_KEY_ALT.
TABPFN_ALT_KEY_HORIZONS = (480, 240)
# Every TabPFN snapshot is also a predictor snapshot: TabPFN fires this long
# after the predictor run that covers it, so that run enters the runner's queue
# first. ``as_of`` keeps the snapshot.
TABPFN_DELAY_MINUTES = 2
# Keep coinciding TabPFN dispatches at least one minute apart.
TABPFN_SPACING_MINUTES = 1
FINISHED_MATCHES = Dispatch("update_finished_matches_daily.yml")
PAST_SNAPSHOTS_WORKFLOW = "nba_predict_past_snapshots.yml"
# Tip-off to final buzzer: about 2 h 15 min, more with overtime.
GAME_LENGTH_MINUTES = 150
# After the last game's estimated end.
PAST_SNAPSHOTS_AFTER_END_MINUTES = 30
FINISHED_MATCHES_AFTER_END_MINUTES = 120
POOL_PREFIX = "Scorecast - snapshot trigger "
# TabPFN payloads differ by tip-off and horizon and cannot share a pool job.
# Leave room for both GETs under cron-job.org's 100-request daily allowance.
MAX_POOL = 90
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
    """One (pool index, jobId or None to create) per group, and the (pool index,
    jobId) of the jobs to disable."""

    assign: list[tuple[int, int | None]]
    disable: list[tuple[int, int]]


def pool_title(index: int, group: ScheduleGroup | None = None) -> str:
    title = f"{POOL_PREFIX}{index:02d}"
    if group is None:
        return title
    dispatch = group.dispatch
    inputs = dict(dispatch.inputs)
    if dispatch.workflow_file == TABPFN_WORKFLOW:
        if inputs["line"] == "closing":
            at = group.fire_times()[0].astimezone(MADRID)
            purpose = f"TabPFN closing {dispatch.label} | run {at:%d/%m %H:%M} Madrid"
        else:
            snapshot = datetime.fromisoformat(inputs["as_of"]).astimezone(MADRID)
            purpose = (
                f"TabPFN T-{inputs['horizon_minutes']} {dispatch.label} | "
                f"snapshot {snapshot:%d/%m %H:%M} Madrid"
            )
    elif dispatch.workflow_file == PREDICTOR.workflow_file:
        times = ", ".join(f"{at.astimezone(MADRID):%H:%M}" for at in group.fire_times())
        purpose = f"Predictor {times} Madrid"
        if "tabpfn" in inputs:
            purpose += f" + TabPFN {dispatch.label}"
    elif dispatch.workflow_file == PAST_SNAPSHOTS_WORKFLOW:
        purpose = f"Past snapshots {inputs['game_date']}"
    elif dispatch == FINISHED_MATCHES:
        purpose = "Final scores"
    else:
        purpose = dispatch.workflow_file
    return f"{title} | {purpose}"


def pool_index(title: str) -> int | None:
    rest = title[len(POOL_PREFIX) :] if title.startswith(POOL_PREFIX) else ""
    index = rest.partition(" | ")[0]
    return int(index) if index.isdigit() else None


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


@dataclass(frozen=True)
class TabPFNSnapshot:
    """One TabPFN run anchored before the last tip-off, with its anchor games
    and covering predictor run (None when none does). ``horizon`` is the
    dispatch lead time; T-240 runs closing rather than an intermediate slot.
    """

    at: datetime
    horizon: int
    game_ids: tuple[str, ...]
    label: str
    covered_by: datetime | None

    @property
    def fits(self) -> TabPFNFits:
        """This run's fits (``TABPFN_FITS``)."""
        return TABPFN_FITS[self.horizon]

    @property
    def alt_key(self) -> bool:
        """T-8 and T-4 fit with TABPFN_API_KEY_ALT."""
        return self.horizon in TABPFN_ALT_KEY_HORIZONS

    @property
    def intermediate(self) -> bool:
        """T-4 runs only closing; earlier runs also reconstruct intermediate."""
        return bool(self.fits.intermediate)

    def jobs(self) -> list[str]:
        """This run's jobs, as a predictor run carries them (``--tabpfn``)."""
        closing = format_job(0, fits=self.fits.closing)
        if not self.intermediate:
            return [closing]
        return [
            format_job(self.horizon, self.at, self.game_ids, self.fits.intermediate),
            closing,
        ]

    def dispatch(self, game_date: str) -> Dispatch:
        """The TabPFN workflow's dispatch of this snapshot."""
        first = self.fits.intermediate or self.fits.closing
        return Dispatch(
            TABPFN_WORKFLOW,
            (
                ("fits", format_fits(first)),
                ("line", "intermediate" if self.intermediate else "closing"),
                ("use_alt_key", str(self.alt_key).lower()),
                ("include_closing", str(self.intermediate).lower()),
                ("update_db", "false"),
                ("game_date", game_date),
            )
            + (
                (
                    ("closing_fits", format_fits(self.fits.closing)),
                    ("horizon_minutes", str(self.horizon)),
                    ("as_of", self.at.isoformat()),
                    ("snapshot_game_ids", " ".join(self.game_ids)),
                )
                if self.intermediate
                else ()
            ),
            label=self.label,
        )


def tabpfn_snapshots(
    games: pd.DataFrame,
    predictor_runs: Mapping[datetime, tuple[str, ...]] | None = None,
) -> list[TabPFNSnapshot]:
    """One per horizon of the date's last tip-off, in snapshot order.

    ``predictor_runs`` is ``trigger_times``' plan: it groups snapshots and fires
    each group at its last one, so a snapshot's predictor run can be up to
    ``GROUP_MINUTES`` after it. Games sharing a tip-off share each horizon's
    snapshot. Intermediate predicts earlier games at their own, already passed
    snapshot of the same horizon; closing uses current eligible pending games.
    Earlier games do not book extra runs.
    """
    valid = games.loc[
        ~games["game_status"]
        .fillna("")
        .astype(str)
        .str.upper()
        .map(lambda status: any(skip in status for skip in SKIP_STATUSES))
        & games["tipoff_utc"].notna()
    ]
    if valid.empty:
        return []
    games = valid.loc[valid["tipoff_utc"].eq(valid["tipoff_utc"].max())]
    covering = {
        reason: run
        for run, reasons in (predictor_runs or {}).items()
        for reason in reasons
    }
    tipoff = pd.Timestamp(games["tipoff_utc"].max())
    anchors = {
        str(game.game_id): f"{game.team_away}@{game.team_home}"
        for game in games.itertuples(index=False)
    }
    game_ids = tuple(sorted(anchors))
    label = ", ".join(anchors[game] for game in game_ids[:2])
    if len(game_ids) > 2:
        label += f" +{len(game_ids) - 2}"
    return [
        TabPFNSnapshot(
            at=(tipoff - timedelta(minutes=horizon))
            .to_pydatetime()
            .astimezone(UTC)
            .replace(second=0, microsecond=0),
            horizon=horizon,
            game_ids=game_ids,
            label=label,
            covered_by=max(
                (
                    covering[f"{matchup} T-{horizon}"]
                    for matchup in anchors.values()
                    if f"{matchup} T-{horizon}" in covering
                ),
                default=None,
            ),
        )
        for horizon in TABPFN_HORIZONS
    ]


def _spaced(runs: list[tuple[datetime, Dispatch]], now: datetime):
    """``runs`` still ahead, in order, each at least
    ``TABPFN_SPACING_MINUTES`` after the one before; the snapshot cutoffs of
    their inputs stay as they are."""
    ordered = sorted(
        ((fire, dispatch) for fire, dispatch in runs if fire > now + MIN_LEAD),
        key=lambda run: (
            run[0],
            dict(run[1].inputs).get("as_of", ""),
            -int(dict(run[1].inputs).get("horizon_minutes", "0")),
            run[1].label,
        ),
    )
    spaced: list[tuple[datetime, Dispatch]] = []
    for fire, dispatch in ordered:
        if spaced:
            fire = max(fire, spaced[-1][0] + timedelta(minutes=TABPFN_SPACING_MINUTES))
        spaced.append((fire, dispatch))
    return spaced


def tabpfn_runs(
    games: pd.DataFrame,
    game_date: str,
    now: datetime,
    predictor_runs: Mapping[datetime, tuple[str, ...]] | None = None,
) -> list[tuple[datetime, Dispatch]]:
    """Runs per (snapshot minute, horizon), fired ``TABPFN_DELAY_MINUTES``
    after the predictor run that covers the snapshot. Intermediate reconstructs
    every game of the slate, including started/final games; ``snapshot_game_ids``
    is anchor metadata, not a filter. T-240 runs closing on eligible pending games.

    A snapshot no predictor run covers fires after the snapshot itself.
    Coinciding dispatches are spaced at least one minute apart without
    changing their snapshot cutoffs. Runs already past are skipped. An
    explicit game date keeps a queued dispatch tied to its original slate even
    if the runner starts it after midnight.
    """
    delay = timedelta(minutes=TABPFN_DELAY_MINUTES)
    return _spaced(
        [
            (max(snap.at, snap.covered_by or snap.at) + delay, snap.dispatch(game_date))
            for snap in tabpfn_snapshots(games, predictor_runs)
        ],
        now,
    )


def joined_tabpfn_runs(
    games: pd.DataFrame,
    game_date: str,
    now: datetime,
    predictor_runs: Mapping[datetime, tuple[str, ...]],
) -> tuple[dict[datetime, Dispatch], list[tuple[datetime, Dispatch]]]:
    """The TabPFN runs as jobs of the predictor runs that cover them.

    Returns each such predictor run's dispatch, carrying its jobs in its
    ``tabpfn`` input (``nba_ou.prediction.tabpfn_run.parse_jobs``), and, as
    ``tabpfn_runs`` books them, the TabPFN runs no predictor run covers.
    """
    jobs: dict[datetime, list[TabPFNSnapshot]] = defaultdict(list)
    alone = []
    for snap in tabpfn_snapshots(games, predictor_runs):
        if snap.covered_by in predictor_runs:
            jobs[snap.covered_by].append(snap)
        else:
            alone.append((snap.at, snap.dispatch(game_date)))
    joined = {
        run: Dispatch(
            PREDICTOR.workflow_file,
            (
                (
                    "tabpfn",
                    # Each job names its own fits (FITS of its ``tabpfn`` item).
                    JOB_SEPARATOR.join(job for snap in snaps for job in snap.jobs()),
                ),
                # The four slate horizons are two hours apart, so their
                # covering predictor runs (grouped by <=30 min) are distinct.
                ("tabpfn_use_alt_key", str(snaps[0].alt_key).lower()),
            ),
            label="; ".join(f"T-{snap.horizon} {snap.label}" for snap in snaps),
        )
        for run, snaps in sorted(jobs.items())
    }
    delay = timedelta(minutes=TABPFN_DELAY_MINUTES)
    return joined, _spaced([(at + delay, dispatch) for at, dispatch in alone], now)


def post_game_runs(
    tipoffs: Iterable[tuple[str, datetime]], game_date: str, now: datetime
) -> list[tuple[datetime, Dispatch]]:
    """The runs after the last game's estimated end: the date's past-snapshot
    predictions, then the final scores, capped at the next morning's
    FINISHED_MATCHES_LATEST_AT in the configured timezone. Empty without
    games; a time already past is left out."""
    tips = [tipoff for _, tipoff in tipoffs]
    if not tips:
        return []
    end = max(tips) + timedelta(minutes=GAME_LENGTH_MINUTES)
    latest = datetime.combine(
        date.fromisoformat(game_date) + timedelta(days=1),
        clock_time.fromisoformat(
            SETTINGS.config.get(
                "Schedule", "FINISHED_MATCHES_LATEST_AT", fallback="07:50"
            ).strip()
        ),
        tzinfo=ZoneInfo(
            SETTINGS.config.get("Schedule", "TIMEZONE", fallback="Europe/Madrid")
        ),
    ).astimezone(UTC)
    runs = [
        (
            end + timedelta(minutes=PAST_SNAPSHOTS_AFTER_END_MINUTES),
            Dispatch(PAST_SNAPSHOTS_WORKFLOW, (("game_date", game_date),)),
        ),
        (
            min(end + timedelta(minutes=FINISHED_MATCHES_AFTER_END_MINUTES), latest),
            FINISHED_MATCHES,
        ),
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
        (i, pool[i]["jobId"] if i in pool else None) for i in range(1, n_groups + 1)
    ]
    disable = [
        (index, job["jobId"])
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
    for (index, job_id), group in zip(plan.assign, groups, strict=True):
        title = pool_title(index, group)
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
    # A disabled job loses its day's purpose: back to the bare pool title.
    for index, job_id in plan.disable:
        cronjob_request(
            "PATCH",
            f"/jobs/{job_id}",
            api_key,
            {"job": {"enabled": False, "title": pool_title(index)}},
        )
        print(f"Disabled job {job_id}")
    return job_ids


def booking_mismatches(
    groups: list[ScheduleGroup], job_ids: list[int], jobs: list[dict], now: datetime
) -> list[str]:
    """Verify enabled jobs against their next execution, or their exact stored
    schedule when cron-job.org cannot predict that execution."""
    by_id = {job["jobId"]: job for job in jobs}
    problems = []
    for group, job_id in zip(groups, job_ids, strict=True):
        expected = min(at for at in group.fire_times() if at > now)
        job = by_id.get(job_id, {})
        actual = job.get("nextExecution")
        # The provider's PHP predictor can skip a shorter month when adding
        # one month to the 31st (e.g. Oct 31 -> Dec 1), returning null despite
        # a valid booking. In that case verify every stored schedule field.
        if (
            job.get("enabled")
            and actual is None
            and job.get("schedule") == group.schedule()
        ):
            print(
                f"OK   job {job_id}: next run unavailable; stored schedule "
                f"matches plan (next {_clock(expected)})"
            )
            continue
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
    parser.add_argument(
        "--tabpfn",
        choices=("separate", "joined", "none"),
        default=SETTINGS.tabpfn_runs,
        help=(
            "TabPFN runs as dispatches of their own, as jobs of the predictor "
            "runs covering their snapshots, or none ([Schedule] TABPFN_RUNS, "
            f"now {SETTINGS.tabpfn_runs})"
        ),
    )
    parser.add_argument(
        "--no-tabpfn", action="store_true", help="book no TabPFN runs (--tabpfn none)"
    )
    args = parser.parse_args()
    tabpfn_mode = "none" if args.no_tabpfn else args.tabpfn

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
    tabpfn = []
    joined: dict[datetime, Dispatch] = {}
    if tabpfn_mode == "separate":
        tabpfn = tabpfn_runs(games, game_date, now, triggers)
    elif tabpfn_mode == "joined":
        joined, tabpfn = joined_tabpfn_runs(games, game_date, now, triggers)
    after_games = post_game_runs(tipoffs, game_date, now)
    # A predictor run carrying TabPFN jobs has inputs of its own: a job of its
    # own too, not one shared with other hours.
    groups = group_schedules(at for at in triggers if at not in joined)
    for at, dispatch in joined.items():
        groups += group_schedules([at], dispatch)
    for at, dispatch in tabpfn:
        groups += group_schedules([at], dispatch)
    for at, dispatch in after_games:
        groups += group_schedules([at], dispatch)
    groups.sort(key=lambda group: group.fire_times()[0])
    print(f"\n{len(triggers)} predictor run(s) still ahead:")
    for at, reasons in triggers.items():
        print(f"  {_clock(at)}  {', '.join(reasons)}")
        if at in joined:
            print(f"      + TabPFN jobs: {joined[at].label}")
    print(f"{len(tabpfn)} TabPFN run(s) still ahead:")
    for at, dispatch in tabpfn:
        inputs = dict(dispatch.inputs)
        if inputs["line"] == "closing":
            print(
                f"  {_clock(at)}  TabPFN closing ({inputs['fits']}): "
                f"eligible pending games (game date {inputs['game_date']})"
            )
            continue
        print(
            f"  {_clock(at)}  TabPFN ({inputs['fits']}) "
            f"T-{inputs['horizon_minutes']} (snapshot "
            f"{_clock(datetime.fromisoformat(inputs['as_of']))} of "
            f"{dispatch.label}): every game of the slate "
            f"(game date {inputs['game_date']}), then closing "
            f"({inputs['closing_fits']})"
        )
    for at, dispatch in after_games:
        print(f"After the last game: {dispatch.workflow_file} at {_clock(at)}")
    print(f"{len(groups)} job(s)")
    if len(groups) > MAX_POOL:
        sys.exit(f"{len(groups)} jobs needed, more than MAX_POOL={MAX_POOL}")

    if not args.apply:
        for i, group in enumerate(groups, start=1):
            print(
                f"  {pool_title(i, group)} -> {group.dispatch.workflow_file}: "
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
