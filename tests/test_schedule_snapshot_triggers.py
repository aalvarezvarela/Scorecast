"""The daily booking of snapshot-time predictor runs on cron-job.org."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
from nba_ou.modeling.registry_paths import ModelSlot

import scripts.cronjobs.schedule_snapshot_triggers as job
import scripts.cronjobs.setup_cronjobs as setup


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def _slot(horizon):
    return ModelSlot(schema_version="2_5", target="line_error", horizon_minutes=horizon)


def test_horizons_leave_out_closing_and_pooled_slots():
    slots = [_slot(0), _slot(360), _slot(30), _slot(360), _slot(None)]
    assert job.snapshot_horizons(slots) == [30, 360]


def test_snapshot_times_after_the_offset():
    tip = _utc("2026-10-22 23:00")
    triggers = job.trigger_times(
        [("CLE@PHI", tip)], [30, 120], _utc("2026-10-22 12:00"), 2
    )
    assert triggers == {
        _utc("2026-10-22 21:02"): ("CLE@PHI T-120",),
        _utc("2026-10-22 22:32"): ("CLE@PHI T-30",),
    }


def test_no_run_before_tip_off_other_than_the_snapshots():
    tip = _utc("2026-10-22 23:00")
    triggers = job.trigger_times([("A@B", tip)], [30, 60], _utc("2026-10-22 12:00"))
    assert list(triggers) == [_utc("2026-10-22 22:00"), _utc("2026-10-22 22:30")]


def test_past_snapshots_are_left_out():
    tip = _utc("2026-10-22 23:00")
    # T-120 (21:00) is past; T-30 (22:30) is still ahead.
    triggers = job.trigger_times([("A@B", tip)], [30, 120], _utc("2026-10-22 21:30"))
    assert triggers == {_utc("2026-10-22 22:30"): ("A@B T-30",)}
    # Less than MIN_LEAD ahead counts as past.
    almost = job.trigger_times([("A@B", tip)], [30], _utc("2026-10-22 22:29:30"))
    assert almost == {}


def test_predictor_window_is_the_pacific_date():
    # PDT (UTC-7) in October, PST (UTC-8) in December.
    assert job.predictor_window("2026-10-22") == (
        _utc("2026-10-22 07:00"),
        _utc("2026-10-23 07:00"),
    )
    assert job.predictor_window("2026-12-25") == (
        _utc("2026-12-25 08:00"),
        _utc("2026-12-26 08:00"),
    )


def test_triggers_before_pacific_midnight_are_left_out():
    # A noon-ET Christmas game (17:00 UTC): its T-720 is 05:00 UTC, 21:00 the
    # evening before in Pacific time, when the predictor still predicts the
    # day before's slate.
    tip = _utc("2026-12-25 17:00")
    window = job.predictor_window("2026-12-25")
    triggers = job.trigger_times(
        [("A@B", tip)], [720, 540], _utc("2026-12-24 12:00"), 0, window
    )
    # T-720 (05:00) is before Pacific midnight (08:00 UTC); T-540 is exactly it.
    assert triggers == {_utc("2026-12-25 08:00"): ("A@B T-540",)}


def test_games_sharing_a_tip_share_their_run():
    tip = _utc("2026-10-22 23:00")
    triggers = job.trigger_times(
        [("A@B", tip), ("C@D", tip)], [60], _utc("2026-10-22 12:00")
    )
    assert triggers == {_utc("2026-10-22 22:00"): ("A@B T-60", "C@D T-60")}


def test_snapshots_at_most_30_min_apart_share_the_last_ones_run():
    # T-240 snapshots at 19:00, 19:10, 19:30 and 19:45.
    tips = [
        ("A@B", _utc("2026-10-22 23:00")),
        ("C@D", _utc("2026-10-22 23:10")),
        ("E@F", _utc("2026-10-22 23:30")),
        ("G@H", _utc("2026-10-22 23:45")),
    ]
    triggers = job.trigger_times(tips, [240], _utc("2026-10-22 12:00"))
    # 19:00, 19:10 and 19:30 are within 30 min of the first: one run at 19:30.
    # 19:45 is 45 min after it: a group of its own.
    assert triggers == {
        _utc("2026-10-22 19:30"): ("A@B T-240", "C@D T-240", "E@F T-240"),
        _utc("2026-10-22 19:45"): ("G@H T-240",),
    }


def test_groups_never_span_more_than_30_minutes():
    # Snapshots every 10 min: 0..30 | 40..70 -- a chain is cut after 30.
    start = _utc("2026-10-22 15:00")
    items = [(start + timedelta(minutes=m), f"s{m}") for m in range(0, 80, 10)]
    groups = job.cluster_times(items)
    assert [[reason for _, reason in group] for group in groups] == [
        ["s0", "s10", "s20", "s30"],
        ["s40", "s50", "s60", "s70"],
    ]


def test_t30_keeps_its_own_time():
    tips = [("A@B", _utc("2026-10-22 23:00")), ("C@D", _utc("2026-10-22 23:15"))]
    triggers = job.trigger_times(tips, [30, 60], _utc("2026-10-22 12:00"))
    assert triggers == {
        # T-60 at 22:00 and 22:15: grouped, run at 22:15.
        _utc("2026-10-22 22:15"): ("A@B T-60", "C@D T-60"),
        # T-30 never grouped, even 15 min apart.
        _utc("2026-10-22 22:30"): ("A@B T-30",),
        _utc("2026-10-22 22:45"): ("C@D T-30",),
    }


def test_a_group_ending_on_a_t30_minute_is_the_same_run():
    # A@B's T-30 (22:30) and the group of T-60s at 22:00 / 22:30.
    tips = [("A@B", _utc("2026-10-22 23:00")), ("C@D", _utc("2026-10-22 23:30"))]
    triggers = job.trigger_times(tips, [30, 60], _utc("2026-10-22 12:00"))
    assert triggers[_utc("2026-10-22 22:30")] == ("A@B T-30", "A@B T-60", "C@D T-60")


def test_half_hour_tips_give_hourly_runs():
    # Tips every half hour: snapshots of one horizon 30 min apart pair up.
    tips = [
        (f"G{i}", _utc("2026-10-22 23:00") + timedelta(minutes=30 * i))
        for i in range(4)
    ]
    triggers = job.trigger_times(tips, [240], _utc("2026-10-22 12:00"))
    assert list(triggers) == [_utc("2026-10-22 19:30"), _utc("2026-10-22 20:30")]


def test_groups_fire_at_exactly_the_trigger_times():
    # A 7:00 PM ET tip, a 10:30 PM ET one (next UTC date) and an odd 7:10 PM.
    tips = [
        ("A@B", _utc("2026-10-22 23:00")),
        ("C@D", _utc("2026-10-23 02:30")),
        ("E@F", _utc("2026-10-22 23:10")),
    ]
    horizons = [30, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720]
    triggers = job.trigger_times(tips, horizons, _utc("2026-10-22 09:00"))
    groups = job.group_schedules(triggers)

    fired = [at for group in groups for at in group.fire_times()]
    assert sorted(fired) == sorted(triggers)
    assert len(fired) == len(set(fired))
    assert {group.day.isoformat() for group in groups} == {"2026-10-22", "2026-10-23"}
    firsts = [group.fire_times()[0] for group in groups]
    assert firsts == sorted(firsts)


def test_group_schedule_is_one_utc_date_that_expires_with_it():
    group = job.ScheduleGroup(
        day=pd.Timestamp("2026-10-23").date(), minute=2, hours=(0, 1)
    )
    assert group.schedule() == {
        "timezone": "UTC",
        "expiresAt": 20261023235959,
        "hours": [0, 1],
        "minutes": [2],
        "mdays": [23],
        "months": [10],
        "wdays": [-1],
    }


def test_game_tipoffs_skip_games_without_a_tip_time():
    games = pd.DataFrame(
        {
            "team_away": ["CLE", "DEN", "LAL"],
            "team_home": ["PHI", "OKC", "BOS"],
            "tipoff_utc": pd.to_datetime(
                ["2026-10-23 01:30", "2026-10-22 23:00", "2026-10-23 04:00"], utc=True
            ),
            "game_status": ["7:30 pm ET", "7:00 pm ET", "TBD"],
        }
    )
    tipoffs = job.game_tipoffs(games)
    assert [label for label, _ in tipoffs] == ["DEN@OKC", "CLE@PHI"]
    assert tipoffs[0][1] == _utc("2026-10-22 23:00")


def test_pool_patches_creates_and_disables_only_its_own_jobs():
    existing = [
        {"jobId": 11, "title": job.pool_title(1), "enabled": True},
        {"jobId": 13, "title": job.pool_title(3), "enabled": True},
        {"jobId": 14, "title": job.pool_title(4), "enabled": False},
        {"jobId": 15, "title": job.pool_title(5), "enabled": True},
        {"jobId": 99, "title": "Scorecast - update databases daily", "enabled": True},
        {"jobId": 98, "title": job.POOL_PREFIX + "x", "enabled": True},
    ]
    plan = job.plan_pool(existing, 3)
    assert plan.assign == [
        (job.pool_title(1), 11),
        (job.pool_title(2), None),
        (job.pool_title(3), 13),
    ]
    # 04 is already disabled; 99 and 98 are not pool jobs.
    assert plan.disable == [15]


def test_no_games_disables_the_whole_pool():
    existing = [{"jobId": 11, "title": job.pool_title(1), "enabled": True}]
    assert job.plan_pool(existing, 0) == job.PoolPlan(assign=[], disable=[11])


def test_payload_dispatches_the_daily_predictor():
    group = job.ScheduleGroup(
        day=pd.Timestamp("2026-10-22").date(), minute=32, hours=(14,)
    )
    payload = job.job_payload(job.pool_title(1), group, "secret")
    assert payload["url"].endswith(
        "/actions/workflows/nba_predictor_daily.yml/dispatches"
    )
    assert payload["requestMethod"] == 1
    assert payload["extendedData"]["body"] == '{"ref": "main"}'
    assert payload["extendedData"]["headers"]["Authorization"] == "Bearer secret"
    assert payload["schedule"] == group.schedule()


def test_apply_pool_spaces_creates_and_disables_surplus(monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(job.time, "sleep", sleeps.append)

    def fake_request(method, path, api_key, body=None):
        calls.append((method, path, body))
        return {"jobId": 50 + len(calls)} if method == "PUT" else {}

    monkeypatch.setattr(job, "cronjob_request", fake_request)
    day = pd.Timestamp("2026-10-22").date()
    groups = [job.ScheduleGroup(day=day, minute=m, hours=(20,)) for m in (2, 32, 50)]
    plan = job.PoolPlan(
        assign=[
            (job.pool_title(1), 11),
            (job.pool_title(2), None),
            (job.pool_title(3), None),
        ],
        disable=[15],
    )
    assert job.apply_pool(plan, groups, "key", "token") == [11, 52, 53]

    assert [(m, p) for m, p, _ in calls] == [
        ("PATCH", "/jobs/11"),
        ("PUT", "/jobs"),
        ("PUT", "/jobs"),
        ("PATCH", "/jobs/15"),
    ]
    assert calls[-1][2] == {"job": {"enabled": False}}
    # One wait, between the two creates.
    assert sleeps == [job.CREATE_SPACING_S]


def test_setup_cronjobs_payload_unchanged_by_the_shared_client():
    payload = setup.job_payload(setup.JOBS[0], "secret")
    assert payload["url"] == (
        "https://api.github.com/repos/aalvarezvarela/Scorecast"
        "/actions/workflows/update_database_daily.yml/dispatches"
    )
    assert payload["schedule"]["timezone"] == "Europe/Madrid"
    assert payload["extendedData"]["body"] == '{"ref": "main"}'
    assert setup.redacted(payload)["extendedData"]["headers"]["Authorization"] == (
        "Bearer <redacted>"
    )


def test_booking_check_compares_next_execution_with_the_plan():
    day = pd.Timestamp("2026-10-22").date()
    groups = [
        job.ScheduleGroup(day=day, minute=30, hours=(14, 20)),
        job.ScheduleGroup(day=day, minute=0, hours=(21,)),
    ]
    now = _utc("2026-10-22 15:00")
    booked = [
        # 14:30 has passed, so the next run is 20:30.
        {
            "jobId": 1,
            "enabled": True,
            "nextExecution": int(_utc("2026-10-22 20:30").timestamp()),
        },
        # Off by an hour, as a wrong time zone would be.
        {
            "jobId": 2,
            "enabled": True,
            "nextExecution": int(_utc("2026-10-22 22:00").timestamp()),
        },
    ]
    problems = job.booking_mismatches(groups, [1, 2], booked, now)
    assert len(problems) == 1 and problems[0].startswith("job 2:")

    booked[1]["nextExecution"] = int(_utc("2026-10-22 21:00").timestamp())
    assert job.booking_mismatches(groups, [1, 2], booked, now) == []
    booked[1]["enabled"] = False
    assert len(job.booking_mismatches(groups, [1, 2], booked, now)) == 1
