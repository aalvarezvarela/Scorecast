"""The daily booking of snapshot-time predictor runs on cron-job.org."""

from __future__ import annotations

import configparser
import json
from datetime import UTC, date, datetime, timedelta
from unittest.mock import Mock
from zoneinfo import ZoneInfoNotFoundError

import pandas as pd
import pytest
from nba_ou.modeling.registry_paths import ModelSlot

import scripts.check_games_today as check
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
    assert plan.assign == [(1, 11), (2, None), (3, 13)]
    # 04 is already disabled; 99 and 98 are not pool jobs.
    assert plan.disable == [(5, 15)]


def test_no_games_disables_the_whole_pool():
    existing = [{"jobId": 11, "title": job.pool_title(1), "enabled": True}]
    assert job.plan_pool(existing, 0) == job.PoolPlan(assign=[], disable=[(1, 11)])


def test_pool_reuses_legacy_and_descriptive_titles_without_duplicates():
    group = job.ScheduleGroup(day=date(2026, 10, 22), minute=0, hours=(20,))
    existing = [
        {"jobId": 11, "title": job.pool_title(1), "enabled": True},
        {"jobId": 12, "title": job.pool_title(2, group), "enabled": True},
        {"jobId": 13, "title": job.pool_title(3, group), "enabled": True},
        {"jobId": 99, "title": "Other job | Predictor", "enabled": True},
    ]
    plan = job.plan_pool(existing, 2)
    assert plan.assign == [(1, 11), (2, 12)]
    assert plan.disable == [(3, 13)]
    assert job.pool_index(job.pool_title(2, group)) == 2
    assert job.pool_index(job.POOL_PREFIX + "x | Predictor") is None
    assert job.pool_index("Other job | Predictor") is None


def test_predictor_title_lists_its_madrid_fire_times():
    group = job.ScheduleGroup(day=date(2026, 10, 26), minute=0, hours=(21, 22, 23))
    assert job.pool_title(1, group) == (
        "Scorecast - snapshot trigger 01 | Predictor 22:00, 23:00, 00:00 Madrid"
    )


def test_postgame_titles_describe_the_workflow():
    day = date(2026, 10, 22)
    past = job.Dispatch(job.PAST_SNAPSHOTS_WORKFLOW, (("game_date", "2026-10-22"),))
    for dispatch, purpose in (
        (past, "Past snapshots 2026-10-22"),
        (job.FINISHED_MATCHES, "Final scores"),
    ):
        group = job.ScheduleGroup(day=day, minute=0, hours=(20,), dispatch=dispatch)
        assert job.pool_title(1, group).endswith(purpose)


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
    plan = job.PoolPlan(assign=[(1, 11), (2, None), (3, None)], disable=[(5, 15)])
    assert job.apply_pool(plan, groups, "key", "token") == [11, 52, 53]

    assert [(m, p) for m, p, _ in calls] == [
        ("PATCH", "/jobs/11"),
        ("PUT", "/jobs"),
        ("PUT", "/jobs"),
        ("PATCH", "/jobs/15"),
    ]
    # Each job is titled with its day's purpose; a disabled one gets the bare title.
    assert calls[0][2]["job"]["title"] == job.pool_title(1, groups[0])
    assert calls[-1][2] == {"job": {"enabled": False, "title": job.pool_title(5)}}
    # One wait, between the two creates.
    assert sleeps == [job.CREATE_SPACING_S]


def _setup_job(workflow_file):
    (found,) = [j for j in setup.JOBS if j.workflow_file == workflow_file]
    return found


def _configured_time(option: str) -> tuple[int, int]:
    clock = datetime.strptime(
        setup.SETTINGS.config.get("Schedule", option).strip(), "%H:%M"
    )
    return clock.hour, clock.minute


def test_setup_cronjobs_payload_dispatches_on_main_in_configured_timezone():
    payload = setup.job_payload(_setup_job("schedule_snapshot_triggers.yml"), "secret")
    assert payload["url"] == (
        "https://api.github.com/repos/aalvarezvarela/Scorecast"
        "/actions/workflows/schedule_snapshot_triggers.yml/dispatches"
    )
    assert (
        payload["schedule"]["timezone"]
        == setup.SETTINGS.config.get("Schedule", "TIMEZONE").strip()
    )
    hour, minute = _configured_time("SCHEDULE_PREDICTIONS_AT")
    assert (payload["schedule"]["hours"], payload["schedule"]["minutes"]) == (
        [hour],
        [minute],
    )
    assert payload["extendedData"]["body"] == '{"ref": "main"}'
    assert setup.redacted(payload)["extendedData"]["headers"]["Authorization"] == (
        "Bearer <redacted>"
    )


def test_setup_cronjobs_retrain_updates_the_database_and_promotes():
    job = _setup_job("retrain_prediction_models.yml")
    assert (job.hour, job.minute) == _configured_time("UPDATE_AND_RETRAIN_AT")
    body = json.loads(setup.job_payload(job, "secret")["extendedData"]["body"])
    assert body == {
        "ref": "main",
        "inputs": {"update_database": "true", "promote": "true"},
    }


def test_setup_cronjobs_daily_jobs_and_retired_update_job():
    times = {job.workflow_file: (job.hour, job.minute) for job in setup.JOBS}
    assert times == {
        "retrain_prediction_models.yml": _configured_time("UPDATE_AND_RETRAIN_AT"),
        "schedule_snapshot_triggers.yml": _configured_time("SCHEDULE_PREDICTIONS_AT"),
    }
    # The standalone update would run beside the retrain's own.
    assert setup.RETIRED_TITLES == ["Scorecast - update databases daily"]
    # The scheduler books the post-game workflows instead of fixed times.
    assert {
        "nba_predict_past_snapshots.yml",
        "update_finished_matches_daily.yml",
    } <= set(setup.BOOKED_WORKFLOWS)


def _schedule_config(update="08:00", booking="09:30", timezone="Europe/Madrid"):
    config = configparser.ConfigParser()
    config["Schedule"] = {
        "TIMEZONE": timezone,
        "UPDATE_AND_RETRAIN_AT": update,
        "SCHEDULE_PREDICTIONS_AT": booking,
    }
    return config


def test_setup_cronjobs_reads_the_daily_times_from_config():
    timezone, jobs = setup.read_schedule(_schedule_config("08:15", "09:45"))
    assert timezone == "Europe/Madrid"
    assert {j.workflow_file: (j.hour, j.minute) for j in jobs} == {
        "retrain_prediction_models.yml": (8, 15),
        "schedule_snapshot_triggers.yml": (9, 45),
    }


@pytest.mark.parametrize("text", ["9", "9:5", "24:00", "09:60", "nine"])
def test_setup_cronjobs_rejects_a_malformed_time(text):
    with pytest.raises(ValueError):
        setup.read_schedule(_schedule_config(update=text))


def test_setup_cronjobs_rejects_an_unknown_timezone():
    with pytest.raises(ZoneInfoNotFoundError):
        setup.read_schedule(_schedule_config(timezone="Europe/Madird"))


def test_booking_must_follow_pacific_midnight_all_year():
    # 09:00 Madrid is Pacific midnight in most weeks and 01:00 in the weeks
    # when only one side has changed its clocks.
    assert setup.before_pacific_midnight(9, 0, "Europe/Madrid", 2026) == []
    late = setup.before_pacific_midnight(8, 30, "Europe/Madrid", 2026)
    assert date(2026, 1, 15) in late and date(2026, 3, 10) not in late
    timezone, jobs = setup.read_schedule(_schedule_config(booking="08:30"))
    with pytest.raises(SystemExit, match="before US/Pacific midnight"):
        setup.check_schedule(jobs, timezone, date(2026, 10, 26))
    # The retrain time is not checked.
    timezone, jobs = setup.read_schedule(_schedule_config(update="05:00"))
    setup.check_schedule(jobs, timezone, date(2026, 10, 26))


def test_post_game_runs_follow_the_last_games_estimated_end():
    tips = [("A@B", _utc("2026-10-22 23:00")), ("C@D", _utc("2026-10-23 02:30"))]
    runs = job.post_game_runs(tips, "2026-10-22", _utc("2026-10-22 08:00"))
    # Last tip 02:30 UTC + 150 min = 05:00 estimated end.
    assert [at for at, _ in runs] == [
        _utc("2026-10-23 05:30"),
        _utc("2026-10-23 07:00"),
    ]
    past, finished = (dispatch for _, dispatch in runs)
    assert past.workflow_file == "nba_predict_past_snapshots.yml"
    assert dict(past.inputs) == {"game_date": "2026-10-22"}
    assert finished == job.FINISHED_MATCHES
    assert job.post_game_runs([], "2026-10-22", _utc("2026-10-22 08:00")) == []
    # Already past (a scheduler run late in the night): left out.
    assert len(job.post_game_runs(tips, "2026-10-22", _utc("2026-10-23 05:45"))) == 1


def test_post_game_payload_passes_the_game_date():
    ((at, dispatch),) = job.post_game_runs(
        [("A@B", _utc("2026-10-22 23:00"))], "2026-10-22", _utc("2026-10-23 02:05")
    )
    (group,) = job.group_schedules([at], dispatch)
    payload = job.job_payload(job.pool_title(1), group, "secret")
    assert payload["url"].endswith(
        "/actions/workflows/update_finished_matches_daily.yml/dispatches"
    )
    assert json.loads(payload["extendedData"]["body"]) == {"ref": "main"}
    assert group.fire_times() == [_utc("2026-10-23 03:30")]


def test_refit_only_on_a_game_day(monkeypatch):
    frames = {
        "2026-10-22": pd.DataFrame({"game_id": ["0022600006"]}),
        "2026-07-15": pd.DataFrame({"game_id": []}),
        "2026-10-23": None,  # scoreboard unavailable
    }
    monkeypatch.setattr(check, "fetch_tipoffs_for_date", frames.get)
    assert check.should_refit("2026-10-22") is True
    assert check.should_refit("2026-07-15") is False
    assert check.should_refit("2026-10-23") is True


def _tabpfn_games():
    return pd.DataFrame(
        {
            "game_id": ["0022600001", "0022600002", "0022600003"],
            "team_away": ["A", "C", "E"],
            "team_home": ["B", "D", "F"],
            "tipoff_utc": pd.to_datetime(
                [
                    "2026-10-22 23:00",
                    "2026-10-22 23:00",
                    "2026-10-23 01:00",
                ],
                utc=True,
            ),
            "game_status": ["7:00 pm ET", "7:00 pm ET", "9:00 pm ET"],
        }
    )


#: ``_tabpfn_games``' matchups, as a TabPFN dispatch's label names them.
TABPFN_GAME_IDS = {"A@B": "0022600001", "C@D": "0022600002", "E@F": "0022600003"}


def _snapshot_games(dispatch) -> list[str]:
    """The game IDs whose snapshot a TabPFN dispatch is; its label names them."""
    games = dict(dispatch.inputs)["snapshot_game_ids"].split()
    assert games == [TABPFN_GAME_IDS[label] for label in dispatch.label.split(", ")]
    return games


def test_tabpfn_runs_each_game_at_each_exact_horizon_with_both_targets():
    games = _tabpfn_games()
    now = _utc("2026-10-22 08:00")
    runs = job.tabpfn_runs(games, "2026-10-22", now)
    assert len(runs) == 8  # two tipoff times, four horizons, two games share a tip
    expected = {
        (str(game.game_id), horizon): game.tipoff_utc.to_pydatetime()
        - timedelta(minutes=horizon)
        for game in games.itertuples()
        for horizon in (720, 480, 360, 240)
    }
    actual = {}
    for at, dispatch in runs:
        inputs = dict(dispatch.inputs)
        assert dispatch.workflow_file == job.TABPFN_WORKFLOW
        assert inputs["target"] == "line_error total_points"
        assert inputs["line"] == "intermediate"
        assert inputs["update_db"] == "false"
        assert inputs["game_date"] == "2026-10-22"
        # Fired after the predictor run of the same snapshot; as_of keeps it.
        snapshot = datetime.fromisoformat(inputs["as_of"])
        assert at >= snapshot + timedelta(minutes=job.TABPFN_DELAY_MINUTES)
        # Every game not started is predicted; no game IDs pin the run, the
        # snapshot's games ride along as metadata.
        assert "game_ids" not in inputs
        horizon = int(inputs["horizon_minutes"])
        for game_id in _snapshot_games(dispatch):
            assert (game_id, horizon) not in actual
            actual[(game_id, horizon)] = snapshot
    assert actual == expected
    # Coinciding snapshots keep separate payloads but fire a minute apart.
    coinciding = [
        (at, dispatch)
        for at, dispatch in runs
        if dict(dispatch.inputs)["as_of"] == "2026-10-22T17:00:00+00:00"
    ]
    assert [at for at, _ in coinciding] == [
        _utc("2026-10-22 17:02"),
        _utc("2026-10-22 17:03"),
    ]
    assert {dict(d.inputs)["horizon_minutes"] for _, d in coinciding} == {"360", "480"}
    assert len({d.label for _, d in coinciding}) == 2


def test_tabpfn_nearby_tipoffs_are_not_delayed_or_grouped():
    games = _tabpfn_games().iloc[:2].copy()
    games.loc[1, "tipoff_utc"] += pd.Timedelta(minutes=15)
    runs = job.tabpfn_runs(games, "2026-10-22", _utc("2026-10-22 08:00"))
    assert len(runs) == 8
    assert [at for at, _ in runs[:2]] == [
        _utc("2026-10-22 11:02"),
        _utc("2026-10-22 11:17"),
    ]


def test_tabpfn_fires_after_the_grouped_predictor_run_of_its_snapshot():
    # Tips at 23:00 and 23:30: the predictor groups the 17:00 and 17:30
    # snapshots and fires at 17:30, so the 23:00 game's T-360 TabPFN waits for
    # that run instead of firing at 17:02 ahead of it.
    games = _tabpfn_games().iloc[:2].copy()
    games.loc[1, "tipoff_utc"] = pd.Timestamp("2026-10-22 23:30", tz="UTC")
    now = _utc("2026-10-22 08:00")
    tipoffs = job.game_tipoffs(games)
    predictor = job.trigger_times(tipoffs, [720, 480, 360, 240, 30], now)
    assert predictor[_utc("2026-10-22 17:30")] == ("A@B T-360", "C@D T-360")
    runs = job.tabpfn_runs(games, "2026-10-22", now, predictor)
    t360 = [
        (at, dict(dispatch.inputs)["as_of"], dispatch.label)
        for at, dispatch in runs
        if dict(dispatch.inputs)["horizon_minutes"] == "360"
    ]
    assert t360 == [
        (_utc("2026-10-22 17:32"), "2026-10-22T17:00:00+00:00", "A@B"),
        (_utc("2026-10-22 17:33"), "2026-10-22T17:30:00+00:00", "C@D"),
    ]
    covering = {reason: run for run, reasons in predictor.items() for reason in reasons}
    for at, dispatch in runs:
        inputs = dict(dispatch.inputs)
        for label in dispatch.label.split(", "):
            reason = f"{label} T-{inputs['horizon_minutes']}"
            assert at >= covering[reason] + timedelta(minutes=job.TABPFN_DELAY_MINUTES)


def test_two_t240_snapshots_have_descriptive_names_and_fire_at_2202_and_2203():
    games = _tabpfn_games().iloc[:2].copy()
    games["game_id"] = ["0022600125", "0022600126"]
    games["tipoff_utc"] = pd.to_datetime(["2026-10-27T00:30Z", "2026-10-27T01:00Z"])
    now = _utc("2026-10-26 20:00")
    normal = job.trigger_times(job.game_tipoffs(games), [240], now)
    runs = [
        (at, dispatch)
        for at, dispatch in job.tabpfn_runs(games, "2026-10-26", now, normal)
        if dict(dispatch.inputs)["horizon_minutes"] == "240"
    ]
    assert [at.astimezone(job.MADRID).strftime("%H:%M") for at, _ in runs] == [
        "22:02",
        "22:03",
    ]
    assert [dict(dispatch.inputs)["as_of"] for _, dispatch in runs] == [
        "2026-10-26T20:30:00+00:00",
        "2026-10-26T21:00:00+00:00",
    ]
    titles = []
    for index, (at, dispatch) in enumerate(runs, 2):
        (group,) = job.group_schedules([at], dispatch)
        payload = job.job_payload(job.pool_title(index, group), group, "secret")
        titles.append(payload["title"])
        assert "label" not in json.loads(payload["extendedData"]["body"])["inputs"]
    assert titles == [
        "Scorecast - snapshot trigger 02 | TabPFN T-240 A@B | snapshot 26/10 21:30 Madrid",
        "Scorecast - snapshot trigger 03 | TabPFN T-240 C@D | snapshot 26/10 22:00 Madrid",
    ]


def test_staggering_resolves_cascading_collisions_in_a_stable_order():
    games = _tabpfn_games()
    games["tipoff_utc"] = pd.to_datetime(
        [
            "2026-10-22T23:00Z",
            "2026-10-22T23:30Z",
            "2026-10-22T23:31Z",
        ]
    )
    now = _utc("2026-10-22 08:00")
    normal = job.trigger_times(job.game_tipoffs(games), [360], now)
    runs = job.tabpfn_runs(games, "2026-10-22", now, normal)
    assert runs == job.tabpfn_runs(games.iloc[::-1], "2026-10-22", now, normal)
    t360 = [
        (at, dispatch)
        for at, dispatch in runs
        if dict(dispatch.inputs)["horizon_minutes"] == "360"
    ]
    assert [at for at, _ in t360] == [
        _utc("2026-10-22 17:32"),
        _utc("2026-10-22 17:33"),
        _utc("2026-10-22 17:34"),
    ]
    assert all(
        b[0] - a[0] >= timedelta(minutes=1)
        for a, b in zip(runs, runs[1:], strict=False)
    )


def test_staggering_can_cross_utc_midnight_without_changing_the_snapshot():
    games = _tabpfn_games().iloc[:2].copy()
    games["tipoff_utc"] = pd.to_datetime(["2026-10-23T05:57Z", "2026-10-23T07:57Z"])
    runs = job.tabpfn_runs(games, "2026-10-22", _utc("2026-10-22 22:00"))
    coinciding = [
        (at, dispatch)
        for at, dispatch in runs
        if dict(dispatch.inputs)["as_of"] == "2026-10-22T23:57:00+00:00"
    ]
    assert [at for at, _ in coinciding] == [
        _utc("2026-10-22 23:59"),
        _utc("2026-10-23 00:00"),
    ]
    for at, dispatch in coinciding:
        (group,) = job.group_schedules([at], dispatch)
        assert group.day == at.date()
        assert dict(dispatch.inputs)["game_date"] == "2026-10-22"


def test_tabpfn_skips_past_snapshots_and_postponed_games():
    games = _tabpfn_games()
    games.loc[2, "game_status"] = "PPD"
    runs = job.tabpfn_runs(games, "2026-10-22", _utc("2026-10-22 15:00"))
    # The 15:00 T-480 is booked at its minute: it fires at 15:02.
    assert [at for at, _ in runs] == [
        _utc("2026-10-22 15:02"),
        _utc("2026-10-22 17:02"),
        _utc("2026-10-22 19:02"),
    ]
    runs = job.tabpfn_runs(games, "2026-10-22", _utc("2026-10-22 15:05"))
    assert [at for at, _ in runs] == [
        _utc("2026-10-22 17:02"),
        _utc("2026-10-22 19:02"),
    ]
    assert job.tabpfn_runs(games.iloc[:0], "2026-10-22", _utc("2026-10-22 08:00")) == []


def test_tabpfn_slots_before_daily_booking_are_lost():
    games = _tabpfn_games().iloc[:1].copy()
    games["tipoff_utc"] = pd.Timestamp("2026-12-25 17:00", tz="UTC")
    # 09:30 Madrid: T-720 has already passed. No catch-up dispatch.
    runs = job.tabpfn_runs(games, "2026-12-25", _utc("2026-12-25 08:30"))
    assert [at for at, _ in runs] == [
        _utc("2026-12-25 09:02"),
        _utc("2026-12-25 11:02"),
        _utc("2026-12-25 13:02"),
    ]
    assert {dict(dispatch.inputs)["horizon_minutes"] for _, dispatch in runs} == {
        "480",
        "360",
        "240",
    }


def test_tabpfn_payload_carries_date_horizon_cutoff_and_snapshot_games_only():
    ((at, dispatch), *_) = job.tabpfn_runs(
        _tabpfn_games(), "2026-10-22", _utc("2026-10-22 08:00")
    )
    (group,) = job.group_schedules([at], dispatch)
    payload = job.job_payload(job.pool_title(1), group, "secret")
    assert payload["url"].endswith(
        "/actions/workflows/nba_predictor_daily_tabpfn_client.yml/dispatches"
    )
    assert json.loads(payload["extendedData"]["body"]) == {
        "ref": "main",
        "inputs": {
            "target": "line_error total_points",
            "line": "intermediate",
            "horizon_minutes": "720",
            "update_db": "false",
            "game_date": "2026-10-22",
            "as_of": "2026-10-22T11:00:00+00:00",
            "snapshot_game_ids": "0022600001 0022600002",
        },
    }


def test_both_prediction_workflows_queue_all_pending_runs():
    from pathlib import Path

    import yaml

    for name in ("nba_predictor_daily.yml", "nba_predictor_daily_tabpfn_client.yml"):
        workflow = yaml.safe_load(Path(".github/workflows", name).read_text())
        assert workflow["concurrency"]["cancel-in-progress"] is False
        assert workflow["concurrency"]["queue"] == "max"
        assert workflow["jobs"]["run"]["runs-on"] == "self-hosted"


@pytest.mark.parametrize("include_tabpfn", [True, False])
def test_scheduler_books_exact_tabpfn_payloads_and_verifies_the_pool(
    monkeypatch, include_tabpfn
):
    """Run the booking entrypoint with fake scoreboard and cron-job.org I/O."""
    now = _utc("2026-10-22 08:00")

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz)

    games = _tabpfn_games()
    fetch = Mock(return_value=games)
    monkeypatch.setattr(job, "datetime", Clock)
    monkeypatch.setattr(job, "fetch_tipoffs_for_date", fetch)
    monkeypatch.setattr(job, "read_secret", lambda key: "test-secret")
    monkeypatch.setattr(
        job.sys,
        "argv",
        [
            "schedule_snapshot_triggers.py",
            "--apply",
            "--game-date",
            "2026-10-22",
            *([] if include_tabpfn else ["--no-tabpfn"]),
        ],
    )
    groups = []

    def apply(plan, scheduled, api_key, github_token):
        groups.extend(scheduled)
        return list(range(1, len(groups) + 1))

    monkeypatch.setattr(job, "apply_pool", apply)

    def request(method, endpoint, *args):
        assert (method, endpoint) == ("GET", "/jobs")
        return {
            "jobs": [
                {
                    "jobId": index,
                    "enabled": True,
                    "nextExecution": int(group.fire_times()[0].timestamp()),
                }
                for index, group in enumerate(groups, 1)
            ]
        }

    monkeypatch.setattr(job, "cronjob_request", request)
    job.main()
    fetch.assert_called_once_with("2026-10-22")
    tabpfn = [
        group for group in groups if group.dispatch.workflow_file == job.TABPFN_WORKFLOW
    ]
    assert len(tabpfn) == (8 if include_tabpfn else 0)
    assert any(group.dispatch == job.PREDICTOR for group in groups)
    assert any(group.dispatch == job.FINISHED_MATCHES for group in groups)
    assert any(
        group.dispatch.workflow_file == job.PAST_SNAPSHOTS_WORKFLOW for group in groups
    )
    assert len(groups) <= job.MAX_POOL
    for group in tabpfn:
        assert len(group.fire_times()) == 1
        # After the predictor run covering the snapshot, never before it.
        snapshot = datetime.fromisoformat(dict(group.dispatch.inputs)["as_of"])
        assert group.fire_times()[0] >= snapshot + timedelta(
            minutes=job.TABPFN_DELAY_MINUTES
        )


def test_booked_tabpfn_inputs_match_the_workflow_and_cli(monkeypatch):
    from pathlib import Path

    import yaml

    from scripts import predict_tabpfn as predictor

    workflow = yaml.load(
        Path(".github/workflows/nba_predictor_daily_tabpfn_client.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    supported = workflow["on"]["workflow_dispatch"]["inputs"]
    predict = Mock()
    monkeypatch.setattr(predictor, "predict_tabpfn", predict)
    for at, dispatch in job.tabpfn_runs(
        _tabpfn_games(), "2026-10-22", _utc("2026-10-22 08:00")
    ):
        inputs = dict(dispatch.inputs)
        assert inputs.keys() <= supported.keys()
        predictor.main(
            [
                "--targets",
                inputs["target"],
                "--line",
                inputs["line"],
                "--horizon-minutes",
                inputs["horizon_minutes"],
                "--game-date",
                inputs["game_date"],
                "--as-of",
                inputs["as_of"],
                "--snapshot-game-ids",
                inputs["snapshot_game_ids"],
                "--skip-db-update",
            ]
        )
        options = predict.call_args.kwargs
        assert options["targets"] == ["line_error", "total_points"]
        assert options["horizon_minutes"] == int(inputs["horizon_minutes"])
        assert options["game_ids"] is None
        assert options["snapshot_game_ids"] == inputs["snapshot_game_ids"].split()
        assert options["game_date"] == "2026-10-22"
        assert pd.Timestamp(options["as_of"]) == pd.Timestamp(inputs["as_of"])
        assert pd.Timestamp(options["as_of"]) <= at - timedelta(
            minutes=job.TABPFN_DELAY_MINUTES
        )
        assert options["update_db"] is False


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
