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
        "update_finished_matches_daily.yml": _configured_time(
            "FINISHED_MATCHES_RETRY_AT"
        ),
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
        "update_finished_matches_daily.yml": (11, 11),
    }


def test_final_score_retry_is_an_independent_daily_job():
    config = _schedule_config()
    config.set("Schedule", "FINISHED_MATCHES_RETRY_AT", "11:25")
    _, jobs = setup.read_schedule(config)
    retry = next(
        j for j in jobs if j.workflow_file == "update_finished_matches_daily.yml"
    )
    payload = setup.job_payload(retry, "secret")
    assert retry.title == "Scorecast - retry final scores"
    assert payload["schedule"]["hours"] == [11]
    assert payload["schedule"]["minutes"] == [25]
    assert payload["schedule"]["mdays"] == [-1]
    assert json.loads(payload["extendedData"]["body"]) == {"ref": "main"}


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
    # Last tip 02:30 UTC + 150 min = 05:00 estimated end. Final scores
    # would fire at 07:00 UTC, so cap them at 07:50 Madrid (05:50 UTC).
    assert [at for at, _ in runs] == [
        _utc("2026-10-23 05:30"),
        _utc("2026-10-23 05:50"),
    ]
    past, finished = (dispatch for _, dispatch in runs)
    assert past.workflow_file == "nba_predict_past_snapshots.yml"
    assert dict(past.inputs) == {"game_date": "2026-10-22"}
    assert finished == job.FINISHED_MATCHES
    assert job.post_game_runs([], "2026-10-22", _utc("2026-10-22 08:00")) == []
    # Already past (a scheduler run late in the night): left out.
    assert len(job.post_game_runs(tips, "2026-10-22", _utc("2026-10-23 05:45"))) == 1


@pytest.mark.parametrize(
    ("game_date", "last_tip", "latest_utc"),
    [
        ("2026-10-22", "2026-10-23 04:00", "2026-10-23 05:50"),
        ("2026-10-29", "2026-10-30 04:00", "2026-10-30 06:50"),
        ("2026-03-28", "2026-03-29 04:00", "2026-03-29 05:50"),
        ("2026-10-24", "2026-10-25 04:00", "2026-10-25 06:50"),
    ],
)
def test_final_scores_are_capped_before_eight_including_dst_changes(
    game_date, last_tip, latest_utc
):
    runs = job.post_game_runs(
        [("A@B", _utc(last_tip))], game_date, _utc(f"{game_date} 08:00")
    )
    final_at = next(at for at, dispatch in runs if dispatch == job.FINISHED_MATCHES)
    assert final_at == _utc(latest_utc)
    assert final_at.astimezone(job.MADRID).strftime("%H:%M") == "07:50"


def test_final_score_cap_does_not_move_an_earlier_dispatch_later():
    runs = job.post_game_runs(
        [("A@B", _utc("2026-10-22 23:00"))],
        "2026-10-22",
        _utc("2026-10-22 08:00"),
    )
    assert next(at for at, d in runs if d == job.FINISHED_MATCHES) == _utc(
        "2026-10-23 03:30"
    )


def test_final_scores_are_not_rebooked_after_the_morning_deadline():
    runs = job.post_game_runs(
        [("A@B", _utc("2026-10-23 04:00"))],
        "2026-10-22",
        _utc("2026-10-23 06:00"),
    )
    assert all(dispatch != job.FINISHED_MATCHES for _, dispatch in runs)


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


def test_tabpfn_runs_four_times_from_the_last_tip_with_all_targets():
    games = _tabpfn_games()
    runs = job.tabpfn_runs(games, "2026-10-22", _utc("2026-10-22 08:00"))
    assert len(runs) == 4
    assert [(at, dict(d.inputs)["line"]) for at, d in runs] == [
        (_utc("2026-10-22 15:02"), "intermediate"),
        (_utc("2026-10-22 17:02"), "intermediate"),
        (_utc("2026-10-22 19:02"), "intermediate"),
        (_utc("2026-10-22 21:02"), "closing"),
    ]
    for at, dispatch in runs:
        inputs = dict(dispatch.inputs)
        assert dispatch.workflow_file == job.TABPFN_WORKFLOW
        assert inputs["target"] == "line_error total_points spread_error"
        assert inputs["update_db"] == "false"
        assert inputs["game_date"] == "2026-10-22"
        assert "game_ids" not in inputs
        if inputs["line"] == "closing":
            assert not {"horizon_minutes", "as_of", "snapshot_game_ids"} & inputs.keys()
            assert inputs["include_closing"] == "false"
            assert inputs["modes"] == "standard"
            assert inputs["use_alt_key"] == "false"
            continue
        assert _snapshot_games(dispatch) == ["0022600003"]
        horizon = int(inputs["horizon_minutes"])
        cutoff = datetime.fromisoformat(inputs["as_of"])
        assert cutoff == _utc("2026-10-23 01:00") - timedelta(minutes=horizon)
        assert at == cutoff + timedelta(minutes=job.TABPFN_DELAY_MINUTES)
        # Every earlier game's own snapshot has passed by this cutoff.
        assert all(
            tip - timedelta(minutes=horizon) <= cutoff
            for _, tip in job.game_tipoffs(games)
        )


def test_nearby_tipoffs_do_not_book_extra_tabpfn_runs():
    games = _tabpfn_games().iloc[:2].copy()
    games.loc[1, "tipoff_utc"] += pd.Timedelta(minutes=15)
    runs = job.tabpfn_runs(games, "2026-10-22", _utc("2026-10-22 08:00"))
    assert len(runs) == 4
    assert [at for at, _ in runs] == [
        _utc("2026-10-22 13:17"),
        _utc("2026-10-22 15:17"),
        _utc("2026-10-22 17:17"),
        _utc("2026-10-22 19:17"),
    ]
    assert all(dispatch.label == "C@D" for _, dispatch in runs)


def test_only_t8_adds_high_thinking_in_both_booking_modes():
    from nba_ou.prediction.tabpfn_run import parse_jobs

    games = _tabpfn_games()
    now = _utc("2026-10-22 08:00")
    expected = {
        600: ("standard",),
        480: ("standard", "thinking_high"),
        360: ("standard",),
    }
    separate = job.tabpfn_runs(games, "2026-10-22", now)
    assert {
        int(dict(dispatch.inputs)["horizon_minutes"]): tuple(
            dict(dispatch.inputs)["modes"].split()
        )
        for _, dispatch in separate
        if dict(dispatch.inputs)["line"] == "intermediate"
    } == expected

    _, joined, alone = _joined(games, now)
    assert alone == []
    actual = {}
    for dispatch in joined.values():
        inputs = dict(dispatch.inputs)
        jobs = parse_jobs(inputs["tabpfn"], modes=inputs["tabpfn_modes"].split())
        for item in jobs:
            assert item.targets == ("line_error", "total_points", "spread_error")
            actual[item.horizon] = item.modes
    assert actual == {**expected, 0: ("standard",)}


def test_tabpfn_follows_the_covering_predictor_but_keeps_the_last_games_cutoff():
    games = _tabpfn_games().iloc[:1].copy()
    now = _utc("2026-10-22 08:00")
    predictor = job.trigger_times(
        job.game_tipoffs(games), [600, 590, 480, 360, 30], now
    )
    runs = job.tabpfn_runs(games, "2026-10-22", now, predictor)
    at, dispatch = runs[0]
    assert at == _utc("2026-10-22 13:12")
    assert dict(dispatch.inputs)["as_of"] == "2026-10-22T13:00:00+00:00"


def test_all_tabpfn_times_book_closing_and_only_t8_selects_the_alternate_key():
    from nba_ou.prediction.tabpfn_run import parse_jobs

    games = _tabpfn_games()
    now = _utc("2026-10-22 08:00")
    _, joined, alone = _joined(games, now)
    assert len(joined) == 4 and not alone
    for at, dispatch in joined.items():
        inputs = dict(dispatch.inputs)
        jobs = parse_jobs(inputs["tabpfn"], modes=inputs["tabpfn_modes"].split())
        horizon = int((_utc("2026-10-23 01:00") - at).total_seconds() / 60)
        assert [item.horizon for item in jobs] == (
            [horizon, 0] if horizon != 240 else [0]
        )
        assert inputs["tabpfn_use_alt_key"] == str(horizon == 480).lower()
        closing = jobs[-1]
        assert closing.dataset_type == "closing_line"
        assert closing.as_of is None
        assert closing.snapshot_games is None
        assert closing.modes == (
            ("standard", "thinking_high") if horizon == 480 else ("standard",)
        )

    for _, dispatch in job.tabpfn_runs(games, "2026-10-22", now):
        inputs = dict(dispatch.inputs)
        if inputs["line"] == "closing":
            assert inputs["include_closing"] == "false"
            continue
        horizon = int(inputs["horizon_minutes"])
        assert inputs["include_closing"] == "true"
        assert inputs["use_alt_key"] == str(horizon == 480).lower()


def test_games_sharing_the_last_tip_share_the_anchor_metadata():
    games = _tabpfn_games()
    games.loc[1, "tipoff_utc"] = games.loc[2, "tipoff_utc"]
    runs = job.tabpfn_runs(games, "2026-10-22", _utc("2026-10-22 08:00"))
    assert len(runs) == 4
    assert all(
        _snapshot_games(dispatch) == ["0022600002", "0022600003"]
        for _, dispatch in runs
        if dict(dispatch.inputs)["line"] == "intermediate"
    )
    assert runs == job.tabpfn_runs(
        games.iloc[::-1], "2026-10-22", _utc("2026-10-22 08:00")
    )


def test_tabpfn_titles_name_the_last_games_horizon_and_cutoff():
    ((at, dispatch), *_) = job.tabpfn_runs(
        _tabpfn_games(), "2026-10-22", _utc("2026-10-22 08:00")
    )
    (group,) = job.group_schedules([at], dispatch)
    title = job.pool_title(2, group)
    assert (
        title
        == "Scorecast - snapshot trigger 02 | TabPFN T-600 E@F | snapshot 22/10 17:00 Madrid"
    )


def test_tabpfn_skips_past_snapshots_and_postponed_last_games():
    games = _tabpfn_games()
    games.loc[2, "game_status"] = "PPD"
    runs = job.tabpfn_runs(games, "2026-10-22", _utc("2026-10-22 15:00"))
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
    games["game_status"] = "PPD"
    assert job.tabpfn_runs(games, "2026-10-22", _utc("2026-10-22 08:00")) == []


def test_tabpfn_slots_before_daily_booking_are_lost():
    games = _tabpfn_games().iloc[:1].copy()
    games["tipoff_utc"] = pd.Timestamp("2026-12-25 17:00", tz="UTC")
    runs = job.tabpfn_runs(games, "2026-12-25", _utc("2026-12-25 08:30"))
    assert [at for at, _ in runs] == [
        _utc("2026-12-25 09:02"),
        _utc("2026-12-25 11:02"),
        _utc("2026-12-25 13:02"),
    ]
    assert {dict(d.inputs).get("horizon_minutes") for _, d in runs} == {
        "480",
        "360",
        None,
    }


def test_tabpfn_payload_carries_date_horizon_cutoff_and_anchor_games_only():
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
            "target": "line_error total_points spread_error",
            "line": "intermediate",
            "horizon_minutes": "600",
            "modes": "standard",
            "use_alt_key": "false",
            "include_closing": "true",
            "update_db": "false",
            "game_date": "2026-10-22",
            "as_of": "2026-10-22T15:00:00+00:00",
            "snapshot_game_ids": "0022600003",
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
            # Separate dispatches, whatever [Schedule] TABPFN_RUNS says.
            *(["--tabpfn", "separate"] if include_tabpfn else ["--no-tabpfn"]),
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
    assert len(tabpfn) == (4 if include_tabpfn else 0)
    assert any(group.dispatch == job.PREDICTOR for group in groups)
    assert any(group.dispatch == job.FINISHED_MATCHES for group in groups)
    assert any(
        group.dispatch.workflow_file == job.PAST_SNAPSHOTS_WORKFLOW for group in groups
    )
    assert len(groups) <= job.MAX_POOL
    for group in tabpfn:
        assert len(group.fire_times()) == 1
        if dict(group.dispatch.inputs)["line"] == "closing":
            assert group.fire_times() == [_utc("2026-10-22 21:02")]
            continue
        # After the predictor run covering the snapshot, never before it.
        snapshot = datetime.fromisoformat(dict(group.dispatch.inputs)["as_of"])
        assert group.fire_times()[0] >= snapshot + timedelta(
            minutes=job.TABPFN_DELAY_MINUTES
        )


def test_booked_tabpfn_inputs_match_the_workflow_and_cli(monkeypatch, tmp_path):
    import os
    import subprocess
    from pathlib import Path

    import yaml

    from scripts import predict_tabpfn as predictor

    workflow = yaml.load(
        Path(".github/workflows/nba_predictor_daily_tabpfn_client.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    supported = workflow["on"]["workflow_dispatch"]["inputs"]
    step = next(
        s
        for s in workflow["jobs"]["run"]["steps"]
        if s.get("name") == "Run TabPFN predictor"
    )
    capture = tmp_path / "args"
    script = 'python() { printf "%s\\n" "$@" > "$CAPTURE_ARGS"; }\n' + step["run"]
    script = script.replace(
        "${{ !inputs.update_db && '--skip-db-update' || '' }}", "--skip-db-update"
    ).replace("${{ !inputs.persist && '--no-persist' || '' }}", "")
    predict = Mock()
    monkeypatch.setattr(predictor, "predict_tabpfn", predict)
    for at, dispatch in job.tabpfn_runs(
        _tabpfn_games(), "2026-10-22", _utc("2026-10-22 08:00")
    ):
        inputs = dict(dispatch.inputs)
        assert inputs.keys() <= supported.keys()
        subprocess.run(
            ["bash", "-c", script],
            env={
                **os.environ,
                "CAPTURE_ARGS": str(capture),
                "TABPFN_TARGETS": inputs["target"],
                "TABPFN_LINE": inputs["line"],
                "TABPFN_HORIZON": inputs.get("horizon_minutes", "360"),
                "TABPFN_MODES": inputs["modes"],
                "TABPFN_GAME_DATE": inputs["game_date"],
                "TABPFN_GAME_IDS": "",
                "TABPFN_AS_OF": inputs.get("as_of", ""),
                "TABPFN_SNAPSHOT_GAME_IDS": inputs.get("snapshot_game_ids", ""),
            },
            check=True,
            capture_output=True,
        )
        args = capture.read_text().splitlines()
        assert args[0] == "scripts/predict_tabpfn.py"
        predictor.main(args[1:])
        options = predict.call_args.kwargs
        assert options["targets"] == ["line_error", "total_points", "spread_error"]
        assert options["line"] == inputs["line"]
        assert options["modes"] == inputs["modes"].split()
        assert options["game_ids"] is None
        assert options["game_date"] == "2026-10-22"
        assert options["update_db"] is False
        if inputs["line"] == "closing":
            assert options["horizon_minutes"] is None
            assert options["as_of"] is None
            assert options["snapshot_game_ids"] is None
        else:
            assert options["horizon_minutes"] == int(inputs["horizon_minutes"])
            assert options["snapshot_game_ids"] == inputs["snapshot_game_ids"].split()
            assert pd.Timestamp(options["as_of"]) == pd.Timestamp(inputs["as_of"])
            assert pd.Timestamp(options["as_of"]) <= at - timedelta(
                minutes=job.TABPFN_DELAY_MINUTES
            )


def _joined(games, now):
    predictor = job.trigger_times(
        job.game_tipoffs(games), [600, 480, 360, 240, 30], now
    )
    return predictor, *job.joined_tabpfn_runs(games, "2026-10-22", now, predictor)


def test_joined_tabpfn_jobs_ride_in_the_predictor_run_covering_their_snapshot():
    from nba_ou.prediction.tabpfn_run import parse_jobs

    games = _tabpfn_games()
    now = _utc("2026-10-22 08:00")
    predictor, joined, alone = _joined(games, now)
    # Every TabPFN snapshot is also a predictor snapshot.
    assert alone == []
    assert set(joined) <= set(predictor)
    expected = {
        (str(game.game_id), horizon): game.tipoff_utc.to_pydatetime()
        - timedelta(minutes=horizon)
        for game in games.iloc[[-1]].itertuples()
        for horizon in (600, 480, 360)
    }
    actual = {}
    for run, dispatch in joined.items():
        assert dispatch.workflow_file == job.PREDICTOR.workflow_file
        inputs = dict(dispatch.inputs)
        assert inputs.keys() == {"tabpfn", "tabpfn_modes", "tabpfn_use_alt_key"}
        for tabpfn_job in parse_jobs(
            inputs["tabpfn"], modes=inputs["tabpfn_modes"].split()
        ):
            if tabpfn_job.horizon == 0:
                assert tabpfn_job.as_of is None
                assert tabpfn_job.snapshot_games is None
                continue
            # The run covering the snapshot fires at it or after it.
            assert tabpfn_job.as_of <= pd.Timestamp(run)
            for game_id in tabpfn_job.snapshot_games:
                actual[(game_id, tabpfn_job.horizon)] = tabpfn_job.as_of
    assert actual == expected
    assert len(joined) == 4
    assert joined[_utc("2026-10-22 17:00")].label == "T-480 E@F"


def test_a_tabpfn_snapshot_no_predictor_run_covers_keeps_its_own_dispatch():
    games = _tabpfn_games()
    now = _utc("2026-10-22 08:00")
    joined, alone = job.joined_tabpfn_runs(games, "2026-10-22", now, {})
    assert joined == {}
    assert alone == job.tabpfn_runs(games, "2026-10-22", now)


def test_a_predictor_run_with_tabpfn_jobs_books_a_job_of_its_own(monkeypatch):
    """Joined: no TabPFN dispatch; each carrying predictor run fires once,
    with inputs the predictor workflow takes."""
    from pathlib import Path

    import yaml

    now = _utc("2026-10-22 08:00")

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz)

    monkeypatch.setattr(job, "datetime", Clock)
    monkeypatch.setattr(
        job, "fetch_tipoffs_for_date", Mock(return_value=_tabpfn_games())
    )
    monkeypatch.setattr(
        job.sys,
        "argv",
        [
            "schedule_snapshot_triggers.py",
            "--game-date",
            "2026-10-22",
            "--tabpfn",
            "joined",
        ],
    )
    groups = []
    monkeypatch.setattr(job, "MAX_POOL", 0)  # stop after planning, before booking
    original = job.group_schedules

    def recorded(times, dispatch=job.PREDICTOR):
        made = original(times, dispatch)
        groups.extend(made)
        return made

    monkeypatch.setattr(job, "group_schedules", recorded)
    with pytest.raises(SystemExit):
        job.main()

    assert not any(g.dispatch.workflow_file == job.TABPFN_WORKFLOW for g in groups)
    carrying = [g for g in groups if "tabpfn" in dict(g.dispatch.inputs)]
    assert carrying and all(len(g.fire_times()) == 1 for g in carrying)
    workflow = yaml.load(
        Path(".github/workflows/nba_predictor_daily.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    supported = workflow["on"]["workflow_dispatch"]["inputs"]
    for group in carrying:
        assert dict(group.dispatch.inputs).keys() <= supported.keys()
        assert " + TabPFN T-" in job.pool_title(1, group)


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
