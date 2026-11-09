"""Step 6 of the coach presser archive: videos to games, and auto-accept."""

from __future__ import annotations

import json

import pytest

from lab.coach_pressers import catalog, match
from lab.coach_pressers.coaches import staff_entry
from tests.support.coach_pressers import config_2021, game, schedule

# SAC plays a back-to-back: Nov 18 at 22:00 ET (03:00 UTC Nov 19), then Nov 19.
GAMES = [
    game("0022100010", "2021-11-18", "PHX", "SAC", tip="2021-11-19T03:00:00Z"),
    game("0022100020", "2021-11-19", "SAC", "DEN", tip="2021-11-20T02:00:00Z"),
    game("0022100030", "2021-11-23", "BOS", "SAC", tip="2021-11-24T03:00:00Z"),
]
COACHES = {
    "teams": {
        "SAC": {
            "staff": [
                staff_entry("Alvin Gentry", "head", "t"),
                staff_entry("Doug Christie", "assistant", "t"),
            ],
            "players": ["De'Aaron Fox"],
        }
    }
}


@pytest.fixture
def setup(tmp_path):
    config = config_2021(tmp_path, teams=("SAC",))
    manifest = catalog.prepare_manifest(config, schedule("2021-22", GAMES))
    for entry in manifest["entries"].values():
        entry["coach_expected"] = "Alvin Gentry" if entry["team"] == "SAC" else None
    return config, manifest


def video(vid, title, published=None, day=None, **extra):
    return {
        "id": vid,
        "channel_id": "UC1",
        "title": title,
        "published_at_utc": published,
        "published_date": day or (published[:10] if published else None),
        "duration": 600,
        "chapters": [],
        **extra,
    }


def test_back_to_back_after_utc_midnight_goes_to_the_first_game(setup):
    config, manifest = setup
    videos = {
        "SAC": [video("v1", "Alvin Gentry Postgame", "2021-11-19T06:00:00+00:00")]
    }
    to_judge = match.match_season(config, manifest, COACHES, videos)
    assert list(to_judge) == ["0022100010:SAC"]  # no title evidence: judged
    assert to_judge["0022100010:SAC"][0]["match_method"] == "window_instant"


def test_date_only_back_to_back_is_ambiguous(setup):
    config, manifest = setup
    videos = {"SAC": [video("v1", "Alvin Gentry Postgame", day="2021-11-19")]}
    to_judge = match.match_season(config, manifest, COACHES, videos)
    assert set(to_judge) == {"0022100010:SAC", "0022100020:SAC"}
    assert {c["match_method"] for f in to_judge.values() for c in f} == {
        "ambiguous_window"
    }


def test_title_and_window_agree_is_auto_accepted(setup):
    config, manifest = setup
    videos = {
        "SAC": [
            video(
                "v2",
                "Kings vs Nuggets Postgame: Alvin Gentry",
                "2021-11-20T05:00:00+00:00",
                chapters=[{"start_time": 0, "end_time": 600, "title": "Gentry"}],
            )
        ]
    }
    to_judge = match.match_season(config, manifest, COACHES, videos)
    entry = manifest["entries"]["0022100020:SAC"]
    assert to_judge == {}
    assert entry["status"] == "selected"
    selection = entry["selection"]
    assert (
        selection["decided_by"] == "rules"
        and selection["coach_actual"] == "Alvin Gentry"
    )
    assert selection["chapters"][0]["title"] == "Gentry"
    assert manifest["entries"]["0022100030:SAC"]["status"] == "not_found"


def test_late_upload_is_title_only_and_never_auto(setup):
    config, manifest = setup
    videos = {
        "SAC": [
            video(
                "v3", "Alvin Gentry Postgame | 11.18.2021", "2021-11-28T15:00:00+00:00"
            )
        ]
    }
    to_judge = match.match_season(config, manifest, COACHES, videos)
    assert to_judge["0022100010:SAC"][0]["match_method"] == "title_only"


def test_another_staff_member_goes_to_the_judge(setup):
    config, manifest = setup
    videos = {
        "SAC": [
            video(
                "v4",
                "Kings vs Nuggets Postgame: Doug Christie",
                "2021-11-20T05:00:00+00:00",
            )
        ]
    }
    to_judge = match.match_season(config, manifest, COACHES, videos)
    assert list(to_judge) == ["0022100020:SAC"]


def test_coach_name_without_a_cue_is_always_judged(setup):
    config, manifest = setup
    videos = {
        "SAC": [
            video(
                "v5",
                "Alvin Gentry on Kings Win vs Nuggets | Nov. 19, 2021",
                "2021-11-20T05:00:00+00:00",
            )
        ]
    }
    to_judge = match.match_season(config, manifest, COACHES, videos)
    assert list(to_judge) == ["0022100020:SAC"]
    assert to_judge["0022100020:SAC"][0]["match_method"] == "title+window"
    assert manifest["entries"]["0022100020:SAC"]["status"] == "pending"


@pytest.mark.parametrize(
    "title,duration,flag",
    [
        (
            "Alvin Gentry on Tonight's Win vs Nuggets | 11.19.2021",
            300,
            "keyword:tonights",
        ),
        (
            "Alvin Gentry Postgame on Kings Locker Room | 11.19.2021",
            300,
            "keyword:locker room",
        ),
        ("Alvin Gentry | 11.19.2021", 300, "coach_name_no_cue"),
        ("Alvin Gentry Postgame | 11.19.2021", 32, "too_short"),
        ("Alvin Gentry Postgame | 11.19.2021", 3000, "long"),
        # The length penalty puts this below MIN_SCORE; it must still be judged.
        ("Kings Postgame Availability | 11.19.2021", 3000, "long"),
    ],
)
def test_warning_candidates_reach_the_judge_despite_title_window_agreement(
    setup, title, duration, flag
):
    config, manifest = setup
    videos = {
        "SAC": [video("flagged", title, "2021-11-20T05:00:00+00:00", duration=duration)]
    }
    to_judge = match.match_season(config, manifest, COACHES, videos)
    found = to_judge["0022100020:SAC"]
    assert len(found) == 1 and found[0]["video_id"] == "flagged"
    assert found[0]["match_method"] == "title+window"
    assert found[0]["requires_llm"] and flag in found[0]["notes"]
    entry = manifest["entries"]["0022100020:SAC"]
    assert entry["status"] == "pending" and entry["selection"] is None
    # Even a strong classification must not override an explicit LLM referral.
    assert (
        match.auto_accept(entry, [{**found[0], "class": "coach", "score": 1}]) is None
    )


def test_outlet_videos_about_another_team_are_dropped(setup):
    config, manifest = setup
    videos = {
        "SAC": [
            video(
                "o1",
                "Michael Malone Postgame | Nuggets vs Suns",
                "2021-11-20T05:00:00+00:00",
                channel_kind="outlet",
            ),
            video(
                "o2",
                "Alvin Gentry Postgame | Kings vs Nuggets",
                "2021-11-20T05:10:00+00:00",
                channel_kind="outlet",
            ),
        ]
    }
    match.match_season(config, manifest, COACHES, videos)
    found = json.loads(config.candidates_path.read_bytes())
    assert [c["video_id"] for c in found["0022100020:SAC"]] == ["o2"]


def test_two_candidates_go_to_the_judge(setup):
    config, manifest = setup
    videos = {
        "SAC": [
            video(
                "v5",
                "Kings vs Nuggets Postgame: Alvin Gentry",
                "2021-11-20T05:00:00+00:00",
            ),
            video(
                "v6",
                "Kings vs Nuggets | Full Postgame Media Availability",
                "2021-11-20T05:30:00+00:00",
            ),
        ]
    }
    to_judge = match.match_season(config, manifest, COACHES, videos)
    assert [c["video_id"] for c in to_judge["0022100020:SAC"]] == ["v5", "v6"]


def test_combined_chapter_sets_the_trim():
    staff = COACHES["teams"]["SAC"]["staff"]
    chapters = [
        {"start_time": 0, "end_time": 410, "title": "Coach Gentry"},
        {"start_time": 410, "end_time": 900, "title": "De'Aaron Fox"},
    ]
    assert match.find_coach_chapter(chapters, staff)["start_time"] == 0
    candidate = {
        "video_id": "v",
        "channel_id": "c",
        "title": "t",
        "published_at_utc": None,
        "duration_s": 900,
        "chapters": chapters,
        "coach_chapter": match.find_coach_chapter(chapters, staff),
    }
    selection = match.selection_from(
        candidate, decided_by="rules", coach_actual="Alvin Gentry", kind="combined"
    )
    assert (selection["trim_start_s"], selection["trim_end_s"]) == (0, 410)
    assert selection["needs_trim"] is False
    no_chapter = {**candidate, "coach_chapter": None, "chapters": []}
    assert match.selection_from(
        no_chapter, decided_by="rules", coach_actual="x", kind="combined"
    )["needs_trim"]


def test_playoff_game_number_and_opponent_evidence():
    games = [
        {
            "game_id": "1",
            "date": "2020-09-03",
            "tip_utc": "2020-09-04T01:00:00Z",
            "opponent": "DEN",
            "playoff_game_no": 1,
        },
        {
            "game_id": "2",
            "date": "2020-09-05",
            "tip_utc": "2020-09-06T01:00:00Z",
            "opponent": "DEN",
            "playoff_game_no": 2,
        },
    ]
    late = {
        "title": "Doc Rivers Post-Game Press Conference | Clippers vs. Nuggets Game One",
        "published_at_utc": "2020-09-06T18:00:00+00:00",
    }
    hits, evidence = match.title_evidence(late, games, "LAC")
    assert [g["game_id"] for g in hits] == ["1"] and evidence["game_no"] == 1


def test_the_opponents_head_coach_is_never_a_candidate(setup):
    config, manifest = setup
    coaches = {
        "teams": {
            **COACHES["teams"],
            "DEN": {"staff": [staff_entry("Michael Malone", "head", "t")]},
        }
    }
    assert "michael malone" in match.rival_head_coaches(coaches["teams"], "SAC")
    videos = {
        "SAC": [
            video(
                "r1",
                "Michael Malone PostGame Interview | Sacramento Kings vs Denver Nuggets",
                "2021-11-20T05:00:00+00:00",
                channel_kind="outlet",
            )
        ]
    }
    match.match_season(config, manifest, coaches, videos)
    assert manifest["entries"]["0022100020:SAC"]["status"] == "not_found"
