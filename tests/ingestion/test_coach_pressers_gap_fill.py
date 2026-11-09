"""Step 6b of the coach presser archive: window uploads of games without a candidate."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pytest

from lab.coach_pressers import catalog, crawl, gap_fill, llm_judge, match
from lab.coach_pressers.coaches import staff_entry
from tests.support.coach_pressers import config_2021, game, schedule

GAMES = [
    game("0022100010", "2021-11-18", "PHX", "SAC", tip="2021-11-19T03:00:00Z"),
    game("0022100030", "2021-11-23", "BOS", "SAC", tip="2021-11-24T03:00:00Z"),
]
COACHES = {
    "teams": {
        "SAC": {
            "staff": [staff_entry("Alvin Gentry", "head", "t")],
            "players": ["De'Aaron Fox"],
        }
    }
}
GAP = "0022100030:SAC"
# Newest first, as a channel tab lists them: (id, title, published instant).
TEAM_TAB = [
    ("late", "Kings Highlights", "2021-12-10T05:00:00+00:00"),
    ("p2", "De'Aaron Fox after the win", "2021-11-24T06:00:00+00:00"),
    ("x2", "Kings vs Celtics Highlights", "2021-11-24T05:00:00+00:00"),
    ("k1", "Alvin Gentry Postgame | 11.18.2021", "2021-11-19T06:00:00+00:00"),
    ("old", "Kings Highlights", "2021-11-01T05:00:00+00:00"),
    *((f"o{i}", "Kings Highlights", "2021-10-25T05:00:00+00:00") for i in range(20)),
]
OUTLET_TAB = [
    ("n1", "Kings vs Celtics: what went wrong", "2021-11-24T06:30:00+00:00"),
    ("n2", "Frank Vogel postgame", "2021-11-24T06:40:00+00:00"),
]
CACHED = {"late", "k1", "old", "n1"}  # n1: found by step 4 on the outlet


def record(video_id, title, published, channel_id="UC1"):
    return {
        "id": video_id,
        "channel_id": channel_id,
        "title": title,
        "published_at_utc": published,
        "published_date": published[:10],
        "duration": 600,
        "description": "",
        "chapters": [],
    }


class Extractor:
    def __init__(self):
        self.calls = []
        self.titles = {v: (t, p) for v, t, p in TEAM_TAB + OUTLET_TAB}

    def __call__(self, video_id):
        self.calls.append(video_id)
        title, published = self.titles[video_id]
        stamp = datetime.fromisoformat(published).astimezone(UTC).timestamp()
        return {
            "id": video_id,
            "title": title,
            "timestamp": int(stamp),
            "duration": 900,
        }


@pytest.fixture
def world(tmp_path):
    config = config_2021(tmp_path, teams=("SAC",))
    manifest = catalog.prepare_manifest(config, schedule("2021-22", GAMES))
    for entry in manifest["entries"].values():
        entry["coach_expected"] = "Alvin Gentry"
    for channel_id, tab in (("UC1", TEAM_TAB), ("UC2", OUTLET_TAB)):
        cache = crawl.ChannelCache(config.channels_dir, channel_id)
        cache.save_index(
            {
                "channel_id": channel_id,
                "listed_at_utc": "2022-01-01T00:00:00+00:00",
                "tabs": {"videos": [{"id": v, "title": t} for v, t, _ in tab]},
            }
        )
        for video_id, title, published in tab:
            if video_id in CACHED:
                cache.add(record(video_id, title, published, channel_id))
    rows = [
        {"team": "SAC", "channel_id": "UC1", "priority": 1, "handle": "@kings"},
        {
            "team": "SAC",
            "channel_id": "UC2",
            "priority": 2,
            "handle": "@outlet",
            "kind": "outlet",
        },
    ]
    k1 = next(v for v in TEAM_TAB if v[0] == "k1")
    n1 = next(v for v in OUTLET_TAB if v[0] == "n1")
    videos = {
        "SAC": [
            crawl.tagged(record(*k1), rows[0]),
            crawl.tagged(record(*n1, channel_id="UC2"), rows[1]),
        ]
    }
    to_judge = match.match_season(config, manifest, COACHES, videos)
    return config, manifest, rows, videos, to_judge


def test_only_the_gap_window_is_fetched_and_every_upload_is_a_candidate(world):
    config, manifest, rows, videos, to_judge = world
    assert to_judge == {}
    assert manifest["entries"]["0022100010:SAC"]["selection"]["video_id"] == "k1"
    assert manifest["entries"][GAP]["status"] == "not_found"
    extractor = Extractor()
    gaps = gap_fill.fill_gaps(
        config,
        manifest,
        COACHES,
        rows,
        videos,
        extractor=extractor,
        today=date(2022, 1, 1),
    )
    assert set(gaps) == {GAP}
    # The player-titled video and the highlight both reach the judge; the
    # outlet's step 4 video joins at no cost; the outlet gets no new requests.
    assert {c["video_id"] for c in gaps[GAP]} == {"p2", "x2", "n1"}
    assert all(c["requires_llm"] and "gap_fill" in c["notes"] for c in gaps[GAP])
    assert sorted(extractor.calls) == ["p2", "x2"]
    stored = json.loads(config.candidates_path.read_bytes())
    assert {c["video_id"] for c in stored[GAP]} == {"p2", "x2", "n1"}
    assert "0022100010:SAC" in stored


def test_a_game_a_person_closed_is_not_reopened(world):
    config, manifest, rows, videos, _ = world
    manifest["entries"][GAP]["error"] = "human: no presser"
    extractor = Extractor()
    assert (
        gap_fill.fill_gaps(config, manifest, COACHES, rows, videos, extractor=extractor)
        == {}
    )
    assert extractor.calls == []


def test_merged_windows():
    entries = [{"date": "2021-11-18"}, {"date": "2021-11-20"}, {"date": "2021-12-10"}]
    assert gap_fill.merged_windows(entries, date(2021, 12, 11)) == [
        (date(2021, 11, 18), date(2021, 11, 23)),
        (date(2021, 12, 10), date(2021, 12, 11)),
    ]


def test_known_bounds_uses_cached_dates():
    ids = [f"v{i}" for i in range(20)]
    videos = {
        "v2": {"published_date": "2021-12-05"},
        "v5": {"published_date": "2021-12-01"},  # newest known after the window
        "v12": {"published_date": "2021-11-01"},  # newest known before it
        "v15": {"published_date": "2021-10-01"},
    }
    lo, hi = gap_fill.known_bounds(ids, videos, date(2021, 11, 20), date(2021, 11, 23))
    assert (lo, hi) == (6 - gap_fill.BOUND_MARGIN, 12 + gap_fill.BOUND_MARGIN)
    assert gap_fill.known_bounds(ids, {}, date(2021, 11, 20), date(2021, 11, 23)) == (
        0,
        20,
    )


class StubJudge:
    def __init__(self, answer):
        self.answer, self.calls, self.cached, self.spent_usd = answer, 0, 0, 0.0

    def ask(self, payload, *, game_id="", team=""):
        self.calls += 1
        return {
            "input_hash": "h",
            "model": "m",
            "answer": self.answer(payload),
            "error": None,
        }


def none(payload):
    return {
        "video_id": None,
        "kind": "none",
        "coach_name": None,
        "coach_chapter": None,
        "confidence": 0.9,
        "reason": "No presser.",
    }


def pick(video_id):
    def answer(payload):
        return {
            "video_id": video_id,
            "kind": "combined",
            "coach_name": "Alvin Gentry",
            "coach_chapter": None,
            "confidence": 0.8,
            "reason": "Full availability.",
        }

    return answer


@pytest.mark.parametrize(
    "answer,status", [(none, "not_found"), (pick("p2"), "selected")]
)
def test_the_judge_fills_a_gap_or_leaves_it_not_found(world, answer, status):
    config, manifest, rows, videos, _ = world
    gaps = gap_fill.fill_gaps(
        config,
        manifest,
        COACHES,
        rows,
        videos,
        extractor=Extractor(),
        today=date(2022, 1, 1),
    )
    stats = llm_judge.judge_season(
        config, manifest, COACHES, {}, judge=StubJudge(answer), gaps=gaps
    )
    entry = manifest["entries"][GAP]
    assert entry["status"] == status and entry["llm"]["answer"] is not None
    if status == "selected":
        assert entry["selection"]["video_id"] == "p2"
        assert entry["selection"]["decided_by"] == "llm"
        assert stats["gaps_selected"] == 1
    else:
        assert entry["selection"] is None and stats["gaps_none"] == 1


def test_without_the_llm_gaps_stay_not_found(world):
    config, manifest, rows, videos, _ = world
    gaps = gap_fill.fill_gaps(
        config,
        manifest,
        COACHES,
        rows,
        videos,
        extractor=Extractor(),
        today=date(2022, 1, 1),
    )
    no_llm = config_2021(config.local_root.parent, teams=("SAC",), use_llm=False)
    llm_judge.judge_season(no_llm, manifest, COACHES, {}, gaps=gaps)
    assert manifest["entries"][GAP]["status"] == "not_found"
