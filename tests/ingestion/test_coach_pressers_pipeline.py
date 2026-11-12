"""The coach presser pipeline end to end, with every network client faked."""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from lab.coach_pressers import llm_judge
from lab.coach_pressers.channels import write_channels
from lab.coach_pressers.pipeline import Hooks, run_season
from lab.youtube_downloader import youtube_media as media
from tests.support.coach_pressers import (
    config_2021,
    game,
    players_2021,
    schedule,
    wikipedia_2021,
)
from tests.support.conditional_s3 import ConditionalFakeS3

GAMES = [
    game("0022100010", "2021-11-19", "SAC", "PHX", tip="2021-11-20T02:00:00Z"),
    game("0022100020", "2021-11-21", "PHX", "SAC", tip="2021-11-22T02:00:00Z"),
    game("0022100030", "2021-12-01", "SAC", "PHX", tip="2021-12-02T02:00:00Z"),
    # Playoffs make the season "over", so the cached schedule is reused.
    game("0042100101", "2022-04-17", "SAC", "PHX", tip="2022-04-17T23:00:00Z"),
]
# Newest first, as YouTube lists a channel.
VIDEOS = [
    ("v4", "Kings Highlights vs Suns", "2021-12-02T05:00:00Z"),
    ("v3", "Kings vs Suns Postgame: Alvin Gentry | 12.01.2021", "2021-12-02T05:10:00Z"),
    ("v2", "Alvin Gentry Postgame", "2021-11-22T05:00:00Z"),
    ("v2b", "Kings | Full Postgame Media Availability", "2021-11-22T05:20:00Z"),
    ("v1", "Kings at Suns Postgame: Luke Walton", "2021-11-20T05:00:00Z"),
]


def ts(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


class Channel:
    def __init__(self):
        self.listings, self.extractions = 0, []

    def lister(self, url):
        self.listings += 1
        if url.endswith("/streams"):
            return []
        return [{"id": v, "title": t} for v, t, _ in VIDEOS]

    def extractor(self, video_id):
        self.extractions.append(video_id)
        vid, title, published = next(v for v in VIDEOS if v[0] == video_id)
        return {
            "id": vid,
            "title": title,
            "channel_id": "UCkings",
            "timestamp": ts(published),
            "duration": 600,
            "chapters": [],
        }


def first_candidate(payload):
    candidate = payload["candidates"][0]
    return {
        "video_id": candidate["video_id"],
        "kind": "coach",
        "coach_name": "Alvin Gentry",
        "coach_chapter": None,
        "confidence": 0.9,
    }


class Responses:
    def __init__(self):
        self.calls = 0
        self.payloads = []
        self.answer_for = first_candidate

    def create(self, **kwargs):
        self.calls += 1
        payload = json.loads(kwargs["input"])
        self.payloads.append(payload)
        answer = self.answer_for(payload)
        return SimpleNamespace(
            output_text=json.dumps(answer),
            status="completed",
            output=[],
            usage=SimpleNamespace(input_tokens=1500, output_tokens=100),
        )


class Downloader:
    def __init__(self):
        self.calls = []

    def __call__(self, video_id, directory):
        self.calls.append(video_id)
        source = directory / "source.webm"
        source.write_bytes(video_id.encode())
        return {
            "info": {"id": video_id, "channel_id": "UCkings", "duration": 600},
            "source": source,
            "captions": None,
            "format": {"format_id": "251", "acodec": "opus"},
        }


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(
        media,
        "encode_audio",
        lambda source, dest, *a, **k: dest.write_bytes(b"opus " + source.read_bytes())
        or 600.0,
    )
    monkeypatch.setattr(media, "probe_chapters", lambda path: [])

    def make(**config_kwargs):
        config = config_2021(
            tmp_path, teams=("SAC",), download_sleep=(0.0, 0.0), **config_kwargs
        )
        write_channels(
            config.channels_csv,
            [
                {
                    "team": "SAC",
                    "channel_id": "UCkings",
                    "handle": "@SacramentoKings",
                    "name": "Sacramento Kings",
                    "kind": "team",
                    "priority": 1,
                    "verified_by_human": True,
                }
            ],
        )
        client = SimpleNamespace(responses=Responses())
        judge = llm_judge.Judge(config, client=client)
        hooks = Hooks(
            payload=schedule("2021-22", GAMES),
            wikipedia=wikipedia_2021(),
            players_loader=players_2021,
            lister=channel.lister,
            extractor=channel.extractor,
            judge=judge,
            downloader=downloader,
        )
        return config, hooks, client

    channel, downloader = Channel(), Downloader()
    return make, channel, downloader


def test_dry_run_then_archive(world):
    make, channel, downloader = world
    config, hooks, client = make()
    result = run_season(config, dry_run=True, hooks=hooks)
    entries = json.loads(config.manifest_path.read_bytes())["entries"]
    walton = entries["0022100010:SAC"]
    assert walton["coach_expected"] == "Luke Walton"
    assert walton["selection"]["decided_by"] == "rules"
    assert walton["selection"]["video_id"] == "v1"
    # Two candidates for Nov 21: the judge chose the coach-only video.
    assert entries["0022100020:SAC"]["selection"]["video_id"] == "v2"
    assert entries["0022100020:SAC"]["selection"]["decided_by"] == "llm"
    assert entries["0022100030:SAC"]["selection"]["video_id"] == "v3"
    assert entries["0042100101:SAC"]["status"] == "not_found"
    assert result["llm"]["calls"] == 1
    # Step 4 skips the highlights; step 6b fetches them while looking for the
    # playoff game's presser, finds them outside its window and asks nothing.
    assert channel.extractions.count("v4") == 1
    assert "0042100101:SAC" not in json.loads(config.candidates_path.read_bytes())
    assert downloader.calls == []
    for name in ("coverage.csv", "missing.csv", "review.csv", "coach_changes.csv"):
        assert (config.reports_dir / name).is_file()

    s3 = ConditionalFakeS3()
    remote, hooks, client = make(bucket="b")
    result = run_season(remote, dry_run=False, client=s3, hooks=hooks)
    assert sorted(downloader.calls) == ["v1", "v2", "v3"]
    assert result["summary"]["by_status"]["uploaded"] == 3
    assert client.responses.calls == 0  # the judge's answer came from the cache
    keys = set(s3.objects)
    assert remote.season_file_key("manifest.json") in keys
    assert remote.shared_key("llm_decisions.jsonl") in keys
    assert remote.shared_key("channels/UCkings/videos.jsonl.gz") in keys

    again = run_season(remote, dry_run=False, client=s3, hooks=hooks)
    assert sorted(downloader.calls) == ["v1", "v2", "v3"]
    assert again["summary"]["by_status"]["uploaded"] == 3


def test_unverified_channels_stop_at_checkpoint_one(world):
    make, channel, downloader = world
    config, hooks, client = make()
    rows = [
        {
            "team": "SAC",
            "channel_id": "UCkings",
            "handle": "@k",
            "name": "Sacramento Kings",
            "kind": "team",
            "priority": 1,
            "verified_by_human": False,
        }
    ]
    write_channels(config.channels_csv, rows)
    result = run_season(config, dry_run=True, hooks=hooks)
    assert result["stopped"] == "checkpoint 1: channels not verified"
    assert channel.listings == 0
    assert result["summary"]["by_status"] == {"pending": 8}


@pytest.mark.parametrize("decision", ["coach", "none", "disabled"])
def test_flagged_video_is_decided_by_the_llm_or_left_for_review(
    world, monkeypatch, decision
):
    title = "Alvin Gentry on Tonight's Win vs Suns | 12.01.2021"
    monkeypatch.setitem(
        globals(),
        "VIDEOS",
        [(vid, title if vid == "v3" else text, pub) for vid, text, pub in VIDEOS],
    )
    make, channel, downloader = world
    config, hooks, client = make(use_llm=decision != "disabled")
    if decision == "none":
        client.responses.answer_for = lambda payload: {
            **first_candidate(payload),
            "video_id": None,
            "kind": "none",
            "coach_name": None,
        }
    run_season(config, dry_run=True, hooks=hooks)
    entry = json.loads(config.manifest_path.read_bytes())["entries"]["0022100030:SAC"]
    assert "v3" in channel.extractions and downloader.calls == []
    if decision == "coach":
        assert entry["status"] == "selected"
        assert entry["selection"]["decided_by"] == "llm"
        assert entry["selection"]["video_id"] == "v3"
    else:
        assert entry["status"] == "review" and entry["selection"] is None
    if decision != "disabled":
        payload = next(
            p for p in client.responses.payloads if p["game"]["date"] == "2021-12-01"
        )
        assert payload["candidates"][0]["requires_llm"] is True
        assert "keyword:tonights" in payload["candidates"][0]["rule_flags"]
