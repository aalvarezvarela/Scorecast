"""Step 4 of the coach presser archive: channel listings and metadata cache."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from lab.coach_pressers import crawl
from lab.coach_pressers.coaches import staff_entry

BASE = date(2021, 10, 20)


def timestamp(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, 6, tzinfo=UTC).timestamp())


class FakeChannel:
    """200 days of uploads, newest first: one coach presser and one highlight a day."""

    def __init__(self, days=200, channel_id="UC1"):
        self.channel_id, self.videos = channel_id, []
        for offset in reversed(range(days)):
            day = BASE + timedelta(days=offset)
            self.videos.append(
                {
                    "id": f"c{offset}",
                    "title": f"Mike Brown Postgame | {day:%m.%d.%Y}",
                    "day": day,
                }
            )
            self.videos.append(
                {"id": f"h{offset}", "title": "Kings Highlights", "day": day}
            )
        self.listings, self.extractions = 0, []

    def lister(self, url):
        self.listings += 1
        if url.endswith("/streams"):
            raise RuntimeError("This channel does not have a streams tab")
        return [{"id": v["id"], "title": v["title"]} for v in self.videos]

    def extractor(self, video_id):
        self.extractions.append(video_id)
        video = next(v for v in self.videos if v["id"] == video_id)
        if video_id == "c50":
            raise RuntimeError("Private video")
        return {
            "id": video_id,
            "title": video["title"],
            "channel_id": self.channel_id,
            "timestamp": timestamp(video["day"]),
            "duration": 600,
            "description": "0:00 Mike Brown",
            "chapters": [{"start_time": 0, "end_time": 600, "title": "Mike Brown"}],
            "formats": [{"url": "https://signed"}],
        }


def entries(first: date, last: date, status="pending"):
    out, day = [], first
    while day < last:
        out.append({"team": "SAC", "date": day.isoformat(), "status": status})
        day += timedelta(days=7)
    out.append({"team": "SAC", "date": last.isoformat(), "status": status})
    return out


@pytest.fixture
def setup(tmp_path):
    channel = FakeChannel()
    config_root = tmp_path / "channels"
    coaches = {
        "teams": {
            "SAC": {"staff": [staff_entry("Mike Brown", "head", "t")], "players": []}
        }
    }
    rows = [{"team": "SAC", "channel_id": "UC1", "priority": 1, "handle": "@k"}]
    return channel, config_root, coaches, rows


def run(channel, root, coaches, rows, manifest_entries, today):
    class Config:
        teams = ("SAC",)
        season = "2021-22"
        channels_dir = root

    manifest = {"entries": {str(i): e for i, e in enumerate(manifest_entries)}}
    return crawl.crawl_season(
        Config(),
        manifest,
        coaches,
        rows,
        lister=channel.lister,
        extractor=channel.extractor,
        today=today,
    )


def test_only_the_season_window_is_fetched(setup):
    channel, root, coaches, rows = setup
    first, last = BASE + timedelta(days=100), BASE + timedelta(days=120)
    found = run(channel, root, coaches, rows, entries(first, last), date(2022, 9, 1))
    ids = {r["id"] for r in found["SAC"]}
    # Window = first game .. last game + 3 days; c50 never matters here.
    assert ids == {f"c{d}" for d in range(100, 124)}
    assert not any(v.startswith("h") for v in channel.extractions)
    assert len(channel.extractions) < 24 + 2 * 20  # probes + window + margins
    record = found["SAC"][0]
    assert record["chapters"] == [
        {"start_time": 0.0, "end_time": 600.0, "title": "Mike Brown"}
    ]
    assert "formats" not in record and record["published_date"] == "2022-01-28"


def test_rerun_uses_the_caches(setup):
    channel, root, coaches, rows = setup
    first, last = BASE + timedelta(days=100), BASE + timedelta(days=120)
    today = date(2022, 9, 1)
    run(channel, root, coaches, rows, entries(first, last), today)
    listed, fetched = channel.listings, len(channel.extractions)
    again = run(channel, root, coaches, rows, entries(first, last), today)
    assert channel.listings == listed and len(channel.extractions) == fetched
    assert len(again["SAC"]) == 24


def test_unavailable_videos_are_cached_as_errors(setup):
    channel, root, coaches, rows = setup
    first, last = BASE + timedelta(days=45), BASE + timedelta(days=55)
    found = run(channel, root, coaches, rows, entries(first, last), date(2022, 9, 1))
    assert "c50" not in {r["id"] for r in found["SAC"]}
    cache = crawl.ChannelCache(root, "UC1")
    assert "Private video" in cache.videos()["c50"]["error"]


def test_listing_is_refreshed_only_when_a_window_closed_after_it():
    index = {"listed_at_utc": "2022-01-10T12:00:00+00:00"}
    old = [{"date": "2022-01-01", "status": "pending"}]
    assert not crawl.needs_listing(index, old, date(2022, 1, 20))
    recent = [{"date": "2022-01-08", "status": "pending"}]
    assert crawl.needs_listing(index, recent, date(2022, 1, 20))
    assert not crawl.needs_listing(index, recent, date(2022, 1, 11))
    done = [{"date": "2022-01-08", "status": "uploaded"}]
    assert not crawl.needs_listing(index, done, date(2022, 1, 20))


def test_truncated_listing_is_refused(tmp_path):
    cache = crawl.ChannelCache(tmp_path, "UC1")
    cache.save_index(
        {
            "channel_id": "UC1",
            "listed_at_utc": "2022-01-01T00:00:00+00:00",
            "tabs": {"videos": [{"id": str(i), "title": ""} for i in range(1000)]},
        }
    )
    short = [{"date": "2022-01-05", "status": "pending"}]
    with pytest.raises(RuntimeError, match="truncated"):
        crawl.channel_index(cache, short, lambda url: [], date(2022, 2, 1))
