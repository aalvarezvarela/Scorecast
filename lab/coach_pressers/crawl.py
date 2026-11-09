"""Step 4: the video inventory of each team channel, cached across seasons.

A flat listing of the ``/videos`` and ``/streams`` tabs gives id, title and
duration of every upload, newest first, but no reliable date. The listing is
cached per channel and refreshed only when a pending game's upload window has
closed after it was taken (as in the recap channel index).

Titles are pre-filtered with the classifier; only survivors get a full
metadata request (date, description, chapters). Because each tab is ordered
by publication, the survivors that belong to one season are found with a
binary search on their position, so a season run fetches metadata for its own
uploads plus a few probes, not for the whole channel history.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from lab.coach_pressers.channels import team_channels
from lab.coach_pressers.classify import prefilter
from lab.coach_pressers.config import PresserConfig
from lab.coach_pressers.manifest import MATCHABLE
from lab.game_reports.season_reports import (
    RequestPacer,
    atomic_write,
    json_bytes,
    utc_now,
)
from lab.youtube_downloader import youtube_media as media

log = logging.getLogger("coach_pressers")

TABS = ("videos", "streams")
WINDOW_DAYS = 3
SEARCH_MARGIN = 5
DESCRIPTION_CHARS = 5000
KEEP = (
    "id",
    "channel_id",
    "channel",
    "title",
    "timestamp",
    "release_timestamp",
    "upload_date",
    "duration",
    "live_status",
    "was_live",
    "availability",
    "language",
)
NEW_YORK = ZoneInfo("America/New_York")


# Uploads that need a membership or a login (outlets often gate pressers).
RESTRICTED = {"subscriber_only", "premium_only", "needs_auth"}


def trim_info(info: dict) -> dict:
    """The metadata kept for matching; never formats or signed URLs."""
    record = {k: info.get(k) for k in KEEP}
    record["description"] = (info.get("description") or "")[:DESCRIPTION_CHARS]
    record["chapters"] = [
        {
            "start_time": float(c.get("start_time") or 0),
            "end_time": float(c.get("end_time") or 0),
            "title": c.get("title") or "",
        }
        for c in info.get("chapters") or []
    ]
    published_day, published_at = media.publication(info)
    record["published_date"] = published_day.isoformat() if published_day else None
    record["published_at_utc"] = published_at
    record["fetched_at_utc"] = utc_now()
    return record


class ChannelCache:
    def __init__(self, root: Path, channel_id: str):
        self.directory = root / channel_id
        self.channel_id = channel_id
        self._videos: dict[str, dict] | None = None

    @property
    def index_path(self) -> Path:
        return self.directory / "index.json"

    @property
    def videos_path(self) -> Path:
        return self.directory / "videos.jsonl.gz"

    def index(self) -> dict | None:
        if not self.index_path.is_file():
            return None
        return json.loads(self.index_path.read_bytes())

    def save_index(self, index: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_write(self.index_path, json_bytes(index))

    def videos(self) -> dict[str, dict]:
        if self._videos is None:
            self._videos = {}
            if self.videos_path.is_file():
                with gzip.open(self.videos_path, "rt", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            record = json.loads(line)
                            self._videos[record["id"]] = record
        return self._videos

    def add(self, record: dict) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.videos_path, "at", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.videos()[record["id"]] = record


def default_lister(config: PresserConfig) -> Callable[[str], list[dict]]:
    options = media.youtube_options(config.root, config.js_runtime)
    pacer = RequestPacer(config.youtube_interval)

    def lister(url: str) -> list[dict]:
        return media.list_tab(url, options, pacer)

    return lister


def default_extractor(config: PresserConfig) -> Callable[[str], dict]:
    from yt_dlp import YoutubeDL

    options = {
        **media.youtube_options(config.root, config.js_runtime),
        "skip_download": True,
        "ignore_no_formats_error": True,
    }
    pacer = RequestPacer(config.youtube_interval)

    def extractor(video_id: str) -> dict:
        pacer.wait()
        with YoutubeDL(options) as ydl:
            return ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", download=False
            )

    return extractor


def list_channel(channel_id: str, lister: Callable[[str], list[dict]]) -> dict:
    tabs = {}
    for tab in TABS:
        url = f"https://www.youtube.com/channel/{channel_id}/{tab}"
        try:
            tabs[tab] = lister(url)
        except Exception as exc:
            if tab == "streams":  # many channels have no streams tab
                log.info("%s has no %s tab (%s)", channel_id, tab, exc)
                tabs[tab] = []
            else:
                raise
    return tabs


def needs_listing(index: dict | None, entries: list[dict], today: date) -> bool:
    """True if some pending game's window closed after the cached listing."""
    if index is None:
        return True
    listed = datetime.fromisoformat(index["listed_at_utc"]).astimezone(NEW_YORK).date()
    for entry in entries:
        if entry["status"] not in MATCHABLE:
            continue
        window_end = date.fromisoformat(entry["date"]) + timedelta(days=WINDOW_DAYS)
        if listed <= window_end < today:
            return True
    return False


def channel_index(
    cache: ChannelCache,
    entries: list[dict],
    lister: Callable[[str], list[dict]],
    today: date,
) -> dict:
    cached = cache.index()
    if not needs_listing(cached, entries, today):
        return cached
    listed_at = utc_now()
    tabs = list_channel(cache.channel_id, lister)
    total = sum(len(v) for v in tabs.values())
    if cached:
        before = sum(len(v) for v in cached["tabs"].values())
        if total < 0.99 * before:
            raise RuntimeError(
                f"Listing of {cache.channel_id} returned {total} uploads, fewer than "
                f"the {before} cached; YouTube may have truncated it, retry later"
            )
    index = {"channel_id": cache.channel_id, "listed_at_utc": listed_at, "tabs": tabs}
    cache.save_index(index)
    log.info("step 4: listed %s: %d uploads", cache.channel_id, total)
    return index


class MetadataSource:
    """Full metadata of one video: cache first, then one request."""

    def __init__(self, cache: ChannelCache, extractor: Callable[[str], dict]):
        self.cache, self.extractor, self.fetched = cache, extractor, 0

    def record(self, video_id: str) -> dict:
        cached = self.cache.videos().get(video_id)
        if cached is not None:
            return cached
        try:
            info = self.extractor(video_id)
            record = trim_info(info) if info else {"id": video_id, "error": "empty"}
        except Exception as exc:  # removed, private, members-only, region
            record = {
                "id": video_id,
                "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                "fetched_at_utc": utc_now(),
            }
        record["id"] = video_id
        self.cache.add(record)
        self.fetched += 1
        return record

    def day(self, video_id: str) -> date | None:
        value = self.record(video_id).get("published_date")
        return date.fromisoformat(value) if value else None


def _first_at_or_before(ids: list[str], limit: date, source: MetadataSource) -> int:
    """Smallest position whose publication date is <= limit (dates decrease)."""

    def day_at(position: int) -> date:
        for i in range(position, len(ids)):
            found = source.day(ids[i])
            if found is not None:
                return found
        return date.min

    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if day_at(mid) <= limit:
            hi = mid
        else:
            lo = mid + 1
    return lo


def in_window(
    ids: list[str], first: date, last: date, source: MetadataSource
) -> list[dict]:
    """Records of the listed ids published within [first, last]."""
    if not ids:
        return []
    start = _first_at_or_before(ids, last, source)
    end = _first_at_or_before(ids, first - timedelta(days=1), source)
    start, end = max(0, start - SEARCH_MARGIN), min(len(ids), end + SEARCH_MARGIN)
    records = []
    for video_id in ids[start:end]:
        record = source.record(video_id)
        day = record.get("published_date")
        if day and first.isoformat() <= day <= last.isoformat():
            records.append(record)
    return records


def team_window(entries: list[dict], today: date) -> tuple[date, date]:
    first = min(date.fromisoformat(e["date"]) for e in entries)
    last = max(date.fromisoformat(e["date"]) for e in entries)
    return first, min(last + timedelta(days=WINDOW_DAYS), today)


def crawl_season(
    config: PresserConfig,
    manifest: dict,
    coaches: dict,
    channel_rows: list[dict],
    *,
    lister: Callable[[str], list[dict]] | None = None,
    extractor: Callable[[str], dict] | None = None,
    today: date | None = None,
) -> dict[str, list[dict]]:
    """Team -> metadata of the pre-filtered videos published in its season."""
    today = today or datetime.now(NEW_YORK).date()
    teams = config.teams or tuple(sorted(coaches["teams"]))
    lister = lister or default_lister(config)
    extractor = extractor or default_extractor(config)
    found: dict[str, list[dict]] = {}
    for team in teams:
        entries = [
            e
            for e in manifest["entries"].values()
            if e["team"] == team and e["status"] != "not_final"
        ]
        rows = team_channels(channel_rows, team)
        if not entries or not rows:
            if not rows:
                log.warning("step 4 %s: no channel in the registry", team)
            continue
        team_data = coaches["teams"][team]
        first, last = team_window(entries, today)
        records = []
        for row in rows:
            cache = ChannelCache(config.channels_dir, row["channel_id"])
            index = channel_index(cache, entries, lister, today)
            source = MetadataSource(cache, extractor)
            outlet = row.get("kind", "team") != "team"
            for tab in TABS:
                ids = [
                    v["id"]
                    for v in index["tabs"].get(tab, [])
                    if prefilter(
                        v["title"],
                        team_data["staff"],
                        team_data["players"],
                        team=team if outlet else None,
                    )
                ]
                for record in in_window(ids, first, last, source):
                    if record.get("live_status") in ("is_upcoming", "is_live"):
                        continue
                    if record.get("availability") in RESTRICTED:
                        continue  # members-only uploads cannot be archived
                    records.append(
                        {
                            **record,
                            "channel_priority": row["priority"],
                            "channel_kind": row.get("kind", "team"),
                        }
                    )
            log.info(
                "step 4 %s %s: %d candidates in %s..%s, %d new metadata requests",
                config.season,
                row["handle"],
                len(records),
                first,
                last,
                source.fetched,
            )
        unique = {r["id"]: r for r in records}
        found[team] = sorted(unique.values(), key=lambda r: r["published_date"])
    return found
