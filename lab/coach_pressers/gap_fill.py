"""Step 6b: a second look at the games the rules left without a candidate.

For each ``not_found`` game, every upload of the team's official channels
published in that game's window gets metadata and goes to the judge, whatever
its title says. Outlet channels get no new requests (their pressers nearly
always name the coach, so step 4 already found them); their step 4 videos join
the pool at no cost. The rules only decide which games need this second look;
the judge decides what each video is. A judge's "none" leaves the game
``not_found``.

The window is found from the dates already cached for the channel (the step 4
metadata), so a game costs the requests for the uploads of its own window plus
a few binary-search probes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import date, datetime, timedelta

from lab.coach_pressers import crawl
from lab.coach_pressers.channels import team_channels
from lab.coach_pressers.config import PresserConfig
from lab.coach_pressers.match import candidates_for_team, rival_head_coaches
from lab.game_reports.season_reports import atomic_write, json_bytes

log = logging.getLogger("coach_pressers")

MAX_CANDIDATES = 15
BOUND_MARGIN = 2
CLASS_ORDER = {"coach": 0, "combined": 1, "player": 2, "other": 3}


def gap_keys(manifest: dict, teams) -> dict[str, list[str]]:
    """Team -> keys of its ``not_found`` games, except those a person closed."""
    out: dict[str, list[str]] = {}
    for key, entry in manifest["entries"].items():
        if (
            entry["team"] in teams
            and entry["status"] == "not_found"
            and not (entry.get("error") or "").startswith("human:")
        ):
            out.setdefault(entry["team"], []).append(key)
    return out


def merged_windows(entries: list[dict], today: date) -> list[tuple[date, date]]:
    """Publication-date windows of the games, overlapping ones merged."""
    spans = sorted(
        (
            date.fromisoformat(e["date"]),
            min(
                date.fromisoformat(e["date"]) + timedelta(days=crawl.WINDOW_DAYS), today
            ),
        )
        for e in entries
    )
    merged: list[list[date]] = []
    for first, last in spans:
        if merged and first <= merged[-1][1] + timedelta(days=1):
            merged[-1][1] = max(merged[-1][1], last)
        else:
            merged.append([first, last])
    return [(first, last) for first, last in merged if first <= last]


def known_bounds(
    ids: list[str], videos: dict[str, dict], first: date, last: date
) -> tuple[int, int]:
    """Positions [lo, hi) of newest-first ``ids`` that can hold [first, last].

    Uses only dates already cached: the newest upload known to be after
    ``last`` and the oldest known to be before ``first`` bound the slice.
    """
    lo, hi = 0, len(ids)
    for position, video_id in enumerate(ids):
        day = (videos.get(video_id) or {}).get("published_date")
        if not day:
            continue
        if day > last.isoformat():
            lo = position + 1
        elif day < first.isoformat():
            hi = position
            break
    return max(0, lo - BOUND_MARGIN), min(len(ids), max(lo, hi) + BOUND_MARGIN)


def window_uploads(
    config: PresserConfig,
    row: dict,
    windows: list[tuple[date, date]],
    extractor: Callable[[str], dict],
) -> tuple[list[dict], int]:
    """Archivable uploads of one channel published inside the windows."""
    cache = crawl.ChannelCache(config.channels_dir, row["channel_id"])
    index = cache.index()
    if index is None:
        return [], 0
    source = crawl.MetadataSource(cache, extractor)
    records = []
    for tab in crawl.TABS:
        ids = [v["id"] for v in index["tabs"].get(tab, [])]
        for first, last in windows:
            lo, hi = known_bounds(ids, cache.videos(), first, last)
            records.extend(
                crawl.tagged(record, row)
                for record in crawl.in_window(ids[lo:hi], first, last, source)
                if crawl.archivable(record)
            )
    return records, source.fetched


def _order(candidate: dict) -> tuple:
    return (
        CLASS_ORDER.get(candidate["class"], 9),
        candidate["channel_priority"],
        candidate["published_at_utc"] or candidate["published_date"] or "",
        candidate["video_id"],
    )


def fill_gaps(
    config: PresserConfig,
    manifest: dict,
    coaches: dict,
    channel_rows: list[dict],
    videos: dict[str, list[dict]],
    *,
    extractor: Callable[[str], dict] | None = None,
    today: date | None = None,
) -> dict[str, list[dict]]:
    """Entry key -> window candidates for every gap the judge should look at."""
    today = today or datetime.now(crawl.NEW_YORK).date()
    gaps = gap_keys(manifest, set(videos))
    if not gaps:
        return {}
    extractor = extractor or crawl.default_extractor(config)
    out: dict[str, list[dict]] = {}
    fetched = 0
    for team, keys in sorted(gaps.items()):
        team_data = coaches["teams"][team]
        windows = merged_windows([manifest["entries"][k] for k in keys], today)
        pool = {v["id"]: v for v in videos.get(team, [])}
        for row in team_channels(channel_rows, team):
            if row.get("kind", "team") != "team":
                continue  # outlets: only their step 4 videos, no new requests
            records, requests = window_uploads(config, row, windows, extractor)
            fetched += requests
            for record in records:
                pool.setdefault(record["id"], record)
        found = candidates_for_team(
            team,
            list(pool.values()),
            manifest,
            team_data,
            rival_head_coaches(coaches["teams"], team),
            keep_all=True,
        )
        for key in keys:
            candidates = sorted(found.get(key, []), key=_order)[:MAX_CANDIDATES]
            for candidate in candidates:
                candidate["requires_llm"] = True
                candidate["notes"] = [*candidate["notes"], "gap_fill"]
            if candidates:
                out[key] = candidates
    if out:
        stored = (
            json.loads(config.candidates_path.read_bytes())
            if config.candidates_path.is_file()
            else {}
        )
        stored.update(out)
        atomic_write(config.candidates_path, json_bytes(stored))
    log.info(
        "step 6b %s: %d games without candidates, %d with window uploads for the "
        "judge, %d new metadata requests",
        config.season,
        sum(len(k) for k in gaps.values()),
        len(out),
        fetched,
    )
    return out
