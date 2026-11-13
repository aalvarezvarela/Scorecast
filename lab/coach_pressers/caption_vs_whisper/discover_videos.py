"""Step 1: pick coach postgame pressers with English captions, a few per season.

    python lab/coach_pressers/caption_vs_whisper/discover_videos.py

Searches YouTube for one presser per query (different coaches and channels in
every season), keeps videos of 4-20 minutes uploaded inside the season that
carry English captions, and writes ``videos.json`` under the probe folder.
Resumable: seasons that already have enough videos are skipped.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import yt_dlp
from common import DEFAULT_ROOT, youtube_options

QUERIES = {
    "2019-20": [
        "Nick Nurse postgame press conference 2020",
        "Erik Spoelstra postgame press conference 2020",
        "Frank Vogel postgame press conference 2020",
        "Brad Stevens postgame press conference 2019",
        "Mike Malone postgame press conference 2020",
    ],
    "2020-21": [
        "Monty Williams postgame press conference 2021",
        "Quin Snyder postgame press conference 2021",
        "Doc Rivers postgame press conference 2021",
        "Steve Nash postgame press conference 2021",
        "Tom Thibodeau postgame press conference 2021",
    ],
    "2021-22": [
        "Steve Kerr postgame press conference 2022",
        "Ime Udoka postgame press conference 2022",
        "Taylor Jenkins postgame press conference 2022",
        "J.B. Bickerstaff postgame press conference 2022",
        "Dwane Casey postgame press conference 2022",
    ],
    "2022-23": [
        "Michael Malone postgame press conference 2023",
        "Mike Brown postgame press conference Kings 2023",
        "Gregg Popovich postgame press conference 2023",
        "Tom Thibodeau postgame press conference 2023",
        "Joe Mazzulla postgame press conference 2023",
    ],
    "2023-24": [
        "Mark Daigneault postgame press conference 2024",
        "Chris Finch postgame press conference 2024",
        "Jason Kidd postgame press conference 2024",
        "Joe Mazzulla postgame press conference 2024",
        "Rick Carlisle postgame press conference 2024",
    ],
    "2024-25": [
        "Kenny Atkinson postgame press conference 2025",
        "JJ Redick postgame press conference 2025",
        "Ime Udoka postgame press conference Rockets 2025",
        "Rick Carlisle postgame press conference 2025",
        "Will Hardy postgame press conference 2025",
    ],
    "2025-26": [
        "Mike Brown Knicks Postgame Press Conference 2026",
        "JJ Redick postgame press conference 2026",
        "Joe Mazzulla postgame press conference 2026",
        "Kenny Atkinson Post Game 2026",
        "Nuggets Postgame Media Coach Adelman 2026",
        "Mitch Johnson postgame Spurs 2026",
    ],
}
TITLE = re.compile(r"post[- ]?game|press conference|presser", re.I)
MIN_SECONDS, MAX_SECONDS = 240, 1200


def in_season(upload_date: str, season: str) -> bool:
    start = int(season[:4])
    return f"{start}1001" <= upload_date <= f"{start + 1}1015"


def candidate(info: dict, season: str, query: str) -> dict | None:
    manual = sorted(k for k in (info.get("subtitles") or {}) if k.startswith("en"))
    auto = sorted(
        k for k in (info.get("automatic_captions") or {}) if k in ("en", "en-orig")
    )
    if not in_season(info.get("upload_date") or "", season):
        return None
    if not (manual or auto) or (info.get("language") or "en")[:2] != "en":
        return None
    return {
        "season": season,
        "id": info["id"],
        "title": info["title"],
        "channel": info.get("channel"),
        "upload_date": info["upload_date"],
        "duration": info["duration"],
        "manual_subs": manual,
        "auto_captions": auto,
        "query": query,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--per-season", type=int, default=3)
    parser.add_argument("--js-runtime", default="deno", help="name on PATH or path")
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    out = args.root / "videos.json"
    picked = json.loads(out.read_text()) if out.exists() else []
    base = youtube_options(args.root, args.js_runtime)
    flat = yt_dlp.YoutubeDL({**base, "extract_flat": True})
    full = yt_dlp.YoutubeDL({**base, "skip_download": True})
    for season, queries in QUERIES.items():
        have = sum(p["season"] == season for p in picked)
        for query in queries:
            if have >= args.per_season:
                break
            time.sleep(1.5)
            try:
                found = flat.extract_info(f"ytsearch8:{query}", download=False)
            except Exception as error:  # noqa: BLE001
                print(season, query, "search failed:", error, flush=True)
                continue
            for entry in found.get("entries") or []:
                duration = entry.get("duration") or 0
                if not MIN_SECONDS <= duration <= MAX_SECONDS:
                    continue
                if not TITLE.search(entry["title"]):
                    continue
                if any(p["id"] == entry["id"] for p in picked):
                    continue
                time.sleep(1.5)
                try:
                    info = full.extract_info(entry["id"], download=False)
                except Exception as error:  # noqa: BLE001
                    print(season, entry["id"], "metadata failed:", error, flush=True)
                    continue
                row = candidate(info, season, query)
                if row is None:
                    continue
                picked.append(row)
                have += 1
                out.write_text(json.dumps(picked, indent=1))
                print(json.dumps(row), flush=True)
                break  # one video per query: a different coach each time
    print("total", len(picked))


if __name__ == "__main__":
    main()
