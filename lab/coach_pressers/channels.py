"""Step 3: the channel registry (channels.csv), shared by every season.

Handles are never trusted blindly: each candidate is resolved with yt-dlp and
kept only if the channel's name contains the team's nickname. Every row starts
with ``verified_by_human=false``; the user confirms the list at checkpoint 1.
Secondary channels (local TV, outlets) are added by hand later, with a higher
``priority`` number (1 = the official team channel).
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Callable
from pathlib import Path

from lab.coach_pressers.names import normalize
from lab.coach_pressers.teams import TEAMS
from lab.game_reports.season_reports import RequestPacer, utc_now

log = logging.getLogger("coach_pressers")

FIELDS = (
    "team",
    "channel_id",
    "handle",
    "name",
    "kind",
    "priority",
    "verified_by_human",
    "followers",
    "resolved_at_utc",
)

# Candidate handles per team. Resolution decides: a handle must name the team
# and have MIN_FOLLOWERS (fake channels with the exact team name exist, e.g. a
# 117-follower "Dallas Mavericks" at @dallasmavs); the biggest one wins.
MIN_FOLLOWERS = 10_000
CANDIDATE_HANDLES: dict[str, tuple[str, ...]] = {
    "ATL": ("ATLHawks", "hawks"),
    "BOS": ("celtics", "BostonCeltics"),
    "BKN": ("BrooklynNets", "nets"),
    "CHA": ("hornets", "CharlotteHornets"),
    "CHI": ("chicagobulls", "bulls"),
    "CLE": ("cavs", "Cavaliers"),
    "DAL": ("mavericks", "dallasmavs"),
    "DEN": ("nuggets", "DenverNuggets"),
    "DET": ("PistonsTV", "DetroitPistons"),
    "GSW": ("warriors", "GoldenStateWarriors"),
    "HOU": ("OfficialRockets", "HoustonRockets"),
    "IND": ("Pacers", "IndianaPacers"),
    "LAC": ("LAClippers", "clippers"),
    "LAL": ("Lakers", "LosAngelesLakers"),
    "MEM": ("MemphisGrizzlies", "grizzlies"),
    "MIA": ("MiamiHEAT", "heat"),
    "MIL": ("Bucks", "MilwaukeeBucks"),
    "MIN": ("Timberwolves", "MinnesotaTimberwolves"),
    "NOP": ("NBAPelicans", "PelicansNBA"),
    "NYK": ("NYKnicks", "knicks"),
    "OKC": ("OKCThunder", "thunder"),
    "ORL": ("OrlandoMagic", "magic"),
    "PHI": ("Sixers", "76ers"),
    "PHX": ("Suns", "PhoenixSuns"),
    "POR": ("trailblazers", "PortlandTrailBlazers"),
    "SAC": ("SacramentoKings", "kings"),
    "SAS": ("Spurs", "SanAntonioSpurs"),
    "TOR": ("TorontoRaptors", "Raptors"),
    "UTA": ("utahjazz", "jazz"),
    "WAS": ("WashingtonWizards", "WashWizards"),
}


def read_channels(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["priority"] = int(row.get("priority") or 1)
        row["verified_by_human"] = str(row.get("verified_by_human")).lower() == "true"
    return rows


def write_channels(path: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["team"], int(r["priority"]), r["handle"]))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "verified_by_human": str(bool(row["verified_by_human"])).lower(),
                }
            )
    temporary.replace(path)


def team_channels(rows: list[dict], team: str) -> list[dict]:
    return sorted(
        [r for r in rows if r["team"] == team and r.get("channel_id")],
        key=lambda r: r["priority"],
    )


def default_extract(js_runtime: str, cache_dir: Path) -> Callable[[str], dict]:
    from yt_dlp import YoutubeDL

    from lab.youtube_downloader.youtube_media import youtube_options

    options = {**youtube_options(cache_dir, js_runtime), "skip_download": True}

    def extract(url: str) -> dict:
        with YoutubeDL(options) as ydl:
            return ydl.extract_info(url, download=False, process=False)

    return extract


def matches_team(info: dict, team: str) -> bool:
    nickname = normalize(TEAMS[team][1])
    names = (info.get("channel"), info.get("uploader"), info.get("title"))
    return any(nickname in normalize(str(n or "")) for n in names)


def channel_row(team: str, info: dict, handle: str) -> dict:
    return {
        "team": team,
        "channel_id": info.get("channel_id") or info.get("id"),
        "handle": handle,
        "name": info.get("channel") or info.get("uploader") or "",
        "kind": "team",
        "priority": 1,
        "verified_by_human": False,
        "followers": info.get("channel_follower_count") or "",
        "resolved_at_utc": utc_now(),
    }


def qualifies(info: dict | None, team: str) -> bool:
    followers = (info or {}).get("channel_follower_count") or 0
    return bool(info) and matches_team(info, team) and followers >= MIN_FOLLOWERS


def resolve_team(
    team: str,
    extract: Callable[[str], dict],
    pacer: RequestPacer,
    search: Callable[[str], list[dict]] | None = None,
) -> tuple[dict | None, list[str]]:
    """The team's official channel: best qualifying handle, else a search."""
    tried, found = [], []
    for handle in CANDIDATE_HANDLES[team]:
        pacer.wait()
        try:
            info = extract(f"https://www.youtube.com/@{handle}")
        except Exception as exc:
            tried.append(f"@{handle}: {type(exc).__name__}")
            continue
        if qualifies(info, team):
            found.append(channel_row(team, info, f"@{handle}"))
        else:
            tried.append(
                f"@{handle}: {(info or {}).get('channel')!r}, "
                f"{(info or {}).get('channel_follower_count')} followers"
            )
    if not found and search is not None:
        pacer.wait()
        entries = search(f"{TEAMS[team][0]} postgame press conference")
        counts: dict[str, int] = {}
        for entry in entries:
            if entry.get("channel_id") and matches_team(entry, team):
                counts[entry["channel_id"]] = counts.get(entry["channel_id"], 0) + 1
        for channel_id in sorted(counts, key=counts.get, reverse=True)[:3]:
            pacer.wait()
            try:
                info = extract(f"https://www.youtube.com/channel/{channel_id}")
            except Exception as exc:
                tried.append(f"{channel_id}: {type(exc).__name__}")
                continue
            if qualifies(info, team):
                handle = info.get("uploader_id") or channel_id
                found.append(channel_row(team, info, handle))
            else:
                tried.append(f"{channel_id} (search): not qualifying")
    if not found:
        return None, tried
    best = max(found, key=lambda r: int(r["followers"] or 0))
    return best, tried


def default_search(js_runtime: str, cache_dir: Path) -> Callable[[str], list[dict]]:
    from yt_dlp import YoutubeDL

    from lab.youtube_downloader.youtube_media import youtube_options

    options = {
        **youtube_options(cache_dir, js_runtime),
        "extract_flat": True,
        "skip_download": True,
    }

    def search(query: str) -> list[dict]:
        with YoutubeDL(options) as ydl:
            result = ydl.extract_info(f"ytsearch20:{query}", download=False)
        return [e for e in (result or {}).get("entries", []) if e]

    return search


def resolve_all(
    path: Path,
    extract: Callable[[str], dict],
    pacer: RequestPacer,
    teams: list[str] | None = None,
    *,
    search: Callable[[str], list[dict]] | None = None,
    refresh: bool = False,
) -> tuple[list[dict], dict[str, list[str]]]:
    """Resolve the official channel of every team that has none yet.

    ``refresh`` re-resolves every official row not yet verified by a human.
    """
    rows = read_channels(path)
    if refresh:
        rows = [r for r in rows if r["kind"] != "team" or r["verified_by_human"]]
    have = {r["team"] for r in rows if r["kind"] == "team" and r.get("channel_id")}
    failures = {}
    for team in teams or sorted(TEAMS):
        if team in have:
            continue
        row, tried = resolve_team(team, extract, pacer, search)
        if row:
            rows.append(row)
            log.info("%s -> %s (%s)", team, row["handle"], row["name"])
        else:
            failures[team] = tried
            log.warning("%s: no channel resolved (%s)", team, "; ".join(tried))
    write_channels(path, rows)
    return rows, failures


def table(rows: list[dict]) -> str:
    lines = [
        f"{'team':4}  {'handle':24}  {'channel name':32}  {'channel id':24}  "
        f"{'followers':>9}  verified"
    ]
    for r in sorted(rows, key=lambda r: (r["team"], r["priority"])):
        lines.append(
            f"{r['team']:4}  {r['handle']:24}  {r['name'][:32]:32}  "
            f"{r['channel_id']:24}  {str(r.get('followers') or ''):>9}  "
            f"{'yes' if r['verified_by_human'] else 'no'}"
        )
    return "\n".join(lines)


def main(args) -> int:
    if args.resolve:
        rows, failures = resolve_all(
            args.csv,
            default_extract(args.js_runtime, args.local_root),
            RequestPacer(2.0),
            search=default_search(args.js_runtime, args.local_root),
            refresh=args.refresh,
        )
    else:
        rows, failures = read_channels(args.csv), {}
    print(table(rows))
    missing = sorted(set(TEAMS) - {r["team"] for r in rows if r["kind"] == "team"})
    if missing:
        print(f"\nNo official channel for: {', '.join(missing)}")
        for team, tried in failures.items():
            print(f"  {team}: {'; '.join(tried)}")
    print(
        "\nCheckpoint 1: check every row, then set verified_by_human=true in "
        f"{args.csv}"
    )
    return 1 if missing else 0
