"""Step 6: assign each classified video to one of its team's games.

A video from team T's channel can only belong to one of T's games. The window
is by instant: published at least 90 minutes after tip-off and before T's next
tip-off (at most 3 days later). That separates back-to-backs, where a late
western game is published after midnight UTC on the day of the next game.
Without an instant (``upload_date`` only), the dates ``[day - 2, day]`` are used
and a back-to-back inside them is ambiguous.

Evidence written in the title (a date, the opponent, a playoff ``Game N``)
confirms the window or, for late uploads, places a video outside it
(``title_only``, never auto-accepted).
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, date, datetime, timedelta

from lab.coach_pressers.classify import classify, names_team
from lab.coach_pressers.config import PresserConfig
from lab.coach_pressers.manifest import MATCHABLE, entry_key
from lab.coach_pressers.names import contains_phrase, full_forms, last_name, normalize
from lab.coach_pressers.teams import TEAMS, team_names
from lab.coach_pressers.title_dates import title_dates
from lab.game_reports.season_reports import atomic_write, json_bytes

log = logging.getLogger("coach_pressers")

MIN_DELAY = timedelta(minutes=90)
MAX_DELAY = timedelta(days=3)
AUTO_SCORE = 0.8
MIN_SCORE = 0.4
KEPT_CLASSES = {"coach", "combined"}
GAME_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
}
GAME_RE = re.compile(r"\bgame (one|two|three|four|five|six|seven|[1-7])\b")


def parse_instant(value: str | None) -> datetime | None:
    if not value:
        return None
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return instant if instant.tzinfo else instant.replace(tzinfo=UTC)


def team_games(manifest: dict, team: str) -> list[dict]:
    games = [
        e
        for e in manifest["entries"].values()
        if e["team"] == team and e["status"] != "not_final"
    ]
    return sorted(games, key=lambda e: (e["tip_utc"] or e["date"], e["game_id"]))


def window_games(video: dict, games: list[dict]) -> tuple[list[dict], str]:
    """Games whose publication window contains the video, and how it was decided."""
    published = parse_instant(video.get("published_at_utc"))
    if published is not None:
        for game, following in zip(games, [*games[1:], None], strict=True):
            tip = parse_instant(game["tip_utc"])
            if tip is None:
                continue
            end = tip + MAX_DELAY
            if following and parse_instant(following["tip_utc"]):
                end = min(end, parse_instant(following["tip_utc"]))
            if tip + MIN_DELAY <= published < end:
                return [game], "instant"
        return [], "instant"
    day = video.get("published_date")
    if not day:
        return [], "none"
    upload = date.fromisoformat(day)
    lo = (upload - timedelta(days=2)).isoformat()
    inside = [g for g in games if lo <= g["date"] <= upload.isoformat()]
    return inside, "date"


def opponents_named(title: str, team: str) -> set[str]:
    text = normalize(title)
    own = {normalize(n) for n in team_names(team)}
    found = set()
    for code in TEAMS:
        if code == team:
            continue
        for name in team_names(code):
            phrase = normalize(name)
            if phrase not in own and contains_phrase(text, phrase):
                found.add(code)
    return found


def game_number(title: str) -> int | None:
    match = GAME_RE.search(normalize(title))
    if not match:
        return None
    word = match.group(1)
    return GAME_WORDS.get(word) or int(word)


def title_evidence(video: dict, games: list[dict], team: str) -> tuple[list, dict]:
    """Games the title points to (date, opponent, Game N) near the upload."""
    title = video.get("title") or ""
    published = parse_instant(video.get("published_at_utc"))
    day = video.get("published_date")
    if published is None and day:
        published = datetime.fromisoformat(day).replace(tzinfo=UTC) + timedelta(days=1)

    def before_upload(game):
        tip = parse_instant(game["tip_utc"])
        return published is None or tip is None or tip <= published

    evidence = {"title_dates": [d.isoformat() for d in title_dates(title)]}
    if evidence["title_dates"]:
        hits = [g for g in games if g["date"] in evidence["title_dates"]]
        return [g for g in hits if before_upload(g)], {**evidence, "by": "title_date"}
    opponents = opponents_named(title, team)
    number = game_number(title)
    if opponents or number:
        evidence.update(opponents=sorted(opponents), game_no=number)
        hits = [
            g
            for g in games
            if before_upload(g)
            and (not opponents or g["opponent"] in opponents)
            and (not number or g.get("playoff_game_no") == number)
        ]
        if published is not None:
            hits = [
                g
                for g in hits
                if parse_instant(g["tip_utc"]) is None
                or published - parse_instant(g["tip_utc"]) <= MAX_DELAY
            ]
        return hits[-1:], {**evidence, "by": "opponent" if opponents else "game_no"}
    return [], {**evidence, "by": None}


def find_coach_chapter(chapters: list[dict], staff: list[dict]) -> dict | None:
    """The single chapter whose title names a coach on the staff."""
    hits = []
    for chapter in chapters:
        text = normalize(chapter.get("title") or "")
        for person in staff:
            forms = person.get("forms") or full_forms(person["name"])
            last = person.get("last") or last_name(person["name"])
            if any(contains_phrase(text, f) for f in forms) or contains_phrase(
                text, last
            ):
                hits.append({**chapter, "coach": person["name"]})
                break
    return hits[0] if len(hits) == 1 else None


def rival_head_coaches(teams: dict, team: str) -> set[str]:
    """Normalised full names of the other teams' head coaches, not on our staff.

    Outlets post both coaches of a game; the opponent's presser shares the
    window and often names our team, so a rival head coach in the title or in
    the judge's answer means the video is not ours.
    """
    own = {f for p in teams.get(team, {}).get("staff", []) for f in _forms(p)}
    rivals = set()
    for code, data in teams.items():
        if code == team:
            continue
        for person in data.get("staff", []):
            if person.get("role") == "head":
                rivals.update(_forms(person))
        for spell in data.get("spells", []):
            if spell.get("coach"):
                rivals.update(full_forms(spell["coach"]))
    return rivals - own


def _forms(person: dict) -> list[str]:
    return person.get("forms") or full_forms(person["name"])


def candidates_for_team(
    team: str,
    videos: list[dict],
    manifest: dict,
    data: dict,
    rivals: set[str] = frozenset(),
):
    """(entry key -> list of candidates) for one team's videos."""
    games = team_games(manifest, team)
    out: dict[str, list[dict]] = {}
    for video in videos:
        if video.get("error"):
            continue
        verdict = classify(
            video.get("title") or "",
            data["staff"],
            data["players"],
            video.get("duration"),
        )
        keep = verdict.label in KEPT_CLASSES or "unknown_name" in verdict.notes
        if not keep or verdict.score < MIN_SCORE:
            continue
        if video.get("channel_kind", "team") != "team" and not names_team(
            video.get("title") or "", team, data["staff"], data["players"]
        ):
            continue  # an outlet's video about another team
        text = normalize(video.get("title") or "")
        if not verdict.coach_guess and any(contains_phrase(text, f) for f in rivals):
            continue  # the opponent's head coach at the podium
        window, how = window_games(video, games)
        titled, evidence = title_evidence(video, games, team)
        if titled:
            game = titled[0]
            method = "title+window" if game in window else "title_only"
        elif len(window) == 1:
            game, method = window[0], f"window_{how}"
        elif len(window) > 1:
            for game in window:  # ambiguous back-to-back: candidate for each
                out.setdefault(entry_key(game["game_id"], team), []).append(
                    candidate(video, verdict, "ambiguous_window", evidence, data)
                )
            continue
        else:
            continue
        out.setdefault(entry_key(game["game_id"], team), []).append(
            candidate(video, verdict, method, evidence, data)
        )
    return out


def candidate(video, verdict, method, evidence, data) -> dict:
    chapter = (
        find_coach_chapter(video.get("chapters") or [], data["staff"])
        if verdict.label == "combined"
        else None
    )
    return {
        "video_id": video["id"],
        "channel_id": video.get("channel_id"),
        "channel_priority": video.get("channel_priority", 1),
        "title": video.get("title"),
        "description": (video.get("description") or "")[:500],
        "published_at_utc": video.get("published_at_utc"),
        "published_date": video.get("published_date"),
        "duration_s": video.get("duration"),
        "chapters": video.get("chapters") or [],
        "live_status": video.get("live_status"),
        **verdict.as_dict(),
        "match_method": method,
        "evidence": evidence,
        "coach_chapter": chapter,
    }


def selection_from(
    candidate: dict,
    *,
    decided_by,
    coach_actual,
    kind,
    chapter=None,
    confidence=None,
    reason="",
):
    chapter = chapter if chapter is not None else candidate.get("coach_chapter")
    combined = kind == "combined"
    return {
        "video_id": candidate["video_id"],
        "channel_id": candidate["channel_id"],
        "title": candidate["title"],
        "published_at_utc": candidate["published_at_utc"],
        "duration_s": candidate["duration_s"],
        "kind": kind,
        "coach_actual": coach_actual,
        "decided_by": decided_by,
        "confidence": confidence,
        "reason": reason,
        "coach_chapter": chapter["title"] if chapter else None,
        "trim_start_s": chapter["start_time"] if combined and chapter else None,
        "trim_end_s": chapter["end_time"] if combined and chapter else None,
        "needs_trim": combined and not chapter,
        "chapters": candidate["chapters"],
    }


def auto_accept(entry: dict, found: list[dict]) -> dict | None:
    """The rules' choice, or None when the judge (or a human) must decide."""
    strong = [c for c in found if c["class"] in KEPT_CLASSES]
    if len(strong) != 1 or len(found) != 1:
        return None
    best = strong[0]
    agrees = best["match_method"] == "title+window"
    expected = normalize(entry.get("coach_expected") or "")
    named = normalize(best.get("coach_guess") or "")
    if best["score"] < AUTO_SCORE or not agrees:
        return None
    if named and named != expected:
        return None  # someone else at the podium: let the judge confirm
    if best["class"] == "combined" and not entry.get("coach_expected") and not named:
        return None
    return selection_from(
        best,
        decided_by="rules",
        coach_actual=best.get("coach_guess") or entry["coach_expected"],
        kind=best["class"],
    )


def match_season(
    config: PresserConfig, manifest: dict, coaches: dict, videos: dict[str, list]
) -> dict[str, list[dict]]:
    """Write candidates.json and settle what the rules can; return the rest."""
    candidates: dict[str, list[dict]] = {}
    for team, team_videos in videos.items():
        candidates.update(
            candidates_for_team(
                team,
                team_videos,
                manifest,
                coaches["teams"][team],
                rival_head_coaches(coaches["teams"], team),
            )
        )
    for found in candidates.values():
        found.sort(key=lambda c: (-c["score"], c["channel_priority"], c["video_id"]))
    to_judge = {}
    counts = {"rules": 0, "judge": 0, "not_found": 0}
    for key, entry in manifest["entries"].items():
        if entry["status"] not in MATCHABLE or entry["team"] not in videos:
            continue
        found = candidates.get(key, [])
        if not found:
            entry.update(status="not_found", selection=None, error="")
            counts["not_found"] += 1
            continue
        selection = auto_accept(entry, found)
        if selection:
            entry.update(status="selected", selection=selection, error="")
            counts["rules"] += 1
        else:
            to_judge[key] = found
            counts["judge"] += 1
    atomic_write(config.candidates_path, json_bytes(candidates))
    log.info(
        "step 6 %s: %d candidate pressers; rules %d, to judge %d, not found %d",
        config.season,
        len(candidates),
        counts["rules"],
        counts["judge"],
        counts["not_found"],
    )
    return to_judge
