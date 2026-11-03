"""ESPN written recaps, matched to an official NBA game by date and teams."""

from __future__ import annotations

import re
import threading
from datetime import UTC, datetime

from bs4 import BeautifulSoup

from lab.game_reports.get_game_reports import SOURCE_TIMEZONE, clean_text

API_ROOT = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
TEAM_CODES = {
    "GS": "GSW",
    "SA": "SAS",
    "NO": "NOP",
    "NY": "NYK",
    "UTAH": "UTA",
    "WSH": "WAS",
}


def matches_game(competition: dict, entry: dict) -> bool:
    """Require the US Eastern game date and both home/away team codes."""
    slug = re.search(r"/game/([a-z]+)-vs-([a-z]+)-(\d{10})(?:/|$)", entry["source_url"])
    if not slug or slug[3] != entry["game_id"]:
        raise ValueError("Cannot identify NBA teams from the official game URL")
    moment = datetime.fromisoformat(competition["date"].replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("ESPN game date has no time zone")
    if moment.astimezone(SOURCE_TIMEZONE).date().isoformat() != entry["date"]:
        return False
    teams = {}
    for competitor in competition.get("competitors", []):
        code = competitor["team"]["abbreviation"].upper()
        teams[competitor["homeAway"]] = TEAM_CODES.get(code, code)
    return teams == {"away": slug[1].upper(), "home": slug[2].upper()}


def parse_espn_report(payload: dict, event_id: str, entry: dict) -> dict | None:
    header = payload.get("header") or {}
    competitions = header.get("competitions") or []
    if (
        str(header.get("id")) != event_id
        or len(competitions) != 1
        or str(competitions[0].get("id")) != event_id
        or not matches_game(competitions[0], entry)
    ):
        raise ValueError("ESPN summary belongs to a different game")
    competition = competitions[0]
    if not (competition.get("status", {}).get("type", {}).get("completed")):
        return None
    article = payload.get("article") or {}
    if article.get("type", "").lower() != "recap":
        return None
    if str(article.get("gameId")) != event_id:
        raise ValueError("ESPN article belongs to a different game")
    story = article.get("story") or ""
    paragraphs = [
        text
        for part in re.split(r"\n\s*\n|</p>|<br\s*/?>", story, flags=re.IGNORECASE)
        if (text := " ".join(BeautifulSoup(part, "html.parser").get_text().split()))
    ]
    if not paragraphs:
        return None
    published = article.get("originallyPosted") or article.get("published") or ""
    published_utc = ""
    if published:
        moment = datetime.fromisoformat(published.replace("Z", "+00:00"))
        if moment.tzinfo is None:
            raise ValueError("ESPN publication date has no time zone")
        published_utc = moment.astimezone(UTC).isoformat()
    teams = {c["homeAway"]: c for c in competition["competitors"]}
    text = "\n\n".join(paragraphs)
    return {
        "game_id": entry["game_id"],
        "date": entry["date"],
        "away": entry["away"],
        "home": entry["home"],
        "away_score": int(teams["away"]["score"]),
        "home_score": int(teams["home"]["score"]),
        "source": "ESPN",
        "source_url": f"https://www.espn.com/nba/recap/_/gameId/{event_id}",
        "source_api_url": f"{API_ROOT}/summary?event={event_id}",
        "nba_source_url": entry["source_url"],
        "espn_event_id": event_id,
        "title": clean_text(article.get("headline") or ""),
        "author": clean_text(article.get("byline") or ""),
        "bytitle": clean_text(article.get("source") or ""),
        "agency": clean_text(article.get("source") or ""),
        "published_at": published,
        "published_at_utc": published_utc,
        "source_published_at": article.get("published") or "",
        "source_originally_posted_at": article.get("originallyPosted") or "",
        "source_last_modified_at": article.get("lastModified") or "",
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "text": text,
        "word_count": len(text.split()),
    }


class ESPNRecaps:
    """Cache each day's event index across workers, pace every HTTP request."""

    def __init__(self, wait):
        self.wait = wait
        self.scoreboards = {}
        self.lock = threading.Lock()

    def fetch(self, session, entry: dict, timeout: int) -> dict | None:
        game_date = entry["date"]
        with self.lock:
            if game_date not in self.scoreboards:
                self.wait()
                response = session.get(
                    f"{API_ROOT}/scoreboard",
                    params={"dates": game_date.replace("-", ""), "limit": 100},
                    timeout=timeout,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload.get("events"), list):
                    raise ValueError("ESPN scoreboard has no events list")
                self.scoreboards[game_date] = payload["events"]
            events = self.scoreboards[game_date]
        matching = [
            event
            for event in events
            if any(matches_game(c, entry) for c in event.get("competitions", []))
        ]
        if not matching:
            return None
        if len(matching) != 1:
            raise ValueError("Multiple ESPN events match this NBA game")
        event_id = str(matching[0]["id"])
        self.wait()
        response = session.get(
            f"{API_ROOT}/summary", params={"event": event_id}, timeout=timeout
        )
        response.raise_for_status()
        return parse_espn_report(response.json(), event_id, entry)
