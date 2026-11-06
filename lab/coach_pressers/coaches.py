"""Step 2: who coached each team on each date, and every name a title may carry.

Sources, in increasing precedence:

1. The team-season Wikipedia article: the infobox lists the season's head
   coaches in order ("Luke Walton (fired)", "Alvin Gentry (interim)") and the
   roster template lists the head coach and assistants at season end.
   (stats.nba.com ``CommonTeamRoster`` is not used: for past seasons it returns
   a later staff, e.g. Mike Brown for the 2021-22 Kings.)
2. The league-season article's "In-season" coaching changes, written as
   sentences ("On November 21, 2021, the Sacramento Kings fired head coach Luke
   Walton and named Alvin Gentry as their interim head coach."). Each gives the
   date a new spell starts. A sentence that cannot be parsed is reported, never
   guessed.
3. ``coaches_overrides.csv`` (team, coach, start_date, end_date, kind, note).

Player names (to tell player videos from coach videos) come from the box
scores in the database.
"""

from __future__ import annotations

import csv
import html as html_lib
import json
import logging
import re
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

from lab.coach_pressers.config import PresserConfig, season_label
from lab.coach_pressers.names import full_forms, last_name, normalize
from lab.coach_pressers.teams import TEAMS
from lab.game_reports.season_reports import RequestPacer, atomic_write, json_bytes

log = logging.getLogger("coach_pressers")

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "NBAPresserArchive/0.1 (personal research script)"
DATE_RE = re.compile(r"\bOn ([A-Z][a-z]+\.? \d{1,2}, \d{4}),")
FOOTNOTE_RE = re.compile(r"\[[^\]]{1,12}\]")
DEPARTURE_RE = re.compile(
    r"\b(fired|resigned|stepped down|part ways|parted ways|relieved|dismissed|"
    r"suspended|leave of absence|stroke|surgery)\b"
)
NAME_AFTER = ("named ", "hired ", "promoted ", "appointed ")
NAME_BEFORE = (
    " was named",
    " was hired",
    " was promoted",
    " was appointed",
    " took over",
)
ROLE_WORDS = (
    "assistant coach ",
    "associate head coach ",
    "assistant ",
    "lead assistant ",
)
OVERRIDE_FIELDS = ("team", "coach", "start_date", "end_date", "kind", "note")


# ----------------------------------------------------------------- Wikipedia


def league_title(season_year: int) -> str:
    return f"{season_year}–{(season_year + 1) % 100:02d} NBA season"


def team_title(season_year: int, team: str) -> str:
    return f"{season_year}–{(season_year + 1) % 100:02d} {TEAMS[team][0]} season"


class Wikipedia:
    """Article HTML through the MediaWiki parse API, cached on disk."""

    def __init__(
        self,
        cache_dir: Path,
        pacer: RequestPacer,
        *,
        max_age_days=None,
        sleep=time.sleep,
    ):
        self.cache_dir, self.pacer, self.max_age_days = cache_dir, pacer, max_age_days
        self.sleep = sleep

    def path(self, title: str) -> Path:
        return self.cache_dir / (re.sub(r"[^\w.-]+", "_", title) + ".html")

    def article(self, title: str) -> str | None:
        path = self.path(title)
        if path.is_file() and (
            self.max_age_days is None
            or time.time() - path.stat().st_mtime < self.max_age_days * 86400
        ):
            return path.read_text(encoding="utf-8") or None
        payload = self.request(title)
        text = "" if "error" in payload else payload["parse"]["text"]
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, text.encode("utf-8"))
        return text or None

    def request(self, title: str) -> dict:
        """One parse request; 429/503 wait (Retry-After or 30, 60, 120, 240 s)."""
        import requests

        for attempt in range(5):
            self.pacer.wait()
            response = requests.get(
                WIKIPEDIA_API,
                params={
                    "action": "parse",
                    "page": title,
                    "prop": "text",
                    "format": "json",
                    "formatversion": "2",
                    "redirects": "1",
                    "maxlag": "5",
                },
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            if response.status_code not in (429, 503) or attempt == 4:
                response.raise_for_status()
                payload = response.json()
                if payload.get("error", {}).get("code") != "maxlag":
                    return payload
            retry = response.headers.get("Retry-After", "")
            pause = float(retry) if retry.isdigit() else 30.0 * 2**attempt
            log.info("Wikipedia busy (%s); waiting %.0f s", response.status_code, pause)
            self.sleep(min(pause, 600.0))
        raise RuntimeError(f"Wikipedia kept refusing {title!r}")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", FOOTNOTE_RE.sub("", html_lib.unescape(text))).strip()


def _section_items(doc, heading_id: str) -> list[str]:
    """Text of the list items under a heading, up to the next heading."""
    heads = doc.xpath(f'//*[@id="{heading_id}"]')
    if not heads:
        return []
    head = heads[0]
    parent = head.getparent()
    container = parent if "mw-heading" in (parent.get("class") or "") else head
    items = []
    for sibling in container.itersiblings():
        if sibling.tag in ("h2", "h3", "h4") or "mw-heading" in (
            sibling.get("class") or ""
        ):
            break
        items += [_clean(li.text_content()) for li in sibling.xpath(".//li")]
        if sibling.tag == "p" and _clean(sibling.text_content()):
            items.append(_clean(sibling.text_content()))
    return [i for i in items if i]


def _cap_run(tokens: list[str]) -> list[str]:
    """Leading tokens that look like a name: capitalised words and initials."""
    run = []
    for token in tokens:
        word = token.strip(",;:()")
        if not word or not word[0].isupper():
            break
        if word.endswith(".") and len(word.rstrip(".")) > 1:
            run.append(word.rstrip("."))
            break  # the end of a sentence, not an initial
        run.append(word)
        if token.endswith((",", ";", ")")):
            break
    return run


def _name_after(text: str) -> str | None:
    tokens = text.split()
    for role in ROLE_WORDS:
        if text.lower().startswith(role):
            tokens = tokens[len(role.split()) :]
    run = _cap_run(tokens)
    return " ".join(run) if len(run) >= 2 else None


def _name_before(text: str) -> str | None:
    run = []
    for token in reversed(text.split()):
        word = token.strip(",;:()")
        if not word or not word[0].isupper():
            break
        if word.endswith(".") and len(word.rstrip(".")) > 1:
            break  # the previous sentence's last word
        run.insert(0, word)
    return " ".join(run) if len(run) >= 2 else None


def parse_change(sentence: str, team_hint: str | None = None) -> dict | None:
    """One in-season change sentence -> {date, team, incoming, kind, text}.

    ``team_hint`` names the team when the sentence only names its coach
    ("Gregg Popovich suffered a mild stroke").
    """
    match = DATE_RE.search(sentence)
    if not match:
        return None
    try:
        when = datetime.strptime(match.group(1).replace(".", ""), "%B %d, %Y").date()
    except ValueError:
        try:
            when = datetime.strptime(
                match.group(1).replace(".", ""), "%b %d, %Y"
            ).date()
        except ValueError:
            return None
    teams = [
        code
        for code, (full, nickname) in TEAMS.items()
        if full in sentence or re.search(rf"\b{re.escape(nickname)}\b", sentence)
    ]
    if not teams and team_hint:
        teams = [team_hint]
    if len(teams) != 1:
        return None
    incoming = None
    lower = sentence.lower()
    for anchor in NAME_BEFORE:
        index = lower.find(anchor)
        if index > 0:
            incoming = _name_before(sentence[:index])
            if incoming:
                break
    if not incoming:
        for anchor in NAME_AFTER:
            for found in re.finditer(re.escape(anchor), sentence):
                rest = sentence[found.end() :]
                if "head coach" not in rest[:90].lower():
                    continue
                incoming = _name_after(rest)
                if incoming:
                    break
            if incoming:
                break
    if incoming and normalize(incoming) in {normalize(t[0]) for t in TEAMS.values()}:
        incoming = None
    if not incoming and not DEPARTURE_RE.search(lower):
        return None
    return {
        "date": when.isoformat(),
        "team": teams[0],
        "incoming": incoming,
        "kind": "interim" if incoming and "interim" in lower else "head",
        "text": sentence,
    }


def parse_league_changes(html_text: str | None) -> tuple[list[dict], list[str]]:
    """In-season coaching changes of a league-season article, and unparsed items."""
    if not html_text:
        return [], []
    from lxml import html

    doc = html.fromstring(html_text)
    events, unparsed = [], []
    for item in _section_items(doc, "In-season"):
        event = parse_change(item)
        (events.append(event) if event else unparsed.append(item))
    return sorted(events, key=lambda e: (e["date"], e["team"])), unparsed


def parse_team_season(html_text: str | None) -> dict:
    """Infobox head coaches (with notes) and the season-end roster staff."""
    result = {"infobox": [], "head_coach": None, "assistants": []}
    if not html_text:
        return result
    from lxml import html

    doc = html.fromstring(html_text)
    for row in doc.xpath('//table[contains(@class,"infobox")]//tr'):
        header, data = row.xpath("./th"), row.xpath("./td")
        if not header or not data or header[0].text_content().strip() != "Head coach":
            continue
        raw = html.tostring(data[0], encoding="unicode")
        raw = re.sub(r"<sup.*?</sup>", "", raw, flags=re.S)
        raw = re.sub(r"<br\s*/?>|</li>|</div>|</p>", "\n", raw)
        for line in html.fromstring(f"<div>{raw}</div>").text_content().split("\n"):
            line = _clean(line)
            if not line:
                continue
            note = re.search(r"\(([^)]*)\)\s*$", line)
            name = re.sub(r"\s*\([^)]*\)\s*$", "", line).strip()
            if name:
                result["infobox"].append(
                    {"name": name, "note": note.group(1).lower() if note else ""}
                )
        break
    roster = doc.xpath('//table[contains(@class,"toccolours")]')
    for table in roster:
        labels = table.xpath('.//b[normalize-space()="Head coach"]')
        if not labels:
            continue
        head = labels[0].xpath("following::ul[1]/li")
        result["head_coach"] = _clean(head[0].text_content()) if head else None
        assistants = table.xpath('.//b[normalize-space()="Assistant(s)"]')
        if assistants:
            result["assistants"] = [
                _clean(li.text_content())
                for li in assistants[0].xpath("following::ul[1]/li")
                if _clean(li.text_content())
            ]
        break
    return result


# ------------------------------------------------------------------- spells


def build_spells(
    team_page: dict, events: list[dict], first: date, last: date
) -> tuple[list[dict], list[str]]:
    """Dated head-coach spells of one team-season from Wikipedia."""
    warnings = []
    infobox = [c["name"] for c in team_page["infobox"]]
    opening = infobox[0] if infobox else team_page["head_coach"]
    if not opening:
        return [], ["No head coach on Wikipedia"]
    spells = [
        {
            "coach": opening,
            "start_date": first.isoformat(),
            "kind": "head",
            "source": "wikipedia",
        }
    ]
    for event in events:
        start = date.fromisoformat(event["date"])
        current = spells[-1]
        if current["coach"] is not None and start <= date.fromisoformat(
            current["start_date"]
        ):
            if event["incoming"]:
                current.update(coach=event["incoming"], kind=event["kind"])
            continue
        spells.append(
            {
                "coach": event["incoming"],
                "start_date": start.isoformat(),
                "kind": event["kind"] if event["incoming"] else "vacancy",
                "source": "wikipedia",
            }
        )
    for current, following in zip(spells, spells[1:], strict=False):
        current["end_date"] = (
            date.fromisoformat(following["start_date"]) - timedelta(days=1)
        ).isoformat()
    spells[-1]["end_date"] = last.isoformat()
    spells = [s for s in spells if s["coach"]]
    named = {normalize(s["coach"]) for s in spells}
    missing = [n for n in infobox if normalize(n) not in named]
    if missing:
        warnings.append(
            f"Infobox coaches without a dated spell: {', '.join(missing)}; "
            "add them to coaches_overrides.csv"
        )
    return spells, warnings


def load_overrides(path: Path, first: date, last: date) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    overrides = []
    for row in rows:
        if not row.get("team"):
            continue
        if row["team"] not in TEAMS:
            raise ValueError(f"Unknown team in {path.name}: {row['team']}")
        start = date.fromisoformat(row["start_date"])
        end = date.fromisoformat(row["end_date"]) if row.get("end_date") else last
        if end < first or start > last:
            continue
        overrides.append(
            {
                "team": row["team"],
                "coach": row["coach"].strip(),
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "kind": row.get("kind") or "acting",
                "source": "override",
                "note": row.get("note") or "",
            }
        )
    return overrides


def coach_on(team_data: dict, day: str) -> str | None:
    """The expected coach on a date: overrides (latest start first), then spells."""
    for spell in sorted(
        team_data["overrides"], key=lambda s: s["start_date"], reverse=True
    ):
        if spell["start_date"] <= day <= spell["end_date"]:
            return spell["coach"]
    for spell in team_data["spells"]:
        if spell["start_date"] <= day <= spell["end_date"]:
            return spell["coach"]
    return None


def staff_entry(name: str, role: str, source: str) -> dict:
    return {
        "name": name,
        "role": role,
        "forms": full_forms(name),
        "last": last_name(name),
        "source": source,
    }


def team_staff(team_data: dict, page: dict, events: list[dict]) -> list[dict]:
    people: dict[str, dict] = {}

    def add(name, role, source):
        if not name:
            return
        key = normalize(name)
        if key not in people or (role == "head" and people[key]["role"] != "head"):
            people[key] = staff_entry(name, role, source)

    for spell in team_data["spells"] + team_data["overrides"]:
        add(spell["coach"], "head", spell["source"])
    for coach in page["infobox"]:
        add(coach["name"], "head", "wikipedia")
    add(page["head_coach"], "head", "wikipedia")
    for event in events:
        add(event["incoming"], "head", "wikipedia")
    for name in page["assistants"]:
        add(name, "assistant", "wikipedia")
    return sorted(people.values(), key=lambda p: (p["role"] != "head", p["name"]))


# ------------------------------------------------------------------ players


def default_players(season: str) -> dict[str, list[str]]:
    from nba_ou.postgre_db.players.fetch_players_data_from_db import (
        load_players_from_db,
    )

    frame = load_players_from_db(seasons=[season])
    if frame is None or frame.empty:
        raise RuntimeError(f"No box-score players in the database for {season}")
    columns = {c.lower(): c for c in frame.columns}
    name, team = columns["player_name"], columns["team_abbreviation"]
    return {
        code: sorted(set(group[name].dropna().astype(str)))
        for code, group in frame.groupby(team)
    }


def season_players(
    config: PresserConfig, loader: Callable[[str], dict] | None = None
) -> tuple[dict[str, list[str]], list[str]]:
    path = config.raw_dir / "players" / f"season={config.season}.json"
    if path.is_file():
        return json.loads(path.read_bytes()), []
    try:
        players = (loader or default_players)(config.season)
    except Exception as exc:  # the database is optional for a dry run
        return {}, [f"No player names ({type(exc).__name__}: {exc})"]
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json_bytes(players))
    return players, []


# -------------------------------------------------------------------- build


def resolve_by_coach(
    events: list[dict], unparsed: list[str], pages: dict[str, dict]
) -> tuple[list[dict], list[str]]:
    """Parse sentences without a team name using the coaches each team listed."""
    still = []
    for sentence in unparsed:
        text = normalize(sentence)
        teams = [
            team
            for team, page in pages.items()
            if any(f" {normalize(c['name'])} " in f" {text} " for c in page["infobox"])
        ]
        event = parse_change(sentence, teams[0]) if len(teams) == 1 else None
        (events.append(event) if event else still.append(sentence))
    return sorted(events, key=lambda e: (e["date"], e["team"])), still


def build_coaches(
    config: PresserConfig,
    manifest: dict,
    *,
    wikipedia: Wikipedia | None = None,
    players_loader: Callable[[str], dict] | None = None,
) -> dict:
    """Write coaches.json and set ``coach_expected`` on every manifest entry."""
    if wikipedia is None:
        current = datetime.now().year - (datetime.now().month < 8)
        wikipedia = Wikipedia(
            config.raw_dir / "wikipedia",
            RequestPacer(config.wikipedia_interval),
            max_age_days=1 if config.season_year >= current else None,
        )
    entries = list(manifest["entries"].values())
    events, unparsed = parse_league_changes(
        wikipedia.article(league_title(config.season_year))
    )
    players, warnings = season_players(config, players_loader)
    previous_season = config.season_year - 1
    codes = sorted({e["team"] for e in entries})
    pages = {
        team: parse_team_season(wikipedia.article(team_title(config.season_year, team)))
        for team in codes
    }
    events, unparsed = resolve_by_coach(events, unparsed, pages)
    teams = {}
    for team in codes:
        own = [e["date"] for e in entries if e["team"] == team]
        first, last = date.fromisoformat(min(own)), date.fromisoformat(max(own))
        page = pages[team]
        team_events = [e for e in events if e["team"] == team]
        spells, team_warnings = build_spells(page, team_events, first, last)
        overrides = [
            o
            for o in load_overrides(config.overrides_csv, first, last)
            if o["team"] == team
        ]
        data = {
            "wikipedia_title": team_title(config.season_year, team),
            "infobox": page["infobox"],
            "season_end_head_coach": page["head_coach"],
            "spells": spells,
            "overrides": overrides,
            "warnings": team_warnings,
        }
        staff = team_staff(data, page, team_events)
        if previous_season >= 2018:
            # Assistants who left before season end still appear in the
            # previous season's roster template.
            before = parse_team_season(
                wikipedia.article(team_title(previous_season, team))
            )
            known = {normalize(s["name"]) for s in staff}
            staff += [
                staff_entry(n, "assistant", "wikipedia_previous_season")
                for n in before["assistants"] + [before["head_coach"] or ""]
                if n and normalize(n) not in known
            ]
        data["staff"] = staff
        data["players"] = players.get(team, [])
        teams[team] = data
    for entry in entries:
        entry["coach_expected"] = coach_on(teams[entry["team"]], entry["date"])
    coaches = {
        "season": season_label(config.season_year),
        "sources": {
            "league_article": league_title(config.season_year),
            "players": "box scores (database)",
        },
        "in_season_changes": events,
        "unparsed_changes": unparsed,
        "warnings": warnings,
        "teams": teams,
    }
    config.directory.mkdir(parents=True, exist_ok=True)
    atomic_write(config.coaches_path, json_bytes(coaches))
    return coaches


def write_overrides_template(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(OVERRIDE_FIELDS)
