"""Step 5: is a video a coach presser? Rule-based, tested on real titles.

Classes: ``coach``, ``combined`` (coach and players in one video), ``player``
and ``other``, each with a score from 0 to 1. A coach name plus a postgame cue
is the strongest signal; keywords alone are not enough.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lab.coach_pressers.names import contains_phrase, full_forms, last_name, normalize
from lab.coach_pressers.teams import TEAMS, team_names

EXCLUDE = (
    "pregame",
    "pre game",
    "shootaround",
    "practice",
    "micd up",
    "mic d up",
    "locker room",
    "speech",
    "highlights",
    "full game",
    "game recap",
    "game rewind",
    "1st quarter",
    "2nd quarter",
    "3rd quarter",
    "4th quarter",
    "shorts",
)
CUES = ("postgame", "post game", "presser", "press conference", "availability")
RECAP_VERBS = ("recaps", "reacts", "breaks down", "discusses")
COMBINED_PHRASES = (
    "full postgame media",
    "full post game media",
    "media availability postgame",
    "postgame media availability",
)
# Capitalised words that are not people.
NOT_NAMES = {
    w
    for w in normalize(
        "Coach Head Postgame Post Game Press Conference Media Availability Sound "
        "Interview Full All Recaps Reacts Win Loss Against Vs At Game One Two Three "
        "Four Five Six Seven Playoffs Playoff Round Series First Second Conference "
        "Finals Final NBA Cup Play In Tournament Presser After Before Victory Defeat "
        "Emirates In Season East West Eastern Western Breaks Down Discusses Talks "
        "Jan Feb Mar Apr May Jun Jul Aug Sep Sept Oct Nov Dec January February March "
        "April June July August September October November December The And Of On "
        "Home Road Opening Night LA OT Overtime Double Live"
    ).split()
}


@dataclass
class Verdict:
    label: str
    score: float
    coach_guess: str | None = None
    people: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "class": self.label,
            "score": round(self.score, 2),
            "coach_guess": self.coach_guess,
            "people": self.people,
            "notes": self.notes,
        }


def _staff_hits(text: str, staff: list[dict], player_lasts: set[str]) -> list[tuple]:
    """(staff member, strength) for every staff name in the normalised title."""
    hits = []
    for person in staff:
        forms = person.get("forms") or full_forms(person["name"])
        last = person.get("last") or last_name(person["name"])
        if any(contains_phrase(text, f) for f in forms):
            hits.append((person, 1.0))
        elif contains_phrase(text, f"coach {last}"):
            hits.append((person, 0.9))
        elif last and last not in player_lasts and contains_phrase(text, last):
            hits.append((person, 0.8))
    best: dict[str, tuple] = {}
    for person, strength in hits:
        key = normalize(person["name"])
        if key not in best or strength > best[key][1]:
            best[key] = (person, strength)
    return sorted(
        best.values(), key=lambda h: (-h[1], h[0].get("role") != "head", h[0]["name"])
    )


def _player_hits(text: str, players: list[str], staff_lasts: set[str]):
    full = [p for p in players if any(contains_phrase(text, f) for f in full_forms(p))]
    lasts = {
        last_name(p)
        for p in players
        if len(last_name(p)) > 2
        and last_name(p) not in staff_lasts
        and contains_phrase(text, last_name(p))
    }
    return full, lasts


def _team_words() -> set[str]:
    words = set()
    for team in TEAMS:
        for name in team_names(team):
            words.update(normalize(name).split())
    return words


TEAM_WORDS = _team_words()


def unknown_people(title: str, known: list[str]) -> list[str]:
    """Runs of two or more capitalised words that are neither teams nor known names."""
    known_words = {w for name in known for w in normalize(name).split()}
    runs, current = [], []
    for token in re.split(r"[\s|:,;()\-–—&/]+", title or ""):
        word = normalize(token)
        if (
            token[:1].isupper()
            and word
            and not word.isdigit()
            and not token.isupper()
            and word not in NOT_NAMES
            and word not in TEAM_WORDS
            and word not in known_words
        ):
            current.append(token.strip(".'\""))
        else:
            if len(current) >= 2:
                runs.append(" ".join(current))
            current = []
    if len(current) >= 2:
        runs.append(" ".join(current))
    return runs


def classify(
    title: str,
    staff: list[dict],
    players: list[str],
    duration: float | None = None,
) -> Verdict:
    """Classify one title against a team's coaching staff and players."""
    text = normalize(title)
    for term in EXCLUDE:
        if contains_phrase(text, term) or (
            term == "shorts" and "#shorts" in title.lower()
        ):
            return Verdict("other", 0.95, notes=[f"excluded:{term}"])
    player_lasts = {last_name(p) for p in players}
    staff_lasts = {p.get("last") or last_name(p["name"]) for p in staff}
    cue = any(contains_phrase(text, c) for c in CUES)
    recap = any(contains_phrase(text, v) for v in RECAP_VERBS)
    staff_hits = _staff_hits(text, staff, player_lasts)
    full_players, last_players = _player_hits(text, players, staff_lasts)
    people = [h[0]["name"] for h in staff_hits] + full_players
    many = len(full_players) + len(last_players - {last_name(p) for p in full_players})
    combined_phrase = any(contains_phrase(text, p) for p in COMBINED_PHRASES) or (
        text.startswith("all ") and contains_phrase(text, "postgame media")
    )
    verdict: Verdict
    if staff_hits and (cue or recap):
        person, strength = staff_hits[0]
        if many >= 2 or len(staff_hits) + many >= 3:
            verdict = Verdict("combined", 0.85, person["name"], people)
        elif cue:
            verdict = Verdict("coach", strength, person["name"], people)
        else:
            verdict = Verdict("coach", 0.7, person["name"], people, ["recap_verb"])
        if person.get("role") != "head":
            verdict.notes.append("assistant_or_acting")
    elif cue:
        unknown = unknown_people(title, people)
        if many >= 2:
            verdict = Verdict("combined", 0.6, None, people, ["several_names"])
        elif full_players or last_players:
            verdict = Verdict("player", 0.9, None, people)
        elif unknown:
            verdict = Verdict("player", 0.5, None, unknown, ["unknown_name"])
        elif combined_phrase:
            verdict = Verdict("combined", 0.85, None, people)
        else:
            verdict = Verdict("combined", 0.6, None, people, ["no_name"])
    elif staff_hits:
        verdict = Verdict(
            "other", 0.5, staff_hits[0][0]["name"], people, ["coach_name_no_cue"]
        )
    else:
        return Verdict("other", 0.9)
    if duration:
        if duration < 45:
            return Verdict("other", 0.8, verdict.coach_guess, people, ["too_short"])
        limit = 25 * 60 if verdict.label == "coach" else 45 * 60
        if duration > limit:
            verdict.score = max(0.0, verdict.score - 0.3)
            verdict.notes.append("long")
    return verdict


def prefilter(title: str, staff: list[dict], players: list[str]) -> bool:
    """Is a flat-listing title worth a full metadata request?"""
    verdict = classify(title, staff, players)
    return verdict.label in {"coach", "combined"} or "unknown_name" in verdict.notes
