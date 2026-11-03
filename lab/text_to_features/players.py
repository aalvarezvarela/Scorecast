"""Tag the players of a game with the team label of their side.

Runs on a text whose two teams are already replaced by their labels
(``team_perspective.replace_teams``). Every player listed for the game (box
score and inactive list) is searched by full name, first name or last name; a
match gets a note after it, "Tyler Herro (Target Team player)". Names are kept.
Accents are ignored ("Doncic" finds "Dončić"). A form that players of both teams
carry ("Jalen" in a Knicks-Thunder game) is not used. A lone first or last name
next to another capitalized name is somebody else ("Kobe Bryant") unless that
name is close to the player's other half, a misspelling ("Austin Reeves").

``label_rule`` decides when a sentence already says enough:

- ``"any"``: no note in a sentence that contains either team label;
- ``"own"``: no note when the sentence contains the label of the player's team;
- ``"never"``: a note on every sentence.

Within a sentence only the first mention of each player gets a note. The rule
is checked on the text before any note is added.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher

PLAYER_NOTE = "({label} player)"
LABEL_RULES = ("any", "own", "never")
SUFFIXES = {"Jr.", "Jr", "Sr.", "Sr", "II", "III", "IV", "V"}


@dataclass(frozen=True)
class Player:
    first_name: str
    last_name: str
    team: str  # schedule nickname, as in the documents' ``home`` / ``away``

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()


@dataclass(frozen=True)
class PlayerMention:
    start: int
    end: int
    surface: str
    player: str  # full name; "A / B" when one form names two teammates
    role: str  # "target" / "opponent"
    # "noted", or why not: "label_in_sentence", "repeat", "other_person"
    outcome: str


@dataclass(frozen=True)
class PlayerTagging:
    text: str
    mentions: tuple[PlayerMention, ...]

    def count(self, outcome: str) -> int:
        return sum(mention.outcome == outcome for mention in self.mentions)


def fold(text: str) -> str:
    """Strip accents one character for one character ("Dončić" -> "Doncic"),
    so that offsets in the folded text are offsets in the original."""
    folded = []
    for char in text:
        base = "".join(
            c
            for c in unicodedata.normalize("NFKD", char)
            if not unicodedata.combining(c)
        )
        folded.append(base[:1] or char)
    return "".join(folded)


def name_forms(player: Player) -> dict[str, str]:
    """Folded surface form -> kind ("full", "first", "last")."""
    first = player.first_name.strip()
    last = player.last_name.strip()
    bare_last = " ".join(part for part in last.split() if part not in SUFFIXES)
    firsts = {first}
    if "." in first:  # "T.J." is also written "TJ"
        firsts.add(first.replace(".", ""))
    elif re.fullmatch(r"[A-Z]{2}", first):  # "OG" is also written "O.G."
        firsts.add(".".join(first) + ".")
    forms = {}
    for name in firsts:
        if name and last:
            forms[f"{name} {last}"] = "full"
            forms[f"{name} {bare_last}"] = "full"
    for name in firsts:
        forms.setdefault(name, "first")
    forms.setdefault(last, "last")  # "Jones Jr."
    forms.setdefault(bare_last, "last")
    return {fold(form): kind for form, kind in forms.items() if len(form) >= 2}


def bare_last_name(player: Player) -> str:
    return " ".join(p for p in player.last_name.split() if p not in SUFFIXES)


# Sentence ends: ".", "!" or "?" (plus closing quotes) before a capital, or a
# line break. A period after an abbreviation or an initial does not end one.
BOUNDARY = re.compile(r"[.!?]+[\"'”’)]*\s+(?=[A-Z0-9\"“‘'(])|\n+")
ABBREVIATIONS = {"Jr", "Sr", "St", "vs", "No", "Mr", "Dr", "Mt", "Ft", "Gov"}
ABBREVIATIONS |= {"Jan", "Feb", "Mar", "Apr", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec"}


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for match in BOUNDARY.finditer(text):
        if not match.group().startswith("\n"):
            word = re.search(r"(\S+?)[.!?]*$", text[start : match.start() + 1])
            token = word.group(1).rstrip(".") if word else ""
            if token in ABBREVIATIONS or re.fullmatch(
                r"(?:[A-Za-z]\.)*[A-Za-z]", token
            ):
                continue
        spans.append((start, match.start() + 1))
        start = match.end()
    spans.append((start, len(text)))
    return spans


TITLE_WORD = re.compile(r"[A-Z][a-z][\w'’-]*")
POSSESSIVE = re.compile(r"[’']s?(?!\w)")
SIMILAR = 0.6  # difflib ratio from which a neighbour is the same name misspelt


def capitalized_neighbor(
    text: str, start: int, end: int, sentence_start: int, labels
) -> tuple[str, int, int] | None:
    """The capitalized word right after or right before a lone first or last
    name, as (side, start, end). Not the first word of a sentence (capitalized
    anyway: "Then Edwards ...") nor a team label ("Target Team's Reaves")."""
    after = re.match(r" ([A-Z][\w'’-]*)", text[end:])
    if after and TITLE_WORD.fullmatch(after.group(1)):
        return "after", end + after.start(1), end + after.end(1)
    before = re.search(r"([A-Z][\w'’-]*) $", text[sentence_start:start])
    if not before:
        return None
    word = POSSESSIVE.sub("", before.group(1))
    if not TITLE_WORD.fullmatch(word) or any(word in label.split() for label in labels):
        return None
    if not text[sentence_start : sentence_start + before.start(1)].strip():
        return None
    return "before", sentence_start + before.start(1), sentence_start + before.end(1)


def similar(a: str, b: str) -> bool:
    return SequenceMatcher(None, a.casefold(), b.casefold()).ratio() >= SIMILAR


def tag_players(
    text: str,
    players: Iterable[Player],
    target: str,
    opponent: str,
    *,
    target_label: str,
    opponent_label: str,
    label_rule: str = "any",
    note: str = PLAYER_NOTE,
) -> PlayerTagging:
    if label_rule not in LABEL_RULES:
        raise ValueError(f"label_rule must be one of {LABEL_RULES}: {label_rule!r}")
    labels = {"target": target_label, "opponent": opponent_label}
    roles = {target: "target", opponent: "opponent"}
    owners: dict[str, dict[str, set[str]]] = {}
    # Lone forms -> the other half of each owner's name: the last name after a
    # first name, the first name before a last name.
    halves: dict[str, list[tuple[str, str]]] = {}
    for player in players:
        role = roles.get(player.team)
        if role is None:
            continue
        for form, kind in name_forms(player).items():
            owners.setdefault(form, {}).setdefault(role, set()).add(player.full_name)
            if kind == "first":
                halves.setdefault(form, []).append(
                    ("after", fold(bare_last_name(player)))
                )
            elif kind == "last":
                halves.setdefault(form, []).append(("before", fold(player.first_name)))
    # A form carried by players of both teams says nothing about the side.
    forms = {form: sides for form, sides in owners.items() if len(sides) == 1}
    if not forms:
        return PlayerTagging(text, ())
    known = {word for form in forms for word in form.split()} | SUFFIXES
    pattern = re.compile(
        r"(?<![\w'’-])(?:"
        + "|".join(re.escape(form) for form in sorted(forms, key=len, reverse=True))
        + r")(?![\w-])"
    )

    folded = fold(text)
    spans = sentence_spans(text)
    mentions, inserts, seen = [], [], set()
    for match in pattern.finditer(folded):
        form = match.group(0)
        ((role, names),) = forms[form].items()
        player = " / ".join(sorted(names))
        index = next(i for i, (s, e) in enumerate(spans) if s <= match.start() < e)
        sentence = text[spans[index][0] : spans[index][1]]
        start, end = match.start(), match.end()
        outcome = None
        neighbor = None
        if form in halves:
            neighbor = capitalized_neighbor(
                folded, start, end, spans[index][0], labels.values()
            )
        if neighbor:
            side, word_start, word_end = neighbor
            word = POSSESSIVE.sub("", folded[word_start:word_end])
            if any(
                part_side == side and similar(word, part)
                for part_side, part in halves[form]
            ):
                # The other half, misspelt ("Austin Reeves"): one mention.
                start, end = min(start, word_start), max(end, word_end)
            elif word not in known:
                outcome = "other_person"  # "Kobe Bryant", "Cole Anthony"
        if outcome is None and label_rule != "never":
            present = labels.values() if label_rule == "any" else [labels[role]]
            if any(label in sentence for label in present):
                outcome = "label_in_sentence"
        if outcome is None and (index, player) in seen:
            outcome = "repeat"
        if outcome is None:
            outcome = "noted"
            seen.add((index, player))
            possessive = POSSESSIVE.match(text, end)
            at = possessive.end() if possessive else end
            inserts.append((at, " " + note.format(label=labels[role])))
        mentions.append(
            PlayerMention(
                start,
                end,
                text[start:end],
                player,
                role,
                outcome,
            )
        )
    for at, insertion in reversed(inserts):
        text = text[:at] + insertion + text[at:]
    return PlayerTagging(text, tuple(mentions))
