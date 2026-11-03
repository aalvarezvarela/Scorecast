"""Rewrite a game text from the point of view of one of its two teams.

Every mention of the two teams that played (nickname, city, full name, common
short forms) is replaced by a role label: ``Target Team`` for the team whose
point of view is taken and ``Opponent Team`` for the other. Other teams and the
rest of the text are left as written. The rules are deterministic; no model is
involved.

``perspective_texts()`` gives the three versions of one document: the raw
text, the home team as target, and the away team as target. Given the game's
players, the rewrites also note each player's side (``players.tag_players``).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from lab.text_perspective.players import PLAYER_NOTE, Player, tag_players

TARGET_LABEL = "Target Team"
OPPONENT_LABEL = "Opponent Team"


@dataclass(frozen=True)
class Team:
    """Surface forms of one franchise.

    ``nickname`` is the key used by the schedule and stored with every report
    and transcript (``home`` / ``away``). ``cities`` are place names, dropped
    when both teams of the game share one (Lakers vs Clippers). ``names`` are
    matched in any case; ``words`` are ordinary English words ("Heat",
    "Magic", "Kings") and match only capitalized or in upper case.
    """

    nickname: str
    cities: tuple[str, ...]
    names: tuple[str, ...] = ()
    words: tuple[str, ...] = ()

    @property
    def full_name(self) -> str:
        return f"{self.cities[0]} {self.nickname}"

    def aliases(self) -> dict[str, bool]:
        """Every surface form, mapped to whether it is matched case-sensitively."""
        forms: dict[str, bool] = {}
        nicknames = [self.nickname, *self.names, *self.words]
        for city in self.cities:
            forms[city] = is_short(city)
            for nickname in nicknames:
                forms[f"{city} {nickname}"] = False
        for name in self.names:
            forms[name] = is_short(name)
        for word in self.words:
            forms[word] = True
            forms[word.upper()] = True
        if self.nickname not in forms:
            forms[self.nickname] = False
        return forms


def is_short(alias: str) -> bool:
    # "LA", "OKC" and "C's" would match ordinary words in any case.
    return len(alias.replace(".", "")) <= 3


LA = ("Los Angeles", "LA", "L.A.")

TEAMS = {
    team.nickname: team
    for team in (
        Team("Hawks", ("Atlanta",), words=("Hawks",)),
        Team("Celtics", ("Boston",), names=("Celtics", "Celtic", "Celts", "C's")),
        Team("Nets", ("Brooklyn",), words=("Nets",)),
        Team("Hornets", ("Charlotte",), words=("Hornets",)),
        Team("Bulls", ("Chicago",), words=("Bulls",)),
        Team("Cavaliers", ("Cleveland",), names=("Cavs",), words=("Cavaliers",)),
        Team("Mavericks", ("Dallas",), names=("Mavs",), words=("Mavericks",)),
        Team("Nuggets", ("Denver",), names=("Nugs",), words=("Nuggets",)),
        Team("Pistons", ("Detroit",), words=("Pistons",)),
        Team("Warriors", ("Golden State",), words=("Warriors", "Dubs")),
        Team("Rockets", ("Houston",), words=("Rockets",)),
        Team("Pacers", ("Indiana",), names=("Pacers", "Pacer")),
        Team("Clippers", LA, words=("Clippers", "Clipper", "Clips")),
        Team("Lakers", LA, names=("Lakers", "Laker", "Lake Show", "Purple and Gold")),
        Team("Grizzlies", ("Memphis",), names=("Grizz",), words=("Grizzlies",)),
        Team("Heat", ("Miami",), words=("Heat",)),
        Team("Bucks", ("Milwaukee",), words=("Bucks",)),
        Team(
            "Timberwolves",
            ("Minnesota",),
            names=("Timberwolves", "Timberwolf", "T-Wolves", "Twolves"),
            words=("Wolves",),
        ),
        Team("Pelicans", ("New Orleans",), words=("Pelicans", "Pels")),
        Team("Knicks", ("New York",), names=("Knicks", "Knick", "Knickerbockers")),
        Team("Thunder", ("Oklahoma City", "OKC"), words=("Thunder",)),
        Team("Magic", ("Orlando",), words=("Magic",)),
        Team("76ers", ("Philadelphia", "Philly"), names=("76ers", "Sixers", "Sixer")),
        Team("Suns", ("Phoenix",), words=("Suns",)),
        Team(
            "Trail Blazers",
            ("Portland", "Rip City"),
            names=("Trail Blazers", "Trail Blazer"),
            words=("Blazers", "Blazer"),
        ),
        Team("Kings", ("Sacramento",), words=("Kings",)),
        Team("Spurs", ("San Antonio",), words=("Spurs",)),
        Team("Raptors", ("Toronto",), names=("Raptors", "Raptor"), words=("Raps",)),
        Team("Jazz", ("Utah",), words=("Jazz",)),
        Team("Wizards", ("Washington",), words=("Wizards",)),
    )
}


@dataclass(frozen=True)
class Mention:
    """One matched span of the input text."""

    start: int
    end: int
    surface: str
    # "target" / "opponent" are replaced; "other" (a team that did not play)
    # and "shared" (a city of both teams) are left as written.
    role: str


@dataclass(frozen=True)
class Rewrite:
    text: str
    mentions: tuple[Mention, ...]

    def count(self, role: str) -> int:
        return sum(mention.role == role for mention in self.mentions)


def team(nickname: str) -> Team:
    try:
        return TEAMS[nickname]
    except KeyError:
        raise KeyError(f"Unknown team nickname {nickname!r}") from None


def build_rules(target: str, opponent: str) -> dict[str, tuple[str, bool]]:
    """Alias -> (role, case_sensitive) for one game, from one team's view."""
    if target == opponent:
        raise ValueError(f"Target and opponent are the same team: {target!r}")
    playing = {"target": team(target), "opponent": team(opponent)}
    rules: dict[str, tuple[str, bool]] = {}
    # Teams that did not play are matched only so that their names are kept
    # whole: "LA Clippers" must not become "Target Team Clippers".
    for other in TEAMS.values():
        if other.nickname not in (target, opponent):
            for alias, case_sensitive in other.aliases().items():
                rules[alias] = ("other", case_sensitive)
    for role, side in playing.items():
        rules.update(
            (alias, (role, case_sensitive))
            for alias, case_sensitive in side.aliases().items()
        )
    # A place shared by both teams names neither of them.
    for city in set(playing["target"].cities) & set(playing["opponent"].cities):
        rules[city] = ("shared", rules[city][1])
    return rules


@lru_cache(maxsize=256)
def game_matcher(
    target: str, opponent: str
) -> tuple[dict[str, tuple[str, bool]], re.Pattern[str]]:
    rules = build_rules(target, opponent)
    return rules, compile_rules(rules)


def compile_rules(rules: dict[str, tuple[str, bool]]) -> re.Pattern[str]:
    # Python takes the first alternative that matches, so longer aliases go
    # first: "Los Angeles Lakers" before "Los Angeles", "T-Wolves" before
    # "Wolves".
    alternatives = [
        re.escape(alias) if case_sensitive else f"(?i:{re.escape(alias)})"
        for alias, (_, case_sensitive) in sorted(
            rules.items(), key=lambda item: (-len(item[0]), item[0])
        )
    ]
    return re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)")


def lookup(rules: dict[str, tuple[str, bool]], surface: str) -> str:
    if surface in rules:
        return rules[surface][0]
    folded = surface.casefold()
    for alias, (role, case_sensitive) in rules.items():
        if not case_sensitive and alias.casefold() == folded:
            return role
    raise LookupError(f"No rule matches {surface!r}")


def replace_teams(
    text: str,
    target: str,
    opponent: str,
    *,
    target_label: str = TARGET_LABEL,
    opponent_label: str = OPPONENT_LABEL,
) -> Rewrite:
    """Replace every mention of ``target`` and ``opponent`` with role labels."""
    rules, pattern = game_matcher(target, opponent)
    labels = {"target": target_label, "opponent": opponent_label}
    mentions: list[Mention] = []

    def substitute(match: re.Match[str]) -> str:
        surface = match.group(0)
        role = lookup(rules, surface)
        mentions.append(Mention(match.start(), match.end(), surface, role))
        return labels.get(role, surface)

    rewritten = pattern.sub(substitute, text)
    return Rewrite(rewritten, tuple(mentions))


# "TORONTO (AP) — ", "SACRAMENTO, Calif. (AP) ", "ORLANDO, Fla. - (AP)"
DATELINE = re.compile(r"\A\s*[A-Z][A-Z.'’ -]+(?:,[^()\n]{0,30})?\(AP\)\s*[—–-]*\s*")
# AP footers: a rule of dashes, "(More) AP NBA: <url>", "See AP’s full NBA
# coverage here" and the writer's "Follow ... on Twitter: <url>". Story text
# has no URLs, so any line with one is footer.
FOOTER_LINE = re.compile(
    r"^.*(?:https?://|www\.|twitter\.com/|apnews\.com).*$"
    r"|^\s*(?:-{2,}|See AP\b|(?:More )?AP N[BH][AL]\b).*$",
    re.MULTILINE,
)
# Wishper transcripts start with a stray "- ".
LEADING_DASH = re.compile(r"\A\s*-\s+")


def strip_boilerplate(text: str) -> str:
    """Drop the AP dateline and footer and the transcript's leading dash.

    The dateline is the venue's city in capitals; left in place it would
    become a team label although it names a place.
    """
    text = DATELINE.sub("", text)
    text = LEADING_DASH.sub("", text)
    text = FOOTER_LINE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def compose(title: str, body: str) -> str:
    return f"{title.strip()}\n\n{body}" if title.strip() else body


def intro_sentence(
    target: str,
    opponent: str,
    target_is_home: bool,
    *,
    name_teams: bool = True,
    target_label: str = TARGET_LABEL,
    opponent_label: str = OPPONENT_LABEL,
) -> str:
    """Who each label stands for, e.g. "In this text, Target Team is the Los
    Angeles Lakers (home team) and Opponent Team is the Minnesota Timberwolves
    (visiting team)."""
    sides = ("home team", "visiting team")
    target_side, opponent_side = sides if target_is_home else sides[::-1]
    if not name_teams:
        return (
            f"In this text, {target_label} is the {target_side} and "
            f"{opponent_label} is the {opponent_side}."
        )
    return (
        f"In this text, {target_label} is the {team(target).full_name} "
        f"({target_side}) and {opponent_label} is the "
        f"{team(opponent).full_name} ({opponent_side})."
    )


def perspective_texts(
    text: str,
    home: str,
    away: str,
    *,
    title: str = "",
    strip: bool = True,
    intro: bool = True,
    name_teams: bool = True,
    players: Sequence[Player] | None = None,
    label_rule: str = "any",
    player_note: str = PLAYER_NOTE,
    target_label: str = TARGET_LABEL,
    opponent_label: str = OPPONENT_LABEL,
) -> dict[str, str]:
    """The raw text plus one rewrite from each team's point of view.

    Each version is the title, a blank line and the body; with ``strip`` the
    body of every version, ``raw`` included, loses the AP dateline, footers and
    the transcripts' leading dash. With ``players``,
    their mentions get a note of their side, under ``label_rule`` (see
    ``players``). A rewrite starts with the intro sentence; it is written after
    the replacements, so the team names it gives (``name_teams``) stay as
    written.
    """
    body = strip_boilerplate(text) if strip else text
    labels = {"target_label": target_label, "opponent_label": opponent_label}
    texts = {"raw": compose(title, body)}
    for key, target, opponent in (
        ("home_target", home, away),
        ("away_target", away, home),
    ):
        rewritten = replace_teams(compose(title, body), target, opponent, **labels)
        parts = [rewritten.text]
        if players:
            parts[0] = tag_players(
                rewritten.text,
                players,
                target,
                opponent,
                label_rule=label_rule,
                note=player_note,
                **labels,
            ).text
        if intro:
            parts.insert(
                0,
                intro_sentence(
                    target,
                    opponent,
                    target == home,
                    name_teams=name_teams,
                    **labels,
                ),
            )
        texts[key] = "\n\n".join(parts)
    return texts


def document_perspectives(document: dict, **options) -> dict[str, str]:
    """``perspective_texts()`` for a stored report or transcript."""
    return perspective_texts(
        document["text"],
        document["home"],
        document["away"],
        title=document.get("title") or "",
        **options,
    )
