"""The three model inputs embedded for one coach presser.

| view          | text                                   | instruction           |
| ------------- | -------------------------------------- | --------------------- |
| `raw`         | the captions as they are               | none                  |
| `instructed`  | the captions as they are               | `PRESSER_INSTRUCTION` |
| `target_view` | the coach's team -> `Target Team`, the | `TARGET_INSTRUCTION`  |
|               | other -> `Opponent Team`, player notes |                       |

`raw` -> `instructed` isolates the instruction; `instructed` -> `target_view`
isolates the rewrite (the two instructions say the same, one with the labels).
A presser has one rewrite only: the coach belongs to the Target Team.

The rewrite reuses the report rules (`nba_ou.data_processing.game_texts`),
with two differences. The intro sentence gives the sides, not the franchise
names, so the vector is not told which teams played. Player notes do not
depend on sentences (captions before 2024-25 have no punctuation): a player
gets a note at his first mention and again after `NOTE_GAP_WORDS` words.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

from nba_ou.data_processing.game_texts.players import (
    PLAYER_NOTE,
    POSSESSIVE,
    Player,
    tag_players,
)
from nba_ou.data_processing.game_texts.team_perspective import (
    OPPONENT_LABEL,
    TARGET_LABEL,
    replace_teams,
)

PRESSER_INSTRUCTION = (
    "Represent this NBA head coach's postgame press conference: how his team "
    "and the opponent played (result, offense, defense, effort, pace) and "
    "which players were hurt or unavailable."
)
TARGET_INSTRUCTION = (
    "Represent this NBA head coach's postgame press conference from the "
    "Target Team's point of view: how the Target Team and the Opponent Team "
    "played (result, offense, defense, effort, pace) and which players were "
    "hurt or unavailable."
)
#: view -> (text version, instruction)
VIEWS: dict[str, tuple[str, str | None]] = {
    "raw": ("raw", None),
    "instructed": ("raw", PRESSER_INSTRUCTION),
    "target_view": ("target_view", TARGET_INSTRUCTION),
}
#: Words after which a player's next mention gets a note again.
NOTE_GAP_WORDS = 80


@dataclass(frozen=True)
class PresserRewrite:
    text: str
    target_mentions: int
    opponent_mentions: int
    player_mentions: int
    player_notes: int


def intro_sentence(target_is_home: bool) -> str:
    sides = ("home team", "visiting team")
    target_side, opponent_side = sides if target_is_home else sides[::-1]
    return (
        f"In this text, the head coach of the {TARGET_LABEL} ({target_side}) "
        f"answers reporters after the game against the {OPPONENT_LABEL} "
        f"({opponent_side})."
    )


def note_players(
    text: str, players: Sequence[Player], target: str, opponent: str
) -> tuple[str, int, int]:
    """``text`` with "(Target Team player)" / "(Opponent Team player)" after
    player names; returns it with the number of mentions and of notes."""
    labels = {"target": TARGET_LABEL, "opponent": OPPONENT_LABEL}
    tagging = tag_players(
        text,
        players,
        target,
        opponent,
        target_label=TARGET_LABEL,
        opponent_label=OPPONENT_LABEL,
        label_rule="never",
    )
    mentions = [m for m in tagging.mentions if m.outcome != "other_person"]
    word_starts = [0]
    for index, char in enumerate(text):
        if char.isspace():
            word_starts.append(index + 1)
    noted_at: dict[str, int] = {}
    inserts = []
    for mention in sorted(mentions, key=lambda m: m.start):
        word = bisect_right(word_starts, mention.start)
        last = noted_at.get(mention.player)
        if last is not None and word - last < NOTE_GAP_WORDS:
            continue
        noted_at[mention.player] = word
        possessive = POSSESSIVE.match(text, mention.end)
        at = possessive.end() if possessive else mention.end
        inserts.append((at, " " + PLAYER_NOTE.format(label=labels[mention.role])))
    for at, note in reversed(inserts):
        text = text[:at] + note + text[at:]
    return text, len(mentions), len(inserts)


def rewrite_presser(
    text: str,
    target: str,
    opponent: str,
    target_is_home: bool,
    players: Sequence[Player] | None = None,
) -> PresserRewrite:
    """The presser seen from the coach's team; ``target`` / ``opponent`` are
    schedule nicknames ("Lakers")."""
    rewritten = replace_teams(text, target, opponent)
    body, mentions, notes = rewritten.text, 0, 0
    if players:
        body, mentions, notes = note_players(body, players, target, opponent)
    return PresserRewrite(
        text=f"{intro_sentence(target_is_home)}\n\n{body}",
        target_mentions=rewritten.count("target"),
        opponent_mentions=rewritten.count("opponent"),
        player_mentions=mentions,
        player_notes=notes,
    )


#: What a name is replaced with in ``mask_presser``, in the coach's voice.
OUR_TEAM, THEIR_TEAM = "our team", "the other team"
OUR_PLAYER, THEIR_PLAYER = "our player", "their player"


def mask_presser(
    text: str, target: str, opponent: str, players: Sequence[Player] | None = None
) -> str:
    """The presser without names: both teams and every recognized player are
    replaced by whose they are ("our player", "the other team"), so that the
    vector cannot hold who played, only what is said about them."""
    masked = replace_teams(
        text, target, opponent, target_label=OUR_TEAM, opponent_label=THEIR_TEAM
    ).text
    if not players:
        return masked
    tagging = tag_players(
        masked,
        players,
        target,
        opponent,
        target_label=OUR_TEAM,
        opponent_label=THEIR_TEAM,
        label_rule="never",
    )
    names = {"target": OUR_PLAYER, "opponent": THEIR_PLAYER}
    spans = [m for m in tagging.mentions if m.outcome != "other_person"]
    for mention in sorted(spans, key=lambda m: m.start, reverse=True):
        masked = masked[: mention.start] + names[mention.role] + masked[mention.end :]
    return masked
