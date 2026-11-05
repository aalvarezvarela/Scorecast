"""The exact model inputs embedded for one game report.

Three inputs per report (``VIEWS``):

- ``raw``: title and body without AP dateline and footers, no instruction;
- ``target_view`` for the home team: the ``home_target`` rewrite (intro
  sentence, team labels, player notes) under ``TARGET_VIEW_INSTRUCTION``;
- ``target_view`` for the away team: the ``away_target`` rewrite, same
  instruction.

The instruction is the one the role probes chose for Qwen3-Embedding-8B
(``lab/text_to_features/README.md``). ``input_sha256`` hashes the exact string
sent to the model, instruction included, so a stored vector is re-embedded
only when its input changes (report edited, players corrected in the database,
rewrite rules changed).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from nba_ou.data_processing.game_texts.players import Player
from nba_ou.data_processing.game_texts.team_perspective import document_perspectives

TARGET_VIEW_INSTRUCTION = (
    "Represent this NBA game recap from the Target Team's point of view."
)
INSTRUCTIONS: dict[str, str | None] = {
    "raw": None,
    "target_view": TARGET_VIEW_INSTRUCTION,
}
#: ``perspective_texts`` options of the stored rewrites (the role probes').
REWRITE_OPTIONS = {
    "strip": True,
    "intro": True,
    "name_teams": True,
    "label_rule": "any",
}
#: (view, side, perspective version) of every input of a report.
VIEWS = (
    ("raw", None, "raw"),
    ("target_view", "home", "home_target"),
    ("target_view", "away", "away_target"),
)


def embedding_input(text: str, instruction: str | None = None) -> str:
    """Apply Qwen's instruction format; a missing instruction preserves the text."""
    if instruction is None:
        return text
    if not instruction.strip():
        raise ValueError("instruction must be nonempty or None")
    return f"Instruct: {instruction}\nQuery:{text}"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def team_ids() -> dict[str, int]:
    """Schedule nickname (a report's ``home`` / ``away``) to NBA team id."""
    from nba_api.stats.static import teams

    return {team["nickname"]: int(team["id"]) for team in teams.get_teams()}


@dataclass(frozen=True)
class EmbeddingInput:
    game_id: str
    view: str
    side: str | None  # "home" / "away"; None for the raw text
    team_id: int | None  # the Target Team; None for the raw text
    instruction: str | None
    text: str
    model_input: str
    input_sha256: str


def report_sha256(document: dict) -> str:
    """Hash of the stored report text, to tell an edited report from a new rule."""
    return sha256(f"{document.get('title') or ''}\n\n{document['text']}")


def build_inputs(
    document: dict, players: Sequence[Player] | None
) -> list[EmbeddingInput]:
    """The three inputs of one report, in ``VIEWS`` order."""
    texts = document_perspectives(document, players=players, **REWRITE_OPTIONS)
    ids = team_ids()
    inputs = []
    for view, side, version in VIEWS:
        instruction = INSTRUCTIONS[view]
        model_input = embedding_input(texts[version], instruction)
        inputs.append(
            EmbeddingInput(
                game_id=str(document["game_id"]),
                view=view,
                side=side,
                team_id=ids[document[side]] if side else None,
                instruction=instruction,
                text=texts[version],
                model_input=model_input,
                input_sha256=sha256(model_input),
            )
        )
    return inputs
