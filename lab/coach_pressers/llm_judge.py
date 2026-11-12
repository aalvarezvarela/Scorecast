"""Step 7: an LLM picks among candidates the rules could not separate.

OpenAI Responses API with Structured Outputs (strict JSON schema). The schema
is built per call so ``video_id`` can only be one of the candidates or null;
the answer is validated again before use. Every answer is cached in
``llm_decisions.jsonl`` under a hash of the exact input, so re-runs cost
nothing. The API key comes from ``OPENAI_API_KEY`` (env or
``config.secrets.ini``, section ``[OpenAI]``) and is never logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import timedelta
from pathlib import Path

from lab.coach_pressers.config import PresserConfig
from lab.coach_pressers.match import (
    parse_instant,
    rival_head_coaches,
    selection_from,
    team_games,
)
from lab.coach_pressers.names import normalize
from lab.coach_pressers.teams import TEAMS
from lab.game_reports.season_reports import utc_now

log = logging.getLogger("coach_pressers")

PROMPT_VERSION = 4
MIN_CONFIDENCE = 0.6
SYSTEM = """\
Match NBA YouTube videos to this team's coach speaking to media after THIS
game, including interim, acting or assistant coaches. Compare publication time,
title/description dates, opponent, playoff game number, score and nearby games
to distinguish back-to-backs. Reject actual pregame, practice, shootaround,
highlights, recaps and player-only content. If none fits, return video_id null
and kind "none".

requires_llm and rule_flags are warnings, not rejection rules. Postgame coaches
may discuss the locker room, quarters, tonight's win or the next opponent.
Missing postgame keywords or unusual duration alone do not disqualify a video.
Use all context. gap_fill candidates include unrelated uploads near the game:
require evidence of the coach's postgame media appearance. A player-titled video
can include the coach in its description or chapters.

kind is "coach" for coach-only or "combined" for coach and players together.
coach_name is the actual speaker as coach. For combined videos, coach_chapter
is the coach chapter's chapter_id, or null when absent or unidentified.
confidence is your probability (0-1) that the answer is right.
Return only the schema fields, without explanation.
"""

PROMOTIONAL_LINE = re.compile(
    r"^(?:subscribe\b|follow (?:us|our)\b|connect with us\b|visit our\b|"
    r"(?:instagram|twitter|facebook|tiktok)\s*[:»])",
    re.IGNORECASE,
)


def compact_description(description: str) -> str:
    """Remove links and common promotional lines; retain up to 400 characters."""
    lines = []
    for line in description.splitlines():
        line = line.strip()
        if PROMOTIONAL_LINE.match(line):
            continue
        line = re.sub(r"https?://\S+", "", line)
        lines.append(line)
    return " ".join(" ".join(lines).split())[:400]


class BudgetExceeded(RuntimeError):
    pass


def schema(candidate_ids: list[str], chapter_ids: list[str]) -> dict:
    chapter = (
        {"anyOf": [{"type": "string", "enum": chapter_ids}, {"type": "null"}]}
        if chapter_ids
        else {"type": "null"}
    )
    return {
        "type": "object",
        "properties": {
            "video_id": {
                "anyOf": [{"type": "string", "enum": candidate_ids}, {"type": "null"}]
            },
            "kind": {"type": "string", "enum": ["coach", "combined", "none"]},
            "coach_name": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "coach_chapter": chapter,
            "confidence": {"type": "number"},
        },
        "required": [
            "video_id",
            "kind",
            "coach_name",
            "coach_chapter",
            "confidence",
        ],
        "additionalProperties": False,
    }


def encode_chapters(payload: dict) -> tuple[dict, dict[str, tuple[str, str]]]:
    """Keep free-form titles in input, using safe references in the strict enum."""
    candidates, references = [], {}
    for video_index, candidate in enumerate(payload["candidates"]):
        chapters = []
        for chapter_index, chapter in enumerate(candidate["chapters"]):
            reference = f"v{video_index}_ch{chapter_index}"
            references[reference] = (candidate["video_id"], chapter["title"])
            chapters.append({**chapter, "chapter_id": reference})
        candidates.append({**candidate, "chapters": chapters})
    return {**payload, "candidates": candidates}, references


def build_input(
    entry: dict, found: list[dict], manifest: dict, team_data: dict
) -> dict:
    tip = parse_instant(entry["tip_utc"])
    nearby = []
    for game in team_games(manifest, entry["team"]):
        other = parse_instant(game["tip_utc"])
        if (
            game["game_id"] != entry["game_id"]
            and tip
            and other
            and abs(other - tip) <= timedelta(days=3)
        ):
            nearby.append(
                {
                    "date": game["date"],
                    "tip_utc": game["tip_utc"],
                    "opponent": TEAMS[game["opponent"]][0],
                    "home": game["is_home"],
                }
            )
    return {
        "team": TEAMS[entry["team"]][0],
        "game": {
            "date": entry["date"],
            "tip_utc": entry["tip_utc"],
            "home": entry["is_home"],
            "opponent": TEAMS[entry["opponent"]][0],
            "score": f"{entry['team_score']}-{entry['opponent_score']}",
            "phase": entry["phase"],
            "playoff_game_no": entry["playoff_game_no"],
        },
        "other_games_within_3_days": nearby,
        "coach_expected": entry.get("coach_expected"),
        "staff": [{"name": s["name"], "role": s["role"]} for s in team_data["staff"]],
        "candidates": [
            {
                "video_id": c["video_id"],
                "channel_priority": c["channel_priority"],
                "title": c["title"],
                "published_at_utc": c["published_at_utc"],
                "published_date": c["published_date"],
                "duration_s": c["duration_s"],
                "requires_llm": c.get("requires_llm", False),
                "rule_flags": c.get("notes", []),
                "description": compact_description(c.get("description") or ""),
                "chapters": [
                    {"start_s": ch["start_time"], "title": ch["title"]}
                    for ch in c.get("chapters") or []
                ],
            }
            for c in found
        ],
    }


def input_hash(payload: dict, model: str, reasoning: str = "none") -> str:
    canonical = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "model": model,
            "reasoning": reasoning,
            "input": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate(answer: dict, payload: dict) -> str | None:
    """An error message, or None if the answer can be used."""
    ids = {c["video_id"] for c in payload["candidates"]}
    if answer["kind"] == "none":
        return None if answer["video_id"] is None else "kind none with a video_id"
    if answer["video_id"] not in ids:
        return f"video_id {answer['video_id']!r} is not a candidate"
    if answer["coach_chapter"] is not None:
        chosen = next(
            c for c in payload["candidates"] if c["video_id"] == answer["video_id"]
        )
        if answer["coach_chapter"] not in {ch["title"] for ch in chosen["chapters"]}:
            return "coach_chapter is not a chapter of the chosen video"
    if not 0 <= float(answer["confidence"]) <= 1:
        return "confidence outside 0-1"
    return None


class DecisionCache:
    def __init__(self, path: Path):
        self.path = path
        self.decisions: dict[str, dict] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.decisions[record["input_hash"]] = record

    def get(self, key: str) -> dict | None:
        return self.decisions.get(key)

    def add(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.decisions[record["input_hash"]] = record


def default_client():
    from nba_ou.config.settings import SETTINGS
    from openai import OpenAI

    return OpenAI(api_key=SETTINGS.openai_api_key)


def refusal_of(response) -> str | None:
    for item in getattr(response, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return getattr(part, "refusal", "") or "refused"
    return None


class Judge:
    def __init__(
        self, config: PresserConfig, client=None, cache: DecisionCache | None = None
    ):
        self.config = config
        self._client = client
        self.cache = cache or DecisionCache(config.llm_cache_path)
        self.spent_usd = 0.0
        self.calls = 0
        self.cached = 0

    @property
    def client(self):
        if self._client is None:
            self._client = default_client()
        return self._client

    def ask(self, payload: dict, *, game_id: str = "", team: str = "") -> dict:
        """The cached or fresh decision record for one input."""
        key = input_hash(payload, self.config.llm_model, self.config.llm_reasoning)
        cached = self.cache.get(key)
        if cached is not None:
            self.cached += 1
            return cached
        if self.spent_usd >= self.config.llm_budget_usd:
            raise BudgetExceeded(
                f"LLM spending cap of ${self.config.llm_budget_usd:.2f} reached"
            )
        ids = [c["video_id"] for c in payload["candidates"]]
        # This only changes the API representation. Cache keys and stored answers
        # still use the original titles, so existing decisions remain reusable.
        request_payload, chapter_refs = encode_chapters(payload)
        response = self.client.responses.create(
            model=self.config.llm_model,
            instructions=SYSTEM,
            input=json.dumps(
                request_payload, ensure_ascii=False, separators=(",", ":")
            ),
            reasoning={"effort": self.config.llm_reasoning},
            text={
                "format": {
                    "type": "json_schema",
                    "name": "presser_choice",
                    "strict": True,
                    "schema": schema(ids, list(chapter_refs)),
                }
            },
            store=False,
        )
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "input_tokens", 0) or 0
        tokens_out = getattr(usage, "output_tokens", 0) or 0
        cost = (
            tokens_in * self.config.llm_price_in_per_m
            + tokens_out * self.config.llm_price_out_per_m
        ) / 1e6
        self.spent_usd += cost
        self.calls += 1
        record = {
            "input_hash": key,
            "game_id": game_id,
            "team": team,
            "model": self.config.llm_model,
            "reasoning": self.config.llm_reasoning,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": round(cost, 6),
            "created_at_utc": utc_now(),
            "answer": None,
            "error": None,
        }
        refusal = refusal_of(response)
        if refusal:
            record["error"] = f"refusal: {refusal}"
        elif getattr(response, "status", "completed") != "completed":
            record["error"] = f"response {response.status}"
        else:
            try:
                answer = json.loads(response.output_text)
                reference = answer["coach_chapter"]
                if reference is not None:
                    chapter = chapter_refs.get(reference)
                    if chapter is None or chapter[0] != answer["video_id"]:
                        record["error"] = (
                            "coach_chapter is not a chapter of the chosen video"
                        )
                    else:
                        answer["coach_chapter"] = chapter[1]
                if record["error"] is None:
                    record["error"] = validate(answer, payload)
                record["answer"] = answer
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                record["error"] = f"unparseable answer: {exc}"
        self.cache.add(record)
        return record


def apply_decision(
    entry: dict, found: list[dict], record: dict, rivals: set[str] = frozenset()
) -> None:
    answer, error = record.get("answer"), record.get("error")
    llm = {
        "model": record["model"],
        "input_hash": record["input_hash"],
        "answer": answer,
        "error": error,
    }
    if error or not answer:
        entry.update(status="review", selection=None, llm=llm, error=error or "")
        return
    if answer["kind"] == "none" or answer["confidence"] < MIN_CONFIDENCE:
        entry.update(status="review", selection=None, llm=llm, error="")
        return
    if normalize(answer.get("coach_name") or "") in rivals:
        entry.update(
            status="review",
            selection=None,
            llm=llm,
            error=f"judge chose another team's head coach: {answer['coach_name']}",
        )
        return
    chosen = next(c for c in found if c["video_id"] == answer["video_id"])
    chapter = None
    if answer["coach_chapter"]:
        chapters = chosen["chapters"]
        index = next(
            i for i, ch in enumerate(chapters) if ch["title"] == answer["coach_chapter"]
        )
        chapter = chapters[index]
    entry.update(
        status="selected",
        selection=selection_from(
            chosen,
            decided_by="llm",
            coach_actual=answer["coach_name"] or entry.get("coach_expected"),
            kind=answer["kind"],
            chapter=chapter,
            confidence=answer["confidence"],
            reason=answer.get("reason", ""),
        ),
        llm=llm,
        error="",
    )


def judge_season(
    config: PresserConfig,
    manifest: dict,
    coaches: dict,
    to_judge: dict[str, list[dict]],
    *,
    judge: Judge | None = None,
    gaps: dict[str, list[dict]] | None = None,
) -> dict:
    """Decide every entry the rules left open; ambiguous ones go to review.

    ``gaps`` (step 6b) are judged after them; a gap the judge answers "none"
    for, or one left undecided, stays ``not_found``.
    """
    gaps = gaps or {}
    stats = {"llm_selected": 0, "review": 0, "calls": 0, "cached": 0, "cost_usd": 0.0}
    if gaps:
        stats.update(gaps_selected=0, gaps_review=0, gaps_none=0)
    if not config.use_llm:
        for key in to_judge:
            manifest["entries"][key].update(status="review", selection=None)
        stats["review"] = len(to_judge)
        return stats
    judge = judge or Judge(config)
    try:
        for key, found in sorted(to_judge.items()):
            entry = manifest["entries"][key]
            payload = build_input(
                entry, found, manifest, coaches["teams"][entry["team"]]
            )
            record = judge.ask(payload, game_id=entry["game_id"], team=entry["team"])
            rivals = rival_head_coaches(coaches["teams"], entry["team"])
            apply_decision(entry, found, record, rivals)
            stats["llm_selected" if entry["status"] == "selected" else "review"] += 1
        for key, found in sorted(gaps.items()):
            entry = manifest["entries"][key]
            payload = build_input(
                entry, found, manifest, coaches["teams"][entry["team"]]
            )
            record = judge.ask(payload, game_id=entry["game_id"], team=entry["team"])
            rivals = rival_head_coaches(coaches["teams"], entry["team"])
            apply_decision(entry, found, record, rivals)
            answer = record.get("answer") or {}
            if entry["status"] == "selected":
                stats["gaps_selected"] += 1
            elif not record.get("error") and answer.get("kind") == "none":
                entry.update(status="not_found")
                stats["gaps_none"] += 1
            else:
                stats["gaps_review"] += 1
    except BudgetExceeded as exc:
        log.warning("step 7 %s: %s; the rest goes to review", config.season, exc)
        for key in to_judge:
            entry = manifest["entries"][key]
            if entry["status"] not in {"selected", "review"}:
                entry.update(status="review", error=str(exc))
        stats["stopped"] = str(exc)
    stats.update(
        calls=judge.calls, cached=judge.cached, cost_usd=round(judge.spent_usd, 4)
    )
    log.info("step 7 %s: %s", config.season, json.dumps(stats))
    return stats


# ------------------------------------------------------------------ llm-check


def check_main(args) -> int:
    """Run the labelled cases against the live API: agreement, tokens, cost."""
    from lab.coach_pressers.llm_cases import CASES

    config = PresserConfig(
        2025,
        args.local_root,
        llm_model=args.llm_model,
        llm_reasoning=args.llm_reasoning,
    )
    judge = Judge(config, cache=DecisionCache(config.root / "llm_check.jsonl"))
    agree, rows = 0, []
    for case in CASES:
        record = judge.ask(case["input"], game_id=case["name"], team="check")
        answer = record.get("answer") or {}
        ok = (
            not record.get("error")
            and answer.get("video_id") == case["expected"]["video_id"]
            and answer.get("kind") == case["expected"]["kind"]
        )
        agree += ok
        rows.append(
            f"{'ok ' if ok else 'BAD'} {case['name']:28} expected "
            f"{case['expected']['video_id']}/{case['expected']['kind']}, got "
            f"{answer.get('video_id')}/{answer.get('kind')} "
            f"({answer.get('confidence')}) {record.get('error') or ''}"
        )
    print("\n".join(rows))
    tokens_in = sum(
        judge.cache.get(input_hash(c["input"], config.llm_model, config.llm_reasoning))[
            "tokens_in"
        ]
        for c in CASES
    )
    print(
        f"\n{agree}/{len(CASES)} agree; {judge.calls} calls, {judge.cached} cached, "
        f"${judge.spent_usd:.4f} this run, {tokens_in} input tokens in total"
    )
    return 0 if agree >= 0.9 * len(CASES) else 1
