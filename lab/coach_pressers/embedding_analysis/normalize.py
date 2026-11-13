"""Rewrite a presser transcript as a short report of fixed structure.

The model returns a JSON object with one short paragraph per topic
(``FIELDS``); a topic the coach gave no content for is left out. ``render``
turns the object into the text to embed, always in the same order and with
the same headings, so the model decides the content and this code the form.

    .venv/bin/python -m lab.coach_pressers.embedding_analysis.normalize --mode prepare --n 4
    .venv/bin/python -m lab.coach_pressers.embedding_analysis.normalize --mode submit
    .venv/bin/python -m lab.coach_pressers.embedding_analysis.normalize --mode status
    .venv/bin/python -m lab.coach_pressers.embedding_analysis.normalize --mode collect

Writes ``normalized/<model>/<presser_id>.model.json`` (the model's non-null fields),
``.json`` (the same with team names replaced by code) and ``.txt`` (the
rendered report) in the analysis folder and prints each report
with its word count and any limit it broke. Uses OpenAI GPT-6 Luna, low
reasoning and strict Structured Outputs (``[OpenAI] OPENAI_API_KEY``).
The default mode only prepares a local Batch API input; ``submit`` sends it.
``--mode sync`` runs requests directly, with one correction attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.game_texts.team_perspective import replace_teams

from lab.coach_pressers.embedding_analysis.run_embeddings import DEFAULT_OUT
from lab.coach_pressers.teams import TEAMS

DEFAULT_MODEL = "gpt-6-luna"
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 4096  # Includes reasoning tokens as well as the report.
#: field -> (heading in the rendered report, most words allowed)
FIELDS: dict[str, tuple[str, int]] = {
    "overall": ("Overall", 40),
    "offense": ("Offense", 40),
    "defense": ("Defense", 40),
    "effort_and_mindset": ("Effort and mindset", 40),
    "game_flow": ("Game flow", 40),
    "opponent": ("Opponent", 40),
    "players": ("Players", 40),
    "looking_ahead": ("Looking ahead", 40),
    "officiating": ("Officiating", 40),
}
#: What ``render`` writes for a missing field when asked to keep every heading.
NOT_DISCUSSED = "Not discussed."

PROMPT = """Rewrite the transcript of an NBA head coach's postgame press conference as a short structured report of his assessment of the game.

The reports of thousands of press conferences will be turned into text embeddings and compared across games. Transcripts differ in length, in how much the reporters talk and in each coach's way of speaking; the reports must differ only in what the coach thinks and how strongly he feels it. We want what the box score does not show: how the team played, what worked, what failed and why, according to the coach.

The input is an automatically generated transcript and may contain typographical errors, misheard words and mistranscribed names. It has no speaker labels: questions and answers run together, often without punctuation. Correct obvious transcription errors only when the intended meaning is clear from context; do not invent details to resolve ambiguous passages.

Return one JSON object with all the fields below. Each non-null field is one or two sentences, at most 40 words; "officiating" is one sentence of at most 30. When the coach gave nothing real for a field, return null. Do not pad a field, and do not write a placeholder. A short press conference gives fewer non-null fields; a long one does not give longer fields.

- "overall": his verdict on how his team played, apart from the result. A team can play well and lose.
- "offense": how his team attacked and why it worked or failed: shot creation and selection, ball movement, pace, finishing, turnovers.
- "defense": how his team defended and rebounded, and why: coverages, rotations, communication, the glass, fouls.
- "effort_and_mindset": energy, physicality, focus, composure, togetherness, and how they showed in the game.
- "game_flow": how the game changed (stretches, runs, quarters, adjustments by either side) and what caused it.
- "opponent": what the other team did that mattered, and how he rates it.
- "players": individual contributions or problems and why they mattered to the team; not a list of everyone he mentioned.
- "looking_ahead": what he says must change or continue.
- "officiating": only if he talks about the referees, including a pointed refusal to talk about them. Say what he objected to or accepted (specific calls, a foul or free-throw imbalance, the physicality allowed, the late-game whistle), how strongly, and whether he tied it to the result.

Rules:
- Only the coach's views. A reporter's premise that he does not take up is not his view.
- Keep his level of detail and his reasons. Do not add tactics, causes or context he did not give; if he is general, stay general.
- Avoid numerical statistics, scores, exact margins, run totals, counts, minutes, and dates. Preserve standard basketball terminology involving numbers (e.g., three-point shooting, two-big lineups, 2-3 zone defense) and meaningful game phases (e.g., early fourth quarter, final possessions). Replace unnecessary numerical detail with qualitative descriptions.
- Keep what he limits to one part of the game (a quarter, a stretch) limited to it, and keep the direction of his explanations: what led to what.
- Keep the strength of each judgment: harsh stays harsh, mild praise stays mild, a doubt stays a doubt. Do not keep his wording, fillers or repetition; a point he repeats is said once.
- Prefer the specific to the generic: leave out praise or criticism with no content ("we played hard", "they are a good team") unless it is all he says on that topic.
- Put each point in the one field it fits best; do not repeat it in "overall".
- Third person, past tense, plain English. Write "the coach", "his team" and "the opponent", never a team's name or city.
- Keep player names as they appear in the transcript, correcting obvious transcription errors only when the player's identity is clear from context. Do not guess full names or identities when they are ambiguous.
- Leave out injuries and who was available, the schedule, the media and general philosophy about the season, unless he gives one of them as the reason for something that happened in this game.

Output only the JSON object, without code fences."""


def user_message(row, raw: str) -> str:
    """The game's context and the transcript, as the model's input."""
    side = "home" if row.is_home else "visiting"
    return (
        f"Coach's team: {TEAMS[row.team][0]} ({side} team).\n"
        f"Opponent: {TEAMS[row.opponent][0]}.\n\n"
        f"Transcript:\n{raw}"
    )


def parse_report(content: str) -> dict[str, str]:
    """Validate the strict object locally, then omit topics returned as null."""
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("The model did not return a JSON object")
    if set(data) != set(FIELDS):
        raise ValueError("Report fields do not match the JSON schema")
    if any(value is not None and not isinstance(value, str) for value in data.values()):
        raise ValueError("Report fields must be strings or null")
    return {
        field: " ".join(str(data[field]).split())
        for field in FIELDS
        if isinstance(data.get(field), str) and data[field].strip()
    }


OWN, OTHER = "\x00own\x00", "\x00other\x00"
OWN_TEAM, OPPONENT = "his team", "the opponent"


def anonymize_teams(text: str, team: str, opponent: str) -> tuple[str, int]:
    """Replace any mention of the two teams (names, cities, short forms; see
    ``team_perspective``) with "his team" / "the opponent", absorbing a
    preceding "the". ``team`` / ``opponent`` are schedule nicknames. Returns
    the text and the number of replacements."""
    rewrite = replace_teams(
        text, team, opponent, target_label=OWN, opponent_label=OTHER
    )
    text = rewrite.text
    for token, label in ((OWN, OWN_TEAM), (OTHER, OPPONENT)):
        text = re.sub(rf"\b[Tt]he {token}", label, text).replace(token, label)
    # A sentence that now starts with a label starts in lower case.
    text = re.sub(
        r"(^|[.!?]\s+)(his team|the opponent)",
        lambda m: m.group(1) + m.group(2)[0].upper() + m.group(2)[1:],
        text,
    )
    return text, rewrite.count("target") + rewrite.count("opponent")


def normalize_report(
    report: dict[str, str], team: str, opponent: str
) -> tuple[dict[str, str], int]:
    """``anonymize_teams`` on every field; the total replacements."""
    out, total = {}, 0
    for field, text in report.items():
        out[field], count = anonymize_teams(text, team, opponent)
        total += count
    return out, total


#: Digits or number words from two up (``"one"`` is too common to flag).
NUMBER = re.compile(
    r"\d|\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|hundred)\b(?![- ]pointers?\b|-point\b)",
    re.IGNORECASE,
)


def problems(report: dict[str, str]) -> list[str]:
    """Rules the code can check: fields over their word limit
    (``"offense: 47 words, at most 40"``) and fields with a number."""
    found = [
        f"{field}: {len(text.split())} words, at most {FIELDS[field][1]}"
        for field, text in report.items()
        if len(text.split()) > FIELDS[field][1]
    ]
    found += [
        f"{field}: contains a number ({NUMBER.search(text).group(0)!r})"
        for field, text in report.items()
        if NUMBER.search(text)
    ]
    return found


def render(report: dict[str, str], *, keep_empty: bool = False) -> str:
    """The text to embed: one "Heading: text" line per field, in ``FIELDS``
    order. A missing field is left out, or with ``keep_empty`` written as
    ``NOT_DISCUSSED`` so that every report has every heading."""
    lines = []
    for field, (heading, _) in FIELDS.items():
        text = report.get(field)
        if text or keep_empty:
            lines.append(f"{heading}: {text or NOT_DISCUSSED}")
    return "\n".join(lines)


def default_client():
    from nba_ou.config.settings import SETTINGS
    from openai import OpenAI

    return OpenAI(api_key=SETTINGS.openai_api_key, timeout=300)


def request_body(model: str, messages: list[dict]) -> dict:
    """Identical Responses API parameters for synchronous and batch calls."""
    return {
        "model": model,
        "input": messages,
        "reasoning": {"effort": REASONING_EFFORT},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "coach_presser_report",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        field: {
                            "type": ["string", "null"],
                            "description": f"{heading}, at most {limit} words; null if not discussed.",
                        }
                        for field, (heading, limit) in FIELDS.items()
                    },
                    "required": list(FIELDS),
                    "additionalProperties": False,
                },
            }
        },
    }


def messages_for(message: str) -> list[dict]:
    return [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": message},
    ]


def request_hash(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def response_report(body: dict) -> dict[str, str]:
    """Reject refusals, API errors and truncated output before parsing JSON."""
    if body.get("status") != "completed" or body.get("error"):
        raise ValueError(
            f"Response not completed: {body.get('status')}; "
            f"{body.get('error') or body.get('incomplete_details')}"
        )
    texts = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "refusal":
                raise ValueError(f"Model refused: {part.get('refusal')}")
            if part.get("type") == "output_text":
                texts.append(part["text"])
    if not texts:
        raise ValueError("Response contains no report text")
    return parse_report("".join(texts))


def save_report(folder: Path, presser_id: str, report: dict, metadata: dict) -> None:
    """Shared output format for synchronous calls and collected batch results."""
    normalized, replaced = normalize_report(report, metadata["team"], metadata["opponent"])
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        ".model.json": report,
        ".json": normalized,
        ".meta.json": {
            **metadata,
            "team_names_replaced": replaced,
            "problems": problems(normalized),
        },
    }
    for suffix, value in data.items():
        (folder / f"{presser_id}{suffix}").write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    text = render(normalized)
    (folder / f"{presser_id}.txt").write_text(text + "\n", encoding="utf-8")
    print(f"{presser_id}: {len(text.split())} words; problems: {problems(normalized) or 'none'}")


def rewrite(
    client, model: str, message: str, *, retries: int = 1
) -> tuple[dict[str, str], dict, list[str]]:
    """The model's report, the token usage summed over the calls, and the
    rules it still breaks. A report that breaks a rule ``problems`` can check
    is sent back once with the list of problems to fix."""
    messages = messages_for(message)
    usage = {"input_tokens": 0, "output_tokens": 0}
    for attempt in range(retries + 1):
        body = client.responses.create(**request_body(model, messages)).model_dump()
        for key in usage:
            usage[key] += (body.get("usage") or {}).get(key, 0)
        report = response_report(body)
        found = problems(report)
        if not found or attempt == retries:
            usage["calls"] = attempt + 1
            return report, usage, found
        messages += [
            {"role": "assistant", "content": json.dumps({field: report.get(field) for field in FIELDS})},
            {
                "role": "user",
                "content": "Your report breaks these rules:\n- "
                + "\n- ".join(found)
                + "\nReturn the corrected JSON object. Change only what is "
                "needed: write sizes in words instead of numbers and shorten "
                "fields that are too long.",
            },
        ]
    raise AssertionError("unreachable")


def pick(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """A spread of pressers: by result and by length, across seasons."""
    frame = frame.assign(
        cell=(frame["target_margin"] > 0).astype(int) * 2
        + (frame["n_words"] > frame["n_words"].median()).astype(int)
    )
    per_cell = -(-n // 4)
    parts = [
        group.sample(min(per_cell, len(group)), random_state=seed)
        for _, group in frame.groupby("cell")
    ]
    return pd.concat(parts).head(n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--mode", choices=("prepare", "submit", "status", "collect", "watch", "sync"),
        default="prepare",
    )
    parser.add_argument("--all", action="store_true", help="all sampled pressers instead of --n")
    parser.add_argument("--batch-dir", type=Path, help="batch files; defaults to normalized/<model>/batch")
    args = parser.parse_args()
    folder = args.out / "normalized" / args.model.split("/")[-1]
    batch_dir = args.batch_dir or folder / "batch"
    from lab.coach_pressers.embedding_analysis import normalize_batch as batch

    if args.mode in ("submit", "status", "collect", "watch"):
        getattr(batch, args.mode)(batch_dir)
        return
    if args.n < 1:
        parser.error("--n must be positive")
    from lab.coach_pressers.embedding_analysis import analysis

    run = analysis.load_run(args.out)
    chosen = run.frame if args.all else pick(run.frame, args.n, args.seed)
    jobs = []
    for index, row in chosen.iterrows():
        raw = run.texts["raw"].iloc[index]
        body = request_body(args.model, messages_for(user_message(row, raw)))
        jobs.append({
            "presser_id": row.presser_id,
            "team": TEAMS[row.team][1], "opponent": TEAMS[row.opponent][1],
            "body": body, "request_hash": request_hash(body),
        })
    if args.mode == "prepare":
        batch.prepare(batch_dir, folder, jobs)
        return
    client = default_client()
    for job in jobs:
        report, usage, _ = rewrite(client, args.model, job["body"]["input"][1]["content"])
        save_report(folder, job["presser_id"], report, {
            "team": job["team"], "opponent": job["opponent"],
            "model": args.model, "reasoning_effort": REASONING_EFFORT,
            "request_hash": job["request_hash"], "usage": usage, "mode": "sync",
        })


if __name__ == "__main__":
    main()
