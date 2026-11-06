"""Step 7 of the coach presser archive: the OpenAI judge, offline."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lab.coach_pressers import llm_judge
from lab.coach_pressers.llm_cases import CASES
from tests.support.coach_pressers import config_2021


class FakeResponses:
    def __init__(self, answer_for):
        self.answer_for, self.calls = answer_for, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        payload = json.loads(kwargs["input"])
        answer = self.answer_for(payload)
        if isinstance(answer, SimpleNamespace):
            return answer
        return SimpleNamespace(
            output_text=json.dumps(answer),
            status="completed",
            output=[],
            usage=SimpleNamespace(input_tokens=1500, output_tokens=100),
        )


class FakeClient:
    def __init__(self, answer_for):
        self.responses = FakeResponses(answer_for)


def labelled(payload):
    case = next(c for c in CASES if c["input"] == payload)
    expected = case["expected"]
    chapter = "Darko Rajakovic" if case["name"] == "combined_with_chapter" else None
    return {
        "video_id": expected["video_id"],
        "kind": expected["kind"],
        "coach_name": None if expected["kind"] == "none" else "Coach",
        "coach_chapter": chapter,
        "confidence": 0.9,
        "reason": "Label.",
    }


def judge(tmp_path, answer_for, **config):
    cfg = config_2021(tmp_path, **config)
    return llm_judge.Judge(cfg, client=FakeClient(answer_for))


def test_twenty_labelled_cases_validate(tmp_path):
    assert len(CASES) == 20
    j = judge(tmp_path, labelled)
    for case in CASES:
        record = j.ask(case["input"])
        assert record["error"] is None, case["name"]
        assert record["answer"]["video_id"] == case["expected"]["video_id"]
    assert j.calls == 20
    assert j.spent_usd == pytest.approx(20 * (1500 * 0.10 + 100 * 0.50) / 1e6)


def test_request_uses_a_strict_schema_with_the_candidates(tmp_path):
    j = judge(tmp_path, labelled)
    case = next(c for c in CASES if c["name"] == "combined_with_chapter")
    j.ask(case["input"])
    request = j.client.responses.calls[0]
    fmt = request["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    properties = fmt["schema"]["properties"]
    assert properties["video_id"]["anyOf"][0]["enum"] == ["m2"]
    assert properties["coach_chapter"]["anyOf"][0]["enum"] == sorted(
        ["Darko Rajakovic", "Scottie Barnes", "RJ Barrett"]
    )
    assert fmt["schema"]["additionalProperties"] is False
    assert set(fmt["schema"]["required"]) == set(properties)
    assert request["store"] is False and request["model"] == "gpt-6-luna"


def test_cache_hit_costs_nothing(tmp_path):
    j = judge(tmp_path, labelled)
    case = CASES[0]
    j.ask(case["input"])
    again = llm_judge.Judge(j.config, client=FakeClient(labelled))
    again.ask(case["input"])
    assert again.calls == 0 and again.cached == 1 and again.spent_usd == 0


def test_budget_stop(tmp_path):
    j = judge(tmp_path, labelled, llm_budget_usd=0.0)
    with pytest.raises(llm_judge.BudgetExceeded):
        j.ask(CASES[0]["input"])


def test_refusal_and_bad_answers_are_errors(tmp_path):
    refusal = SimpleNamespace(
        output_text="",
        status="completed",
        output=[
            SimpleNamespace(content=[SimpleNamespace(type="refusal", refusal="no")])
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=1),
    )
    j = judge(tmp_path, lambda payload: refusal)
    assert j.ask(CASES[0]["input"])["error"] == "refusal: no"

    def wrong(payload):
        return {**labelled(payload), "video_id": "not-a-candidate"}

    j = judge(tmp_path / "b", wrong)
    assert "not a candidate" in j.ask(CASES[0]["input"])["error"]


def found_for(case):
    return [
        {
            "video_id": c["video_id"],
            "channel_id": "UC",
            "channel_priority": 1,
            "title": c["title"],
            "published_at_utc": c["published_at_utc"],
            "published_date": c["published_date"],
            "duration_s": c["duration_s"],
            "description": c["description"],
            "chapters": [
                {
                    "start_time": ch["start_s"],
                    "end_time": ch["start_s"] + 100,
                    "title": ch["title"],
                }
                for ch in c["chapters"]
            ],
            "class": "combined",
            "coach_chapter": None,
        }
        for c in case["input"]["candidates"]
    ]


def record(answer, error=None):
    return {"model": "m", "input_hash": "h", "answer": answer, "error": error}


def test_apply_decision():
    case = next(c for c in CASES if c["name"] == "combined_with_chapter")
    entry = {"coach_expected": "Darko Rajaković"}
    answer = {
        "video_id": "m2",
        "kind": "combined",
        "coach_name": "Darko Rajaković",
        "coach_chapter": "Darko Rajakovic",
        "confidence": 0.92,
        "reason": "Chapter one is the coach.",
    }
    llm_judge.apply_decision(entry, found_for(case), record(answer))
    assert entry["status"] == "selected"
    selection = entry["selection"]
    assert selection["decided_by"] == "llm" and selection["trim_start_s"] == 0
    assert selection["trim_end_s"] == 100 and selection["confidence"] == 0.92

    low = {**answer, "confidence": 0.5}
    entry = {"coach_expected": "x"}
    llm_judge.apply_decision(entry, found_for(case), record(low))
    assert entry["status"] == "review" and entry["selection"] is None

    none = {**answer, "video_id": None, "kind": "none", "coach_chapter": None}
    entry = {"coach_expected": "x"}
    llm_judge.apply_decision(entry, found_for(case), record(none))
    assert entry["status"] == "review"


def test_without_llm_everything_goes_to_review(tmp_path):
    config = config_2021(tmp_path, use_llm=False)
    manifest = {"entries": {"g:SAC": {"status": "pending"}}}
    stats = llm_judge.judge_season(config, manifest, {}, {"g:SAC": [{}]})
    assert stats["review"] == 1 and manifest["entries"]["g:SAC"]["status"] == "review"
