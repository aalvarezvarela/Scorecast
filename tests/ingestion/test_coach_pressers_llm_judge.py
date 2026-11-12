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
    original = {
        **payload,
        "candidates": [
            {
                **candidate,
                "chapters": [
                    {k: v for k, v in ch.items() if k != "chapter_id"}
                    for ch in candidate["chapters"]
                ],
            }
            for candidate in payload["candidates"]
        ],
    }
    case = next(c for c in CASES if c["input"] == original)
    expected = case["expected"]
    chapter = "v0_ch0" if case["name"] == "combined_with_chapter" else None
    return {
        "video_id": expected["video_id"],
        "kind": expected["kind"],
        "coach_name": None if expected["kind"] == "none" else "Coach",
        "coach_chapter": chapter,
        "confidence": 0.9,
    }


def judge(tmp_path, answer_for, **config):
    cfg = config_2021(tmp_path, **config)
    return llm_judge.Judge(cfg, client=FakeClient(answer_for))


def test_labelled_cases_validate(tmp_path):
    assert len(CASES) == 23
    j = judge(tmp_path, labelled)
    for case in CASES:
        record = j.ask(case["input"])
        assert record["error"] is None, case["name"]
        assert record["answer"]["video_id"] == case["expected"]["video_id"]
    assert j.calls == 23
    assert j.spent_usd == pytest.approx(23 * (1500 * 0.10 + 100 * 0.50) / 1e6)


def test_request_uses_a_strict_schema_with_the_candidates(tmp_path):
    j = judge(tmp_path, labelled)
    case = next(c for c in CASES if c["name"] == "combined_with_chapter")
    j.ask(case["input"])
    request = j.client.responses.calls[0]
    fmt = request["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    properties = fmt["schema"]["properties"]
    assert properties["video_id"]["anyOf"][0]["enum"] == ["m2"]
    assert properties["coach_chapter"]["anyOf"][0]["enum"] == [
        "v0_ch0",
        "v0_ch1",
        "v0_ch2",
    ]
    sent = json.loads(request["input"])
    assert sent["candidates"][0]["chapters"][0] == {
        "start_s": 0,
        "title": "Darko Rajakovic",
        "chapter_id": "v0_ch0",
    }
    assert fmt["schema"]["additionalProperties"] is False
    assert set(fmt["schema"]["required"]) == set(properties)
    assert "reason" not in properties
    assert request["input"] == json.dumps(
        sent, ensure_ascii=False, separators=(",", ":")
    )
    assert request["store"] is False and request["model"] == "gpt-6-luna"
    assert request["reasoning"] == {"effort": "none"}


def test_cache_hit_costs_nothing(tmp_path):
    j = judge(tmp_path, labelled)
    case = CASES[0]
    j.ask(case["input"])
    again = llm_judge.Judge(j.config, client=FakeClient(labelled))
    again.ask(case["input"])
    assert again.calls == 0 and again.cached == 1 and again.spent_usd == 0


@pytest.mark.parametrize("title", ['Coach "Kenny"', "Coach \\ Name\n🎤"])
def test_free_form_chapter_titles_are_decoded_cached_and_trimmed(tmp_path, title):
    case = next(c for c in CASES if c["name"] == "combined_with_chapter")
    payload = json.loads(json.dumps(case["input"]))
    payload["candidates"][0]["chapters"][0]["title"] = title
    original = json.dumps(payload, sort_keys=True)

    def choose_coach(sent):
        assert sent["candidates"][0]["chapters"][0]["title"] == title
        return {
            "video_id": "m2",
            "kind": "combined",
            "coach_name": "Darko Rajaković",
            "coach_chapter": "v0_ch0",
            "confidence": 0.9,
        }

    j = judge(tmp_path, choose_coach)
    decision = j.ask(payload)
    assert decision["error"] is None
    assert decision["answer"]["coach_chapter"] == title
    assert json.dumps(payload, sort_keys=True) == original
    fmt = j.client.responses.calls[0]["text"]["format"]
    assert fmt["schema"]["properties"]["coach_chapter"]["anyOf"][0]["enum"] == [
        "v0_ch0",
        "v0_ch1",
        "v0_ch2",
    ]

    entry = {"coach_expected": "Darko Rajaković"}
    llm_judge.apply_decision(entry, found_for({"input": payload}), decision)
    assert entry["selection"]["coach_chapter"] == title
    assert entry["selection"]["trim_start_s"] == 0
    assert entry["selection"]["trim_end_s"] == 100

    cached = llm_judge.Judge(j.config, client=FakeClient(choose_coach))
    assert cached.ask(payload) == decision
    assert cached.cached == 1 and cached.calls == 0


def test_chapter_reference_cannot_belong_to_another_video(tmp_path):
    payload = json.loads(json.dumps(CASES[0]["input"]))
    for candidate in payload["candidates"]:
        candidate["chapters"] = [{"title": 'Coach "Name"', "start_s": 0}]

    def wrong_video(sent):
        return {
            "video_id": sent["candidates"][1]["video_id"],
            "kind": "combined",
            "coach_name": "Jordan Ott",
            "coach_chapter": "v0_ch0",
            "confidence": 0.9,
        }

    j = judge(tmp_path, wrong_video)
    assert "not a chapter of the chosen video" in j.ask(payload)["error"]


def test_pre_fix_cached_chapter_title_is_reused(tmp_path):
    case = next(c for c in CASES if c["name"] == "combined_with_chapter")
    j = judge(tmp_path, labelled)
    key = llm_judge.input_hash(case["input"], j.config.llm_model, "none")
    cached = {
        "input_hash": key,
        "answer": {
            "video_id": "m2",
            "kind": "combined",
            "coach_name": "Darko Rajaković",
            "coach_chapter": "Darko Rajakovic",
            "confidence": 0.9,
        },
        "error": None,
    }
    j.cache.add(cached)
    assert j.ask(case["input"]) == cached
    assert j.calls == 0 and j.cached == 1


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


def test_compact_input_keeps_game_identity_and_coach_chapter():
    case = next(c for c in CASES if c["name"] == "combined_with_chapter")
    found = found_for(case)
    found[0]["description"] = (
        "Subscribe: https://example.com/subscribe\n"
        "Follow us on Instagram https://example.com/social\n"
        "Darko Rajakovic after Raptors at Nuggets, March 11, 2024.\n"
        "0:00 Darko Rajakovic https://example.com/watch\n"
    )
    entry = {
        **case["input"]["game"],
        "game_id": "g1",
        "team": "TOR",
        "opponent": "DEN",
        "is_home": False,
        "team_score": 110,
        "opponent_score": 104,
        "status": "pending",
        "coach_expected": "Darko Rajaković",
    }
    following = {
        **entry,
        "game_id": "g2",
        "date": "2024-03-12",
        "tip_utc": "2024-03-13T02:00:00Z",
        "opponent": "PHX",
    }
    payload = llm_judge.build_input(
        entry,
        found,
        {"entries": {"g1:TOR": entry, "g2:TOR": following}},
        {"staff": case["input"]["staff"]},
    )
    candidate = payload["candidates"][0]
    assert candidate["description"] == (
        "Darko Rajakovic after Raptors at Nuggets, March 11, 2024. "
        "0:00 Darko Rajakovic"
    )
    assert candidate["title"] == found[0]["title"]
    assert candidate["chapters"] == case["input"]["candidates"][0]["chapters"]
    assert payload["staff"] == case["input"]["staff"]
    assert payload["other_games_within_3_days"][0]["date"] == "2024-03-12"
    assert payload["game"]["date"] == "2024-03-11"


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
    }
    llm_judge.apply_decision(entry, found_for(case), record(answer))
    assert entry["status"] == "selected"
    selection = entry["selection"]
    assert selection["decided_by"] == "llm" and selection["trim_start_s"] == 0
    assert selection["trim_end_s"] == 100 and selection["confidence"] == 0.92
    assert selection["reason"] == ""

    legacy = {**answer, "reason": "Previously cached explanation."}
    llm_judge.apply_decision(entry, found_for(case), record(legacy))
    assert entry["selection"]["reason"] == legacy["reason"]

    low = {**answer, "confidence": 0.5}
    entry = {"coach_expected": "x"}
    llm_judge.apply_decision(entry, found_for(case), record(low))
    assert entry["status"] == "review" and entry["selection"] is None

    none = {**answer, "video_id": None, "kind": "none", "coach_chapter": None}
    entry = {"coach_expected": "x"}
    llm_judge.apply_decision(entry, found_for(case), record(none))
    assert entry["status"] == "review"

    entry = {"coach_expected": "Darko Rajaković"}
    llm_judge.apply_decision(
        entry, found_for(case), record(answer), rivals={"darko rajakovic"}
    )
    assert entry["status"] == "review" and "another team" in entry["error"]


def test_without_llm_everything_goes_to_review(tmp_path):
    config = config_2021(tmp_path, use_llm=False)
    manifest = {"entries": {"g:SAC": {"status": "pending"}}}
    stats = llm_judge.judge_season(config, manifest, {}, {"g:SAC": [{}]})
    assert stats["review"] == 1 and manifest["entries"]["g:SAC"]["status"] == "review"
