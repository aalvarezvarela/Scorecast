"""Offline checks for strict responses and Batch collection; no live API."""

import json
from types import SimpleNamespace

import pytest

from lab.coach_pressers.embedding_analysis import normalize as N
from lab.coach_pressers.embedding_analysis import normalize_batch as B


def response(overall="The coach praised his team's ball movement."):
    report = dict.fromkeys(N.FIELDS)
    report["overall"] = overall
    return {
        "id": "resp_test", "status": "completed",
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(report)},
        ]}],
        "usage": {"input_tokens": 100, "output_tokens": 80},
    }


def job(presser_id):
    body = N.request_body(N.DEFAULT_MODEL, N.messages_for("Automatic transcript"))
    return {
        "presser_id": presser_id, "team": "Knicks", "opponent": "Bulls",
        "body": body, "request_hash": N.request_hash(body),
    }


def test_strict_nullable_schema_and_no_player_list():
    row = SimpleNamespace(team="NYK", opponent="CHI", is_home=True)
    message = N.user_message(row, "Brunson moved the ball.")
    assert "Players:" not in message
    assert message.endswith("Transcript:\nBrunson moved the ball.")
    body = N.request_body(N.DEFAULT_MODEL, N.messages_for(message))
    assert body["model"] == "gpt-6-luna"
    assert body["reasoning"] == {"effort": "low"}
    fmt = body["text"]["format"]
    assert fmt["strict"] is True
    assert fmt["schema"]["additionalProperties"] is False
    assert fmt["schema"]["required"] == list(N.FIELDS)
    assert all(p["type"] == ["string", "null"] for p in fmt["schema"]["properties"].values())
    assert list(N.response_report(response())) == ["overall"]


@pytest.mark.parametrize("status", ["incomplete", "failed", "in_progress"])
def test_non_completed_response_is_not_a_report(status):
    body = response()
    body["status"] = status
    with pytest.raises(ValueError, match="not completed"):
        N.response_report(body)


def test_refusal_and_schema_mismatch_are_rejected():
    body = response()
    body["output"][0]["content"] = [{"type": "refusal", "refusal": "Refused"}]
    with pytest.raises(ValueError, match="refused"):
        N.response_report(body)
    with pytest.raises(ValueError, match="schema"):
        N.parse_report('{"overall": "Incomplete set of fields"}')
    invalid = dict.fromkeys(N.FIELDS)
    invalid["offense"] = ["not a paragraph"]
    with pytest.raises(ValueError, match="strings or null"):
        N.parse_report(json.dumps(invalid))


def test_prepare_is_local_and_detects_modified_input(tmp_path, monkeypatch):
    def forbidden():
        pytest.fail("Preparation must not construct an API client")

    monkeypatch.setattr(B, "default_client", forbidden)
    batch_dir = tmp_path / "batch"
    B.prepare(batch_dir, tmp_path / "reports", [job("a")])
    manifest = B.read_json(batch_dir / "manifest.json")
    B.validate_input(batch_dir, manifest)
    request_file = batch_dir / "requests.jsonl"
    record = json.loads(request_file.read_text())
    record["body"]["reasoning"]["effort"] = "high"
    request_file.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="no longer matches"):
        B.validate_input(batch_dir, manifest)
    with pytest.raises(FileExistsError):
        B.prepare(batch_dir, tmp_path / "reports", [job("a")])


def test_collect_matches_ids_and_records_partial_failures(tmp_path, monkeypatch):
    batch_dir, reports = tmp_path / "batch", tmp_path / "reports"
    B.prepare(batch_dir, reports, [job(name) for name in ("a", "b", "c", "d")])
    B.write_json(batch_dir / "batch.json", {"id": "batch_test"})
    incomplete = response()
    incomplete["status"] = "incomplete"
    records = [
        {"custom_id": "c", "response": {"status_code": 200, "body": incomplete}},
        {"custom_id": "b", "response": {"status_code": 200, "body": response("Strong defense.")}},
        {"custom_id": "a", "response": {"status_code": 200, "body": response("Good offense.")}},
    ]
    content = ("\n".join(json.dumps(r) for r in records) + "\n").encode()
    batch = {
        "id": "batch_test", "status": "expired", "output_file_id": "file_test",
        "error_file_id": None,
    }
    client = SimpleNamespace(
        batches=SimpleNamespace(retrieve=lambda _: SimpleNamespace(model_dump=lambda: batch)),
        files=SimpleNamespace(content=lambda _: SimpleNamespace(content=content)),
    )
    monkeypatch.setattr(B, "default_client", lambda: client)
    B.collect(batch_dir)
    assert B.read_json(reports / "a.json")["overall"] == "Good offense."
    assert B.read_json(reports / "b.json")["overall"] == "Strong defense."
    assert not (reports / "c.txt").exists()
    summary = B.read_json(batch_dir / "collection.json")
    assert set(summary["saved"]) == {"a", "b"}
    assert set(summary["errors"]) == {"c", "d"}


def test_submit_with_saved_id_does_not_send_again(tmp_path, monkeypatch):
    batch_dir = tmp_path / "batch"
    B.prepare(batch_dir, tmp_path / "reports", [job("a")])
    B.write_json(batch_dir / "batch.json", {"id": "batch_existing"})

    def forbidden():
        pytest.fail("A saved batch must not be submitted again")

    monkeypatch.setattr(B, "default_client", forbidden)
    B.submit(batch_dir)
