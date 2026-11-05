import numpy as np
import pytest
from nba_ou.data_processing.game_texts import nebius
from nba_ou.data_processing.game_texts.nebius import NebiusEmbedder, batches


class FakeTokenizer:
    def __call__(self, texts):
        return {"input_ids": [text.split() for text in texts]}


class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code, self.ok, self._body = status, status == 200, body
        self.text = "" if body else "busy"

    def json(self):
        return self._body


class FakeSession:
    """Embeds each text as [word count, 1], rows returned in reverse order."""

    def __init__(self, failures=0):
        self.failures, self.inputs = failures, []

    def post(self, url, json, timeout):
        if self.failures:
            self.failures -= 1
            return FakeResponse(429)
        self.inputs.append(json["input"])
        data = [
            {"index": i, "embedding": [len(text.split()), 1.0]}
            for i, text in enumerate(json["input"])
        ]
        tokens = sum(len(text.split()) for text in json["input"])
        return FakeResponse(200, {"data": data[::-1], "usage": {"prompt_tokens": tokens}})


def embedder(session, **kwargs):
    return NebiusEmbedder(
        cache_dir=".", session=session, tokenizer=FakeTokenizer(), **kwargs
    )


def test_batches_respect_text_and_token_limits():
    assert batches([5, 5, 5, 5, 5], max_texts=2, max_tokens=100) == [[0, 1], [2, 3], [4]]
    assert batches([6, 6, 20, 1], max_texts=10, max_tokens=12) == [[0, 1], [2], [3]]


def test_encode_keeps_order_formats_instruction_and_normalizes():
    session = FakeSession()
    texts = ["a", "a b", "a b c", "a b c d", "a b c d e"]
    vectors, lengths = embedder(session, max_texts=2, workers=3).encode(
        texts, instruction="Do it", progress_every=0
    )
    assert len(session.inputs) == 3
    assert session.inputs[0][0] == "Instruct: Do it\nQuery:a"
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert lengths == [4, 5, 6, 7, 8]  # "Instruct:", "Do", "it", "Query:a", ...
    assert np.allclose(vectors[:, 0] / vectors[:, 1], lengths)  # rows in order


def test_encode_retries_rate_limits(monkeypatch):
    monkeypatch.setattr(nebius.time, "sleep", lambda seconds: None)
    session = FakeSession(failures=2)
    vectors, _ = embedder(session).encode(["a b"], progress_every=0)
    assert vectors.shape == (1, 2)


def test_encode_gives_up_after_the_retries(monkeypatch):
    monkeypatch.setattr(nebius.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="Gave up"):
        embedder(FakeSession(failures=5), retries=2).encode(["a"], progress_every=0)
