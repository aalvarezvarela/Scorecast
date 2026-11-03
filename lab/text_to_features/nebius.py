"""Qwen3-Embedding-8B through the Nebius Token Factory API.

Same interface as ``embeddings.QwenEmbedder`` (``encode``, ``metadata``), so
``run_role_probes`` can use either. The API is OpenAI-compatible
(``POST /v1/embeddings``) and returns L2-normalized last-token vectors; the
instruction is applied here, in Qwen's ``Instruct: ...\\nQuery:...`` format,
exactly as in the local runs. Token counts come from the local Qwen3 tokenizer
(the 0.6B, 4B and 8B models share it) and are checked against what the API
bills. See ``nebius_embedding_api_test.ipynb`` for the smoke test.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from lab.text_to_features.embeddings import MODELS, embedding_input

BASE_URL = "https://api.tokenfactory.nebius.com/v1"
MODEL_ID = "Qwen/Qwen3-Embedding-8B"
TOKENIZER_ID = "Qwen/Qwen3-Embedding-0.6B"
MAX_TOKENS = 32768
RETRY_STATUS = (408, 409, 429, 500, 502, 503, 504)


def batches(lengths: list[int], max_texts: int, max_tokens: int) -> list[list[int]]:
    """Consecutive text indices, at most ``max_texts`` texts and ``max_tokens``
    tokens per request (a longer text goes alone)."""
    out, current, tokens = [], [], 0
    for index, length in enumerate(lengths):
        if current and (len(current) == max_texts or tokens + length > max_tokens):
            out.append(current)
            current, tokens = [], 0
        current.append(index)
        tokens += length
    if current:
        out.append(current)
    return out


class NebiusEmbedder:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        cache_dir: str | Path,
        model_id: str = MODEL_ID,
        max_texts: int = 16,
        max_batch_tokens: int = 16_000,
        workers: int = 4,
        retries: int = 6,
        session: requests.Session | None = None,
        tokenizer=None,
    ):
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_ID,
                revision=MODELS[TOKENIZER_ID][0],
                cache_dir=str(cache_dir),
            )
        if session is None:
            if api_key is None:
                from nba_ou.config.settings import SETTINGS

                api_key = SETTINGS.nebius_api_key
            if not api_key:
                raise RuntimeError("Set [Nebius] NEBIUS_API_KEY in config.secrets.ini")
            session = requests.Session()
            session.headers["Authorization"] = f"Bearer {api_key}"
        self.session = session
        self.tokenizer = tokenizer
        self.max_texts = max_texts
        self.max_batch_tokens = max_batch_tokens
        self.workers = workers
        self.retries = retries
        self.last_call: dict = {}
        self.metadata = {
            "model_id": model_id,
            "provider": "Nebius Token Factory",
            "endpoint": f"{BASE_URL}/embeddings",
            "pooling": "server side (last token, L2 normalized); renormalized in FP32",
            "tokenizer": f"{TOKENIZER_ID}@{MODELS[TOKENIZER_ID][0]}",
            "instruction": None,
            "max_texts_per_request": max_texts,
            "max_tokens_per_request": max_batch_tokens,
            "max_tokens": MAX_TOKENS,
            "truncation": False,
        }

    def _post(self, inputs: list[str]) -> tuple[np.ndarray, int | None]:
        for attempt in range(self.retries + 1):
            try:
                response = self.session.post(
                    f"{BASE_URL}/embeddings",
                    json={
                        "model": self.metadata["model_id"],
                        "input": inputs,
                        "encoding_format": "float",
                    },
                    timeout=180,
                )
            except (requests.ConnectionError, requests.Timeout) as error:
                failure = repr(error)
            else:
                if response.ok:
                    body = response.json()
                    rows = sorted(body["data"], key=lambda row: row["index"])
                    if len(rows) != len(inputs):
                        raise RuntimeError(
                            f"Asked for {len(inputs)} embeddings, got {len(rows)}"
                        )
                    vectors = np.array([row["embedding"] for row in rows], np.float32)
                    return vectors, body.get("usage", {}).get("prompt_tokens")
                if response.status_code not in RETRY_STATUS:
                    raise RuntimeError(
                        f"{response.status_code}: {response.text[:500]}"
                    )
                failure = f"{response.status_code}: {response.text[:200]}"
            if attempt == self.retries:
                raise RuntimeError(f"Gave up after {attempt + 1} tries: {failure}")
            wait = min(60.0, 2.0**attempt)
            print(f"retry in {wait:.0f} s ({failure})", flush=True)
            time.sleep(wait)
        raise AssertionError("unreachable")

    def encode(
        self,
        texts: list[str],
        *,
        instruction: str | None = None,
        progress_every: int = 1,
    ) -> tuple[np.ndarray, list[int]]:
        """``progress_every`` prints when a multiple of n texts is done, and at
        the end; 0 is silent."""
        self.metadata.update(
            {
                "instruction": instruction,
                "instruction_format": "Instruct: {instruction}\\nQuery:{text}"
                if instruction
                else None,
            }
        )
        inputs = [embedding_input(text, instruction) for text in texts]
        lengths = [len(ids) for ids in self.tokenizer(inputs)["input_ids"]]
        for index, length in enumerate(lengths):
            if length > MAX_TOKENS:
                raise ValueError(f"Text {index} has {length} tokens; refusing to truncate.")
        plan = batches(lengths, self.max_texts, self.max_batch_tokens)
        started = time.perf_counter()
        vectors: list[np.ndarray | None] = [None] * len(plan)
        billed: list[int | None] = [None] * len(plan)
        done = 0
        with ThreadPoolExecutor(self.workers) as pool:
            futures = {
                pool.submit(self._post, [inputs[i] for i in batch]): number
                for number, batch in enumerate(plan)
            }
            for future in futures:  # in order; completion order does not matter
                number = futures[future]
                vectors[number], billed[number] = future.result()
                before, done = done, done + len(plan[number])
                if progress_every and (
                    done // progress_every > before // progress_every
                    or done == len(texts)
                ):
                    print(f"Embedded {done}/{len(texts)}", flush=True)
        stacked = np.concatenate(vectors).astype(np.float32)
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        if not np.isfinite(stacked).all() or (norms == 0).any():
            raise RuntimeError("Invalid embedding returned by the API")
        billed_tokens = None if None in billed else int(sum(billed))
        self.last_call = {
            "requests": len(plan),
            "billed_tokens": billed_tokens,
            "local_tokens": int(sum(lengths)),
            "api_norm_min": float(norms.min()),
            "api_norm_max": float(norms.max()),
            "api_seconds": time.perf_counter() - started,
        }
        if billed_tokens is not None and billed_tokens != sum(lengths):
            print(
                f"warning: API billed {billed_tokens} tokens, local count "
                f"{sum(lengths)}",
                flush=True,
            )
        return stacked / norms, lengths
