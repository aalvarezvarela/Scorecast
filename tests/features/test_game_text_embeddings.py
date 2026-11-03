import gzip
import json

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.game_texts import documents, vector_codec
from nba_ou.data_processing.game_texts.embed_reports import (
    DIMENSION,
    store_settings,
    update_season,
)
from nba_ou.data_processing.game_texts.embedding_inputs import (
    TARGET_VIEW_INSTRUCTION,
    build_inputs,
    team_ids,
)
from nba_ou.data_processing.game_texts.embedding_store import (
    ROW_COLUMNS,
    EmbeddingStore,
    EmbeddingTable,
)
from nba_ou.data_processing.game_texts.nebius import exact_float32
from nba_ou.data_processing.game_texts.players import Player

from tests.support.fake_s3 import FakeS3Client

REPORT = {
    "game_id": "0022500001",
    "season": "2025-26",
    "date": "2025-10-21",
    "home": "Thunder",
    "away": "Rockets",
    "title": "Thunder edge Rockets in double overtime",
    "text": "OKLAHOMA CITY (AP) — Shai Gilgeous-Alexander scored 35 as the "
    "Thunder beat Houston 125-124. Kevin Durant had 23 in his debut.",
    "published_at": "2025-10-22T00:07:19",
}
PLAYERS = [
    Player("Shai", "Gilgeous-Alexander", "Thunder"),
    Player("Kevin", "Durant", "Rockets"),
]


def bf16_vector(rng, dimension=DIMENSION) -> np.ndarray:
    """A vector built like the API's: bfloat16 values over a float32 norm."""
    codes = vector_codec.bf16_codes(rng.normal(size=dimension).astype(np.float32))
    values = vector_codec.bf16_values(codes)
    return values / np.float32(np.linalg.norm(values))


# --- codec -----------------------------------------------------------------


def test_codec_reproduces_api_like_vectors_bit_for_bit():
    rng = np.random.default_rng(0)
    for _ in range(20):
        vector = bf16_vector(rng)
        vector[:3] = [0.0, -0.0, vector[3]]
        codes, divisor = vector_codec.encode(vector)
        assert codes.dtype == np.uint16
        assert vector_codec.same_bits(vector_codec.decode(codes, divisor), vector)


def test_codec_refuses_vectors_it_cannot_reproduce():
    rng = np.random.default_rng(1)
    assert vector_codec.encode(rng.normal(size=64).astype(np.float32)) is None
    assert vector_codec.encode(np.zeros(8, np.float32)) is None
    assert vector_codec.encode(np.array([1.0, np.nan], np.float32)) is None


def test_exact_float32_refuses_values_float32_would_round():
    assert exact_float32([[0.5, float(np.float32(0.1))]]).dtype == np.float32
    with pytest.raises(RuntimeError, match="float32"):
        exact_float32([[0.1]])


# --- inputs ----------------------------------------------------------------


def test_build_inputs_gives_raw_and_both_target_views():
    raw, home, away = build_inputs(REPORT, PLAYERS)
    assert (raw.view, raw.side, raw.team_id, raw.instruction) == (
        "raw",
        None,
        None,
        None,
    )
    assert raw.model_input == raw.text and "(AP)" not in raw.text
    assert (
        home.team_id == team_ids()["Thunder"] and away.team_id == team_ids()["Rockets"]
    )
    assert home.model_input.startswith(f"Instruct: {TARGET_VIEW_INSTRUCTION}\nQuery:")
    assert "In this text, Target Team is the Oklahoma City Thunder" in home.text
    assert "In this text, Target Team is the Houston Rockets" in away.text
    assert "Durant (Opponent Team player)" in home.text
    assert [i.input_sha256 for i in build_inputs(REPORT, PLAYERS)] == [
        i.input_sha256 for i in (raw, home, away)
    ]
    assert build_inputs(REPORT, [])[1].input_sha256 != home.input_sha256


def test_published_at_utc_converts_eastern_time():
    assert documents.published_at_utc("2025-10-22T00:07:19") == (
        "2025-10-22T04:07:19+00:00"
    )
    assert documents.published_at_utc("2026-01-10T22:00:00") == (
        "2026-01-11T03:00:00+00:00"
    )
    assert documents.published_at_utc(None) == ""


# --- store -----------------------------------------------------------------


def table_of(vectors, game_ids) -> EmbeddingTable:
    rows = []
    for game_id in game_ids:
        for view, side, team in (
            ("raw", None, None),
            ("target_view", "home", 1),
            ("target_view", "away", 2),
        ):
            rows.append(
                {c: None for c in ROW_COLUMNS}
                | {
                    "game_id": game_id,
                    "season": "2025-26",
                    "view": view,
                    "side": side,
                    "team_id": team,
                    "input_sha256": f"{game_id}-{view}-{side}",
                    "n_tokens": 10,
                    "n_players": 2,
                    "api_norm": 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    frame["team_id"] = frame["team_id"].astype("Int64")
    return EmbeddingTable(frame, vectors)


def test_store_round_trip_is_bit_exact_for_both_encodings():
    rng = np.random.default_rng(2)
    vectors = np.stack(
        [
            bf16_vector(rng, 16),
            rng.normal(size=16).astype(np.float32),
            bf16_vector(rng, 16),
        ]
    )
    s3 = FakeS3Client()
    store = EmbeddingStore(s3, "bucket", "emb")
    settings = store_settings() | {"vector_dimension": 16}
    entry = store.write_season("2025-26", table_of(vectors, ["1"]), settings)
    assert (entry["bf16_rows"], entry["f32_rows"]) == (2, 1)
    read = store.read_season("2025-26")
    assert vector_codec.same_bits(read.vectors, vectors)
    assert read.rows["team_id"].tolist()[1:] == [1, 2]
    assert pd.isna(read.rows["side"].iloc[0])

    with pytest.raises(RuntimeError, match="settings differ"):
        store.write_season(
            "2025-26", table_of(vectors, ["1"]), settings | {"model_id": "x"}
        )
    s3.store["emb/season=2025-26.parquet"] += b"x"
    with pytest.raises(RuntimeError, match="sha256"):
        store.read_season("2025-26")


# --- incremental update ----------------------------------------------------


class FakeTokenizer:
    def __call__(self, texts):
        return {"input_ids": [text.split() for text in texts]}


class FakeEmbedder:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.texts: list[str] = []
        self.rng = np.random.default_rng(3)

    def encode(self, texts, *, instruction=None, progress_every=1, normalize=True):
        self.texts += texts
        vectors = np.stack([bf16_vector(self.rng) for _ in texts])
        lengths = [len(t.split()) for t in texts]
        self.last_call = {
            "billed_tokens": sum(lengths),
            "local_tokens": sum(lengths),
            "api_seconds": 0.0,
        }
        return vectors, lengths


def put_report(s3, report):
    key = (
        f"{documents.SOURCES['report'][0]}/season={report['season']}/reports/"
        f"{report['game_id']}.json.gz"
    )
    s3.store[key] = gzip.compress(json.dumps(report).encode())
    return key


def test_update_season_embeds_only_what_is_new_or_changed():
    s3 = FakeS3Client()
    store = EmbeddingStore(s3, "bucket", "emb")
    second = REPORT | {"game_id": "0022500002", "home": "Lakers", "away": "Warriors"}
    put_report(s3, REPORT)
    key = put_report(s3, second)
    embedder = FakeEmbedder()

    def run():
        embedder.texts = []
        return update_season(
            "2025-26",
            client=s3,
            store=store,
            embedder=embedder,
            chunk_size=4,
            load_players=lambda ids: {"0022500001": PLAYERS},
            log=lambda line: None,
        )

    first = run()
    assert (first["embedded"], first["reports"]) == (6, 2)
    assert first["games_without_players"] == 1
    stored = store.read_season("2025-26")
    assert len(stored) == 6 and stored.rows["n_tokens"].gt(0).all()
    assert stored.rows["published_at_utc"].eq("2025-10-22T04:07:19+00:00").all()
    assert store.manifest()["seasons"]["2025-26"]["bf16_rows"] == 6

    assert run()["embedded"] == 0 and embedder.texts == []

    put_report(s3, second | {"text": second["text"] + " Edited."})
    changed = run()
    assert changed["embedded"] == 3 and len(store.read_season("2025-26")) == 6
    unchanged = store.read_season("2025-26").rows.query("game_id == '0022500001'")
    assert unchanged["embedded_at_utc"].nunique() == 1

    del s3.store[key]
    removed = run()
    assert (removed["embedded"], removed["orphans"]) == (0, 3)
    assert len(store.read_season("2025-26")) == 6
