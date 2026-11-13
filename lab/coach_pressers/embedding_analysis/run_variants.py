"""More embeddings of the sampled pressers: chunks, instructions, reports.

    .venv/bin/python -m lab.coach_pressers.embedding_analysis.run_variants
    .venv/bin/python -m lab.coach_pressers.embedding_analysis.run_variants --dry-run

Needs the sample of ``run_embeddings``. Adds, in the same folder (a file that
exists is skipped):

- ``chunks_<text>.parquet``: every presser cut into pieces of about
  ``CHUNK_WORDS`` words (``text`` is ``raw`` or ``masked``, the presser
  without team and player names);
- ``vectors/chunks_<condition>.npy`` (float16): one vector per chunk under
  each instruction of ``CHUNK_CONDITIONS``. The mean of a presser's chunks
  does not depend on its length the way one vector of the whole text does,
  and two halves of the chunks measure how repeatable a representation is;
- ``vectors/whole_<condition>.npy``: the whole presser under ``WHOLE_CONDITIONS``;
- ``vectors/anchors_<condition>.npy`` + ``anchors.parquet``: the concept
  anchor sentences (``anchors.py``);
- ``vectors/report_target_view.npy`` + ``report_rows.parquet``: the stored
  NBA.com report vector of the same game and team (NaN rows where missing).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lab.coach_pressers.embedding_analysis.anchors import AXES
from lab.coach_pressers.embedding_analysis.presser_inputs import (
    PRESSER_INSTRUCTION,
    mask_presser,
)
from lab.coach_pressers.embedding_analysis.run_embeddings import DEFAULT_OUT, REPO_ROOT
from lab.coach_pressers.teams import TEAMS

CHUNK_WORDS = 220
INSTRUCTIONS: dict[str, str | None] = {
    "none": None,
    "basketball": PRESSER_INSTRUCTION,
    "intangibles": (
        "Represent what this NBA head coach's postgame press conference "
        "reveals about his team beyond the box score: style of play, effort, "
        "intensity, physicality, focus, chemistry and mood."
    ),
    "style": (
        "Represent the style of play this NBA head coach describes in his "
        "postgame press conference: pace, ball movement, shot selection, "
        "physicality and defensive scheme."
    ),
    "mood": (
        "Represent the mood and attitude of this NBA head coach in his "
        "postgame press conference: satisfaction or frustration with his "
        "team's effort, focus and execution."
    ),
}
#: condition -> (text, instruction name)
CHUNK_CONDITIONS: dict[str, tuple[str, str]] = {
    "none": ("raw", "none"),
    "basketball": ("raw", "basketball"),
    "intangibles": ("raw", "intangibles"),
    "style": ("raw", "style"),
    "mood": ("raw", "mood"),
    "masked_intangibles": ("masked", "intangibles"),
}
WHOLE_CONDITIONS: dict[str, tuple[str, str]] = {
    "intangibles": ("raw", "intangibles"),
    "masked_intangibles": ("masked", "intangibles"),
}
ANCHOR_CONDITIONS = ("intangibles",)  # the masked condition shares the instruction


def log(message: str) -> None:
    print(f"[{datetime.now(UTC):%H:%M:%S}] {message}", flush=True)


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS) -> list[str]:
    """Pieces of equal size, about ``chunk_words`` words each."""
    words = text.split()
    pieces = max(1, round(len(words) / chunk_words))
    return [" ".join(part) for part in np.array_split(np.array(words, object), pieces)]


def masked_texts(sample: pd.DataFrame, texts: pd.DataFrame) -> list[str]:
    from nba_ou.data_processing.game_texts.documents import load_game_players

    players = load_game_players(sample["game_id"])
    return [
        mask_presser(
            raw, TEAMS[row.team][1], TEAMS[row.opponent][1], players.get(row.game_id)
        )
        for row, raw in zip(sample.itertuples(index=False), texts["raw"], strict=True)
    ]


def chunk_table(presser_ids: pd.Series, texts: list[str]) -> pd.DataFrame:
    rows = []
    for position, (presser_id, text) in enumerate(zip(presser_ids, texts, strict=True)):
        for index, piece in enumerate(chunk_text(text)):
            rows.append(
                {
                    "presser": position,
                    "presser_id": presser_id,
                    "chunk": index,
                    "text": piece,
                }
            )
    return pd.DataFrame(rows)


def report_vectors(sample: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """The stored ``target_view`` report vector of each presser's game and team."""
    from nba_ou.config.settings import SETTINGS
    from nba_ou.data_processing.game_texts.documents import s3_client
    from nba_ou.data_processing.game_texts.embedding_inputs import team_ids
    from nba_ou.data_processing.game_texts.embedding_store import EmbeddingStore

    store = EmbeddingStore(s3_client(), SETTINGS.s3_bucket)
    ids = team_ids()
    wanted = {
        (row.game_id, ids[TEAMS[row.team][1]]): position
        for position, row in enumerate(sample.itertuples(index=False))
    }
    vectors = np.full((len(sample), 4096), np.nan, np.float32)
    found = []
    for season in sorted(sample["season"].unique()):
        table = store.read_season(season)
        rows = table.rows.reset_index(drop=True)
        for index, row in rows[rows["view"] == "target_view"].iterrows():
            position = wanted.get((str(row["game_id"]), int(row["team_id"])))
            if position is not None:
                vector = table.vectors[index].astype(np.float64)
                vectors[position] = vector / np.linalg.norm(vector)
                found.append(
                    {
                        "presser": position,
                        "n_tokens": row["n_tokens"],
                        "published_at_utc": row["published_at_utc"],
                    }
                )
    return vectors, pd.DataFrame(found).sort_values("presser")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true", help="tokens and cost only")
    args = parser.parse_args()

    from nba_ou.data_processing.game_texts import nebius
    from nba_ou.data_processing.game_texts.embedding_inputs import embedding_input

    out: Path = args.out
    sample = pd.read_parquet(out / "sample.parquet")
    texts = pd.read_parquet(out / "texts.parquet")

    masked_file = out / "texts_masked.parquet"
    if masked_file.exists():
        masked = pd.read_parquet(masked_file)["masked"].tolist()
    else:
        masked = masked_texts(sample, texts)
        pd.DataFrame({"presser_id": sample["presser_id"], "masked": masked}).to_parquet(
            masked_file, index=False
        )
    whole = {"raw": texts["raw"].tolist(), "masked": masked}
    chunks = {}
    for name, full in whole.items():
        file = out / f"chunks_{name}.parquet"
        if not file.exists():
            chunk_table(sample["presser_id"], full).to_parquet(file, index=False)
        chunks[name] = pd.read_parquet(file)
        log(f"chunks_{name}: {len(chunks[name])} chunks")

    report_file = out / "vectors" / "report_target_view.npy"
    if not report_file.exists() and not args.dry_run:
        vectors, rows = report_vectors(sample)
        np.save(report_file, vectors)
        rows.to_parquet(out / "report_rows.parquet", index=False)
        log(f"reports: {len(rows)} of {len(sample)} pressers have one")

    anchors = pd.DataFrame(
        [
            {"axis": axis, "pole": pole, "text": sentence}
            for axis, poles in AXES.items()
            for pole, sentences in poles.items()
            for sentence in sentences
        ]
    )
    anchors.to_parquet(out / "anchors.parquet", index=False)

    jobs = [
        (f"chunks_{name}", chunks[text]["text"].tolist(), instruction, np.float16)
        for name, (text, instruction) in CHUNK_CONDITIONS.items()
    ]
    jobs += [
        (f"whole_{name}", whole[text], instruction, np.float32)
        for name, (text, instruction) in WHOLE_CONDITIONS.items()
    ]
    jobs += [
        (
            f"anchors_{name}",
            anchors["text"].tolist(),
            CHUNK_CONDITIONS[name][1],
            np.float32,
        )
        for name in ANCHOR_CONDITIONS
    ]
    run_file = out / "run_variants.json"
    run = json.loads(run_file.read_text()) if run_file.exists() else {}
    run["chunk_words"] = CHUNK_WORDS
    run["instructions"] = INSTRUCTIONS
    tokenizer = nebius.load_tokenizer(REPO_ROOT / "artifacts" / "huggingface")
    embedder = None
    total = 0
    for name, inputs, instruction_name, dtype in jobs:
        instruction = INSTRUCTIONS[instruction_name]
        file = out / "vectors" / f"{name}.npy"
        if args.dry_run:
            tokens = sum(
                nebius.token_counts(
                    tokenizer, [embedding_input(text, instruction) for text in inputs]
                )
            )
            total += tokens
            log(f"{name}: {len(inputs)} texts, {tokens:,} tokens")
            continue
        if file.exists():
            log(f"{name}: stored")
            continue
        if embedder is None:
            embedder = nebius.NebiusEmbedder(tokenizer=tokenizer)
        started = time.perf_counter()
        vectors, lengths = embedder.encode(
            inputs, instruction=instruction, progress_every=1000
        )
        np.save(file, vectors.astype(dtype))
        run[name] = {
            "texts": len(inputs),
            "instruction": instruction_name,
            "tokens": int(sum(lengths)),
            "usd": sum(lengths) / 1e6 * nebius.USD_PER_M_TOKENS,
            "seconds": round(time.perf_counter() - started, 1),
        }
        run_file.write_text(json.dumps(run, indent=2))
        log(f"{name}: {vectors.shape} in {run[name]['seconds']} s")
    if args.dry_run:
        log(
            f"total {total:,} tokens, about ${total / 1e6 * nebius.USD_PER_M_TOKENS:.2f}"
        )
    log("done")


if __name__ == "__main__":
    main()
