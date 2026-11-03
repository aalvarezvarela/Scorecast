"""Embed the role-probe sample under every instruction (``role_probes``).

    .venv/bin/python -m lab.text_to_features.run_role_probes --model 0.6B
    .venv/bin/python -m lab.text_to_features.run_role_probes --model 4B \\
        --conditions none neutral target_focus
    .venv/bin/python -m lab.text_to_features.run_role_probes --model 8B

0.6B and 4B run on the local GPU; 8B through the Nebius API (``nebius``,
key in ``[Nebius] NEBIUS_API_KEY``).

The first run draws the sample (``--per-source`` documents of each source from
``--seasons``, games where both teams have played ``role_probes.WINDOW`` season
games) and writes it with its labels and texts to ``--out``; later runs, any
model, reuse it. Embeddings go to ``--out/<model>/``, one file per condition
and setting; a file that exists is skipped, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from lab.text_to_features import role_probes as rp
from lab.text_to_features.documents import (
    list_keys,
    load_document,
    load_game_players,
    s3_client,
)
from lab.text_to_features.team_perspective import document_perspectives

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/text_to_features/role_probes"
MODELS = {
    "0.6B": "Qwen/Qwen3-Embedding-0.6B",
    "4B": "Qwen/Qwen3-Embedding-4B",
    "8B": "Qwen/Qwen3-Embedding-8B",
}
REMOTE = ("8B",)
SEASONS = ("2023-24", "2024-25", "2025-26")
SOURCES = ("report", "transcript")
VIEWS = ("raw", "home_target", "away_target")
SAMPLE_FILES = ("sample.parquet", "texts.parquet", "labels.parquet", "sample.json")


def box_score_labels(seasons: list[str]) -> pd.DataFrame:
    from nba_ou.postgre_db.games.fetch_data_from_db.fetch_data_from_games_db import (
        load_games_from_db,
    )
    from nba_ou.postgre_db.injuries_refs.fetch_injury_db.get_injury_data_from_db import (  # noqa: E501
        get_injury_data_from_db,
    )
    from nba_ou.postgre_db.players.fetch_players_data_from_db import (
        load_players_from_db,
    )

    games = load_games_from_db(seasons=seasons)
    box = load_players_from_db(seasons=seasons)
    inactive = get_injury_data_from_db(seasons=seasons)
    if games is None or box is None or inactive is None:
        raise RuntimeError("Could not load games, box scores or inactive lists")
    games = games.assign(game_id=games["game_id"].astype(str))
    minutes = pd.DataFrame(
        {
            "game_id": box["game_id"].astype(str),
            "team_id": box["team_id"].astype(str),
            "player_id": box["player_id"].astype(str),
            "minutes": box["min"].map(rp.parse_minutes),
        }
    )
    listed = pd.concat(
        [
            minutes[["game_id", "team_id", "player_id"]],
            pd.DataFrame(
                {
                    "game_id": inactive["GAME_ID"].astype(str),
                    "team_id": inactive["TEAM_ID"].astype(str),
                    "player_id": inactive["PLAYER_ID"].astype(str),
                }
            ),
        ]
    )
    order = rp.team_game_order(games)
    out = rp.regulars_out(order, minutes, listed)
    return rp.game_labels(games, out, order)


def draw_sample(out: Path, per_source: int, seasons: list[str], seed: int) -> None:
    started = time.perf_counter()
    labels = box_score_labels(seasons)
    games = rp.eligible_games(labels)
    client = s3_client()
    keys = {
        source: [key for season in seasons for key in list_keys(client, source, season)]
        for source in SOURCES
    }
    picked = rp.sample_keys(keys, games, per_source, seed)
    documents = [load_document(client, key, source) for source, key in picked]
    players = load_game_players(doc["game_id"] for doc in documents)
    rows, texts = [], []
    for index, doc in enumerate(documents):
        rows.append(
            {
                "doc_index": index,
                "source": doc["source"],
                "s3_key": doc["s3_key"],
                "game_id": str(doc["game_id"]),
                "date": str(doc["date"]),
                "home": doc["home"],
                "away": doc["away"],
                "home_score": doc["home_score"],
                "away_score": doc["away_score"],
                "word_count": doc.get("word_count"),
                "players": len(players[str(doc["game_id"])]),
            }
        )
        views = document_perspectives(
            doc, players=players[str(doc["game_id"])], label_rule="any"
        )
        texts.append({"doc_index": index, **{view: views[view] for view in VIEWS}})
    sample = pd.DataFrame(rows)
    frame = rp.view_labels(sample, labels)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(texts).to_parquet(out / "texts.parquet", index=False)
    frame.to_parquet(out / "labels.parquet", index=False)
    (out / "sample.json").write_text(
        json.dumps(
            {
                "per_source": per_source,
                "seasons": seasons,
                "seed": seed,
                "eligible_games": len(games),
                "label_rule": "any",
                "regular": {
                    "window": rp.WINDOW,
                    "min_played": rp.MIN_PLAYED,
                    "min_minutes": rp.MIN_MINUTES,
                },
                "seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )
    sample.to_parquet(out / "sample.parquet", index=False)  # last: marks it complete
    print(f"Sample: {len(sample)} documents, {len(games)} eligible games", flush=True)


def embed(path: Path, embedder, texts: list[str], instruction, log: dict) -> None:
    if path.exists():
        print(f"skip {path.name}", flush=True)
        return
    started = time.perf_counter()
    vectors, lengths = embedder.encode(
        texts, instruction=instruction, progress_every=100
    )
    np.save(path, vectors.astype(np.float32))
    log[path.stem] = {
        "texts": len(texts),
        "seconds": time.perf_counter() - started,
        "max_tokens": max(lengths),
        "mean_tokens": float(np.mean(lengths)),
        **getattr(embedder, "last_call", {}),
    }
    print(f"wrote {path.name} in {log[path.stem]['seconds']:.0f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="0.6B")
    parser.add_argument("--conditions", nargs="+", default=list(rp.INSTRUCTIONS))
    parser.add_argument("--per-source", type=int, default=250)
    parser.add_argument("--seasons", nargs="+", default=list(SEASONS))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--skip-documents", action="store_true")
    args = parser.parse_args()
    unknown = set(args.conditions) - set(rp.INSTRUCTIONS)
    if unknown:
        parser.error(f"unknown conditions {sorted(unknown)}")

    if not all((args.out / name).exists() for name in SAMPLE_FILES):
        draw_sample(args.out, args.per_source, args.seasons, args.seed)
    texts = pd.read_parquet(args.out / "texts.parquet").sort_values("doc_index")

    cache_dir = ROOT / "artifacts/huggingface"
    if args.model in REMOTE:
        from lab.text_to_features.nebius import NebiusEmbedder

        embedder = NebiusEmbedder(cache_dir=cache_dir, model_id=MODELS[args.model])
    else:
        from lab.text_to_features.embeddings import QwenEmbedder

        embedder = QwenEmbedder(cache_dir=cache_dir, model_id=MODELS[args.model])
    model_dir = args.out / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    meta_path = model_dir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    log = meta.setdefault("files", {})
    meta["model"] = {
        k: v for k, v in embedder.metadata.items() if k not in ("instruction",)
    }
    meta["instructions"] = rp.INSTRUCTIONS

    for condition in args.conditions:
        instruction = rp.INSTRUCTIONS[condition]
        for setting, in_context in (("short", False), ("context", True)):
            items = rp.triplets(in_context=in_context)
            path = model_dir / f"triplets_{setting}_{condition}.npy"
            embed(path, embedder, rp.triplet_texts(items), instruction, log)
        meta_path.write_text(json.dumps(meta, indent=2))
    if not args.skip_documents:
        for condition in args.conditions:
            # The raw text is only a baseline: embedded without instruction.
            views = VIEWS if condition == "none" else rp.ROLE_VIEWS
            batch = [text for _, row in texts.iterrows() for text in row[list(views)]]
            path = model_dir / f"documents_{condition}.npy"
            embed(path, embedder, batch, rp.INSTRUCTIONS[condition], log)
            meta_path.write_text(json.dumps(meta, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
