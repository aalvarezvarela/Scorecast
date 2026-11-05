#!/usr/bin/env python3
"""Embed the archived NBA.com game reports with Qwen3-Embedding-8B (Nebius).

Three vectors per report (raw text; home and away ``target_view`` rewrites),
stored in S3 one Parquet file per season
(``nba_ou.data_processing.game_texts``, ``docs/pipeline/text_embeddings.md``)::

    # Reports archived and vectors stored, per season (no database, no API)
    python scripts/text_embeddings/embed_game_reports.py status

    # What an update would embed and cost (database and S3 reads, no API)
    python scripts/text_embeddings/embed_game_reports.py update --dry-run

    # Embed what is missing or changed; one or more seasons, default all
    python scripts/text_embeddings/embed_game_reports.py update --season 2019-20

    # Re-read every stored season and check it
    python scripts/text_embeddings/embed_game_reports.py verify

The rewrites read each game's players from the database. A run resumes where
a stopped one left off: only inputs whose exact text is not stored are sent.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from nba_ou.config.settings import SETTINGS
from nba_ou.data_processing.game_texts import documents
from nba_ou.data_processing.game_texts.embed_reports import (
    SOURCE,
    plan_season,
    update_season,
)
from nba_ou.data_processing.game_texts.embedding_store import EmbeddingStore
from nba_ou.data_processing.game_texts.nebius import (
    USD_PER_M_TOKENS,
    NebiusEmbedder,
    load_tokenizer,
)


def report_seasons(client) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in documents.list_keys(client, SOURCE):
        season = documents.key_season(key)
        counts[season] = counts.get(season, 0) + 1
    return dict(sorted(counts.items()))


def status(client, store: EmbeddingStore) -> int:
    reports = report_seasons(client)
    stored = (store.manifest() or {}).get("seasons", {})
    print(f"{'season':<9} {'reports':>8} {'games':>7} {'rows':>7} {'MB':>7}")
    for season in sorted(set(reports) | set(stored)):
        entry = stored.get(season, {})
        print(
            f"{season:<9} {reports.get(season, 0):>8} {entry.get('games', 0):>7} "
            f"{entry.get('rows', 0):>7} {entry.get('bytes', 0) / 1e6:>7.1f}"
        )
    return 0


def update(client, store: EmbeddingStore, args) -> int:
    seasons = args.season or list(report_seasons(client))
    if args.dry_run:
        tokenizer = load_tokenizer()
        total_tokens = 0
        for season in seasons:
            summary = plan_season(
                season, client=client, store=store, tokenizer=tokenizer
            ).summary()
            total_tokens += summary["tokens_to_embed"]
            print(json.dumps(summary), flush=True)
        print(
            f"total: {total_tokens:,} tokens, ${total_tokens / 1e6 * USD_PER_M_TOKENS:.3f}"
        )
        return 0
    embedder = NebiusEmbedder(workers=args.workers)
    for season in seasons:
        summary = update_season(
            season,
            client=client,
            store=store,
            embedder=embedder,
            chunk_size=args.chunk_size,
            log=lambda line: print(line, flush=True),
        )
        print(json.dumps(summary), flush=True)
    return 0


def verify(store: EmbeddingStore) -> int:
    failures = 0
    for season in store.seasons():
        table = store.read_season(season)  # checks the manifest sha256
        rows = table.rows
        norms = np.linalg.norm(table.vectors.astype(np.float64), axis=1)
        per_game = rows.groupby("game_id").size()
        problems = {
            "games without 3 rows": int((per_game != 3).sum()),
            "duplicated keys": int(rows.duplicated(["game_id", "view", "side"]).sum()),
            "non-finite vectors": int((~np.isfinite(table.vectors).all(axis=1)).sum()),
            "norm far from 1": int((np.abs(norms - 1) > 1e-3).sum()),
            "norm != stored api_norm": int(
                (
                    ~np.isclose(norms, rows["api_norm"].to_numpy(), rtol=0, atol=1e-9)
                ).sum()
            ),
            "rewrite without team_id": int(
                (rows["view"].ne("raw") & rows["team_id"].isna()).sum()
            ),
        }
        bad = {name: count for name, count in problems.items() if count}
        failures += bool(bad)
        print(
            f"{season}: {len(rows)} rows, {rows['game_id'].nunique()} games",
            bad or "OK",
        )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    update_parser = commands.add_parser("update")
    update_parser.add_argument("--season", action="append", help="e.g. 2025-26")
    update_parser.add_argument("--dry-run", action="store_true")
    update_parser.add_argument("--chunk-size", type=int, default=1200)
    update_parser.add_argument("--workers", type=int, default=4)
    commands.add_parser("verify")
    args = parser.parse_args()

    client = documents.s3_client()
    store = EmbeddingStore(client, SETTINGS.s3_bucket)
    if args.command == "status":
        return status(client, store)
    if args.command == "update":
        return update(client, store, args)
    return verify(store)


if __name__ == "__main__":
    sys.exit(main())
