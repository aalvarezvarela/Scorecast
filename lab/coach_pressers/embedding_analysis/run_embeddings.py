"""Embed a sample of coach pressers in three views and collect their labels.

    .venv/bin/python -m lab.coach_pressers.embedding_analysis.run_embeddings
    .venv/bin/python -m lab.coach_pressers.embedding_analysis.run_embeddings --dry-run

Steps, each skipped when its file exists (delete the file to redo it):

1. `sample.parquet`: random games per season from the S3 manifests, every
   usable captioned presser of those games; captions cached in `captions/`.
2. `texts.parquet`: the caption text and its Target Team rewrite.
3. `labels.parquet`: same-game statistics seen from the coach's team.
4. `vectors/<view>.npy`: Qwen3-Embedding-8B (Nebius), L2-normalized float32,
   rows in `sample.parquet` order. `--dry-run` only counts tokens and cost.

Needs S3, the database (players, box scores) and `[Nebius] NEBIUS_API_KEY`.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lab.coach_pressers.config import S3_PREFIX
from lab.coach_pressers.embedding_analysis import labels as presser_labels
from lab.coach_pressers.embedding_analysis.captions import (
    caption_text,
    punctuation_per_100_words,
)
from lab.coach_pressers.embedding_analysis.presser_inputs import (
    NOTE_GAP_WORDS,
    VIEWS,
    rewrite_presser,
)
from lab.coach_pressers.teams import TEAMS
from lab.text_to_features.role_probes import GAME_TYPES

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "coach_pressers" / "embedding_analysis"
SEASONS = ("2023-24", "2024-25", "2025-26")
MIN_SECONDS, MAX_SECONDS = 60, 2400  # of the coach's part
MIN_WORDS = 150


def log(message: str) -> None:
    print(f"[{datetime.now(UTC):%H:%M:%S}] {message}", flush=True)


def usable_pressers(manifest: dict) -> list[dict]:
    """Manifest entries with captions whose coach part is known: coach-only
    videos, and combined videos with the coach's start and end."""
    rows = []
    for entry in manifest["entries"].values():
        archive = entry.get("archive") or {}
        selection = entry.get("selection") or {}
        if not (archive.get("captions") or {}).get("key"):
            continue
        if entry["game_id"][:3] not in GAME_TYPES:
            continue
        start, end = selection.get("trim_start_s"), selection.get("trim_end_s")
        if selection.get("kind") == "coach":
            start = end = None
            seconds = archive.get("duration_s") or selection.get("duration_s") or 0
        elif start is not None and end is not None:
            seconds = end - start
        else:
            continue  # combined video without a coach chapter
        if not MIN_SECONDS <= seconds <= MAX_SECONDS:
            continue
        rows.append(
            {
                "presser_id": f"{entry['game_id']}_{entry['team']}",
                "game_id": entry["game_id"],
                "season": manifest["season"],
                "date": entry["date"],
                "phase": entry["phase"],
                "team": entry["team"],
                "opponent": entry["opponent"],
                "is_home": bool(entry["is_home"]),
                "coach": selection.get("coach_actual") or entry["coach_expected"],
                "kind": selection.get("kind"),
                "video_id": selection.get("video_id"),
                "title": selection.get("title"),
                "published_at_utc": selection.get("published_at_utc"),
                "trim_start_s": start,
                "trim_end_s": end,
                "seconds": float(seconds),
                "captions_key": archive["captions"]["key"],
            }
        )
    return rows


def draw_sample(client, bucket: str, per_season: int, seed: int) -> pd.DataFrame:
    """Whole games in random order until the season has `per_season` pressers."""
    rng = random.Random(seed)
    picked = []
    for season in SEASONS:
        key = f"{S3_PREFIX}/season={season}/manifest.json"
        manifest = json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
        rows = usable_pressers(manifest)
        by_game: dict[str, list[dict]] = {}
        for row in rows:
            by_game.setdefault(row["game_id"], []).append(row)
        games = sorted(by_game)
        rng.shuffle(games)
        chosen: list[dict] = []
        for game_id in games:
            if len(chosen) >= per_season:
                break
            chosen += by_game[game_id]
        log(f"{season}: {len(rows)} usable pressers, {len(chosen)} sampled")
        picked += chosen
    return pd.DataFrame(picked)


def fetch_captions(client, bucket: str, sample: pd.DataFrame, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)

    def fetch(row) -> None:
        path = folder / f"{row.presser_id}.vtt"
        if not path.exists():
            body = client.get_object(Bucket=bucket, Key=row.captions_key)["Body"]
            path.write_bytes(body.read())

    with ThreadPoolExecutor(16) as pool:
        list(pool.map(fetch, sample.itertuples(index=False)))


def build_texts(sample: pd.DataFrame, folder: Path) -> pd.DataFrame:
    """`presser_id`, `raw`, `target_view` and the rewrite's counts."""
    from nba_ou.data_processing.game_texts.documents import load_game_players

    players = load_game_players(sample["game_id"])
    rows = []
    for row in sample.itertuples(index=False):
        raw = caption_text(
            (folder / f"{row.presser_id}.vtt").read_text(encoding="utf-8"),
            None if pd.isna(row.trim_start_s) else row.trim_start_s,
            None if pd.isna(row.trim_end_s) else row.trim_end_s,
        )
        rewrite = rewrite_presser(
            raw,
            TEAMS[row.team][1],
            TEAMS[row.opponent][1],
            row.is_home,
            players.get(row.game_id),
        )
        rows.append(
            {
                "presser_id": row.presser_id,
                "raw": raw,
                "target_view": rewrite.text,
                "n_words": len(raw.split()),
                "punct_per_100_words": punctuation_per_100_words(raw),
                "target_mentions": rewrite.target_mentions,
                "opponent_mentions": rewrite.opponent_mentions,
                "player_mentions": rewrite.player_mentions,
                "player_notes": rewrite.player_notes,
                "game_players": len(players.get(row.game_id) or ()),
            }
        )
    return pd.DataFrame(rows)


def newest_closing_file() -> Path | None:
    files = sorted(
        (REPO_ROOT / "data" / "train_data").glob("closing_line_data_2_5_*.parquet")
    )
    return files[-1] if files else None


def build_labels(sample: pd.DataFrame, closing_file: Path | None) -> pd.DataFrame:
    games, out = presser_labels.load_box_scores(list(SEASONS))
    labels = presser_labels.team_game_labels(games, out)
    if closing_file is None:
        log("warning: no closing-line file, market labels left out")
    else:
        labels = presser_labels.add_market(labels, closing_file)
    frame = sample[["presser_id", "game_id", "team"]].merge(
        labels, on=["game_id", "team"], how="left", validate="one_to_one"
    )
    missing = int(frame["target_points"].isna().sum())
    if missing:
        log(f"warning: {missing} pressers without a box score")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--pressers-per-season", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--views", nargs="+", default=list(VIEWS), choices=list(VIEWS))
    parser.add_argument("--closing-file", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="tokens and cost only")
    args = parser.parse_args()

    from nba_ou.config.settings import SETTINGS
    from nba_ou.data_processing.game_texts import nebius
    from nba_ou.data_processing.game_texts.documents import s3_client
    from nba_ou.data_processing.game_texts.embedding_inputs import embedding_input

    out: Path = args.out
    (out / "vectors").mkdir(parents=True, exist_ok=True)
    client, bucket = s3_client(), SETTINGS.s3_bucket

    sample_file, texts_file = out / "sample.parquet", out / "texts.parquet"
    if sample_file.exists() and texts_file.exists():
        sample, texts = pd.read_parquet(sample_file), pd.read_parquet(texts_file)
        log(f"sample: {len(sample)} pressers (stored)")
    else:
        drawn = draw_sample(client, bucket, args.pressers_per_season, args.seed)
        fetch_captions(client, bucket, drawn, out / "captions")
        texts = build_texts(drawn, out / "captions")
        keep = (texts["n_words"] >= MIN_WORDS).to_numpy()
        log(f"dropped {int((~keep).sum())} pressers under {MIN_WORDS} words")
        sample = drawn[keep].reset_index(drop=True)
        texts = texts[keep].reset_index(drop=True)
        texts.to_parquet(texts_file, index=False)
        sample.to_parquet(sample_file, index=False)
        log(f"sample: {len(sample)} pressers, {sample['game_id'].nunique()} games")

    labels_file = out / "labels.parquet"
    if not labels_file.exists():
        closing = args.closing_file or newest_closing_file()
        build_labels(sample, closing).to_parquet(labels_file, index=False)
        log(f"labels: {labels_file.name} (closing lines: {closing})")

    run_file = out / "run.json"
    run = json.loads(run_file.read_text()) if run_file.exists() else {"views": {}}
    run.update(
        {
            "seasons": SEASONS,
            "seed": args.seed,
            "pressers": len(sample),
            "note_gap_words": NOTE_GAP_WORDS,
            "min_words": MIN_WORDS,
        }
    )
    embedder = None
    tokenizer = nebius.load_tokenizer(REPO_ROOT / "artifacts" / "huggingface")
    for view in args.views:
        version, instruction = VIEWS[view]
        vector_file = out / "vectors" / f"{view}.npy"
        view_texts = texts[version].tolist()
        if args.dry_run:
            inputs = [embedding_input(text, instruction) for text in view_texts]
            tokens = sum(nebius.token_counts(tokenizer, inputs))
            usd = tokens / 1e6 * nebius.USD_PER_M_TOKENS
            log(f"{view}: {tokens:,} tokens, about ${usd:.3f}")
            continue
        if vector_file.exists():
            log(f"{view}: stored")
            continue
        if embedder is None:
            embedder = nebius.NebiusEmbedder(tokenizer=tokenizer)
        started = time.perf_counter()
        vectors, lengths = embedder.encode(
            view_texts, instruction=instruction, progress_every=200
        )
        np.save(vector_file, vectors.astype(np.float32))
        run["views"][view] = {
            "text": version,
            "instruction": instruction,
            "tokens": int(sum(lengths)),
            "usd": sum(lengths) / 1e6 * nebius.USD_PER_M_TOKENS,
            "seconds": round(time.perf_counter() - started, 1),
            "embedded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            **embedder.metadata,
        }
        run_file.write_text(json.dumps(run, indent=2))
        log(f"{view}: {vectors.shape} in {run['views'][view]['seconds']} s")
    log("done")


if __name__ == "__main__":
    main()
