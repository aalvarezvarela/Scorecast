"""Embed NBA.com game reports and keep the vectors in the S3 store.

``embed_documents`` is the unit a daily job needs: given reports (and their
players), it returns their rows and vectors. ``update_season`` runs it over one
archived season incrementally: every report's inputs are rebuilt, an input
whose ``input_sha256`` is already stored is kept, and only new or changed
inputs are embedded, in chunks written as they finish, so a stopped run
resumes where it stopped. Stored rows of reports no longer in the archive are
kept and counted, never dropped.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from nba_ou.data_processing.game_texts import documents
from nba_ou.data_processing.game_texts.embedding_inputs import (
    INSTRUCTIONS,
    REWRITE_OPTIONS,
    VIEWS,
    EmbeddingInput,
    build_inputs,
    report_sha256,
)
from nba_ou.data_processing.game_texts.embedding_store import (
    KEY,
    EmbeddingStore,
    EmbeddingTable,
)
from nba_ou.data_processing.game_texts.nebius import (
    BASE_URL,
    MODEL_ID,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    USD_PER_M_TOKENS,
    NebiusEmbedder,
    token_counts,
)
from nba_ou.data_processing.game_texts.players import Player

SOURCE = "report"
DIMENSION = 4096
CHUNK_SIZE = 1200


def store_settings() -> dict:
    """What the stored vectors depend on, besides the texts (manifest fields)."""
    return {
        "source": SOURCE,
        "model_id": MODEL_ID,
        "endpoint": f"{BASE_URL}/embeddings",
        "tokenizer": f"{TOKENIZER_ID}@{TOKENIZER_REVISION}",
        "instructions": INSTRUCTIONS,
        "rewrite_options": REWRITE_OPTIONS,
        "views": [list(view) for view in VIEWS],
        "vector_dimension": DIMENSION,
        "vectors": "API values, bit for bit (bf16 codes / float32 divisor, or float32)",
    }


def input_rows(
    reports: Sequence[dict], players: dict[str, list[Player]]
) -> tuple[list[EmbeddingInput], pd.DataFrame]:
    """Every input of ``reports`` and its row metadata (no tokens or norm yet)."""
    inputs, rows = [], []
    for report in reports:
        game_players = players.get(str(report["game_id"]), [])
        report_hash = report_sha256(report)
        for item in build_inputs(report, game_players):
            inputs.append(item)
            rows.append(
                {
                    "game_id": item.game_id,
                    "season": report.get("season"),
                    "game_date": report.get("date"),
                    "view": item.view,
                    "side": item.side,
                    "team_id": item.team_id,
                    "published_at": report.get("published_at"),
                    "published_at_utc": report.get("published_at_utc"),
                    "report_sha256": report_hash,
                    "input_sha256": item.input_sha256,
                    "n_players": len(game_players),
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame):
        frame["team_id"] = frame["team_id"].astype("Int64")
    return inputs, frame


def embed_inputs(
    inputs: Sequence[EmbeddingInput],
    rows: pd.DataFrame,
    embedder: NebiusEmbedder,
    progress_every: int = 0,
) -> tuple[EmbeddingTable, dict]:
    """Embed ``inputs`` (one call per instruction) and complete their rows."""
    vectors = np.zeros((len(inputs), DIMENSION), np.float32)
    lengths = np.zeros(len(inputs), np.int64)
    billed, seconds = 0, 0.0
    for instruction in dict.fromkeys(item.instruction for item in inputs):
        where = [i for i, item in enumerate(inputs) if item.instruction == instruction]
        found, counts = embedder.encode(
            [inputs[i].text for i in where],
            instruction=instruction,
            progress_every=progress_every,
            normalize=False,
        )
        if found.shape[1] != DIMENSION:
            raise RuntimeError(f"Expected {DIMENSION} dimensions, got {found.shape}")
        vectors[where], lengths[where] = found, counts
        call = embedder.last_call
        billed += call["billed_tokens"] or call["local_tokens"]
        seconds += call["api_seconds"]
    done = rows.reset_index(drop=True).copy()
    done["n_tokens"] = lengths
    done["api_norm"] = np.linalg.norm(vectors.astype(np.float64), axis=1)
    done["embedded_at_utc"] = datetime.now(UTC).isoformat()
    return EmbeddingTable(done, vectors), {"billed_tokens": billed, "seconds": seconds}


def embed_documents(
    reports: Sequence[dict],
    embedder: NebiusEmbedder,
    players: dict[str, list[Player]] | None = None,
) -> EmbeddingTable:
    """Rows and vectors of every input of ``reports``; players are read from
    the database when not given."""
    if players is None:
        players = documents.load_game_players(str(r["game_id"]) for r in reports)
    inputs, rows = input_rows(reports, players)
    return embed_inputs(inputs, rows, embedder)[0]


def season_reports(client, season: str) -> list[dict]:
    return documents.load_documents(
        client, documents.list_keys(client, SOURCE, season), SOURCE
    )


def row_keys(rows: pd.DataFrame, columns=KEY) -> list[tuple]:
    if not len(rows):
        return []
    values = (rows[c].astype(object).where(rows[c].notna(), None) for c in columns)
    return list(zip(*values, strict=True))


def replace_rows(stored: EmbeddingTable, new: EmbeddingTable) -> EmbeddingTable:
    """``stored`` with every row whose key is in ``new`` replaced by it."""
    replaced = set(row_keys(new.rows))
    keep = [key not in replaced for key in row_keys(stored.rows)]
    return EmbeddingTable.concat([stored.take(keep), new])


@dataclass
class SeasonPlan:
    season: str
    reports: list[dict]
    inputs: list[EmbeddingInput]
    rows: pd.DataFrame
    stored: EmbeddingTable
    todo: np.ndarray  # positions in ``inputs`` to embed
    tokens: int
    orphans: int
    games_without_players: list[str]

    def summary(self) -> dict:
        return {
            "season": self.season,
            "reports": len(self.reports),
            "inputs": len(self.inputs),
            "stored": len(self.stored),
            "to_embed": len(self.todo),
            "orphans": self.orphans,
            "tokens_to_embed": self.tokens,
            "usd_to_embed": round(self.tokens / 1e6 * USD_PER_M_TOKENS, 4),
            "games_without_players": len(self.games_without_players),
        }


def plan_season(
    season: str,
    *,
    client,
    store: EmbeddingStore,
    tokenizer,
    load_players: Callable[[list[str]], dict[str, list[Player]]] | None = None,
) -> SeasonPlan:
    """What ``update_season`` would embed, and its tokens; no API call."""
    reports = season_reports(client, season)
    load_players = load_players or documents.load_game_players
    players = load_players([str(r["game_id"]) for r in reports]) if reports else {}
    inputs, rows = input_rows(reports, players)
    stored = store.read_season(season) or EmbeddingTable.empty(DIMENSION)
    have = set(row_keys(stored.rows, (*KEY, "input_sha256")))
    todo = np.array(
        [
            index
            for index, key in enumerate(row_keys(rows, (*KEY, "input_sha256")))
            if key not in have
        ],
        np.int64,
    )
    tokens = sum(token_counts(tokenizer, [inputs[i].model_input for i in todo]))
    current = set(row_keys(rows))
    return SeasonPlan(
        season=season,
        reports=reports,
        inputs=inputs,
        rows=rows,
        stored=stored,
        todo=todo,
        tokens=int(tokens),
        orphans=sum(key not in current for key in row_keys(stored.rows)),
        games_without_players=sorted(
            str(r["game_id"]) for r in reports if not players.get(str(r["game_id"]))
        ),
    )


def update_season(
    season: str,
    *,
    client,
    store: EmbeddingStore,
    embedder: NebiusEmbedder,
    chunk_size: int = CHUNK_SIZE,
    load_players: Callable[[list[str]], dict[str, list[Player]]] | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Embed what one season lacks and write it to the store after each chunk.

    A changed input keeps its old row until the chunk that re-embeds it is
    written, so a stopped run loses nothing."""
    plan = plan_season(
        season,
        client=client,
        store=store,
        tokenizer=embedder.tokenizer,
        load_players=load_players,
    )
    table, billed, seconds = plan.stored, 0, 0.0
    for start in range(0, len(plan.todo), chunk_size):
        chunk = plan.todo[start : start + chunk_size]
        new, call = embed_inputs(
            [plan.inputs[i] for i in chunk], plan.rows.iloc[chunk], embedder
        )
        table = replace_rows(table, new)
        billed += call["billed_tokens"]
        seconds += call["seconds"]
        entry = store.write_season(season, table, store_settings())
        log(
            f"{season}: embedded {start + len(chunk)}/{len(plan.todo)} inputs "
            f"({billed:,} tokens, {seconds:.0f} s); {entry['rows']} rows stored, "
            f"{entry['bf16_rows']} as bf16 + divisor, {entry['f32_rows']} as float32"
        )
    return {
        **plan.summary(),
        "embedded": len(plan.todo),
        "billed_tokens": billed,
        "usd": round(billed / 1e6 * USD_PER_M_TOKENS, 4),
        "api_seconds": round(seconds, 1),
    }
