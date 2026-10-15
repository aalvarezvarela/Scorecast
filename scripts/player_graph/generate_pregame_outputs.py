"""Out-of-fold pregame encoder outputs for closing games (phase 7).

Every game is encoded by the checkpoint of its own month's fold: an encoder
trained by ``fm_oof.py`` on stints dated before the month began (monthly
expanding walk-forward, warm-started), so no game is ever encoded by a model
that saw it, and the XGBoost training rows get the same out-of-fold outputs as
the rows it predicts. A game whose fold checkpoint is missing gets no row (and
is counted); a checkpoint whose training cutoff is not before the game's date
stops the run.

Per game (``pregame_encoder``): each availability scenario and the full-health
graph laid out as 48 one-minute stint graphs, encoded, averaged; tonight is the
probability-weighted scenario mean. Outputs (``--outputs``):

* ``style``: predicted turnovers per possession, 3PA/FGA, FTA/FGA and
  offensive rebound rate per offensive side, tonight and the absence shift
  (tonight - full health): ``GE_STYLE_*`` (16 columns);
* ``embedding``: the 16-dimensional game embedding, each side's pooled
  embedding on the fold's 8 aligned principal directions:
  ``<embedding-prefix>_<k>_BEFORE_TEAM_{HOME,AWAY}``.

The fold projections are fitted on each fold's last ``SAMPLE`` training
stints and chained by Procrustes from the run's first fold, in order; they are
cached under ``outputs/<name>/projection/`` and reused.

Writes ``data/player_graph/pregame/v1/outputs/<name>/season=YYYY.parquet``
(``GAME_ID`` + features + provenance: encoder run, fold, checkpoint date) and
``manifest.json`` (coverage per season and fold, the encoder's configuration,
inputs). Resumable: a season already written is skipped unless ``--force``.

    # development smoke (2018-19, dev checkpoints)
    python scripts/player_graph/generate_pregame_outputs.py --encoder-run fm_gnn_full_s0 \\
        --name smoke_full --seasons 2018 --outputs style embedding \\
        --embedding-prefix GE_GRAPH_EMB --max-games 40

Reads the pregame store, the pretraining dataset and the encoder's checkpoints;
no database.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.player_graph.fm import load_checkpoint, stint_tensors
from nba_ou.data_processing.player_graph.pregame_encoder import (
    STYLE_NAMES,
    STYLE_TARGETS,
    ScenarioInput,
    aggregate,
    align,
    embedding_columns,
    encode_slots,
    fold_projection,
    game_embedding,
    scenario_means,
    slot_stints,
    style_columns,
)
from nba_ou.data_processing.player_graph.pretraining_data import ROOT as PRETRAINING
from nba_ou.data_processing.player_graph.pretraining_data import (
    VERSION,
    load_stints,
    seasons_before,
)
from nba_ou.data_processing.player_graph.pretraining_oof import Fold

STORE = Path("player_graph") / "pregame" / "v1"
EVALUATION_START = pd.Timestamp("2019-07-01")
SAMPLE = 20_000


def season_range(text: str) -> list[int]:
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def read_store(root: Path, season: int) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(root / name / f"season={season}.parquet")
        for name in ("scenarios", "nodes", "guards", "node_features")
    }


class Encoders:
    """Fold checkpoints of one encoder run, loaded once each."""

    def __init__(self, root: Path, run: str) -> None:
        self.dir = root / "fm" / run
        self.cache: dict[str, tuple] = {}

    def labels(self) -> list[str]:
        return sorted(p.stem.split("=")[1] for p in self.dir.glob("fold=*.pt"))

    def get(self, label: str):
        if label not in self.cache:
            path = self.dir / f"fold={label}.pt"
            if not path.exists():
                return None
            self.cache[label] = load_checkpoint(path)
        return self.cache[label]


def side_pooled(model, vocab, std, stints) -> np.ndarray:
    """``(2n, dim)`` centered side-pooled embeddings of stints (both sides)."""
    import torch

    tensors = stint_tensors(stints, vocab, std, guard=True)
    model.eval()
    with torch.no_grad():
        h = model.uncentered(tensors)["embedding"] - model.centering_buffer("embedding")
        pooled = torch.cat([h[:, :5].mean(1), h[:, 5:].mean(1)]).numpy()
    return pooled.astype(np.float64)


def projections(encoders, labels, stints, cache_dir: Path) -> dict[str, np.ndarray]:
    """The aligned fold projections, chained in fold order from the first."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    bases: dict[str, np.ndarray] = {}
    previous = None  # (label, sample index)
    for label in labels:
        path = cache_dir / f"fold={label}.npz"
        checkpoint = pd.Timestamp(f"{label}-01")
        before = np.flatnonzero(pd.to_datetime(stints.game_date) < checkpoint)
        sample = before[-SAMPLE:]
        if path.exists():
            bases[label] = np.load(path)["basis"]
        else:
            model, vocab, std, _ = encoders.get(label)
            basis = fold_projection(
                side_pooled(model, vocab, std, stints.subset(sample))
            )
            if previous is not None:
                anchor = stints.subset(previous[1])
                prev_model, prev_vocab, prev_std, _ = encoders.get(previous[0])
                basis = align(
                    basis,
                    side_pooled(model, vocab, std, anchor),
                    bases[previous[0]],
                    side_pooled(prev_model, prev_vocab, prev_std, anchor),
                )
            np.savez(path, basis=basis, sample_until=str(checkpoint.date()))
            bases[label] = basis
        previous = (label, sample)
    return bases


def encode_season(
    store, encoders, bases, outputs, prefix, max_games=None
) -> tuple[pd.DataFrame, dict]:
    scenarios = store["scenarios"].copy()
    scenarios["game_date"] = pd.to_datetime(scenarios["game_date"]).dt.normalize()
    scenarios["as_of_date"] = pd.to_datetime(scenarios["as_of_date"]).dt.normalize()
    nodes = dict(tuple(store["nodes"].groupby(["game_id", "scenario_id"], sort=False)))
    guards = dict(tuple(store["guards"].groupby("game_id", sort=False)))
    features = store["node_features"].assign(
        as_of_date=lambda f: pd.to_datetime(f["as_of_date"]).dt.normalize(),
        player_id=lambda f: f["player_id"].astype(str),
    )
    by_date = {d: f.set_index("player_id") for d, f in features.groupby("as_of_date")}
    rows, missing = [], {"no_checkpoint": 0}
    games = list(scenarios.groupby("game_id", sort=True))
    for game_id, game in games[: max_games or len(games)]:
        date = game["game_date"].iloc[0]
        fold = Fold(date.replace(day=1), date.replace(day=1) + pd.offsets.MonthBegin(1))
        loaded = encoders.get(fold.label)
        if loaded is None:
            missing["no_checkpoint"] += 1
            continue
        model, vocab, std, extra = loaded
        if extra.get("fold") != fold.label or fold.checkpoint > date:
            raise SystemExit(
                f"{game_id}: checkpoint {extra.get('fold')} is not as of {date.date()}"
            )
        per, weights = {}, {}
        for scenario in game.itertuples(index=False):
            frame = nodes[(game_id, scenario.scenario_id)]
            inputs = ScenarioInput(
                game_id,
                scenario.scenario_id,
                frame.loc[frame["side"].eq("home"), ["player_id", "minutes"]],
                frame.loc[frame["side"].eq("away"), ["player_id", "minutes"]],
            )
            stints = slot_stints(inputs, by_date[scenario.as_of_date], guards[game_id])
            per[scenario.scenario_id] = scenario_means(
                encode_slots(model, vocab, std, stints)
            )
            weights[scenario.scenario_id] = scenario.weight
        full = int(game.loc[game["is_full_health"], "scenario_id"].iloc[0])
        result = aggregate(per, weights, full)
        row = {
            "GAME_ID": game_id,
            "game_date": date,
            "encoder_fold": fold.label,
            "encoder_checkpoint": fold.checkpoint,
            "n_scenarios": int(len(game) - 1),
        }
        if "style" in outputs:
            values = np.concatenate(
                [
                    result["style_tonight"].T.ravel(),
                    result["style_absence_shift"].T.ravel(),
                ]
            )
            row.update(dict(zip(style_columns(), values, strict=True)))
        if "embedding" in outputs:
            z = game_embedding(result["pooled_tonight"], bases[fold.label])
            row.update(dict(zip(embedding_columns(prefix), z, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows), missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--encoder-run", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seasons", type=season_range, required=True)
    parser.add_argument(
        "--outputs", nargs="+", choices=["style", "embedding"], required=True
    )
    parser.add_argument("--embedding-prefix", default="GE_GRAPH_EMB")
    parser.add_argument("--max-games", type=int, help="Smoke: games per season")
    parser.add_argument("--allow-evaluation-seasons", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if max(args.seasons) >= EVALUATION_START.year and not args.allow_evaluation_seasons:
        raise SystemExit("Seasons from 2019-20 need --allow-evaluation-seasons")
    store_root = args.local_root / STORE
    pretraining = args.local_root / PRETRAINING / VERSION
    out = store_root / "outputs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    encoders = Encoders(pretraining, args.encoder_run)
    labels = encoders.labels()
    if not labels:
        raise SystemExit(f"No checkpoints under {encoders.dir}")
    bases = {}
    if "embedding" in args.outputs:
        started = time.time()
        last = pd.Timestamp(f"{labels[-1]}-01")
        stints = load_stints(pretraining, seasons_before(last), until=last)
        bases = projections(encoders, labels, stints, out / "projection")
        print(
            f"{len(bases)} fold projections ({time.time() - started:.0f}s)", flush=True
        )
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    first_log = json.loads((encoders.dir / f"fold={labels[0]}.json").read_text())
    for season in args.seasons:
        path = out / f"season={season}.parquet"
        if path.exists() and not args.force:
            print(f"{season}: exists, skipped", flush=True)
            continue
        started = time.time()
        store = read_store(store_root, season)
        table, missing = encode_season(
            store, encoders, bases, args.outputs, args.embedding_prefix, args.max_games
        )
        table.to_parquet(path, index=False)
        manifest.setdefault("seasons", {})[str(season)] = {
            "games_in_store": int(store["scenarios"]["game_id"].nunique()),
            "games_encoded": len(table),
            "missing": missing,
            "folds": table["encoder_fold"].value_counts().sort_index().to_dict(),
            "feature_missingness": (
                float(table.drop(columns=["game_date"]).isna().mean().max())
                if len(table)
                else None
            ),
            "max_games": args.max_games,
            "built_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        }
        manifest.update(
            {
                "encoder_run": args.encoder_run,
                "encoder_variant": first_log.get("variant"),
                "encoder_config": first_log.get("config"),
                "encoder_seed": first_log.get("seed"),
                "outputs": args.outputs,
                "style_targets": {t: STYLE_NAMES[t] for t in STYLE_TARGETS},
                "embedding_prefix": args.embedding_prefix if bases else None,
                "projection": (
                    {
                        "per_side": 8,
                        "sample": f"last {SAMPLE} training stints of each fold",
                        "alignment": "orthogonal Procrustes to the previous fold",
                    }
                    if bases
                    else None
                ),
                "adapter": "48 one-minute lineups per team and scenario (Madow "
                "systematic sampling on expected minutes), slot k vs slot k",
                "store": str(store_root),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        print(
            f"{season}: {len(table):,} games encoded, missing {missing} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
