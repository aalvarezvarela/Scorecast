"""Phase 2 smoke test: a tiny GNN on a few months of v0 stint graphs.

Runs the plumbing end to end on real data, with no claim about signal
(``nba_ou.data_processing.player_graph.smoke_gnn``): tables -> tensors (finite
inputs, every node with a profile, label coverage), training (the loss must go
down), per-label loss against the constant baseline on a later held-out
window, invariance to the order of the players within a side, and a
checkpoint that reloads to identical predictions. Exits non-zero if a check
fails.

Stays inside 2018-19: the 2_7 walk-forward starts in 2019-20, so nothing here
may read it.

    python scripts/player_graph/smoke_test_gnn.py

Reads the stint store, ``expected_guard/v0`` and ``node_profiles``; no
database. Writes ``data/player_graph/smoke_gnn/checkpoint.pt``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nba_ou.data_processing.lineups.stint_store import read_stints
from nba_ou.data_processing.player_graph.expected_guard import read_seasons
from nba_ou.data_processing.player_graph.smoke_gnn import (
    SmokeConfig,
    SmokeGNN,
    Standardizer,
    evaluate,
    guard_inputs,
    load_checkpoint,
    node_inputs,
    predict,
    save_checkpoint,
    to_tensors,
    train,
)
from nba_ou.data_processing.player_graph.stint_graph import LABELS, build_stint_graphs

SEASON = 2018
#: First day the smoke test may not read (2_7 evaluation starts in 2019-20).
EVALUATION_START = pd.Timestamp("2019-07-01")
PERMUTATION = [4, 2, 0, 3, 1]


def build(stints, expected, profiles):
    graphs = build_stint_graphs(stints, expected)
    return graphs, node_inputs(graphs, profiles)


def check(condition: bool, message: str, failures: list[str]) -> None:
    print(f"  [{'ok' if condition else 'FAIL'}] {message}")
    if not condition:
        failures.append(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--train-from", default="2018-10-16")
    parser.add_argument("--train-to", default="2019-01-31")
    parser.add_argument("--holdout-to", default="2019-03-31")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    train_from, train_to, holdout_to = map(
        pd.Timestamp, (args.train_from, args.train_to, args.holdout_to)
    )
    if holdout_to >= EVALUATION_START:
        raise SystemExit(f"The smoke test stays before {EVALUATION_START.date()}")
    out = args.out or args.local_root / "player_graph" / "smoke_gnn" / "checkpoint.pt"
    failures: list[str] = []
    torch.manual_seed(args.seed)

    print("1. tables -> tensors")
    started = time.time()
    stints = read_stints([SEASON], local_root=args.local_root)
    stints = stints.loc[stints["game_date"].between(train_from, holdout_to)]
    is_train = stints["game_date"].le(train_to)
    expected, metadata = read_seasons(
        args.local_root / "player_graph" / "expected_guard" / "v0", [SEASON]
    )
    expected = expected.loc[expected["game_id"].isin(set(stints["game_id"]))]
    profiles = pd.read_parquet(
        args.local_root / "player_graph" / "node_profiles" / f"season={SEASON}.parquet"
    )
    profiles = profiles.loc[profiles["as_of_date"].between(train_from, holdout_to)]
    train_graphs, train_nodes = build(stints.loc[is_train], expected, profiles)
    holdout_stints = stints.loc[~is_train]
    holdout_graphs, holdout_nodes = build(holdout_stints, expected, profiles)
    stats = Standardizer.fit(train_nodes, guard_inputs(train_graphs), train_graphs)
    train_data = to_tensors(train_graphs, train_nodes, stats)
    holdout_data = to_tensors(holdout_graphs, holdout_nodes, stats)
    print(
        f"  expected_guard {metadata}; {time.time() - started:.0f}s\n"
        f"  train   {train_from.date()} -> {train_to.date()}: {len(train_data):,} "
        f"stints, {stints.loc[is_train, 'game_id'].nunique()} games\n"
        f"  holdout {train_to.date()} -> {holdout_to.date()} (exclusive start): "
        f"{len(holdout_data):,} stints, {holdout_stints['game_id'].nunique()} games\n"
        f"  nodes {tuple(train_data.nodes.shape)}, guard {tuple(train_data.guard.shape)}, "
        f"labels {tuple(train_data.labels.shape)}"
    )
    coverage = train_data.mask.float().mean(dim=(0, 1)).numpy()
    print(
        "  label coverage: "
        + ", ".join(f"{n} {c:.1%}" for n, c in zip(LABELS, coverage, strict=True))
    )
    print(
        f"  guard fallback {train_graphs.guard_fallback.mean():.1%}, "
        f"pair history {train_graphs.guard_has_history.mean():.1%}, "
        f"zero-weight sides {(train_graphs.label_weight == 0).mean():.1%}"
    )
    check(
        all(
            torch.isfinite(t).all()
            for d in (train_data, holdout_data)
            for t in (d.nodes, d.guard, d.share, d.labels, d.weight)
        ),
        "every input, label and weight is finite",
        failures,
    )

    print("\n2. message passing -> embeddings -> pooling -> head -> loss -> backprop")
    config = SmokeConfig(dim=args.dim, layers=args.layers)
    model = SmokeGNN(config)
    n_parameters = sum(p.numel() for p in model.parameters())
    started = time.time()
    history = train(
        model,
        train_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )
    print(
        f"  {n_parameters:,} parameters, {args.epochs} epochs in "
        f"{time.time() - started:.0f}s; epoch loss "
        + " ".join(f"{loss:.4f}" for loss in history)
    )
    check(history[-1] < history[0], "training loss goes down", failures)
    embeddings = model.embed(
        holdout_data.nodes[:4], holdout_data.guard[:4], holdout_data.share[:4]
    )
    print(f"  node embeddings {tuple(embeddings.shape)}")

    print("\n3. per-label weighted MSE on standardized labels (constant = 1 x var)")
    rows = {}
    for name, data in (("train", train_data), ("holdout", holdout_data)):
        result = evaluate(model, data)
        rows[f"{name} model"] = result["model"]
        rows[f"{name} constant"] = result["constant"]
        rows[f"{name} ratio"] = result["model"] / result["constant"]
    table = pd.DataFrame(rows, index=list(LABELS)).T
    table["mean"] = table.mean(axis=1)
    print(table.round(4).to_string())

    print("\n4. invariance to player order within a side")
    permuted_stints = holdout_stints.assign(
        home_lineup=holdout_stints["home_lineup"].map(
            lambda lineup: np.asarray(lineup)[PERMUTATION]
        )
    )
    permuted_graphs, permuted_nodes = build(permuted_stints, expected, profiles)
    permuted = to_tensors(permuted_graphs, permuted_nodes, stats)
    a, b = predict(model, holdout_data), predict(model, permuted)
    check(
        not torch.equal(holdout_data.nodes, permuted.nodes)
        and torch.allclose(a, b, atol=1e-4),
        f"home lineups reordered: max |diff| {(a - b).abs().max().item():.2e}",
        failures,
    )

    print("\n5. checkpoint save / reload")
    extra = {
        "season": SEASON,
        "train": [str(train_from.date()), str(train_to.date())],
        "holdout_to": str(holdout_to.date()),
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "history": history,
        "expected_guard": metadata,
    }
    save_checkpoint(out, model, stats, extra)
    loaded, loaded_stats, loaded_extra = load_checkpoint(out)
    check(
        loaded_stats == stats
        and loaded_extra == extra
        and asdict(loaded.config) == asdict(config)
        and torch.equal(predict(loaded, holdout_data), a),
        f"reloaded {out} ({out.stat().st_size / 1024:.0f} KiB): identical predictions",
        failures,
    )

    if failures:
        raise SystemExit(f"{len(failures)} check(s) failed: {failures}")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
