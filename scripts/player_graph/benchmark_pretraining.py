"""Hardware benchmark for phase 6 pretraining (FM and GNN), CPU vs GPU.

Measures training throughput (stints per second, forward + backward + Adam
step) and peak GPU memory for the closed step 6a FM (dimension 16) and the
step 6b GNN (``gnn.py``): the representative configuration (width 32, two
layers) in its three capacity-matched ablations (node-only, teammate /
opponent, full with guard edges), plus wider and deeper full GNNs for
headroom, at several batch sizes. A configuration that runs out of GPU
memory is recorded, not fatal. It needs **no data**: inputs are random
tensors with the dataset's shapes (10 nodes, 50 guard edges with their
attributes, 27 node features, 6 targets per side), so it runs on the GPU
machine before anything is copied there.

It projects wall time for one run (one variant, one seed, 3 epochs) over the
monthly folds of the pretraining dataset v1 (every fold trains on all earlier
stints: 1.63 M stint passes per epoch over the 25 development folds, 17.4 M
over all 88), and for the development GNN ladder (4 variants x 3 seeds).

    python scripts/player_graph/benchmark_pretraining.py --out artifacts/experiments/pretraining_benchmark.json

A few minutes. Nothing here changes a model or a protocol; GPU and CPU
results differ slightly, so one comparison keeps one device.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from nba_ou.data_processing.player_graph.fm import (
    FactorizationMachine,
    FMConfig,
    StintTensors,
    stint_loss,
)
from nba_ou.data_processing.player_graph.gnn import GNN, N_GUARD_FEATURES, GNNConfig
from nba_ou.data_processing.player_graph.pretraining_data import NODE_COLUMNS

N_FEATURES = len(NODE_COLUMNS)
N_TARGETS = 6
N_PLAYERS = 3000
RESIDUAL = (0, 2)  # points and pace: interaction-only residuals
#: Stint passes per epoch over the monthly folds (pretraining v1).
SCHEDULE = {"development_25_folds": 1_633_308, "all_88_folds": 17_436_199}
LADDER_RUNS = 4 * 3  # B, C, D, E x three seeds

_GNN = GNNConfig(no_additive=RESIDUAL, absorb_additive=True)
MODELS = {
    "fm 16": FMConfig(edges=False, no_additive=RESIDUAL, center_interactions=True),
    "gnn node-only 32x2": GNNConfig(**{**_GNN.__dict__, "relations": ()}),
    "gnn no-guard 32x2": GNNConfig(
        **{**_GNN.__dict__, "relations": ("teammate", "opponent")}
    ),
    "gnn full 32x2": _GNN,
    "gnn full 64x2": GNNConfig(**{**_GNN.__dict__, "dim": 64}),
    "gnn full 64x3": GNNConfig(**{**_GNN.__dict__, "dim": 64, "layers": 3}),
    "gnn full 128x3": GNNConfig(**{**_GNN.__dict__, "dim": 128, "layers": 3}),
}
BATCHES = (1024, 4096, 16384, 32768)


def _batch(size: int, device: str) -> StintTensors:
    share = torch.rand(size, 2, 5, 5, device=device)
    share = share / share.sum(-1, keepdim=True)
    return StintTensors(
        x=torch.randn(size, 10, N_FEATURES, device=device),
        ids=torch.randint(1, N_PLAYERS, (size, 10), device=device),
        share=share.reshape(size, 50),
        y=torch.randn(size, 2, N_TARGETS, device=device),
        w=torch.rand(size, 2, N_TARGETS, device=device),
        guard=torch.randn(size, 50, N_GUARD_FEATURES, device=device),
    )


def measure(name: str, batch_size: int, device: str, steps: int) -> dict:
    torch.manual_seed(0)
    config = MODELS[name]
    model_class = GNN if isinstance(config, GNNConfig) else FactorizationMachine
    model = model_class(config, N_PLAYERS).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _batch(batch_size, device)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def step():
        loss = stint_loss(model, batch).mean() + model.penalty(100_000)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    try:
        for _ in range(3):  # warm-up
            step()
        if device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(steps):
            step()
        if device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"oom": True}
    out = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "stints_per_second": steps * batch_size / seconds,
    }
    if device == "cuda":
        out["peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=3, help="For the projection")
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    devices = args.devices or ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "machine": platform.node(),
        "cpu_threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_mib": (
            torch.cuda.get_device_properties(0).total_memory / 2**20
            if torch.cuda.is_available()
            else None
        ),
    }
    print(info, flush=True)
    rows = []
    for device in devices:
        for name in MODELS:
            for batch_size in BATCHES:
                result = measure(name, batch_size, device, args.steps)
                row = {"device": device, "model": name, "batch": batch_size, **result}
                label = f"{device:4} {name:18} batch {batch_size:>5}: "
                if result.get("oom"):
                    rows.append(row)
                    print(label + "out of GPU memory", flush=True)
                    continue
                rate = result["stints_per_second"]
                row["hours_per_run"] = {
                    k: v * args.epochs / rate / 3600 for k, v in SCHEDULE.items()
                }
                row["hours_dev_ladder"] = (
                    row["hours_per_run"]["development_25_folds"] * LADDER_RUNS
                )
                rows.append(row)
                print(
                    label
                    + f"{rate:>9,.0f} stints/s, "
                    + (
                        f"peak {result['peak_mib']:.0f} MiB, "
                        if "peak_mib" in result
                        else ""
                    )
                    + f"run: dev {row['hours_per_run']['development_25_folds']:.2f} h, "
                    f"all folds {row['hours_per_run']['all_88_folds']:.2f} h; "
                    f"dev ladder {row['hours_dev_ladder']:.1f} h",
                    flush=True,
                )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"info": info, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
