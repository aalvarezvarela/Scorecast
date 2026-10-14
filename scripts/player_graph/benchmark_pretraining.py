"""Hardware benchmark for phase 6 pretraining (FM and a relational GNN), CPU vs GPU.

Measures training throughput (stints per second, forward + backward + Adam
step) and peak GPU memory for the step 6a factorization machine and a
representative relational GNN (``smoke_gnn.SmokeGNN``: typed message passing
over teammate, opponent and guard edges; the phase 6b architecture is not
decided, so this measures capacity, not a design), over a grid of sizes and
batch sizes. A configuration that runs out of GPU memory is recorded, not
fatal. It needs **no data**: inputs are random tensors with the dataset's
shapes (10 nodes, 50 guard edges, 27 node features, 6 targets per side), so it
runs on the GPU machine before anything is copied there.

It then projects wall time for the monthly out-of-fold schedule of the
pretraining dataset v1 (88 folds from 2016-12; every fold trains on all
earlier stints, 17.4 M stint passes per epoch over all folds, 1.6 M over the
25 development folds before 2019-07).

    python scripts/player_graph/benchmark_pretraining.py --out artifacts/experiments/pretraining_benchmark.json

Short (a few minutes). Nothing here changes a model or a protocol; GPU and CPU
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
    weighted_mse,
)
from nba_ou.data_processing.player_graph.pretraining_data import NODE_COLUMNS
from nba_ou.data_processing.player_graph.smoke_gnn import (
    GUARD_ATTRIBUTES,
    SmokeConfig,
    SmokeGNN,
)

N_FEATURES = len(NODE_COLUMNS)
N_TARGETS = 6
N_PLAYERS = 3000
#: Stint passes per epoch over the monthly folds (pretraining v1, all / development).
SCHEDULE = {"all_88_folds": 17_436_199, "development_25_folds": 1_633_308}

FM_DIMS = (8, 16, 32)
GNN_SIZES = ((32, 2), (64, 2), (64, 3), (128, 3))
BATCHES = (1024, 4096, 16384)


def _batch(size: int, device: str) -> dict[str, torch.Tensor]:
    share = torch.rand(size, 2, 5, 5, device=device)
    share = share / share.sum(-1, keepdim=True)
    return {
        "x": torch.randn(size, 10, N_FEATURES, device=device),
        "ids": torch.randint(1, N_PLAYERS, (size, 10), device=device),
        "share": share.reshape(size, 50),
        "guard": torch.randn(size, 50, len(GUARD_ATTRIBUTES), device=device),
        "y": torch.randn(size, 2, N_TARGETS, device=device),
        "w": torch.rand(size, 2, N_TARGETS, device=device),
    }


def _step_fn(kind: str, model, batch):
    if kind == "fm":
        return lambda: weighted_mse(
            model(batch["x"], batch["ids"], batch["share"]), batch["y"], batch["w"]
        ).mean()
    nodes = torch.cat(
        [batch["x"], torch.zeros_like(batch["x"][..., :1])], dim=-1
    )  # + is_home
    return lambda: weighted_mse(
        model(nodes, batch["guard"], batch["share"]), batch["y"], batch["w"]
    ).mean()


def measure(kind: str, size, batch_size: int, device: str, steps: int) -> dict:
    torch.manual_seed(0)
    if kind == "fm":
        model = FactorizationMachine(FMConfig(dim=size, edges=False), N_PLAYERS)
    else:
        dim, layers = size
        model = SmokeGNN(
            SmokeConfig(
                n_node_features=N_FEATURES + 1,
                n_labels=N_TARGETS,
                dim=dim,
                layers=layers,
            )
        )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = _batch(batch_size, device)
    loss_fn = _step_fn(kind, model, batch)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def step():
        loss = loss_fn()
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
    grid = [("fm", d) for d in FM_DIMS] + [("gnn", s) for s in GNN_SIZES]
    for device in devices:
        for kind, size in grid:
            for batch_size in BATCHES:
                result = measure(kind, size, batch_size, device, args.steps)
                row = {
                    "device": device,
                    "model": kind,
                    "size": size,
                    "batch": batch_size,
                    **result,
                }
                if not result.get("oom"):
                    rate = result["stints_per_second"]
                    row["projected_hours"] = {
                        k: v * args.epochs / rate / 3600 for k, v in SCHEDULE.items()
                    }
                rows.append(row)
                label = f"{device:4} {kind:3} {str(size):9} batch {batch_size:>5}: "
                if result.get("oom"):
                    print(label + "out of GPU memory", flush=True)
                else:
                    print(
                        label
                        + f"{result['stints_per_second']:>9,.0f} stints/s, "
                        + (
                            f"peak {result['peak_mib']:.0f} MiB, "
                            if "peak_mib" in result
                            else ""
                        )
                        + f"{args.epochs} epochs over all folds "
                        + f"{row['projected_hours']['all_88_folds']:.2f} h",
                        flush=True,
                    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"info": info, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
