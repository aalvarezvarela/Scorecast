"""Pre-registered reading of the pregame graph campaign (arms A-E, not run yet).

Arms (``experiments/pregame_graph_v1_closing/``, decided 2026-10-15):

A. 2_5 (phase 5's arm A); B. 2_5 + the frozen phase 5 PI primary block;
C. B + the learned style outputs and their absence shifts (``GE_STYLE_*``);
D. C + the node-only neural embedding (``GE_NODE_EMB_*``);
E. C + the full graph embedding (``GE_GRAPH_EMB_*``).

Comparisons, each read with phase 5's rule (``report_pi_walk_forward``: mean
of ``|e_worse| - |e_better|`` over games, each arm's prediction the mean of its
ten seeds, SE clustered by game date; **improves** if the 95% interval is
above 0 and at least 8 of the 10 per-seed differences are positive, **hurts**
if the mirror, else **null**):

* B vs A (phase 5's question again, on this file);
* **C vs B**: do learned style outputs add to the PI block;
* **D vs C**: does a neural embedding add (capacity, no relations);
* **E vs D**: do graph relations add beyond neural capacity (the graph
  question);
* E vs C.

Each pair must differ by exactly the expected column family (checked from the
runs' feature lists). Nothing here selects an arm; five comparisons on one
window are read as a family (no single result is a confirmation on its own).

    python scripts/player_graph/report_pregame_walk_forward.py \\
        --out artifacts/experiments/pregame_graph_v1
"""

from __future__ import annotations

import argparse
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

_spec = spec_from_file_location(
    "report_pi_walk_forward", Path(__file__).with_name("report_pi_walk_forward.py")
)
phase5 = module_from_spec(_spec)
_spec.loader.exec_module(phase5)

ARMS = ("a_baseline", "b_pi_primary", "c_style", "d_node_emb", "e_graph_emb")
#: (better, worse, column family the better arm adds)
COMPARISONS = (
    ("b_pi_primary", "a_baseline", "PI_"),
    ("c_style", "b_pi_primary", "GE_STYLE_"),
    ("d_node_emb", "c_style", "GE_NODE_EMB_"),
    ("e_graph_emb", "d_node_emb", None),  # swaps GE_NODE_EMB_ for GE_GRAPH_EMB_
    ("e_graph_emb", "c_style", "GE_GRAPH_EMB_"),
)
SEEDS = phase5.SEEDS


def feature_set(out: Path, arm: str) -> frozenset:
    names = {
        frozenset(json.loads(p.read_text())["feature_names"])
        for p in sorted((out / arm).glob("seed=*/run.json"))
    }
    if len(names) != 1:
        raise SystemExit(f"{arm}: seeds ran with different (or no) feature sets")
    return next(iter(names))


def check_pair(better: frozenset, worse: frozenset, family: str | None) -> list[str]:
    if family is None:  # E vs D: same base, node embedding swapped for graph
        base_b = {f for f in better if not f.startswith("GE_GRAPH_EMB_")}
        base_w = {f for f in worse if not f.startswith("GE_NODE_EMB_")}
        if base_b != base_w:
            raise SystemExit("E and D differ beyond the embedding family")
        return sorted(better - worse)
    if not worse <= better:
        raise SystemExit(f"the {family} arm drops columns of its reference")
    added = sorted(better - worse)
    if not added or any(not f.startswith(family) for f in added):
        raise SystemExit(f"the {family} arm adds {added[:3]}")
    return added


def compare(rows, better, worse, seeds):
    keys = {
        arm: {
            seed: frozenset(part["GAME_ID"]) for seed, part in rows[arm].groupby("seed")
        }
        for arm in (better, worse)
    }
    common = frozenset.intersection(*(g for k in keys.values() for g in k.values()))
    wide = {
        arm: rows[arm]
        .loc[rows[arm]["GAME_ID"].isin(common)]
        .pivot_table(index="GAME_ID", columns="seed", values="y_pred")
        for arm in (better, worse)
    }
    info = (
        rows[worse]
        .loc[rows[worse]["GAME_ID"].isin(common)]
        .drop_duplicates("GAME_ID")
        .set_index("GAME_ID")[["date", "y_true"]]
        .loc[wide[worse].index]
    )
    info["date"] = pd.to_datetime(info["date"]).dt.normalize()
    y = info["y_true"]
    e_w = (wide[worse][seeds].mean(axis=1) - y).abs()
    e_b = (wide[better][seeds].mean(axis=1) - y).abs()
    mean, se = phase5.clustered_mean((e_w - e_b).to_numpy(), info["date"].to_numpy())
    per_seed = np.array(
        [
            (wide[worse][s] - y).abs().mean() - (wide[better][s] - y).abs().mean()
            for s in seeds
        ]
    )
    low, high = mean - 1.96 * se, mean + 1.96 * se
    seasons = phase5.season_of(info["date"])
    by_season = {}
    for season, idx in info.groupby(seasons).groups.items():
        m, s = phase5.clustered_mean(
            (e_w - e_b).loc[idx].to_numpy(), info.loc[idx, "date"].to_numpy()
        )
        by_season[int(season)] = {"delta": m, "se": s, "games": len(idx)}
    return {
        "games": int(len(info)),
        "delta_abs_error": mean,
        "se": se,
        "ci95": [low, high],
        "per_seed": per_seed.round(5).tolist(),
        "seeds_positive": int((per_seed > 0).sum()),
        "decision": phase5.decide(low, high, per_seed),
        "by_season": by_season,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/experiments/pregame_graph_v1")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    devices = {
        json.loads(p.read_text())["device"]
        for arm in ARMS
        for p in (args.out / arm).glob("seed=*/run.json")
    }
    if len(devices) > 1:
        raise SystemExit(f"runs used different devices: {devices}")
    features = {arm: feature_set(args.out, arm) for arm in ARMS}
    rows = {arm: phase5.load(args.out, arm, args.seeds) for arm in ARMS}
    results = {}
    lines = ["# Pregame graph campaign v1: pre-registered comparisons\n"]
    for better, worse, family in COMPARISONS:
        added = check_pair(features[better], features[worse], family)
        result = compare(rows, better, worse, sorted(args.seeds))
        result["added_columns"] = len(added)
        results[f"{better} vs {worse}"] = result
        lines.append(
            f"- **{better} vs {worse}** ({len(added)} columns): "
            f"{result['delta_abs_error']:+.4f} ± {result['se']:.4f} "
            f"(95% {result['ci95'][0]:+.4f} .. {result['ci95'][1]:+.4f}), "
            f"{result['seeds_positive']}/{len(args.seeds)} seeds positive -> "
            f"**{result['decision']}**"
        )
    report = args.out / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "metrics.json").write_text(
        json.dumps({"devices": sorted(devices), "comparisons": results}, indent=2)
        + "\n"
    )
    (report / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
