"""Build the pregame-graph research file and the five arm configs (not a run).

Joins the generated out-of-fold encoder outputs onto phase 5's pinned research
file by ``GAME_ID`` (games without outputs get NaN, as phase 5's PI block):

* ``GE_STYLE_*`` (16) from ``--style-run``: style rates and absence shifts;
* ``GE_NODE_EMB_*`` (16) from ``--node-run``: the node-only embedding;
* ``GE_GRAPH_EMB_*`` (16) from ``--graph-run``: the full graph embedding.

Before writing, every row's encoder checkpoint must be on or before the first
day of the game's month and strictly after nothing it encodes (the monthly
expanding walk-forward), and each output's manifest must name the corrected
matchup history's store. Writes
``data/train_data/closing_line_research_2_6_pi_v1_ge_v1_<date>.parquet`` with
``.provenance.json`` (parent and its checksum, each output run's manifest,
coverage per season), then the five configs under
``experiments/pregame_graph_v1_closing/`` derived from phase 5's two (same
promoted spec, hyperparameters, walk-forward and seeds; only the data file,
its checksum and the exclusions differ):

A ``a_baseline``  phase 5 arm A + ``GE_`` excluded;
B ``b_pi_primary`` phase 5 arm B + ``GE_`` excluded;
C ``c_style``     B, with the style block;
D ``d_node_emb``  C + the node-only embedding;
E ``e_graph_emb`` C + the graph embedding.

    python scripts/player_graph/prepare_pregame_campaign.py \\
        --style-run gnn_node_gen --node-run gnn_node_gen --graph-run gnn_full_gen
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from training_pipeline.data import compute_file_checksum

PARENT = Path("data/train_data/closing_line_research_2_6_pi_v1_20261014.parquet")
PARENT_CHECKSUM = "sha256:32fcdc51ef979b53"
OUTPUTS = Path("data/player_graph/pregame/v1/outputs")
PHASE5 = Path("experiments/pi_block_v1_closing_2026_10")
CAMPAIGN = Path("experiments/pregame_graph_v1_closing")
SEASONS = range(2019, 2026)
FAMILIES = {"style": "GE_STYLE_", "node": "GE_NODE_EMB_", "graph": "GE_GRAPH_EMB_"}
ARMS = {
    "a_baseline": ("a_baseline_2_5.yaml", ["GE_"]),
    "b_pi_primary": ("b_pi_primary.yaml", ["GE_"]),
    "c_style": ("b_pi_primary.yaml", ["GE_NODE_EMB_", "GE_GRAPH_EMB_"]),
    "d_node_emb": ("b_pi_primary.yaml", ["GE_GRAPH_EMB_"]),
    "e_graph_emb": ("b_pi_primary.yaml", ["GE_NODE_EMB_"]),
}


def load_outputs(run: str, family: str) -> tuple[pd.DataFrame, dict]:
    root = OUTPUTS / run
    manifest = json.loads((root / "manifest.json").read_text())
    frames = [pd.read_parquet(root / f"season={s}.parquet") for s in SEASONS]
    table = pd.concat(frames, ignore_index=True)
    table["GAME_ID"] = table["GAME_ID"].astype(str).str.zfill(10)
    dates = pd.to_datetime(table["game_date"]).dt.normalize()
    checkpoints = pd.to_datetime(table["encoder_checkpoint"])
    if (checkpoints > dates).any() or (
        checkpoints != dates.dt.to_period("M").dt.start_time
    ).any():
        raise SystemExit(f"{run}: an encoder checkpoint is not its game's month start")
    columns = [c for c in table.columns if c.startswith(family)]
    if len(columns) != 16:
        raise SystemExit(f"{run}: {len(columns)} {family} columns, expected 16")
    if table["GAME_ID"].duplicated().any():
        raise SystemExit(f"{run}: duplicated games")
    return table.set_index("GAME_ID")[columns], manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style-run", required=True)
    parser.add_argument("--node-run", required=True)
    parser.add_argument("--graph-run", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/train_data"))
    args = parser.parse_args()
    if compute_file_checksum(PARENT) != PARENT_CHECKSUM:
        raise SystemExit("The phase 5 research file is not the pinned one")
    parent = pd.read_parquet(PARENT)
    if any(c.startswith("GE_") for c in parent.columns):
        raise SystemExit("The parent already has GE_ columns")
    keys = parent["GAME_ID"].astype(str).str.zfill(10)
    joined, provenance = parent.copy(), {}
    for kind, run in (
        ("style", args.style_run),
        ("node", args.node_run),
        ("graph", args.graph_run),
    ):
        block, manifest = load_outputs(run, FAMILIES[kind])
        for column in block.columns:
            joined[column] = keys.map(block[column]).to_numpy()
        seasons = pd.to_numeric(parent["SEASON_YEAR"], errors="coerce")
        coverage = joined[block.columns[0]].notna().groupby(seasons).mean().round(4)
        provenance[kind] = {
            "run": run,
            "columns": list(block.columns),
            "coverage_by_season": {int(k): float(v) for k, v in coverage.items()},
            "manifest": manifest,
        }
    stamp = date.today().strftime("%Y%m%d")
    path = args.out_dir / f"closing_line_research_2_6_pi_v1_ge_v1_{stamp}.parquet"
    joined.to_parquet(path, index=False)
    checksum = compute_file_checksum(path)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    path.with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "parent": str(PARENT),
                "parent_checksum": PARENT_CHECKSUM,
                "checksum": checksum,
                "outputs": provenance,
                "commit": commit,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(f"{path}: {len(joined):,} rows, checksum {checksum}")
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    for arm, (source, extra) in ARMS.items():
        config = yaml.safe_load((PHASE5 / source).read_text())
        config["experiment_name"] = f"pregame_graph_v1_closing_line_error_{arm}"
        config["comparison_group"] = "pregame_graph_v1_closing"
        config["training_version"] = f"2.18-pregame-graph-v1-{stamp}"
        config["hypothesis"] = (
            f"arm {arm}: see experiments/pregame_graph_v1_closing/README.md "
            "(pre-registered); phase 5's promoted spec, hyperparameters fixed."
        )
        config["data"]["csv_path"] = str(path)
        config["data"]["data_version"] = f"{stamp}-closing-research-2_6-pi-v1-ge-v1"
        config["data"]["expected_checksum"] = checksum
        excluded = list(config["cleaning"]["exclude_cols_containing"])
        config["cleaning"]["exclude_cols_containing"] = excluded + extra
        (CAMPAIGN / f"{arm}.yaml").write_text(
            f"# pregame_graph_v1_closing / {arm} (generated by "
            "scripts/player_graph/prepare_pregame_campaign.py)\n"
            + yaml.safe_dump(config, sort_keys=False)
        )
    print(f"configs written to {CAMPAIGN}")


if __name__ == "__main__":
    main()
