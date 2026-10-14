"""Build the phase 6 edge-level target table (labels only), 2017-18 onward.

Writes ``data/player_graph/pretraining/<version>/edges/season=YYYY.parquet``
(``nba_ou.data_processing.player_graph.edge_targets``) and records it under
``edges`` in the dataset's ``manifest.json``. Run after
``build_pretraining_dataset.py``: every attacker-game is checked against the
stored node features (attacker and matched defenders as of the game date).

The 69 corrupt matchup games (detected over the whole stored pair history)
and the two stint-rejected games have no rows.

    python scripts/player_graph/build_edge_targets.py --max-season 2025

Reads ``pair_game`` and the pretraining tables; no database.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.player_graph.edge_targets import (
    DIAGNOSTICS,
    TARGETS,
    check_edges,
    edge_table,
    to_edge_batch,
)
from nba_ou.data_processing.player_graph.matchup_defense import (
    doubled_exposure_games,
)
from nba_ou.data_processing.player_graph.pair_game import read_pair_game
from nba_ou.data_processing.player_graph.pretraining_data import (
    FIRST_MATCHUP_SEASON,
    NODE_COLUMNS,
    ROOT,
    STINT_REJECTED_GAMES,
    VERSION,
    read_manifest,
    season_path,
    write_manifest,
)

#: Seasons scanned for corrupt matchup games (the stored pair history).
CORRUPT_SCAN_SEASONS = range(2017, 2026)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--min-season", type=int, default=FIRST_MATCHUP_SEASON)
    parser.add_argument("--max-season", type=int, required=True)
    args = parser.parse_args()
    if args.min_season < FIRST_MATCHUP_SEASON:
        raise SystemExit(f"No matchups before {FIRST_MATCHUP_SEASON}")
    root = args.local_root / ROOT / VERSION
    manifest = read_manifest(root)
    corrupt = doubled_exposure_games(
        read_pair_game(CORRUPT_SCAN_SEASONS, local_root=args.local_root)
    )
    recorded = set(manifest["exclusions"]["corrupt_matchup_games"]["games"])
    if corrupt != recorded:
        raise SystemExit("Corrupt games differ from the dataset's record")
    excluded = corrupt | set(STINT_REJECTED_GAMES)
    seasons = {}
    for season in range(args.min_season, args.max_season + 1):
        pairs = read_pair_game([season], local_root=args.local_root)
        edges = edge_table(pairs, excluded)
        summary = check_edges(edges, excluded)
        nodes = pd.read_parquet(season_path(root, "nodes", season))
        _, dropped = to_edge_batch(edges, nodes, NODE_COLUMNS)
        stints = pd.read_parquet(
            season_path(root, "stints", season), columns=["game_id"]
        )
        without_stints = set(edges["game_id"]) - set(stints["game_id"])
        if without_stints:
            raise SystemExit(
                f"{season}: edge games without stints {sorted(without_stints)[:3]}"
            )
        path = season_path(root, "edges", season)
        path.parent.mkdir(parents=True, exist_ok=True)
        edges.to_parquet(path, index=False)
        removed = sorted(excluded & set(pairs["game_id"].astype(str)))
        seasons[str(season)] = {
            **summary,
            "excluded_games_in_season": removed,
            "dropped_without_node_features": dropped,
        }
        means = ", ".join(f"{k} {v:.3f}" for k, v in summary["target_means"].items())
        print(
            f"{season}: {summary['attacker_games']:,} attacker-games in "
            f"{summary['games']} games ({len(removed)} excluded), {means}; "
            f"dropped without node features {dropped['attacker_games']}",
            flush=True,
        )
    manifest["edges"] = {
        "module": "nba_ou.data_processing.player_graph.edge_targets",
        "targets": list(TARGETS),
        "attribution": "time: attacker-game rate, scored as sum_j s_ij f(i, j) "
        "with s_ij the defender's share of the attacker's partial possessions",
        "exposure": "attacker partial possessions",
        "diagnostics": list(DIAGNOSTICS),
        "labels_only": True,
        "first_season": FIRST_MATCHUP_SEASON,
        "excluded_games": sorted(excluded),
        "seasons": seasons,
    }
    write_manifest(root, manifest)


if __name__ == "__main__":
    main()
