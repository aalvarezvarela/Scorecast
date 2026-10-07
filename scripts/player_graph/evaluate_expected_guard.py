"""Benchmark guard-edge weights against observed in-game guarding shares.

Reports the weighted TVD (see ``nba_ou.data_processing.player_graph.
guard_benchmark``) overall and per season for the baselines (constant rate,
position prior only, pair history only) and a grid of shrinkage strengths k,
plus the error by ``prior_weight`` bucket at one k.

    # v0 k was chosen on 2018-19 only (the 2_7 walk-forward starts in 2019-20)
    python scripts/player_graph/evaluate_expected_guard.py --seasons 2018
    python scripts/player_graph/evaluate_expected_guard.py --seasons 2019-2025

Reads ``pair_game`` and ``expected_guard/<version>``; no database.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.player_graph.guard_benchmark import (
    benchmark_frame,
    estimator_errors,
    prior_weight_buckets,
)
from nba_ou.data_processing.player_graph.pair_game import read_pair_game

DEFAULT_K_GRID = (60.0, 150.0, 300.0, 600.0, 1200.0, 2400.0, 5000.0)


def season_range(text: str) -> list[int]:
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--seasons", type=season_range, required=True, help="2018 or 2019-2025"
    )
    parser.add_argument("--version", default="v0")
    parser.add_argument("--k-grid", type=float, nargs="+", default=list(DEFAULT_K_GRID))
    parser.add_argument(
        "--bucket-k",
        type=float,
        help="k for the prior_weight buckets (default: the table's)",
    )
    args = parser.parse_args()

    root = args.local_root / "player_graph" / "expected_guard" / args.version
    metadata = json.loads((root / "metadata.json").read_text())
    expected = pd.concat(
        [pd.read_parquet(root / f"season={season}.parquet") for season in args.seasons],
        ignore_index=True,
    )
    frame = benchmark_frame(
        read_pair_game(args.seasons, local_root=args.local_root), expected
    )
    bucket_k = args.bucket_k if args.bucket_k is not None else metadata["k"]

    pd.set_option("display.width", 200)
    print(f"provider: {metadata}")
    print(
        f"seasons {args.seasons[0]}-{args.seasons[-1]}: "
        f"{frame.groupby(['game_id', 'off_player_id']).ngroups:,} attacker-games, "
        f"{len(frame):,} pairs"
    )
    print(
        "\nWeighted TVD, expected vs observed in-game guarding shares (lower is better):"
    )
    print(estimator_errors(frame, args.k_grid).round(4).to_string())
    print(f"\nTVD by attacker prior_weight bucket (k={bucket_k:g}):")
    print(prior_weight_buckets(frame, bucket_k).round(3).to_string())


if __name__ == "__main__":
    main()
