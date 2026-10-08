"""Benchmark expected overlaps (pair lift vs independence) on actual shared time.

For every game of the evaluated seasons, each pair's lift is computed as of the
game's date from earlier games only, then compared with the time the pair
actually shared given the players' actual seconds
(``nba_ou.data_processing.player_graph.overlap_benchmark``).

    # v0 k is chosen on 2018-19 only (the 2_7 walk-forward starts in 2019-20)
    python scripts/player_graph/evaluate_overlap.py --seasons 2018
    python scripts/player_graph/evaluate_overlap.py --seasons 2019-2025 --k-grid 1200

Reads only the local stint store.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.lineups.stint_store import read_stints
from nba_ou.data_processing.player_graph.as_of import FIRST_SEASON, PointInTimeData
from nba_ou.data_processing.player_graph.overlap import (
    OverlapParams,
    build_overlap_game,
    expected_lift,
)
from nba_ou.data_processing.player_graph.overlap_benchmark import (
    estimator_lifts,
    overlap_errors,
)

DEFAULT_K_GRID = (150.0, 300.0, 600.0, 1200.0, 2400.0, 5000.0, 10000.0)


def season_range(text: str) -> list[int]:
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def benchmark_pairs(
    seasons: list[int], params: OverlapParams, local_root: Path
) -> pd.DataFrame:
    """Every evaluated pair-game with its as-of evidence columns."""
    history_years = -(-params.window_days // 365)
    loaded = range(max(FIRST_SEASON, seasons[0] - history_years), seasons[-1] + 1)
    stints = read_stints(list(loaded), local_root=local_root)
    overlaps = pd.concat(
        [build_overlap_game(stints.loc[stints["season_year"].eq(s)]) for s in loaded],
        ignore_index=True,
    )
    data = PointInTimeData.from_frames(
        stints, pd.DataFrame(), pd.DataFrame(), overlap_game=overlaps
    )
    targets = data.overlap_game.loc[data.overlap_game["season_year"].isin(seasons)]
    frames = []
    for date, today in targets.groupby("game_date", sort=True):
        frames.append(expected_lift(data.as_of(date), today, params=params))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--seasons", type=season_range, required=True)
    parser.add_argument("--k-grid", type=float, nargs="+", default=list(DEFAULT_K_GRID))
    args = parser.parse_args()

    started = time.time()
    frame = benchmark_pairs(args.seasons, OverlapParams(), args.local_root)
    print(
        f"seasons {args.seasons[0]}-{args.seasons[-1]}: {len(frame):,} pair-games, "
        f"{frame['game_id'].nunique():,} games ({time.time() - started:.0f}s)"
    )
    print(
        "relation lift (league, as of each game): "
        + ", ".join(
            f"{relation} {lift:.3f}"
            for relation, lift in frame.groupby("relation")["relation_lift"]
            .mean()
            .items()
        )
    )
    pd.set_option("display.width", 200)
    rows = {}
    for name, lift in estimator_lifts(frame, args.k_grid).items():
        errors = overlap_errors(frame, lift)
        for (relation, metric), values in errors.iterrows():
            rows[(relation, metric, name)] = values
    table = pd.DataFrame(rows).T.sort_index(level=[0, 1], sort_remaining=False)
    print("\nlower is better:")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
