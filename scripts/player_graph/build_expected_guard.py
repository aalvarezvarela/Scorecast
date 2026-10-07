"""Build v0 expected guarding rates for every pair that shared a stint.

For each game date D, the pairs that shared the floor in D's games get their
expected rate as of D: positions, position prior and pair history all come
from games strictly before D. These are the guard-edge weights of the stint
(training) graphs.

    python scripts/player_graph/build_expected_guard.py --min-season 2017 --max-season 2025

Needs ``pair_game`` (``build_pair_game.py``) and the database (box scores for
the profile positions). Output:
``data/player_graph/expected_guard/v0/season=YYYY.parquet`` plus
``metadata.json`` with the provider parameters.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.player_graph.as_of import PointInTimeData
from nba_ou.data_processing.player_graph.expected_guard import (
    VERSION,
    ExpectedGuardParams,
    expected_guard,
    provider_metadata,
    shrink,
)
from nba_ou.data_processing.player_graph.positions import positions_as_of


def build_season(
    data: PointInTimeData, season: int, params: ExpectedGuardParams
) -> pd.DataFrame:
    pairs = data.pair_game
    pairs = pairs.loc[
        pairs["season_year"].eq(season) & pairs["cofloor_seconds"].gt(0),
        ["game_id", "game_date", "off_player_id", "def_player_id"],
    ]
    frames = []
    for date, today in pairs.groupby("game_date", sort=True):
        view = data.as_of(date)
        positions = positions_as_of(view, window_days=params.window_days)
        frames.append(
            expected_guard(
                view,
                today.drop(columns="game_date"),
                params=params,
                positions=positions,
            ).assign(game_date=date)
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def reshrink(out_dir: Path, seasons: range, k: float) -> None:
    """Apply a new k to stored tables; the decayed evidence and prior are kept.

    Half-life and window are part of the evidence columns, so changing them
    needs a full rebuild.
    """
    metadata_path = out_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    for season in seasons:
        path = out_dir / f"season={season}.parquet"
        table = pd.read_parquet(path)
        table["r_hat"], table["prior_weight"] = shrink(
            table["hist_matchup_seconds_decayed"],
            table["hist_cofloor_seconds_decayed"],
            table["r_prior"],
            k,
        )
        table.to_parquet(path, index=False)
        print(f"{season}: k {metadata['k']:g} -> {k:g} ({len(table):,} pairs)")
    metadata["k"] = k
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--min-season", type=int, default=2017)
    parser.add_argument("--max-season", type=int, required=True)
    defaults = ExpectedGuardParams()
    parser.add_argument("--k", type=float, default=defaults.k)
    parser.add_argument("--half-life-days", type=float, default=defaults.half_life_days)
    parser.add_argument("--window-days", type=int, default=defaults.window_days)
    parser.add_argument(
        "--reshrink",
        action="store_true",
        help="Only recompute r_hat and prior_weight of the stored tables for --k",
    )
    args = parser.parse_args()
    params = ExpectedGuardParams(args.k, args.half_life_days, args.window_days)
    out_dir = args.local_root / "player_graph" / "expected_guard" / VERSION
    if args.reshrink:
        reshrink(out_dir, range(args.min_season, args.max_season + 1), args.k)
        return

    # History reaches back one window before the first season built.
    first = args.min_season - -(-params.window_days // 365)
    data = PointInTimeData.load(
        range(max(first, 2017), args.max_season + 1), local_root=args.local_root
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(
        json.dumps(provider_metadata(params), indent=2) + "\n"
    )
    for season in range(args.min_season, args.max_season + 1):
        started = time.time()
        table = build_season(data, season, params)
        path = out_dir / f"season={season}.parquet"
        table.to_parquet(path, index=False)
        no_history = (~table["has_pair_history"]).mean() if len(table) else float("nan")
        print(
            f"{season}: {table['game_id'].nunique():,} games, {len(table):,} pairs, "
            f"{no_history:.1%} without pair history, "
            f"median prior_weight {table['prior_weight'].median():.2f} "
            f"({time.time() - started:.0f}s) -> {path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
