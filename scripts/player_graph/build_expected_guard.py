"""Build v0 expected guarding rates for every pair that shared a stint.

For each game date D, the pairs that shared the floor in D's stints get their
expected rate as of D: positions, position prior and pair history all come
from games strictly before D. These are the guard-edge weights of the stint
(training) graphs. Pairs come from the stints, not from ``pair_game``: an
expected rate needs no tracking of the game itself, so games without matchups
(all of 2016-17) get rows too (NaN where there is no history at all).

    python scripts/player_graph/build_expected_guard.py --min-season 2016 --max-season 2025

Needs ``pair_game`` (``build_pair_game.py``) and the box scores (database, or
its CSVs before 2018-19) for the profile positions. Output:
``data/player_graph/expected_guard/v0/season=YYYY.parquet``, each next to a
``season=YYYY.json`` with the parameters that built it. ``--reshrink`` applies
a new k to the stored seasons it is given and updates their own JSON only.

``--exclude-corrupt-matchups`` drops the games of
``matchup_defense.doubled_exposure_games`` from the pair history
(``PointInTimeData.without_corrupt_matchups``) and writes to
``data/player_graph/expected_guard/v0_without_corrupt_matchups/`` with the same
parameters, for the phase 4B sensitivity check
(``expected_guard_sensitivity.py``) and phase 5 onward. The stored v0 is not
touched. Every season's JSON lists the corrupt games of the whole stored pair
history (``CORRUPT_SCAN_SEASONS``), not only those inside its own window, so
seasons built separately (2016-17 and 2017-18 for phase 6) carry the same
record and ``read_seasons`` reads them together.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.player_graph.as_of import (
    CORRECTED_MATCHUP_HISTORY,
    FIRST_SEASON,
    ORIGINAL_MATCHUP_HISTORY,
    PointInTimeData,
)
from nba_ou.data_processing.player_graph.expected_guard import (
    ExpectedGuardParams,
    expected_guard,
    provider_metadata,
    season_paths,
    shrink,
    stored_version,
    write_season,
)
from nba_ou.data_processing.player_graph.matchup_defense import (
    doubled_exposure_games,
)
from nba_ou.data_processing.player_graph.pair_game import (
    cofloor_seconds,
    read_pair_game,
)
from nba_ou.data_processing.player_graph.positions import positions_as_of

#: Seasons scanned for corrupt matchup games (the stored pair history,
#: 2017-18 -> 2025-26).
CORRUPT_SCAN_SEASONS = range(2017, 2026)


def build_season(
    data: PointInTimeData, season: int, params: ExpectedGuardParams
) -> pd.DataFrame:
    stints = data.stints.loc[data.stints["season_year"].eq(season)]
    if stints.empty:
        return pd.DataFrame()
    cofloor = cofloor_seconds(stints)
    pairs = pd.concat(
        [
            cofloor.rename(
                columns={
                    "home_player_id": "off_player_id",
                    "away_player_id": "def_player_id",
                }
            ),
            cofloor.rename(
                columns={
                    "away_player_id": "off_player_id",
                    "home_player_id": "def_player_id",
                }
            ),
        ],
        ignore_index=True,
    )[["game_id", "off_player_id", "def_player_id"]]
    dates = stints.drop_duplicates("game_id").set_index("game_id")["game_date"]
    pairs["game_date"] = pairs["game_id"].map(dates)
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
    for season in seasons:
        table_path, metadata_path = season_paths(out_dir, season)
        table = pd.read_parquet(table_path)
        metadata = json.loads(metadata_path.read_text())
        table["r_hat"], table["prior_weight"] = shrink(
            table["hist_matchup_seconds_decayed"],
            table["hist_cofloor_seconds_decayed"],
            table["r_prior"],
            k,
        )
        print(f"{season}: k {metadata['k']:g} -> {k:g} ({len(table):,} pairs)")
        write_season(out_dir, season, table, {**metadata, "k": k})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--min-season", type=int, default=FIRST_SEASON)
    parser.add_argument("--max-season", type=int, required=True)
    defaults = ExpectedGuardParams()
    parser.add_argument("--k", type=float, default=defaults.k)
    parser.add_argument("--half-life-days", type=float, default=defaults.half_life_days)
    parser.add_argument("--window-days", type=int, default=defaults.window_days)
    parser.add_argument(
        "--exclude-corrupt-matchups",
        action="store_true",
        help="Drop doubled-exposure games from the pair history (separate output)",
    )
    parser.add_argument(
        "--reshrink",
        action="store_true",
        help="Only recompute r_hat and prior_weight of the stored tables for --k",
    )
    args = parser.parse_args()
    params = ExpectedGuardParams(args.k, args.half_life_days, args.window_days)
    history = (
        CORRECTED_MATCHUP_HISTORY
        if args.exclude_corrupt_matchups
        else ORIGINAL_MATCHUP_HISTORY
    )
    out_dir = (
        args.local_root / "player_graph" / "expected_guard" / stored_version(history)
    )
    if args.reshrink:
        reshrink(out_dir, range(args.min_season, args.max_season + 1), args.k)
        return

    # History reaches back one window before the first season built.
    first = args.min_season - -(-params.window_days // 365)
    data = PointInTimeData.load(
        range(max(first, FIRST_SEASON), args.max_season + 1), local_root=args.local_root
    )
    excluded: set[str] = set()
    if args.exclude_corrupt_matchups:
        data, dropped = data.without_corrupt_matchups()
        # Game-level detector: the games dropped from the loaded window are
        # exactly the stored history's corrupt games inside it.
        excluded = doubled_exposure_games(
            read_pair_game(CORRUPT_SCAN_SEASONS, local_root=args.local_root)
        )
        if not dropped <= excluded:
            raise SystemExit(f"Unexpected corrupt games: {sorted(dropped - excluded)}")
    metadata = provider_metadata(params, data.matchup_history)
    if excluded:
        metadata["excluded_matchup_games"] = sorted(excluded)
        print(
            f"pair history without {len(excluded)} corrupt matchup games "
            f"({len(dropped)} inside the loaded seasons)",
            flush=True,
        )
    for season in range(args.min_season, args.max_season + 1):
        started = time.time()
        table = build_season(data, season, params)
        write_season(out_dir, season, table, metadata)
        path, _ = season_paths(out_dir, season)
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
