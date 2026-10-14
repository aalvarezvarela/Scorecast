"""Build the phase 6 stint-graph pretraining dataset, one season at a time.

Writes ``data/player_graph/pretraining/<version>/{stints,nodes,guard}/
season=YYYY.parquet`` and ``manifest.json``
(``nba_ou.data_processing.player_graph.pretraining_data``). Every season is
checked before it is recorded:

* the table checks (``check_tables``): ten distinct players per stint, a node
  row as of every stint date, a guard row for every pair on the floor, no
  observed or betting column in an input table, structural rates only on dates
  without any matchup history, stored targets equal to the counts, no
  stint-rejected game;
* **as of the date, recomputed**: on ``--check-dates`` dates of the season
  the node features are rebuilt from scratch (``node_profiles`` with the
  box-start positions) and the expected guarding rates with
  ``expected_guard`` from the corrected history, both from the as-of view, and
  compared with the stored rows.

    python scripts/player_graph/build_pretraining_dataset.py --max-season 2025

Needs the database (games table, box scores), the stint store, the matchup
store, ``pair_game``, the v0 node profiles and
``expected_guard/v0_without_corrupt_matchups`` for every season. A season
already written is skipped unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.player_graph.as_of import (
    CORRECTED_MATCHUP_HISTORY,
    PointInTimeData,
)
from nba_ou.data_processing.player_graph.expected_guard import (
    DEFAULT_PARAMS as GUARD_PARAMS,
)
from nba_ou.data_processing.player_graph.expected_guard import (
    expected_guard,
    read_seasons,
    stored_version,
)
from nba_ou.data_processing.player_graph.node_profiles import (
    PROFILE_COLUMNS,
    node_profiles,
)
from nba_ou.data_processing.player_graph.positions import positions_as_of
from nba_ou.data_processing.player_graph.pretraining_data import (
    FIRST_MATCHUP_SEASON,
    FIRST_SEASON,
    GUARD_SOURCES,
    LABELS,
    NODE_COLUMNS,
    ROOT,
    STINT_REJECTED_GAMES,
    VERSION,
    build_season,
    check_tables,
    directed_pairs,
    read_manifest,
    read_tables,
    season_path,
    write_manifest,
    write_season,
)

TOLERANCE = 1e-9


def check_dates(stints: pd.DataFrame, count: int) -> list[pd.Timestamp]:
    """``count`` dates spread over the season (never its first, where
    everything is prior)."""
    dates = sorted(pd.to_datetime(stints["game_date"]).unique())
    picks = np.linspace(1, len(dates) - 1, count).round().astype(int)
    return [pd.Timestamp(dates[i]) for i in sorted(set(picks))]


def recompute(
    data: PointInTimeData, tables: dict[str, pd.DataFrame], date: pd.Timestamp
) -> dict:
    """Max differences between stored rows of ``date`` and a rebuild from the
    as-of view."""
    view = data.as_of(date)
    nodes = tables["nodes"].loc[pd.to_datetime(tables["nodes"]["as_of_date"]).eq(date)]
    players = pd.Index(nodes["player_id"].astype(str))
    positions = positions_as_of(
        view, window_days=GUARD_PARAMS.window_days, box_starts=True
    )
    fresh = node_profiles(view, players, positions=positions)
    stored = nodes.set_index(nodes["player_id"].astype(str))
    profile_diff = np.nanmax(
        np.abs(
            fresh[list(PROFILE_COLUMNS)].astype(float).to_numpy()
            - stored.loc[fresh.index, list(PROFILE_COLUMNS)].astype(float).to_numpy()
        )
    )
    nan_agree = (
        fresh[list(PROFILE_COLUMNS)].isna().to_numpy()
        == stored.loc[fresh.index, list(PROFILE_COLUMNS)].isna().to_numpy()
    ).all()
    starts_equal = (
        positions["n_starts"].reindex(players).fillna(0).astype(int).to_numpy()
        == stored.loc[players, "position_starts"].to_numpy()
    ).all()
    day = tables["stints"].loc[pd.to_datetime(tables["stints"]["game_date"]).eq(date)]
    lineups = day.assign(
        home_lineup=list(day[[f"home_{k}" for k in range(5)]].astype(str).to_numpy()),
        away_lineup=list(day[[f"away_{k}" for k in range(5)]].astype(str).to_numpy()),
    )
    pairs = directed_pairs(lineups)
    rebuilt = expected_guard(view, pairs, params=GUARD_PARAMS)
    keys = ["game_id", "off_player_id", "def_player_id"]
    joined = rebuilt.merge(tables["guard"], on=keys, suffixes=("", "_stored"))
    guard_diff = {
        column: float(
            np.nanmax(np.abs(joined[column] - joined[f"{column}_stored"]))
            if joined[column].notna().any()
            else 0.0
        )
        for column in ("r_hat", "r_prior", "prior_weight")
    }
    unknown_agree = bool(
        (joined["r_hat"].isna() == joined["r_hat_stored"].isna()).all()
    )
    result = {
        "date": str(date.date()),
        "nodes": len(players),
        "pairs": len(pairs),
        "pairs_matched": len(joined),
        "profile_max_abs_diff": float(profile_diff),
        "profile_nan_pattern_equal": bool(nan_agree),
        "position_starts_equal": bool(starts_equal),
        "guard_max_abs_diff": guard_diff,
        "guard_unknown_pattern_equal": unknown_agree,
    }
    result["ok"] = (
        result["pairs_matched"] == result["pairs"]
        and profile_diff <= TOLERANCE
        and nan_agree
        and starts_equal
        and max(guard_diff.values()) <= TOLERANCE
        and unknown_agree
    )
    return result


def git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return out.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--min-season", type=int, default=FIRST_SEASON)
    parser.add_argument("--max-season", type=int, required=True)
    parser.add_argument("--check-dates", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = args.local_root / ROOT / VERSION
    seasons = list(range(args.min_season, args.max_season + 1))
    guard_root = (
        args.local_root
        / "player_graph"
        / "expected_guard"
        / stored_version(CORRECTED_MATCHUP_HISTORY)
    )
    _, guard_metadata = read_seasons(guard_root, seasons)
    if guard_metadata.get("matchup_history") != CORRECTED_MATCHUP_HISTORY:
        raise SystemExit(f"{guard_root} is not the corrected history: {guard_metadata}")
    excluded = set(guard_metadata["excluded_matchup_games"])
    profile_metadata = {
        season: json.loads(
            (
                args.local_root
                / "player_graph"
                / "node_profiles"
                / f"season={season}.json"
            ).read_text()
        )
        for season in seasons
    }

    started = time.time()
    data, dropped = PointInTimeData.load(
        range(FIRST_SEASON, args.max_season + 1), local_root=args.local_root
    ).without_corrupt_matchups()
    if not dropped <= excluded:
        raise SystemExit(f"Corrupt games not in the guard record: {dropped - excluded}")
    print(
        f"loaded {FIRST_SEASON}-{args.max_season}: {len(data.stints):,} stints, "
        f"pair history without {len(dropped)} corrupt games "
        f"({time.time() - started:.0f}s)",
        flush=True,
    )

    manifest_path = root / "manifest.json"
    manifest = read_manifest(root) if manifest_path.exists() else {"seasons": {}}
    for season in seasons:
        if season_path(root, "stints", season).exists() and not args.force:
            print(f"{season}: exists, skipped", flush=True)
            continue
        season_started = time.time()
        expected, _ = read_seasons(guard_root, [season])
        profiles = pd.read_parquet(
            args.local_root
            / "player_graph"
            / "node_profiles"
            / f"season={season}.parquet"
        )
        tables = build_season(data, season, expected, profiles)
        summary = check_tables(tables, excluded)
        stints = tables["stints"]
        as_of = [
            recompute(data, tables, d) for d in check_dates(stints, args.check_dates)
        ]
        failed = [check for check in as_of if not check["ok"]]
        if failed:
            raise SystemExit(f"{season}: as-of recomputation failed: {failed}")
        write_season(root, season, tables)
        manifest["seasons"][str(season)] = {
            **summary,
            "edge_labels": season >= FIRST_MATCHUP_SEASON,
            "as_of_checks": as_of,
            "node_profiles": profile_metadata[season],
            "built_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "commit": git_commit(),
        }
        manifest.update(
            {
                "dataset": "stint_pretraining",
                "version": VERSION,
                "module": "nba_ou.data_processing.player_graph.pretraining_data",
                "first_season": FIRST_SEASON,
                "first_matchup_season": FIRST_MATCHUP_SEASON,
                "labels": list(LABELS),
                "label_exposures": {
                    "pts_per_poss": "possessions",
                    "tov_per_poss": "possessions",
                    "poss_per_48": "stint seconds / 2 per side (stint-level pace)",
                    "fg3a_per_fga": "FGA",
                    "fta_per_fga": "FGA",
                    "oreb_rate": "OREB + opponent DREB",
                },
                "node_columns": list(NODE_COLUMNS),
                "positions": {
                    "box_starts": True,
                    "window_days": GUARD_PARAMS.window_days,
                },
                "guard": {
                    "provider": guard_metadata,
                    "sources": list(GUARD_SOURCES),
                    "structural_rule": "sum_x p_i(x) p_j(x) on the as-of positions",
                },
                "exclusions": {
                    "corrupt_matchup_games": {
                        "games": sorted(excluded),
                        "role": "out of the guard history and of every observed "
                        "guarding label; their stints keep their graphs",
                    },
                    "stint_rejected_games": {
                        "games": sorted(STINT_REJECTED_GAMES),
                        "role": "no stints, no graphs; never reintroduced",
                    },
                },
            }
        )
        write_manifest(root, manifest)
        sources = ", ".join(
            f"{k} {v:.1%}" for k, v in summary["guard_source_share"].items()
        )
        print(
            f"{season}: {summary['stints']:,} stints, {summary['games']} games, "
            f"{summary['node_rows']:,} node rows, guard {sources}; as-of checks "
            f"{len(as_of)} dates ok ({time.time() - season_started:.0f}s)",
            flush=True,
        )
    # Re-read what is stored, so the manifest describes the files.
    tables = read_tables(root, [int(s) for s in manifest["seasons"]])
    print(
        f"stored: {len(tables['stints']):,} stints in {len(manifest['seasons'])} seasons"
    )


if __name__ == "__main__":
    main()
