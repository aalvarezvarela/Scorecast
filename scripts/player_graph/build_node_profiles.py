"""Build as-of node profiles, one Parquet a season, keyed (as_of_date, player_id).

For every game date of a season and the day before it (the history date of the
early intermediate snapshots), the profile of every player seen in the window
(box scores of counted games, matchups or stints) or playing that date, as of
that date, from games strictly before it
(``nba_ou.data_processing.player_graph.node_profiles``).

    python scripts/player_graph/build_node_profiles.py --min-season 2016 --max-season 2025

Needs the database (box scores). Output:
``data/player_graph/node_profiles/season=YYYY.parquet`` and ``season=YYYY.json``
with the parameters.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.player_graph.as_of import PointInTimeData
from nba_ou.data_processing.player_graph.node_profiles import (
    DEFAULT_PARAMS,
    counted_box_scores,
    node_profiles,
)

VERSION = "v0"


def season_dates(data: PointInTimeData, season: int) -> list[pd.Timestamp]:
    stints = data.stints.loc[data.stints["season_year"].eq(season), "game_date"]
    dates = pd.DatetimeIndex(stints.unique())
    return sorted(set(dates) | set(dates - pd.Timedelta(days=1)))


def players_in_window(view, window_days: int) -> pd.Index:
    since = view.cutoff - pd.Timedelta(days=window_days)
    stints = view.stints(since=since)
    lineups = [
        player
        for column in ("home_lineup", "away_lineup")
        for lineup in stints.get(column, [])
        for player in lineup
    ]
    matchups = view.matchups(since=since)
    # Only games a profile counts: preseason / All-Star rows and zero-minute
    # rows would add players whose profile is just the prior.
    box = counted_box_scores(view.box_scores(since=since))
    box = box.loc[box["MIN"] > 0]
    return pd.Index(
        sorted(
            set(map(str, lineups))
            | set(matchups.get("off_player_id", pd.Series(dtype=str)).astype(str))
            | set(matchups.get("def_player_id", pd.Series(dtype=str)).astype(str))
            | set(box.get("PLAYER_ID", pd.Series(dtype=str)).astype(str))
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--min-season", type=int, default=2016)
    parser.add_argument("--max-season", type=int, required=True)
    args = parser.parse_args()
    params = DEFAULT_PARAMS
    history_years = -(-params.window_days // 365) + 1
    data = PointInTimeData.load(
        range(max(2016, args.min_season - history_years), args.max_season + 1),
        local_root=args.local_root,
    )
    out_dir = args.local_root / "player_graph" / "node_profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    for season in range(args.min_season, args.max_season + 1):
        started = time.time()
        frames = []
        stints = data.stints.loc[data.stints["season_year"].eq(season)]
        playing = {
            date: {
                p
                for lineup in (*day["home_lineup"], *day["away_lineup"])
                for p in lineup
            }
            for date, day in stints.groupby("game_date")
        }
        for date in season_dates(data, season):
            view = data.as_of(date)
            # The graphs' own nodes too (debuts, the first games in the data):
            # their profile is still read strictly before the date.
            players = players_in_window(view, params.window_days).union(
                pd.Index(sorted(playing.get(date, set())))
            )
            profiles = node_profiles(view, players, params)
            frames.append(profiles.reset_index().assign(as_of_date=date))
        table = pd.concat(frames, ignore_index=True)
        table.to_parquet(out_dir / f"season={season}.parquet", index=False)
        (out_dir / f"season={season}.json").write_text(
            json.dumps({"version": VERSION, **asdict(params)}, indent=2) + "\n"
        )
        # Coverage where it matters: the players on the season's stints, as of
        # each of their game dates.
        nodes = pd.DataFrame(
            [
                (date, player)
                for date, home, away in zip(
                    stints["game_date"],
                    stints["home_lineup"],
                    stints["away_lineup"],
                    strict=True,
                )
                for player in (*home, *away)
            ],
            columns=["as_of_date", "player_id"],
        ).drop_duplicates()
        joined = nodes.merge(table, on=["as_of_date", "player_id"], how="left")
        print(
            f"{season}: {table['as_of_date'].nunique()} dates, {len(table):,} rows; "
            f"stint nodes with a profile {joined['prior_weight'].notna().mean():.1%}, "
            f"with box history {joined['has_box_history'].eq(True).mean():.1%} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
