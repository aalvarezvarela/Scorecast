"""Build the phase 5 feature block v1 (``PI_*``) for closing games.

For each season: the frozen phase 4A v1 provider over the history from
2016-17 (``minutes_provider.run_scenario_provider``), closing game graphs with
its minutes over the **corrected** pair history (``exclude_corrupt_matchups``),
read with the frozen phase 3 rating book (``pi_features.pi_features``). Games
without both teams' closing reports, or without a rating date, get no row.

Writes ``data/player_graph/pi_features/season=YYYY.parquet`` (``GAME_ID`` +
``pi_features.FEATURES``) and ``season=YYYY.json`` with every input that
defines the columns: minutes provider, matchup history, rating book and its
metadata, the low-sample threshold, the commit.

    python scripts/player_graph/build_pi_features.py --seasons 2018

Needs the database (box scores, injury reports) and the local stores.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.lineups.features import game_nights, load_rating_book
from nba_ou.data_processing.player_graph.as_of import (
    CLOSING,
    FIRST_SEASON,
    PointInTimeData,
)
from nba_ou.data_processing.player_graph.game_graph import build_game_graphs
from nba_ou.data_processing.player_graph.minutes_provider import (
    V1Minutes,
    profile_positions,
    run_scenario_provider,
)
from nba_ou.data_processing.player_graph.pi_features import (
    FEATURES,
    LOW_SAMPLE_POSSESSIONS,
    pi_features,
)
from nba_ou.data_processing.player_graph.rating_book import BOOK_PATH
from nba_ou.data_processing.player_graph.rotation import team_game_minutes

CLOSING_2_6 = Path("data/train_data/closing_line_data_2_6_20261003.parquet")
OUT = Path("player_graph") / "pi_features"


def season_range(text: str) -> list[int]:
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def season_games(season: int) -> pd.DataFrame:
    file = pd.read_parquet(
        CLOSING_2_6,
        columns=[
            "GAME_ID",
            "GAME_DATE",
            "SEASON_YEAR",
            "TEAM_ID_TEAM_HOME",
            "TEAM_ID_TEAM_AWAY",
        ],
    )
    file = file.loc[pd.to_numeric(file["SEASON_YEAR"], errors="coerce").eq(season)]
    return pd.DataFrame(
        {
            "GAME_ID": file["GAME_ID"].astype(str).str.zfill(10),
            "GAME_DATE": pd.to_datetime(file["GAME_DATE"]).dt.normalize(),
            "HOME_TEAM_ID": file["TEAM_ID_TEAM_HOME"].astype(str),
            "AWAY_TEAM_ID": file["TEAM_ID_TEAM_AWAY"].astype(str),
        }
    ).drop_duplicates("GAME_ID")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--seasons", type=season_range, required=True)
    args = parser.parse_args()
    started = time.time()
    data = PointInTimeData.load(
        range(FIRST_SEASON, args.seasons[-1] + 1),
        closing_injuries=True,
        exclude_corrupt_matchups=True,
    )
    state = data.injury_report(CLOSING)
    minutes = team_game_minutes(data.stints)
    positions = profile_positions(
        args.local_root, range(FIRST_SEASON, args.seasons[-1] + 1)
    )
    run = run_scenario_provider(data, state, minutes, positions, args.seasons)
    provider = V1Minutes(run.provider.team_scenarios, run.provider.team_nodes)
    book_path = args.local_root / BOOK_PATH
    book_rows = pd.read_parquet(book_path)
    book = load_rating_book(book_path)
    book_metadata = json.loads(book_path.with_suffix(".metadata.json").read_text())
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print(f"inputs and provider in {time.time() - started:.0f}s", flush=True)
    out = args.local_root / OUT
    out.mkdir(parents=True, exist_ok=True)
    for season in args.seasons:
        started = time.time()
        games = season_games(season)
        nights = game_nights(
            games, data.box_scores_2_6, run.p_out, excluded=run.excluded
        )
        graphs = build_game_graphs(
            data, games, nights, report_covered=state.covered, minutes=provider
        )
        features = pi_features(graphs, book, book_rows)
        features.to_parquet(out / f"season={season}.parquet", index=False)
        metadata = {
            "features": list(FEATURES),
            "season": season,
            "games_in_closing_file": int(len(games)),
            "games_with_features": int(len(features)),
            "minutes_provider": graphs.metadata["minutes_provider"],
            "matchup_history": graphs.metadata["matchup_history"],
            "graph_version": graphs.metadata["version"],
            "skipped_games": graphs.metadata["skipped_games"],
            "rating_book": str(BOOK_PATH),
            "rating_book_metadata": book_metadata,
            "low_sample_possessions": LOW_SAMPLE_POSSESSIONS,
            "history_from": FIRST_SEASON,
            "commit": commit,
        }
        (out / f"season={season}.json").write_text(
            json.dumps(metadata, indent=2, default=str) + "\n"
        )
        print(
            f"{season}: {len(features):,} of {len(games):,} games with features "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
