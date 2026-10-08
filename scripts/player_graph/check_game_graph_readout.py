"""Reproduce the 2_6 lineup projection from v0 game graphs (integration check).

Builds closing game graphs for a date range from the same inputs the 2_6 layer
uses (box-score rosters, the closing injury report, the rating cache), reads
the 2_6 projection off the graphs' node minutes, and compares it with

1. 2_6's own ``project_lineup_games`` on the same rosters: must agree to
   floating-point precision;
2. the 2_6 closing file's ``LU_ABSENCE_IMPACT_PTS_BEFORE`` and
   ``LU_PROJ_POSS_BEFORE``: agree unless the inputs changed since that build
   (the injury report store is live; its digest is printed next to the
   manifest's).

    python scripts/player_graph/check_game_graph_readout.py \\
        --from 2023-10-24 --to 2023-11-30

Needs the database (box scores, injury reports) and the local stores.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.injury_status.report_state import report_state_digest
from nba_ou.data_processing.lineups.availability import (
    player_out_probabilities,
    roster_exclusions,
)
from nba_ou.data_processing.lineups.features import (
    game_nights,
    load_rating_book,
    project_lineup_games,
)
from nba_ou.data_processing.player_graph.as_of import CLOSING, PointInTimeData
from nba_ou.data_processing.player_graph.game_graph import (
    build_game_graphs,
    readout_2_6,
)
from nba_ou.utils.general_utils import get_season_year_from_date

CLOSING_2_6 = Path("data/train_data/closing_line_data_2_6_20261003.parquet")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="start", required=True)
    parser.add_argument("--to", dest="end", required=True)
    parser.add_argument("--file", type=Path, default=CLOSING_2_6)
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)

    file = pd.read_parquet(
        args.file,
        columns=[
            "GAME_ID",
            "GAME_DATE",
            "TEAM_ID_TEAM_HOME",
            "TEAM_ID_TEAM_AWAY",
            "LU_ABSENCE_IMPACT_PTS_BEFORE",
            "LU_PROJ_POSS_BEFORE",
        ],
    )
    file["GAME_DATE"] = pd.to_datetime(file["GAME_DATE"]).dt.normalize()
    file = file.loc[file["GAME_DATE"].between(start, end)]
    games = pd.DataFrame(
        {
            "GAME_ID": file["GAME_ID"].astype(str).str.zfill(10),
            "GAME_DATE": file["GAME_DATE"],
            "HOME_TEAM_ID": file["TEAM_ID_TEAM_HOME"].astype(str),
            "AWAY_TEAM_ID": file["TEAM_ID_TEAM_AWAY"].astype(str),
        }
    )

    started = time.time()
    first = get_season_year_from_date(start)
    seasons = range(max(2016, first - 3), get_season_year_from_date(end) + 1)
    data = PointInTimeData.load(seasons, closing_injuries=True)
    state = data.injury_report(CLOSING)
    p_out = player_out_probabilities(state.statuses)
    excluded = roster_exclusions(state.statuses)
    nights = game_nights(games, data.box_scores, p_out, excluded=excluded)
    print(f"loaded in {time.time() - started:.0f}s; {len(games)} games")

    started = time.time()
    graphs = build_game_graphs(data, games, nights, report_covered=state.covered)
    print(
        f"game graphs in {time.time() - started:.0f}s: "
        f"{graphs.scenarios['game_id'].nunique()} games, {len(graphs.scenarios)} "
        f"graphs, {len(graphs.nodes):,} nodes, {len(graphs.edges):,} edges; "
        f"skipped {graphs.metadata['skipped_games']}"
    )

    book = load_rating_book()
    readout = readout_2_6(graphs, book).set_index("GAME_ID")
    direct = project_lineup_games(games, data.box_scores, book, nights=nights)
    direct = direct.set_index("GAME_ID").reindex(readout.index)
    print("\n1. graph readout vs 2_6 project_lineup_games on the same rosters:")
    for column, key in (
        ("raw_total", "raw_total"),
        ("possessions", "possessions"),
        ("impact_points", "impact_points"),
    ):
        diff = (readout[column] - direct[key]).abs()
        print(
            f"   {column}: max |diff| {diff.max():.2e} over {diff.notna().sum()} games"
        )

    stored = file.set_index(file["GAME_ID"].astype(str).str.zfill(10))
    file_nan = set(stored.index[stored["LU_ABSENCE_IMPACT_PTS_BEFORE"].isna()])
    no_graph = set(games["GAME_ID"]) - set(readout.index)
    print(
        f"\n   games without a projection: file NaN {len(file_nan)}, no graph "
        f"{len(no_graph)}, in both {len(file_nan & no_graph)}"
    )
    both = readout.join(stored, how="inner").dropna(
        subset=["LU_ABSENCE_IMPACT_PTS_BEFORE"]
    )
    impact = (both["impact_points"] - both["LU_ABSENCE_IMPACT_PTS_BEFORE"]).abs()
    poss = (both["possessions"] - both["LU_PROJ_POSS_BEFORE"]).abs()
    print("\n2. graph readout vs the 2_6 closing file:")
    print(
        f"   impact: {np.mean(impact < 1e-6):.1%} of {len(both)} games equal "
        f"(max |diff| {impact.max():.3g}); possessions: "
        f"{np.mean(poss < 1e-6):.1%} equal (max |diff| {poss.max():.3g})"
    )
    manifest = json.loads(args.file.with_suffix(".manifest.json").read_text())
    print(
        f"   injury report digest now {report_state_digest(state)[:16]}, "
        f"at build {manifest['build_args']['injury_report_state_sha256'][:16]}"
    )
    if (impact >= 1e-6).any():
        worst = both.assign(diff=impact).sort_values("diff").tail(5)
        print(worst[["impact_points", "LU_ABSENCE_IMPACT_PTS_BEFORE", "diff"]])


if __name__ == "__main__":
    main()
