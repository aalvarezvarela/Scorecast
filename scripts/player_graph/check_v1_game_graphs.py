"""Integration check of the phase 5 game graphs on 2018-19 (development).

Closing game graphs built from the same history with two minutes providers:

* ``v0`` (``game_graph.V0Minutes``, 2_6's): read out with 2_6's stored rating
  cache, they must still reproduce the 2_6 closing file
  (``LU_ABSENCE_IMPACT_PTS_BEFORE``, ``LU_PROJ_POSS_BEFORE``);
* ``v1`` (``minutes_provider.V1Minutes`` over ``run_scenario_provider``, the
  frozen phase 4A provider): the scenario-weighted minutes of the graphs'
  nodes must equal the provider's own expected minutes and the frozen
  provider's phase 4A pre-game table
  (``pregame_minutes_roster10_season=2018.parquet``, player MAE 4.656), so
  the player MAE against actual minutes on the same rows reproduces 4A's.

Also checked: the graphs record their minutes provider and matchup history
(phase 5 reads the corrected history), every graph's history date is its game
date (history strictly before it), and the phase 3 rating book's fits end
strictly before each of its dates.

    python scripts/player_graph/check_v1_game_graphs.py

Needs the database (box scores, injury reports) and the local stores.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.lineups.features import game_nights, load_rating_book
from nba_ou.data_processing.player_graph.as_of import (
    CLOSING,
    CORRECTED_MATCHUP_HISTORY,
    PointInTimeData,
)
from nba_ou.data_processing.player_graph.game_graph import (
    FULL_HEALTH,
    build_game_graphs,
    readout_2_6,
)
from nba_ou.data_processing.player_graph.minutes_provider import (
    V1Minutes,
    expected_minutes,
    profile_positions,
    run_scenario_provider,
)
from nba_ou.data_processing.player_graph.rating_book import BOOK_PATH
from nba_ou.data_processing.player_graph.rotation import team_game_minutes

SEASON, FIRST = 2018, 2016
CLOSING_2_6 = Path("data/train_data/closing_line_data_2_6_20261003.parquet")
#: The frozen provider's pre-game table (roster rule 10 / 5, phase 4A step 6);
#: the untagged ``pregame_minutes_season=2018.parquet`` predates the rule. It
#: also predates ``participation.ROW_ORDER``: before it, q's fit depended on
#: the hash seed's set order and runs differed by up to ~0.003 minutes, so
#: the table is reproduced to that tolerance and to its printed MAE.
STORED_TOLERANCE = 0.005
PREGAME = Path(
    "data/player_graph/rotation/pregame_minutes_roster10_season=2018.parquet"
)
KEYS = ["game_id", "team_id", "player_id"]


def season_games() -> pd.DataFrame:
    file = pd.read_parquet(
        CLOSING_2_6,
        columns=[
            "GAME_ID",
            "GAME_DATE",
            "SEASON_YEAR",
            "TEAM_ID_TEAM_HOME",
            "TEAM_ID_TEAM_AWAY",
            "LU_ABSENCE_IMPACT_PTS_BEFORE",
            "LU_PROJ_POSS_BEFORE",
        ],
    )
    file = file.loc[pd.to_numeric(file["SEASON_YEAR"], errors="coerce").eq(SEASON)]
    return pd.DataFrame(
        {
            "GAME_ID": file["GAME_ID"].astype(str).str.zfill(10),
            "GAME_DATE": pd.to_datetime(file["GAME_DATE"]).dt.normalize(),
            "HOME_TEAM_ID": file["TEAM_ID_TEAM_HOME"].astype(str),
            "AWAY_TEAM_ID": file["TEAM_ID_TEAM_AWAY"].astype(str),
            "impact_2_6": file["LU_ABSENCE_IMPACT_PTS_BEFORE"].to_numpy(),
            "poss_2_6": file["LU_PROJ_POSS_BEFORE"].to_numpy(),
        }
    ).drop_duplicates("GAME_ID")


def graph_expected_minutes(graphs) -> pd.DataFrame:
    """Scenario-weighted node minutes per (game, side, player), tonight only,
    weights renormalized per game."""
    scen = graphs.scenarios.loc[~graphs.scenarios["is_full_health"].astype(bool)]
    w = scen.set_index(["game_id", "scenario_id"])["weight"]
    w = w / w.groupby(level="game_id").transform("sum")
    nodes = graphs.nodes.join(w.rename("w"), on=["game_id", "scenario_id"]).dropna(
        subset=["w"]
    )
    nodes["weighted"] = nodes["w"] * nodes["minutes"]
    return nodes.groupby(["game_id", "side", "player_id"])["weighted"].sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    started = time.time()
    data = PointInTimeData.load(
        range(FIRST, SEASON + 1), closing_injuries=True, exclude_corrupt_matchups=True
    )
    state = data.injury_report(CLOSING)
    minutes = team_game_minutes(data.stints)
    positions = profile_positions(args.local_root, range(FIRST, SEASON + 1))
    run = run_scenario_provider(data, state, minutes, positions, [SEASON])
    v1 = V1Minutes(run.provider.team_scenarios, run.provider.team_nodes)
    games = season_games()
    nights = game_nights(
        games[["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID"]],
        data.box_scores_2_6,
        run.p_out,
        excluded=run.excluded,
    )
    graphs = {
        "v0": build_game_graphs(data, games, nights, report_covered=state.covered),
        "v1": build_game_graphs(
            data, games, nights, report_covered=state.covered, minutes=v1
        ),
    }
    print(f"inputs and graphs in {time.time() - started:.0f}s")
    failures = []

    def check(name: str, ok: bool, detail: str) -> None:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    print("\n1. METADATA AND AS-OF")
    for name, g in graphs.items():
        meta = g.metadata
        check(
            f"{name} metadata",
            meta["minutes_provider"] == name
            and meta["matchup_history"] == CORRECTED_MATCHUP_HISTORY,
            f"minutes_provider {meta['minutes_provider']}, matchup_history "
            f"{meta['matchup_history']}, skipped {meta['skipped_games']}",
        )
        same_day = g.scenarios["as_of_date"].eq(g.scenarios["game_date"]).all()
        check(
            f"{name} history date",
            bool(same_day),
            "every graph reads history strictly before its game date",
        )
    book = pd.read_parquet(args.local_root / BOOK_PATH)
    check(
        "phase 3 book",
        bool((book["fit_max_game_date"] < book["as_of_date"]).all()),
        f"{len(book):,} rows, every fit ends before its date",
    )

    print("\n2. v0 REPRODUCES 2_6 (stored 2_6 rating cache)")
    readout = readout_2_6(graphs["v0"], load_rating_book()).set_index("GAME_ID")
    reference = games.set_index("GAME_ID").reindex(readout.index)
    impact = (readout["impact_points"] - reference["impact_2_6"]).abs().max()
    poss = (readout["possessions"] - reference["poss_2_6"]).abs().max()
    check(
        "v0 vs 2_6 closing file",
        impact < 1e-9 and poss < 1e-9,
        f"{len(readout)} games, max |impact diff| {impact:.2e}, max |possessions "
        f"diff| {poss:.2e}",
    )

    print("\n3. v1 REPRODUCES PHASE 4A")
    graph_minutes = graph_expected_minutes(graphs["v1"]).rename("graph").reset_index()
    sides = (
        graphs["v1"]
        .scenarios[["game_id"]]
        .drop_duplicates()
        .merge(games, left_on="game_id", right_on="GAME_ID")
    )
    team_of = pd.concat(
        [
            sides[["game_id", "HOME_TEAM_ID"]]
            .rename(columns={"HOME_TEAM_ID": "team_id"})
            .assign(side="home"),
            sides[["game_id", "AWAY_TEAM_ID"]]
            .rename(columns={"AWAY_TEAM_ID": "team_id"})
            .assign(side="away"),
        ]
    )
    graph_minutes = graph_minutes.merge(team_of, on=["game_id", "side"])
    provider = expected_minutes(run.provider.team_scenarios, run.provider.team_nodes)
    joined = graph_minutes.merge(
        provider.rename(columns={"expected": "provider"}), on=KEYS, how="outer"
    ).fillna(0.0)
    joined = joined.loc[joined["game_id"].isin(set(graph_minutes["game_id"]))]
    diff = (joined["graph"] - joined["provider"]).abs().max()
    check(
        "graph vs provider expected minutes",
        diff < 1e-9,
        f"{graph_minutes['game_id'].nunique()} games, {len(joined):,} player rows, "
        f"max |diff| {diff:.2e}",
    )
    published = pd.read_parquet(PREGAME)
    table = published.merge(graph_minutes[[*KEYS, "graph"]], on=KEYS, how="left")
    table["graph"] = table["graph"].fillna(0.0)
    diff = (table["graph"] - table["v1"]).abs().max()
    mae_graph = (table["graph"] - table["actual"]).abs().mean()
    mae_published = (table["v1"] - table["actual"]).abs().mean()
    check(
        "graph vs published 4A pre-game table",
        diff <= STORED_TOLERANCE and round(mae_graph, 3) == round(mae_published, 3),
        f"{len(table):,} rows, max |diff| {diff:.2e}; player MAE graph "
        f"{mae_graph:.3f}, published {mae_published:.3f}",
    )
    healthy = graphs["v1"].scenarios["scenario_id"].eq(FULL_HEALTH).sum()
    print(
        f"\n  v1 graphs: {graphs['v1'].scenarios['game_id'].nunique()} games, "
        f"{healthy} full-health graphs, {len(graphs['v1'].edges):,} edges"
    )
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'FAILED: {failures}'}")
    print(f"{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
