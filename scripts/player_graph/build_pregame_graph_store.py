"""Store the closing pregame game graphs and their node features (phase 7 input).

The graphs are phase 5's: the frozen phase 4A v1 minutes provider (report
scenarios with ``chance_out``, questionable / doubtful included, plus the
full-health graph), the **corrected** expected-guard history, closing reports
only, games where both teams had filed (``game_graph``). The node features are
the players' as-of profiles on the game date with box-start positions, the
definition the encoders were trained on (``pretraining_data``):
``node_profiles(view, players, positions=positions_as_of(view, box_starts=True))``.

Writes, per season, ``data/player_graph/pregame/v1/{scenarios,nodes,guards,
node_features}/season=YYYY.parquet`` and ``manifest.json`` (provenance,
coverage, checks). Checks before writing: each team's minutes sum to 240 in
every scenario and nobody exceeds 48; every node has features as of its game
date; every opposing pair of a scenario has an expected guard row; no input
table holds an observed or betting column. A game where some team-scenario
cannot fill 240 minutes (fewer than five available players with a minutes
baseline: the provider's infeasible case, e.g. Toronto's 2021-12-26 COVID
roster) is skipped whole and listed in the manifest.

    python scripts/player_graph/build_pregame_graph_store.py --seasons 2018

Needs the database (box scores, injury reports) and the local stores. Reads
nothing dated on or after a game's date for that game.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.lineups.features import game_nights
from nba_ou.data_processing.player_graph.as_of import (
    CLOSING,
    FIRST_SEASON,
    PointInTimeData,
)
from nba_ou.data_processing.player_graph.expected_guard import (
    DEFAULT_PARAMS as GUARD_PARAMS,
)
from nba_ou.data_processing.player_graph.game_graph import (
    FULL_HEALTH,
    build_game_graphs,
)
from nba_ou.data_processing.player_graph.minutes_provider import (
    V1Minutes,
    profile_positions,
    run_scenario_provider,
)
from nba_ou.data_processing.player_graph.node_profiles import node_profiles
from nba_ou.data_processing.player_graph.positions import positions_as_of
from nba_ou.data_processing.player_graph.pretraining_data import NODE_COLUMNS
from nba_ou.data_processing.player_graph.rotation import team_game_minutes
from nba_ou.data_processing.player_graph.stint_graph import check_encoder_inputs

ROOT = Path("player_graph") / "pregame" / "v1"
GUARD_FIELDS = (
    "r_hat",
    "r_hat_known",
    "guard_prior_weight",
    "log_exposure",
    "has_pair_history",
)


def season_range(text: str) -> list[int]:
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def season_games(season: int) -> pd.DataFrame:
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(
        "build_pi_features", Path(__file__).with_name("build_pi_features.py")
    )
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.season_games(season)


def infeasible_games(nodes: pd.DataFrame) -> list[str]:
    """Games with a team-scenario whose minutes cannot reach 240."""
    totals = nodes.groupby(["game_id", "scenario_id", "side"])["minutes"].sum()
    short = totals.loc[(totals - 240.0).abs() > 1e-6]
    return sorted(short.index.get_level_values("game_id").unique())


def node_feature_rows(
    data: PointInTimeData, nodes: pd.DataFrame, scenarios
) -> pd.DataFrame:
    """Raw node features of every graph node as of its game date."""
    dates = scenarios.drop_duplicates("game_id").set_index("game_id")["as_of_date"]
    frame = nodes.assign(as_of_date=nodes["game_id"].map(dates))
    rows = []
    for date, day in frame.groupby("as_of_date", sort=True):
        view = data.as_of(date)
        players = pd.Index(sorted(day["player_id"].astype(str).unique()))
        positions = positions_as_of(
            view, window_days=GUARD_PARAMS.window_days, box_starts=True
        )
        profiles = node_profiles(view, players, positions=positions)
        profiles["position_starts"] = (
            positions["n_starts"].reindex(players).fillna(0).astype(int).to_numpy()
        )
        rows.append(profiles.reset_index().assign(as_of_date=date))
    out = pd.concat(rows, ignore_index=True)
    return out[["as_of_date", "player_id", *NODE_COLUMNS]]


def check(scenarios, nodes, guards, features) -> dict:
    failures = []
    totals = nodes.groupby(["game_id", "scenario_id", "side"])["minutes"].sum()
    off = (totals - 240.0).abs()
    if (off > 1e-6).any():
        failures.append(f"{int((off > 1e-6).sum())} team-scenarios not at 240 minutes")
    if (nodes["minutes"] > 48 + 1e-9).any():
        failures.append("a player above 48 minutes")
    dates = scenarios.drop_duplicates("game_id").set_index("game_id")["as_of_date"]
    keys = nodes.assign(as_of_date=nodes["game_id"].map(dates))[
        ["as_of_date", "player_id"]
    ]
    found = keys.merge(
        features[["as_of_date", "player_id"]].assign(ok=True), how="left"
    )
    if found["ok"].isna().any():
        failures.append(f"{int(found['ok'].isna().sum())} nodes without features")
    pairs = nodes.merge(nodes, on=["game_id", "scenario_id"], suffixes=("_off", "_def"))
    pairs = pairs.loc[pairs["side_off"] != pairs["side_def"]]
    pairs = pairs[["game_id", "player_id_off", "player_id_def"]].drop_duplicates()
    have = pairs.merge(
        guards[["game_id", "off_player_id", "def_player_id"]].assign(ok=True),
        left_on=["game_id", "player_id_off", "player_id_def"],
        right_on=["game_id", "off_player_id", "def_player_id"],
        how="left",
    )
    if have["ok"].isna().any():
        failures.append(
            f"{int(have['ok'].isna().sum())} opposing pairs without guard rows"
        )
    for name, table in (("guards", guards), ("node_features", features)):
        try:
            check_encoder_inputs(table)
        except ValueError as error:
            failures.append(f"{name}: {error}")
    weights = (
        scenarios.loc[~scenarios["is_full_health"]].groupby("game_id")["weight"].sum()
    )
    if (weights <= 0).any():
        failures.append("a game with no positive scenario weight")
    if failures:
        raise SystemExit("Pregame store checks failed: " + "; ".join(failures))
    return {
        "games": int(scenarios["game_id"].nunique()),
        "scenarios_per_game": float(
            scenarios.loc[~scenarios["is_full_health"]].groupby("game_id").size().mean()
        ),
        "nodes_per_team_scenario": float(
            nodes.groupby(["game_id", "scenario_id", "side"]).size().mean()
        ),
        "max_minutes_error": float(off.max()),
    }


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
    positions = profile_positions(
        args.local_root, range(FIRST_SEASON, args.seasons[-1] + 1)
    )
    run = run_scenario_provider(
        data, state, team_game_minutes(data.stints), positions, args.seasons
    )
    provider = V1Minutes(run.provider.team_scenarios, run.provider.team_nodes)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print(f"inputs and provider in {time.time() - started:.0f}s", flush=True)
    root = args.local_root / ROOT
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for season in args.seasons:
        started = time.time()
        games = season_games(season)
        nights = game_nights(
            games, data.box_scores_2_6, run.p_out, excluded=run.excluded
        )
        graphs = build_game_graphs(
            data, games, nights, report_covered=state.covered, minutes=provider
        )
        infeasible = infeasible_games(graphs.nodes)
        scenarios = graphs.scenarios.loc[
            ~graphs.scenarios["game_id"].isin(infeasible)
        ].copy()
        nodes = graphs.nodes.loc[~graphs.nodes["game_id"].isin(infeasible)].copy()
        nodes["player_id"] = nodes["player_id"].astype(str)
        edges = graphs.edges.loc[
            graphs.edges["relation"].eq("guards")
            & ~graphs.edges["game_id"].isin(infeasible)
        ]
        guards = (
            edges.rename(columns={"src": "def_player_id", "dst": "off_player_id"})[
                ["game_id", "off_player_id", "def_player_id", *GUARD_FIELDS]
            ]
            .astype({"off_player_id": str, "def_player_id": str})
            .drop_duplicates(["game_id", "off_player_id", "def_player_id"])
        )
        features = node_feature_rows(data, nodes, scenarios)
        summary = check(scenarios, nodes, guards, features)
        for name, table in (
            ("scenarios", scenarios),
            ("nodes", nodes),
            ("guards", guards),
            ("node_features", features),
        ):
            path = root / name / f"season={season}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_parquet(path, index=False)
        manifest.setdefault("seasons", {})[str(season)] = {
            **summary,
            "games_in_closing_file": int(len(games)),
            "skipped_games": graphs.metadata["skipped_games"],
            "infeasible_minutes_games": infeasible,
            "built_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "commit": commit,
        }
        manifest.update(
            {
                "store": "pregame closing game graphs",
                "version": "v1",
                "minutes_provider": graphs.metadata["minutes_provider"],
                "matchup_history": graphs.metadata["matchup_history"],
                "graph_version": graphs.metadata["version"],
                "full_health_scenario_id": FULL_HEALTH,
                "node_features": "node_profiles with box-start positions as of the "
                "game date (the pretraining dataset's definition)",
                "node_columns": list(NODE_COLUMNS),
                "history_from": FIRST_SEASON,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        print(
            f"{season}: {summary['games']:,} of {len(games):,} games, "
            f"{summary['scenarios_per_game']:.2f} scenarios per game, "
            f"{summary['nodes_per_team_scenario']:.1f} players per team "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
