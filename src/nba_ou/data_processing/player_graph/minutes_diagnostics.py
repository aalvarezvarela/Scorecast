"""Projected vs actual minutes (plan phase 4A, step 1: diagnostic only).

The projection is 2_6's, unchanged: each roster's ``PlayerNight`` list (recent
average minutes, pre-game ``p_out``) is spread over availability scenarios
(``enumerate_scenarios``), and in each scenario ``allocate_minutes`` rescales
the available players' recent minutes to 240. The projected minutes of a
player are the scenario-weighted mean (0 in scenarios where he sits), exactly
what the v0 game graph's nodes carry; the full-health allocation (nobody
sits) is the counterfactual every absence is measured against.

Actual minutes are **regulation** minutes from the stints (periods 1-4), so a
team's actual minutes sum to 240 like the projection's.

Redistribution: for the players available tonight, ``projected - healthy`` is
what ``allocate_minutes`` gives them because of the absences and
``actual - healthy`` is what they really played above their full-health
share. The latter also carries the game's ordinary noise, which games
without absences measure.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from nba_ou.data_processing.lineups.game_projection import (
    PlayerNight,
    allocate_minutes,
    enumerate_scenarios,
)

#: p_out at or above this sits in every scenario (2_6's SCENARIO_HIGH).
OUT = 0.9
#: p_out at or below this plays in every scenario (2_6's SCENARIO_LOW).
AVAILABLE = 0.1
STARTERS = 5


def projected_minutes(
    nights: Mapping[tuple[str, str], list[PlayerNight]],
) -> pd.DataFrame:
    """One row per roster player and team-game: ``base_minutes``, ``p_out``,
    ``healthy`` (full-health allocation), ``projected`` (scenario-weighted
    mean), ``max_scenario`` (largest allocation in any scenario he plays) and
    ``p_over_48`` (probability of a scenario giving him more than 48)."""
    rows = []
    for (game_id, team_id), players in nights.items():
        healthy = allocate_minutes(players)
        if not healthy:
            continue
        expected = dict.fromkeys(healthy, 0.0)
        largest = dict.fromkeys(healthy, 0.0)
        over = dict.fromkeys(healthy, 0.0)
        weight_sum = 0.0
        for sitting, weight in enumerate_scenarios(players):
            minutes = allocate_minutes(players, sitting)
            if not minutes:  # dropped, as in project_game
                continue
            weight_sum += weight
            for player, value in minutes.items():
                expected[player] += weight * value
                largest[player] = max(largest[player], value)
                over[player] += weight * (value > 48.0)
        if weight_sum <= 0:
            continue
        ranked = sorted(healthy, key=lambda p: (-healthy[p], p))
        rank = {player: i + 1 for i, player in enumerate(ranked)}
        for player in players:
            pid = player.player_id
            rows.append(
                {
                    "game_id": game_id,
                    "team_id": team_id,
                    "player_id": pid,
                    "base_minutes": player.base_minutes,
                    "p_out": player.p_out,
                    "healthy": healthy.get(pid, 0.0),
                    "healthy_rank": rank.get(pid),
                    "projected": expected.get(pid, 0.0) / weight_sum,
                    "max_scenario": largest.get(pid, 0.0),
                    "p_over_48": over.get(pid, 0.0) / weight_sum,
                }
            )
    return pd.DataFrame(rows)


def actual_minutes(stints: pd.DataFrame) -> pd.DataFrame:
    """Regulation minutes per ``(game_id, team_id, player_id)`` from the
    stints (periods 1-4 only)."""
    regulation = stints.loc[stints["period"] <= 4]
    parts = []
    for side in ("home", "away"):
        lineups = np.stack(regulation[f"{side}_lineup"].map(np.asarray).to_numpy())
        parts.append(
            pd.DataFrame(
                {
                    "game_id": np.repeat(
                        regulation["game_id"].astype(str).to_numpy(), 5
                    ),
                    "team_id": np.repeat(
                        regulation[f"{side}_team_id"].astype(str).to_numpy(), 5
                    ),
                    "player_id": lineups.astype(str).ravel(),
                    "actual": np.repeat(regulation["seconds"].to_numpy(float), 5) / 60,
                }
            )
        )
    return (
        pd.concat(parts, ignore_index=True)
        .groupby(["game_id", "team_id", "player_id"], as_index=False)["actual"]
        .sum()
    )


def minutes_table(projected: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Projected and actual minutes side by side for every roster player and
    every player who actually played, on the team-games of ``projected``.

    ``in_roster`` False marks players who played without being on the
    projected roster; ``status`` buckets ``p_out`` as 2_6's scenarios do.
    """
    games = projected[["game_id", "team_id"]].drop_duplicates()
    actual = actual.merge(games, on=["game_id", "team_id"])
    table = projected.merge(
        actual, on=["game_id", "team_id", "player_id"], how="outer", indicator=True
    )
    table["in_roster"] = table["_merge"].ne("right_only")
    table = table.drop(columns="_merge")
    for column in ("healthy", "projected", "actual", "max_scenario", "p_over_48"):
        table[column] = table[column].fillna(0.0)
    table["p_out"] = table["p_out"].fillna(0.0)
    table["status"] = np.select(
        [table["p_out"] >= OUT, table["p_out"] > AVAILABLE],
        ["out", "uncertain"],
        "available",
    )
    table.loc[~table["in_roster"], "status"] = "not on roster"
    table["projected_role"] = np.where(
        table["healthy_rank"].le(STARTERS), "starter", "bench"
    )
    table.loc[~table["in_roster"], "projected_role"] = "not on roster"
    table["error"] = table["projected"] - table["actual"]
    table["played"] = table["actual"] > 0
    return table
