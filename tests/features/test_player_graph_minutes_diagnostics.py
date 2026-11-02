"""Phase 4A diagnostic: projected (2_6) vs actual regulation minutes."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.lineups.game_projection import PlayerNight
from nba_ou.data_processing.player_graph.minutes_diagnostics import (
    actual_minutes,
    minutes_table,
    projected_minutes,
)

GAME, TEAM = "0021800001", "100"


def _nights():
    players = [PlayerNight(f"p{k}", 36.0 - 3 * k, 0.0) for k in range(8)]
    players[0] = PlayerNight("p0", 36.0, 0.5)  # a coin flip
    players[1] = PlayerNight("p1", 33.0, 1.0)  # out
    return {(GAME, TEAM): players}


def test_projection_is_the_scenario_weighted_allocation():
    projected = projected_minutes(_nights()).set_index("player_id")
    assert projected["projected"].sum() == pytest.approx(240.0)
    assert projected["healthy"].sum() == pytest.approx(240.0)
    assert projected.loc["p1", "projected"] == 0.0
    # p0 plays half the time: his projection is half his playing allocation.
    assert projected.loc["p0", "projected"] == pytest.approx(
        projected.loc["p0", "max_scenario"] / 2
    )


def _stints():
    rows = []
    for period, seconds in ((1, 720.0), (2, 720.0), (3, 720.0), (4, 720.0), (5, 300.0)):
        rows.append(
            {
                "game_id": GAME,
                "period": period,
                "seconds": seconds,
                "home_team_id": TEAM,
                "away_team_id": "200",
                "home_lineup": np.array(["p0", "p2", "p3", "p4", "x9"]),
                "away_lineup": np.array([f"a{k}" for k in range(5)]),
            }
        )
    return pd.DataFrame(rows)


def test_actual_minutes_are_regulation_only():
    actual = actual_minutes(_stints())
    home = actual.loc[actual["team_id"].eq(TEAM)]
    assert home["actual"].sum() == pytest.approx(240.0)
    assert set(home["player_id"]) == {"p0", "p2", "p3", "p4", "x9"}


def test_table_flags_players_off_the_roster_and_report_status():
    table = minutes_table(projected_minutes(_nights()), actual_minutes(_stints()))
    table = table.set_index("player_id")
    assert not table.loc["x9", "in_roster"] and table.loc["x9", "projected"] == 0
    assert table.loc["x9", "status"] == "not on roster"
    assert table.loc["p1", "status"] == "out"
    assert table.loc["p0", "status"] == "uncertain"
    assert table.loc["p5", "status"] == "available" and not table.loc["p5", "played"]
    assert table["projected"].sum() == pytest.approx(table["actual"].sum())
