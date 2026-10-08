"""Intermediate snapshot graphs: history date, per-horizon availability, coverage."""

from types import SimpleNamespace

import pandas as pd
from nba_ou.data_processing.player_graph.as_of import PointInTimeData
from nba_ou.data_processing.player_graph.snapshots import (
    build_intermediate_graphs,
    snapshot_frame,
)

GAME, DATE = "0022300200", pd.Timestamp("2024-01-10")
HOME, AWAY = "100", "200"
GAMES = pd.DataFrame(
    {
        "GAME_ID": [GAME],
        "GAME_DATE": [DATE],
        "HOME_TEAM_ID": [HOME],
        "AWAY_TEAM_ID": [AWAY],
    }
)


def _box():
    """Two earlier games so both rosters have recent minutes."""
    rows = []
    for days in (3, 6):
        game = f"00223001{days:02d}"
        for team, prefix in ((HOME, "h"), (AWAY, "a")):
            for k in range(8):
                rows.append(
                    {
                        "GAME_ID": game,
                        "GAME_DATE": DATE - pd.Timedelta(days=days),
                        "TEAM_ID": team,
                        "PLAYER_ID": f"{prefix}{k}",
                        "MIN": 34.0 - 3 * k,
                    }
                )
    return pd.DataFrame(rows)


def _cutoffs():
    tip = pd.Timestamp("2024-01-11 00:30", tz="UTC")  # 19:30 ET on DATE
    return pd.DataFrame(
        {
            "GAME_ID": [GAME, GAME],
            "TIME_TO_MATCH_MIN": [0, 1080],
            "SNAPSHOT_TS_UTC": [tip, tip - pd.Timedelta(minutes=1080)],
        }
    )


def _state(out=(), covered=((GAME, HOME), (GAME, AWAY))):
    statuses = pd.DataFrame(
        {
            "game_id": GAME,
            "team_id": HOME,
            "player_id": list(out),
            "status": "out",
            "reason_category": "injury",
        },
        columns=["game_id", "team_id", "player_id", "status", "reason_category"],
    )
    return SimpleNamespace(statuses=statuses, covered=set(covered))


def test_early_cutoffs_read_history_from_the_previous_day():
    frame = snapshot_frame(GAMES, _cutoffs()).set_index("TIME_TO_MATCH_MIN")
    assert frame.loc[0, "AS_OF_DATE"] == DATE
    # 18 h before a 19:30 ET tip is 01:30 ET, before the day has settled.
    assert frame.loc[1080, "AS_OF_DATE"] == DATE - pd.Timedelta(days=1)


def test_each_horizon_uses_its_own_report_and_coverage():
    data = PointInTimeData.from_frames(pd.DataFrame(), pd.DataFrame(), _box())
    frame = snapshot_frame(GAMES, _cutoffs())
    states = {
        0: _state(out=("h0",)),
        1080: _state(covered=((GAME, HOME),)),  # away had not filed yet
    }
    graphs = build_intermediate_graphs(data, frame, states)
    tonight = graphs[0].nodes.loc[graphs[0].nodes["scenario_id"].ne(-1)]
    assert "h0" not in set(tonight["player_id"])
    healthy = graphs[0].nodes.loc[graphs[0].nodes["scenario_id"].eq(-1)]
    assert "h0" in set(healthy["player_id"])
    assert graphs[0].metadata["skipped_games"] == {"uncovered": 0, "no_roster": 0}
    assert graphs[1080].scenarios.empty
    assert graphs[1080].metadata["skipped_games"] == {"uncovered": 1, "no_roster": 0}
    assert (graphs[0].scenarios["as_of_date"] == DATE).all()
