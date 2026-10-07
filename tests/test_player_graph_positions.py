"""Soft G/F/C positions: starts first, profile for the bench, strictly as of D."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.player_graph.as_of import PointInTimeData
from nba_ou.data_processing.player_graph.positions import (
    PROBABILITY_COLUMNS,
    positions_as_of,
    start_counts,
)

#: Per-game box line by archetype: (AST, OREB, DREB, BLK, STL, FG3A, FGA, PF).
ARCHETYPES = {
    "G": (7, 0.5, 3, 0.2, 1.5, 7, 15, 2),
    "F": (3, 1.5, 5, 0.6, 1.0, 4, 12, 2.5),
    "C": (1.5, 3.5, 7, 2.0, 0.6, 0.5, 9, 3.5),
}
STATS = ("AST", "OREB", "DREB", "BLK", "STL", "FG3A", "FGA", "PF")


def _data(n_games=12):
    """Three starters per position plus bench players, one game a day."""
    dates = pd.date_range("2023-11-01", periods=n_games, freq="D")
    starters = {f"{pos}{k}": pos for pos in "GFC" for k in range(3)}
    bench = {"benchC": "C", "benchG": "G"}
    box, matchups = [], []
    for day, date in enumerate(dates):
        game = f"00223{day:05d}"
        for player, pos in {**starters, **bench}.items():
            line = dict(zip(STATS, ARCHETYPES[pos], strict=True))
            box.append(
                {"GAME_ID": game, "GAME_DATE": date, "PLAYER_ID": player, "MIN": 30}
                | line
            )
            matchups.append(
                {
                    "game_id": game,
                    "game_date": date,
                    "off_player_id": player,
                    "off_position": pos if player in starters else "",
                }
            )
    return (
        PointInTimeData.from_frames(
            pd.DataFrame(), pd.DataFrame(matchups), pd.DataFrame(box)
        ),
        dates,
    )


def test_start_counts_one_per_game_and_ignore_the_bench():
    matchups = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g1", "g2"],
            "off_player_id": ["p", "p", "q", "p"],
            "off_position": ["G", "G", "", "F"],
        }
    )
    counts = start_counts(matchups)
    assert counts.loc["p"].tolist() == [1, 1, 0]
    assert "q" not in counts.index


def test_starters_follow_their_starts_and_the_bench_its_profile():
    data, dates = _data()
    pos = positions_as_of(data.as_of(dates[-1] + pd.Timedelta(days=1)))
    assert np.allclose(pos[list(PROBABILITY_COLUMNS)].sum(axis=1), 1.0)
    assert pos.loc["G0", "p_G"] > 0.9 and pos.loc["C2", "p_C"] > 0.9
    assert pos.loc["benchC", "n_starts"] == 0
    assert pos.loc["benchC"][list(PROBABILITY_COLUMNS)].idxmax() == "p_C"
    assert pos.loc["benchG"][list(PROBABILITY_COLUMNS)].idxmax() == "p_G"
    assert pos["has_profile"].all()


def test_no_starts_and_no_box_scores_falls_back_to_league_shares():
    data, dates = _data()
    extra = data.matchups.iloc[:1].assign(
        off_player_id="ghost", off_position="", game_date=dates[0]
    )
    data = PointInTimeData.from_frames(
        data.stints, pd.concat([data.matchups, extra]), data.box_scores
    )
    pos = positions_as_of(data.as_of(dates[-1] + pd.Timedelta(days=1)))
    # League shares: three G, three F and three C starters a game.
    assert pos.loc["ghost", list(PROBABILITY_COLUMNS)].tolist() == pytest.approx(
        [1 / 3] * 3
    )
    assert not pos.loc["ghost", "has_profile"]


def test_starts_on_or_after_the_cutoff_are_not_counted():
    data, dates = _data()
    pos = positions_as_of(data.as_of(dates[5]))
    assert pos.loc["G0", "n_starts"] == 5
    early = positions_as_of(data.as_of(dates[0]))
    assert early.empty
