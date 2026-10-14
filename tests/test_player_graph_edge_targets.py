"""Edge-level targets: time attribution, exclusions, labels kept out of inputs."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.player_graph.edge_targets import (
    TARGETS,
    attacker_games,
    check_edges,
    edge_table,
    to_edge_batch,
)
from nba_ou.data_processing.player_graph.stint_graph import check_encoder_inputs

GAME = "0021700001"
DATE = pd.Timestamp("2017-11-01")


def _pair(
    attacker, defender, pp, fga, fg3a=0, points=0, fta=0, seconds=None, game=GAME
):
    return {
        "game_id": game,
        "game_date": DATE,
        "season_year": 2017,
        "off_team_id": "1",
        "def_team_id": "2",
        "off_player_id": attacker,
        "def_player_id": defender,
        "attacker_is_home": True,
        "cofloor_seconds": 600.0,
        "matchup_seconds": 12.0 * pp if seconds is None else seconds,
        "partial_possessions": pp,
        "player_points": points,
        "matchup_fga": fga,
        "matchup_fg3a": fg3a,
        "matchup_fta": fta,
    }


def _pairs():
    # The attacker's six shots go to the help defender d2 (credited), but d1
    # guarded him for 3/4 of his possessions.
    return pd.DataFrame(
        [
            _pair("i", "d1", pp=6.0, fga=0, seconds=72.0),
            _pair("i", "d2", pp=2.0, fga=6, fg3a=2, points=8, seconds=24.0),
            _pair("i", "d3", pp=0.0, fga=0, seconds=0.0),
            _pair("k", "d1", pp=4.0, fga=2, fg3a=1, points=2),
        ]
    )


def test_time_attribution_spreads_the_whole_game_by_guarding_share():
    edges = edge_table(_pairs(), set()).set_index(["off_player_id", "def_player_id"])
    assert edges.loc[("i", "d1"), "pp_share"] == pytest.approx(0.75)
    assert edges.loc[("i", "d2"), "pp_share"] == pytest.approx(0.25)
    assert edges.loc[("i", "d3"), "pp_share"] == 0.0
    # Same attacker-game rate on every row: 6 FGA over 8 partial possessions,
    # whoever was credited.
    assert edges.loc[("i", "d1"), "fga_per_pp"] == pytest.approx(0.75)
    assert edges.loc[("i", "d2"), "fga_per_pp"] == pytest.approx(0.75)
    assert edges.loc[("i", "d1"), "fg3a_per_pp"] == pytest.approx(0.25)
    # Credited efficiency stays on the credited pair.
    assert edges.loc[("i", "d2"), "pts_per_credited_att"] == pytest.approx(8 / 6)
    assert np.isnan(edges.loc[("i", "d1"), "pts_per_credited_att"])
    # Observed guarding share over the floor (d3 shared it but never guarded).
    shares = edges.loc["i", "observed_share"]
    assert shares.sum() == pytest.approx(1.0) and shares.loc["d3"] == 0.0
    games = attacker_games(edges.reset_index()).set_index("off_player_id")
    assert games.loc["i", "attacker_pp"] == 8.0
    check_edges(edges.reset_index(), set())


def test_excluded_games_have_no_rows():
    other = pd.DataFrame([_pair("i", "d1", pp=3.0, fga=1, game="0021801152")])
    edges = edge_table(pd.concat([_pairs(), other]), {"0021801152"})
    assert set(edges["game_id"]) == {GAME}
    with pytest.raises(ValueError, match="excluded games present"):
        check_edges(edge_table(pd.concat([_pairs(), other]), set()), {"0021801152"})


def test_edge_labels_are_never_encoder_inputs():
    with pytest.raises(ValueError, match="observed"):
        check_encoder_inputs(edge_table(_pairs(), set()).assign(matchup_seconds=1.0))


def _nodes(players):
    return pd.DataFrame(
        {"as_of_date": DATE, "player_id": list(players), "prior_weight": 0.5, "x": 1.0}
    )


def test_batch_scores_the_attacker_game_through_its_shares():
    edges = edge_table(_pairs(), set())
    batch, dropped = to_edge_batch(edges, _nodes(["i", "k", "d1", "d2"]), ["x"])
    assert dropped["attacker_games"] == 0
    assert len(batch) == 2 and len(batch.share) == 3  # d3 had no possessions
    # A per-defender rate f(i, j) aggregates to the attacker-game prediction.
    f = np.where(batch.defender == "d1", 0.5, 1.5)
    predicted = np.bincount(batch.pair_game_index, weights=batch.share * f)
    i = list(batch.attacker).index("i")
    assert predicted[i] == pytest.approx(0.75 * 0.5 + 0.25 * 1.5)
    assert batch.targets.shape == (2, len(TARGETS))


def test_attacker_games_without_node_features_are_dropped_whole():
    edges = edge_table(_pairs(), set())
    batch, dropped = to_edge_batch(edges, _nodes(["i", "k", "d1"]), ["x"])
    assert dropped == {"attacker_games": 1, "attacker_pp": 8.0}
    assert batch.attacker.tolist() == ["k"]
    assert batch.pair_game_index.tolist() == [0]
