"""Stint graphs: fixed shapes, expected guard shares over the floor, contract."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.player_graph.stint_graph import (
    GUARD_EDGES,
    LABELS,
    OPPONENT_EDGES,
    TEAMMATE_EDGES,
    build_stint_graphs,
    message_passing_edges,
)

GAME = "0022300001"
HOME = [f"h{k}" for k in range(5)]
AWAY = [f"a{k}" for k in range(5)]


def _stints():
    return pd.DataFrame(
        {
            "game_id": [GAME],
            "seg_idx": [0],
            "game_date": [pd.Timestamp("2023-10-24")],
            "home_lineup": [np.array(HOME)],
            "away_lineup": [np.array(AWAY)],
            "seconds": [240.0],
            "home_pts": [10],
            "away_pts": [6],
            "home_poss": [8.0],
            "away_poss": [8.0],
            "home_tov": [1],
            "away_tov": [2],
            "home_fga": [8],
            "away_fga": [0],
            "home_fg3a": [4],
            "away_fg3a": [0],
            "home_fta": [2],
            "away_fta": [3],
            "home_oreb": [1],
            "away_oreb": [0],
            "home_dreb": [3],
            "away_dreb": [4],
        }
    )


def _expected(skip=(), extra=()):
    """Every attacker has rate 0.4 on his own index's defender, 0.1 on the rest."""
    rows = []
    for attackers, defenders in ((HOME, AWAY), (AWAY, HOME)):
        for k, att in enumerate(attackers):
            for l, dfn in enumerate(defenders):  # noqa: E741
                if (att, dfn) in skip:
                    continue
                rows.append(
                    {
                        "game_id": GAME,
                        "off_player_id": att,
                        "def_player_id": dfn,
                        "r_hat": 0.4 if k == l else 0.1,
                        "prior_weight": 0.25,
                        "hist_cofloor_seconds_decayed": 1000.0,
                        "has_pair_history": True,
                    }
                )
    rows.extend(extra)
    return pd.DataFrame(rows)


def test_edge_templates():
    assert TEAMMATE_EDGES.shape == (2, 20) and OPPONENT_EDGES.shape == (2, 25)
    assert GUARD_EDGES.shape == (2, 50)
    home = GUARD_EDGES < 5
    # Defender and attacker are always on opposite sides.
    assert (home[0] != home[1]).all()
    # Every node is attacked by five defenders and guards five attackers.
    assert np.bincount(GUARD_EDGES[1], minlength=10).tolist() == [5] * 10
    assert np.bincount(GUARD_EDGES[0], minlength=10).tolist() == [5] * 10


def test_shares_are_normalized_over_the_five_defenders_on_the_floor():
    # A sixth defender with a huge rate is not on the floor and must not count.
    bench = {
        "game_id": GAME,
        "off_player_id": "h0",
        "def_player_id": "a9",
        "r_hat": 5.0,
        "prior_weight": 0.0,
        "hist_cofloor_seconds_decayed": 9.0,
        "has_pair_history": True,
    }
    graphs = build_stint_graphs(_stints(), _expected(extra=[bench]))
    share = graphs.guard_share[0]
    attackers = GUARD_EDGES[1]
    for node in range(10):
        assert share[attackers == node].sum() == pytest.approx(1.0)
    # h0 vs a0 (k == l): 0.4 / (0.4 + 4 * 0.1).
    assert share[0] == pytest.approx(0.5)
    assert not graphs.guard_fallback.any()
    assert graphs.guard_r_hat[0, 0] == 0.4


def test_confidence_attributes_are_carried():
    graphs = build_stint_graphs(_stints(), _expected())
    assert np.allclose(graphs.guard_prior_weight, 0.25)
    assert np.allclose(graphs.guard_log_exposure, np.log1p(1000.0))
    assert graphs.guard_has_history.all()


def test_an_attacker_with_an_unknown_rate_gets_uniform_shares():
    graphs = build_stint_graphs(_stints(), _expected(skip={("a2", "h4")}))
    attacker = GUARD_EDGES[1] == 7  # node 7 = a2
    assert graphs.guard_share[0, attacker] == pytest.approx([0.2] * 5)
    assert graphs.guard_fallback[0, attacker].all()
    assert graphs.guard_fallback[0].sum() == 5
    missing = attacker & (GUARD_EDGES[0] == 4)
    assert graphs.guard_prior_weight[0, missing] == 1.0
    assert not graphs.guard_has_history[0, missing].any()


def test_labels_per_offensive_side():
    graphs = build_stint_graphs(_stints(), _expected())
    home = dict(zip(LABELS, graphs.labels[0, 0], strict=True))
    away = dict(zip(LABELS, graphs.labels[0, 1], strict=True))
    assert home["pts_per_poss"] == pytest.approx(10 / 8)
    assert home["poss_per_48"] == pytest.approx(8 * 2880 / 240)
    assert home["fg3a_per_fga"] == pytest.approx(0.5)
    assert home["oreb_rate"] == pytest.approx(1 / (1 + 4))
    assert np.isnan(away["fg3a_per_fga"])  # no field-goal attempts
    assert graphs.label_weight[0].tolist() == [8.0, 8.0]


@pytest.mark.parametrize(
    "column", ["matchup_seconds", "guard_rate", "TOTAL_LINE_bet365"]
)
def test_observed_or_betting_inputs_are_rejected(column):
    with pytest.raises(ValueError, match="must not hold"):
        build_stint_graphs(_stints(), _expected().assign(**{column: 1.0}))


def test_graph_view_of_one_stint():
    graph = build_stint_graphs(_stints(), _expected()).graph(0)
    assert graph["nodes"]["player_id"].tolist() == HOME + AWAY
    assert graph["guards"]["index"].shape == (2, 50)
    assert graph["teammate"]["weight"].tolist() == [240.0] * 20


def test_sides_without_positive_possessions_get_zero_weight_and_no_labels():
    stints = pd.concat([_stints()] * 2, ignore_index=True)
    stints.loc[1, ["seg_idx", "away_poss", "home_poss"]] = [1, -0.56, 0.0]
    graphs = build_stint_graphs(stints, _expected())
    assert graphs.label_weight[1].tolist() == [0.0, 0.0]
    assert np.isnan(graphs.labels[1]).all()
    assert (graphs.label_weight >= 0).all()
    assert graphs.label_weight[0].tolist() == [8.0, 8.0]


def test_every_guard_attribute_is_finite_with_an_unknown_mask():
    graphs = build_stint_graphs(_stints(), _expected(skip={("a2", "h4")}))
    for values in (
        graphs.guard_share,
        graphs.guard_r_hat,
        graphs.guard_prior_weight,
        graphs.guard_log_exposure,
    ):
        assert np.isfinite(values).all()
    missing = (GUARD_EDGES[1] == 7) & (GUARD_EDGES[0] == 4)
    assert not graphs.guard_r_hat_known[0, missing].any()
    assert graphs.guard_r_hat[0, missing] == 0.0
    assert graphs.guard_r_hat_known[0].sum() == 49


def test_message_passing_lists_undirected_edges_both_ways():
    relations = message_passing_edges(guard_reverse=True)
    for name, template in (("teammate", TEAMMATE_EDGES), ("opponent", OPPONENT_EDGES)):
        index, source = relations[name]["index"], relations[name]["source_edge"]
        assert index.shape == (2, 2 * template.shape[1])
        directed = set(zip(index[0], index[1], strict=True))
        assert all((b, a) in directed for a, b in directed)
        # Each directed edge points back to the template edge it came from.
        assert np.array_equal(
            np.sort(index[:, :], axis=0), np.sort(template[:, source], axis=0)
        )
    assert np.array_equal(relations["guards"]["index"], GUARD_EDGES)
    assert np.array_equal(relations["guarded_by"]["index"], GUARD_EDGES[::-1])
    assert "guarded_by" not in message_passing_edges()
