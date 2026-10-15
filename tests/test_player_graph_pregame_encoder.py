"""Pregame adapter: expected rotation as 48 one-minute stint graphs."""

import numpy as np
import pandas as pd
import pytest
import torch
from nba_ou.data_processing.player_graph.fm import PlayerVocab, Scaler, Standardization
from nba_ou.data_processing.player_graph.gnn import GNN, GNNConfig
from nba_ou.data_processing.player_graph.pregame_encoder import (
    SLOTS,
    ScenarioInput,
    aggregate,
    align,
    embedding_columns,
    encode_slots,
    fold_projection,
    game_embedding,
    lineup_schedule,
    scenario_means,
    schedule_minutes,
    slot_stints,
    style_columns,
)
from nba_ou.data_processing.player_graph.pretraining_data import NODE_COLUMNS

ROTATION = [36.0, 34, 33, 31, 28, 24, 20, 15, 11, 8]  # 240


def test_schedule_is_five_a_slot_and_reproduces_the_minutes():
    players = [f"p{k}" for k in range(len(ROTATION))]
    schedule = lineup_schedule(players, ROTATION)
    assert schedule.shape == (SLOTS, 5)
    assert all(len(set(row)) == 5 for row in schedule)
    realized = schedule_minutes(schedule, len(players))
    assert realized.sum() == pytest.approx(240.0)
    assert np.abs(realized - ROTATION).max() <= 48 / SLOTS + 1e-9
    # Deterministic, and independent of the input order.
    assert np.array_equal(schedule, lineup_schedule(players, ROTATION))
    perm = np.random.default_rng(0).permutation(len(players))
    again = lineup_schedule([players[i] for i in perm], [ROTATION[i] for i in perm])
    assert sorted(map(tuple, np.sort(perm[again], axis=1))) == sorted(
        map(tuple, np.sort(schedule, axis=1))
    )


def test_schedule_rejects_impossible_minutes():
    with pytest.raises(ValueError, match="240"):
        lineup_schedule(["a"] * 5, [40.0] * 5)
    with pytest.raises(ValueError, match="48"):
        lineup_schedule(["a", "b", "c", "d", "e", "f"], [50, 46, 46, 46, 46, 6])


def test_a_48_minute_player_is_in_every_slot():
    minutes = [48.0, 48, 40, 40, 32, 32]
    schedule = lineup_schedule(list("abcdef"), minutes)
    assert (
        np.isin(0, schedule, assume_unique=False) and (schedule == 0).any(axis=1).all()
    )
    assert (schedule == 1).any(axis=1).all()


def _scenario(home_minutes=ROTATION, away_minutes=ROTATION[::-1]):
    home = pd.DataFrame(
        {
            "player_id": [f"h{k}" for k in range(len(home_minutes))],
            "minutes": home_minutes,
        }
    )
    away = pd.DataFrame(
        {
            "player_id": [f"a{k}" for k in range(len(away_minutes))],
            "minutes": away_minutes,
        }
    )
    return ScenarioInput("0021800001", 0, home, away)


def _features(players, rng):
    frame = pd.DataFrame(
        np.abs(rng.normal(size=(len(players), len(NODE_COLUMNS)))),
        columns=list(NODE_COLUMNS),
        index=pd.Index(players, name="player_id"),
    )
    return frame


def _guards(home, away, rng):
    rows = []
    for attackers, defenders in ((home, away), (away, home)):
        for a in attackers:
            for d in defenders:
                rows.append(
                    {
                        "off_player_id": a,
                        "def_player_id": d,
                        "r_hat": rng.uniform(0.02, 0.2),
                        "r_hat_known": True,
                        "guard_prior_weight": rng.uniform(0, 1),
                        "log_exposure": rng.uniform(0, 8),
                        "has_pair_history": bool(rng.integers(0, 2)),
                    }
                )
    return pd.DataFrame(rows)


def test_slot_graphs_keep_sides_and_normalize_guard_shares():
    rng = np.random.default_rng(1)
    scenario = _scenario()
    home, away = list(scenario.home["player_id"]), list(scenario.away["player_id"])
    stints = slot_stints(
        scenario, _features(home + away, rng), _guards(home, away, rng)
    )
    assert stints.players.shape == (SLOTS, 10)
    assert np.isin(stints.players[:, :5], home).all()
    assert np.isin(stints.players[:, 5:], away).all()
    assert np.allclose(stints.guard_share.reshape(SLOTS, 2, 5, 5).sum(axis=3), 1.0)
    assert np.isfinite(stints.nodes).all()
    with pytest.raises(ValueError, match="node features"):
        slot_stints(scenario, _features(home, rng), _guards(home, away, rng))


def _model(players, rng):
    torch.manual_seed(0)
    vocab = PlayerVocab(players)
    std = Standardization(
        Scaler([0.0] * len(NODE_COLUMNS), [1.0] * len(NODE_COLUMNS)),
        Scaler([0.3] * 6, [0.1] * 6),
        None,
    )
    return GNN(GNNConfig(), len(vocab)).eval(), vocab, std


def test_swapping_home_and_away_swaps_the_outputs():
    rng = np.random.default_rng(2)
    scenario = _scenario()
    home, away = list(scenario.home["player_id"]), list(scenario.away["player_id"])
    features, guards = _features(home + away, rng), _guards(home, away, rng)
    model, vocab, std = _model(home + away, rng)  # home term untrained: 0
    out = encode_slots(model, vocab, std, slot_stints(scenario, features, guards))
    swapped = ScenarioInput(scenario.game_id, 0, scenario.away, scenario.home)
    back = encode_slots(model, vocab, std, slot_stints(swapped, features, guards))
    mean, mean_back = scenario_means(out), scenario_means(back)
    assert np.allclose(mean["style"][0], mean_back["style"][1], atol=1e-5)
    assert np.allclose(mean["pooled"][0], mean_back["pooled"][1], atol=1e-5)


def test_tonight_weights_the_scenarios_and_shifts_from_full_health():
    per = {
        0: {"style": np.array([1.0, 2.0])},
        1: {"style": np.array([3.0, 4.0])},
        -1: {"style": np.array([0.0, 0.0])},
    }
    out = aggregate(per, {0: 3.0, 1: 1.0}, full_health=-1)
    assert np.allclose(out["style_tonight"], [1.5, 2.5])
    assert np.allclose(out["style_absence_shift"], [1.5, 2.5])
    with pytest.raises(ValueError):
        aggregate({0: per[0]}, {0: 1.0}, full_health=-1)


def test_projection_is_orthonormal_and_alignment_undoes_a_rotation():
    rng = np.random.default_rng(3)
    pooled = rng.normal(size=(500, 32)) @ np.diag(np.linspace(3, 0.1, 32))
    basis = fold_projection(pooled)
    assert basis.shape == (32, 8)
    assert np.allclose(basis.T @ basis, np.eye(8), atol=1e-8)
    q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    rotated = basis @ q  # same subspace, other axes
    restored = align(rotated, pooled, basis, pooled)
    assert np.allclose(restored, basis, atol=1e-8)
    z = game_embedding(np.stack([pooled[0], pooled[1]]), basis)
    assert z.shape == (16,)


def test_column_names():
    assert len(style_columns()) == 16 and len(set(style_columns())) == 16
    assert all("_BEFORE_TEAM_" in c for c in style_columns())
    assert len(embedding_columns("GE_GRAPH_EMB")) == 16
