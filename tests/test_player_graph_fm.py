"""Phase 6 factorization machine on synthetic stints and edges."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from nba_ou.data_processing.player_graph.fm import (
    FactorizationMachine,
    FMConfig,
    PlayerVocab,
    Scaler,
    Standardization,
    StintTensors,
    _pair_mean,
    edge_tensors,
    fit_standardization,
    load_checkpoint,
    predict,
    predict_edges,
    save_checkpoint,
    stint_tensors,
    train,
    warm_start,
)
from nba_ou.data_processing.player_graph.pretraining_data import NODE_COLUMNS

F = len(NODE_COLUMNS)
T = 6


def _raw_nodes(rng, shape):
    raw = rng.normal(size=(*shape, F))
    return np.abs(raw)  # counts and durations are non-negative


def _stints(rng, n, players=40, share=None):
    """Random stints over ``players`` ids; the home five never overlap the away five."""
    ids = np.stack([rng.choice(players, 10, replace=False) for _ in range(n)])
    return SimpleNamespace(
        players=ids.astype(str),
        nodes=_raw_nodes(rng, (n, 10)),
        guard_share=np.full((n, 50), 0.2) if share is None else share,
        labels=np.zeros((n, 2, T)),
        label_weight=np.ones((n, 2, T)),
    )


def _model(config=None, players=40):
    torch.manual_seed(0)
    return FactorizationMachine(config or FMConfig(), players)


def test_pair_mean_is_the_sum_over_pairs():
    v = torch.randn(3, 5, 4)
    explicit = sum(v[:, i] * v[:, k] for i in range(5) for k in range(i + 1, 5))
    assert torch.allclose(_pair_mean(v), explicit, atol=1e-6)


def test_order_within_a_side_does_not_matter():
    rng = np.random.default_rng(0)
    data = _stints(rng, 8, share=rng.dirichlet(np.ones(5), size=(8, 10)).reshape(8, 50))
    model = _model()
    std = Standardization(Scaler.fit(data.nodes), Scaler([0.0] * T, [1.0] * T), None)
    vocab = PlayerVocab(data.players.reshape(-1))
    tensors = stint_tensors(SimpleNamespace(**data.__dict__), vocab, std)
    base = model(tensors.x, tensors.ids, tensors.share)
    # Permute the home five (nodes and their guard edges, both directions).
    perm = np.array([3, 0, 4, 1, 2])
    nodes = np.r_[perm, np.arange(5, 10)]
    share = tensors.share.reshape(-1, 2, 5, 5).clone()
    share[:, 0] = share[:, 0][:, perm]  # home attackers
    share[:, 1] = share[:, 1][:, :, perm]  # home defenders
    permuted = model(tensors.x[:, nodes], tensors.ids[:, nodes], share.reshape(-1, 50))
    assert torch.allclose(base, permuted, atol=1e-5)


def _fit(model, data, labels, epochs=40):
    data.labels = labels
    std = fit_standardization(data)
    vocab = PlayerVocab(data.players.reshape(-1))
    tensors = stint_tensors(data, vocab, std)
    train(model, tensors, epochs=epochs, batch_size=256, lr=0.03, seed=0)
    return predict(model, tensors, std), vocab, std, tensors


def test_learns_a_planted_additive_effect():
    rng = np.random.default_rng(1)
    data = _stints(rng, 3000)
    skill = rng.normal(0, 1, 40)
    ids = data.players.astype(int)
    labels = np.zeros((3000, 2, T))
    labels[:, 0, 0] = skill[ids[:, :5]].sum(1) - skill[ids[:, 5:]].sum(1) / 2
    labels[:, 1, 0] = skill[ids[:, 5:]].sum(1) - skill[ids[:, :5]].sum(1) / 2
    labels += rng.normal(0, 0.5, labels.shape)
    config = FMConfig(
        chemistry=False, style=False, matchup=False, edges=False, delta_penalty=1.0
    )
    predicted, *_ = _fit(_model(config), data, labels)
    r = np.corrcoef(predicted[:, :, 0].ravel(), labels[:, :, 0].ravel())[0, 1]
    assert r > 0.8


def test_matchup_term_captures_who_guards_whom():
    """A planted attacker x defender interaction that only acts through the
    guard edges: the matchup term learns it, the additive model cannot."""
    rng = np.random.default_rng(2)
    n = 4000
    # Each attacker is guarded by one defender (the same slot on the other side).
    share = np.zeros((n, 2, 5, 5))
    share[:, :, np.arange(5), np.arange(5)] = 1.0
    data = _stints(rng, n, share=share.reshape(n, 50))
    kind = rng.integers(0, 2, 40) * 2 - 1  # +1 / -1 "type"
    ids = data.players.astype(int)
    labels = np.zeros((n, 2, T))
    labels[:, 0, 2] = (kind[ids[:, :5]] * kind[ids[:, 5:]]).sum(1)
    labels[:, 1, 2] = (kind[ids[:, 5:]] * kind[ids[:, :5]]).sum(1)
    labels += rng.normal(0, 0.3, labels.shape)

    def r2(config):
        predicted, *_ = _fit(_model(config), data, labels)
        e = labels[:, :, 2] - predicted[:, :, 2]
        return 1 - e.var() / labels[:, :, 2].var()

    additive = r2(FMConfig(chemistry=False, style=False, matchup=False, edges=False))
    matchup = r2(
        FMConfig(
            chemistry=False, style=False, matchup=True, edges=False, delta_penalty=1.0
        )
    )
    assert additive < 0.1
    assert matchup > 0.5


def _edges(rng, games=50, players=40):
    pairs = rng.integers(2, 5, games)
    index = np.repeat(np.arange(games), pairs)
    share = rng.dirichlet(np.ones(5), size=games)
    share = np.concatenate(
        [share[g, : pairs[g]] / share[g, : pairs[g]].sum() for g in range(games)]
    )
    return SimpleNamespace(
        attacker=rng.integers(0, players, games).astype(str),
        attacker_is_home=rng.random(games) > 0.5,
        attacker_nodes=_raw_nodes(rng, (games,)),
        targets=rng.normal(0.2, 0.05, (games, 2)),
        weight=rng.uniform(5, 20, games),
        pair_game_index=index,
        defender=rng.integers(0, players, len(index)).astype(str),
        defender_nodes=_raw_nodes(rng, (len(index),)),
        share=share,
    )


def test_edge_head_scores_the_share_weighted_defender_rates():
    rng = np.random.default_rng(3)
    stints = _stints(rng, 50)
    edges = _edges(rng)
    std = fit_standardization(stints, edges)
    vocab = PlayerVocab([*stints.players.reshape(-1), *edges.attacker, *edges.defender])
    model = _model()
    tensors = edge_tensors(edges, vocab, std)
    out = model.forward_edges(
        tensors.attacker_x,
        tensors.attacker_ids,
        tensors.home,
        tensors.defender_x,
        tensors.defender_ids,
        tensors.share,
        tensors.pair_index,
    )
    # By hand for game 0: sum_j s_j f(i, j).
    a = model.blocks(tensors.attacker_x[:1], tensors.attacker_ids[:1])
    rows = np.flatnonzero(edges.pair_game_index == 0)
    b = model.blocks(tensors.defender_x[rows], tensors.defender_ids[rows])
    h = 1.0 if edges.attacker_is_home[0] else -1.0
    f = (
        model.mu_edge
        + h * model.hca_edge
        + a["oe"]
        + b["de"]
        + (a["a"] * b["b"]) @ model.readout_edge.T
    )
    by_hand = (
        torch.as_tensor(edges.share[rows], dtype=torch.float32)[:, None] * f
    ).sum(0)
    assert torch.allclose(out[0], by_hand, atol=1e-5)
    # A batch of attacker-games keeps each game's own pairs.
    part = tensors.take(np.array([7, 2]))
    assert torch.equal(
        part.share,
        tensors.share[
            torch.as_tensor(
                np.r_[
                    np.flatnonzero(edges.pair_game_index == 7),
                    np.flatnonzero(edges.pair_game_index == 2),
                ]
            )
        ],
    )
    # Training with edges runs and predictions come back in target units.
    history = train(
        model, stint_tensors(stints, vocab, std), tensors, epochs=2, batch_size=16
    )
    assert history[-1]["edge"] > 0
    assert predict_edges(model, tensors, std).shape == (50, 2)


def test_unseen_players_get_their_profile_only():
    model = _model()
    x = torch.randn(1, 10, F)
    unseen = model.blocks(x, torch.zeros(1, 10, dtype=torch.long))
    with torch.no_grad():
        model.delta.weight[1:] = 5.0
    still = model.blocks(x, torch.zeros(1, 10, dtype=torch.long))
    assert torch.equal(unseen["off"], still["off"])
    assert model.delta.weight[0].abs().sum() == 0


def test_warm_start_moves_deltas_to_the_grown_vocabulary(tmp_path):
    old_vocab = PlayerVocab(["5", "9"])
    old = _model(players=2)
    with torch.no_grad():
        old.delta.weight[1:] = torch.arange(2, dtype=torch.float32)[:, None] + 1
    new_vocab = PlayerVocab(["5", "9"])
    new_vocab.extend(["1", "7"])
    new = _model(players=len(new_vocab))
    warm_start(new, old, new_vocab, old_vocab)
    assert new.delta.weight[new_vocab.index["9"]][0] == 2.0
    assert new.delta.weight[new_vocab.index["7"]].abs().sum() == 0
    assert torch.equal(new.prior.weight, old.prior.weight)

    std = Standardization(
        Scaler([0.0] * F, [1.0] * F), Scaler([0.0] * T, [1.0] * T), None
    )
    save_checkpoint(tmp_path / "fm.pt", new, new_vocab, std, {"fold": "2017-01"})
    loaded, vocab, std_back, extra = load_checkpoint(tmp_path / "fm.pt")
    assert vocab.ids == new_vocab.ids and extra == {"fold": "2017-01"}
    x = torch.randn(2, 10, F)
    ids = torch.ones(2, 10, dtype=torch.long)
    share = torch.full((2, 50), 0.2)
    assert torch.equal(loaded(x, ids, share), new.eval()(x, ids, share))
    assert std_back.targets == std.targets


def test_stint_tensors_mask_missing_labels():
    rng = np.random.default_rng(4)
    data = _stints(rng, 4)
    data.labels[0, 1, 3] = np.nan
    data.label_weight[0, 1, 3] = 0.0
    std = fit_standardization(data)
    tensors = stint_tensors(data, PlayerVocab(data.players.reshape(-1)), std)
    assert isinstance(tensors, StintTensors)
    assert tensors.w[0, 1, 3] == 0 and torch.isfinite(tensors.y).all()


def test_node_features_are_finite():
    from nba_ou.data_processing.player_graph.fm import node_features

    raw = np.ones((2, len(NODE_COLUMNS)))
    raw[0, list(NODE_COLUMNS).index("days_since_last_game")] = np.nan
    assert np.isfinite(node_features(raw)).all()
    raw[1, 0] = np.inf
    with pytest.raises(ValueError):
        node_features(raw)
