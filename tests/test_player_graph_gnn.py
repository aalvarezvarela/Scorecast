"""Phase 6 GNN: capacity-matched ablations, message passing, leakage, plumbing."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from nba_ou.data_processing.player_graph.fm import (
    PlayerVocab,
    fit_standardization,
    load_checkpoint,
    predict,
    save_checkpoint,
    stint_tensors,
    train,
    warm_start,
)
from nba_ou.data_processing.player_graph.gnn import (
    GNN,
    N_GUARD_FEATURES,
    GNNConfig,
    guard_features,
)
from nba_ou.data_processing.player_graph.pretraining_data import NODE_COLUMNS

F = len(NODE_COLUMNS)
T = 6
FULL = GNNConfig()
NO_GUARD = GNNConfig(relations=("teammate", "opponent"))
NODE_ONLY = GNNConfig(relations=())


def _stints(rng, n, players=40, share=None):
    ids = np.stack([rng.choice(players, 10, replace=False) for _ in range(n)])
    if share is None:
        share = rng.dirichlet(np.ones(5), size=(n, 10)).reshape(n, 50)
    return SimpleNamespace(
        players=ids.astype(str),
        nodes=np.abs(rng.normal(size=(n, 10, F))),
        guard_share=share,
        guard_source=rng.integers(0, 4, (n, 50)),
        guard_r_hat=rng.uniform(0, 0.2, (n, 50)),
        guard_prior_weight=rng.uniform(0, 1, (n, 50)),
        guard_log_exposure=rng.uniform(0, 8, (n, 50)),
        labels=np.zeros((n, 2, T)),
        label_weight=np.ones((n, 2, T)),
    )


def _tensors(data):
    std = fit_standardization(data)
    vocab = PlayerVocab(data.players.reshape(-1))
    return stint_tensors(data, vocab, std, guard=True), vocab, std


def _model(config, players=40):
    torch.manual_seed(0)
    return GNN(config, players).eval()


def test_ablations_are_capacity_matched():
    counts = {
        name: sum(p.numel() for p in _model(c).parameters())
        for name, c in (("full", FULL), ("no_guard", NO_GUARD), ("node", NODE_ONLY))
    }
    assert len(set(counts.values())) == 1, counts
    with pytest.raises(ValueError):
        GNNConfig(relations=("guards", "spacing"))


def test_guard_features_are_expected_quantities_only():
    data = _stints(np.random.default_rng(0), 3)
    features = guard_features(data)
    assert features.shape == (3, 50, N_GUARD_FEATURES)
    assert np.allclose(features[..., 0], data.guard_share)
    data.guard_r_hat[0, 0] = np.nan
    with pytest.raises(ValueError):
        guard_features(data)


def test_node_only_model_cannot_see_other_players_or_edges():
    rng = np.random.default_rng(1)
    t, *_ = _tensors(_stints(rng, 4))
    model = _model(NODE_ONLY)
    base = model.embed(t.x, t.ids, t.share, t.guard)
    x = t.x.clone()
    x[:, 1:] = torch.randn_like(x[:, 1:])
    ids = t.ids.clone()
    ids[:, 1:] = 0
    other = model.embed(x, ids, torch.rand_like(t.share), torch.randn_like(t.guard))
    # Node 0's embedding depends on node 0 only.
    assert torch.allclose(base[:, 0], other[:, 0], atol=1e-6)
    out = model(t.x, t.ids, t.share, t.guard)
    assert torch.equal(out, model(t.x, t.ids, torch.rand_like(t.share), None))


def test_teammate_opponent_gnn_reads_neighbours_but_not_guard_edges():
    rng = np.random.default_rng(2)
    t, *_ = _tensors(_stints(rng, 4))
    model = _model(NO_GUARD)
    base = model.embed(t.x, t.ids, t.share, t.guard)
    x = t.x.clone()
    x[:, 1] = torch.randn_like(x[:, 1])  # a teammate of node 0
    assert not torch.allclose(base[:, 0], model.embed(x, t.ids, t.share, t.guard)[:, 0])
    out = model(t.x, t.ids, t.share, t.guard)
    assert torch.allclose(
        out, model(t.x, t.ids, torch.rand_like(t.share), torch.randn_like(t.guard))
    )


def test_full_gnn_depends_on_who_guards_whom():
    rng = np.random.default_rng(3)
    share = np.zeros((4, 2, 5, 5))
    share[:, :, np.arange(5), np.arange(5)] = 1.0
    t, *_ = _tensors(_stints(rng, 4, share=share.reshape(4, 50)))
    model = _model(FULL)
    out = model(t.x, t.ids, t.share, t.guard)
    swapped = share.copy()
    swapped[:, 0, [0, 1]] = swapped[:, 0, [1, 0]]  # home attackers 0, 1 swap men
    t2 = replace(t, share=torch.as_tensor(swapped.reshape(4, 50), dtype=torch.float32))
    assert not torch.allclose(out, model(t2.x, t2.ids, t2.share, t2.guard))
    guard = t.guard.clone()
    guard[:, :, 2] += 1.0  # the edges' expected-rate attributes
    assert not torch.allclose(out, model(t.x, t.ids, t.share, guard))


def test_order_within_a_side_does_not_matter():
    rng = np.random.default_rng(4)
    t, *_ = _tensors(_stints(rng, 6))
    model = _model(FULL)
    perm = np.array([3, 0, 4, 1, 2])
    nodes = np.r_[perm, np.arange(5, 10)]
    share = t.share.reshape(-1, 2, 5, 5).clone()
    guard = t.guard.reshape(-1, 2, 5, 5, N_GUARD_FEATURES).clone()
    share[:, 0], guard[:, 0] = share[:, 0][:, perm], guard[:, 0][:, perm]
    share[:, 1], guard[:, 1] = share[:, 1][:, :, perm], guard[:, 1][:, :, perm]
    permuted = model(
        t.x[:, nodes],
        t.ids[:, nodes],
        share.reshape(-1, 50),
        guard.reshape(-1, 50, N_GUARD_FEATURES),
    )
    assert torch.allclose(model(t.x, t.ids, t.share, t.guard), permuted, atol=1e-5)


def _fit(config, data, labels, epochs=30):
    data.labels = labels
    t, vocab, std = _tensors(data)
    torch.manual_seed(0)
    model = GNN(config, len(vocab))
    train(model, t, epochs=epochs, batch_size=256, lr=0.01, seed=0)
    return predict(model, t, std)


def _r2(predicted, labels, k):
    e = labels[:, :, k] - predicted[:, :, k]
    return 1 - e.var() / labels[:, :, k].var()


def test_centered_residual_cannot_rebuild_an_additive_rating():
    rng = np.random.default_rng(5)
    n = 3000
    data = _stints(rng, n)
    skill = rng.normal(0, 1, 40)
    ids = data.players.astype(int)
    labels = np.zeros((n, 2, T))
    labels[:, 0, 0] = skill[ids[:, :5]].sum(1) - skill[ids[:, 5:]].sum(1)
    labels[:, 1, 0] = -labels[:, 0, 0]
    labels += rng.normal(0, 0.5, labels.shape)
    loose = GNNConfig(
        relations=(), no_additive=(0,), center_interactions=False, delta_penalty=1.0
    )
    rebuilt = _r2(_fit(loose, data, labels), labels, 0)
    centered = _r2(
        _fit(replace(loose, center_interactions=True), data, labels), labels, 0
    )
    assert rebuilt > 0.5
    # The rest comes from the 40-player roster (see the FM's test).
    assert centered < 0.3
    assert centered < rebuilt / 2


def _planted_rating(rng, players, n):
    data = _stints(rng, n, players=players)
    skill = rng.normal(0, 1, players)
    ids = data.players.astype(int)
    labels = np.zeros((n, 2, T))
    labels[:, 0, 0] = skill[ids[:, :5]].sum(1) - skill[ids[:, 5:]].sum(1)
    labels[:, 1, 0] = -labels[:, 0, 0]
    return data, labels + rng.normal(0, 0.5, labels.shape)


def test_absorber_stops_message_passing_from_rebuilding_a_rating():
    """With messages, every node sees its lineup's and the opponents' totals,
    and the centered product readout can rebuild an additive rating from them
    (centering alone does not prevent it). Partialling it out with an
    absorbing additive head, dropped at prediction, does."""
    rng = np.random.default_rng(5)
    data, labels = _planted_rating(rng, players=400, n=12000)
    base = GNNConfig(no_additive=(0,), delta_penalty=1.0)
    leaked = _r2(_fit(base, data, labels, epochs=15), labels, 0)
    absorbed = _r2(
        _fit(replace(base, absorb_additive=True), data, labels, epochs=15), labels, 0
    )
    assert leaked > 0.5  # 0.87 when measured
    assert absorbed < 0.05


def test_guard_edges_carry_a_planted_matchup_signal():
    """Each attacker is guarded by one defender, assigned at random per stint;
    the label is the sum of attacker x defender types. Only the guard edges
    tell which pairs met."""
    rng = np.random.default_rng(6)
    n = 4000
    share = np.zeros((n, 2, 5, 5))
    pairing = np.stack([rng.permutation(5) for _ in range(2 * n)]).reshape(n, 2, 5)
    for s in range(2):
        share[np.arange(n)[:, None], s, np.arange(5)[None, :], pairing[:, s]] = 1.0
    data = _stints(rng, n, share=share.reshape(n, 50))
    kind = rng.integers(0, 2, 40) * 2 - 1.0
    ids = data.players.astype(int)
    # The type shows, noisily, in one profile column (real profiles carry
    # information; an ID-only type is learned too, only far more slowly).
    data.nodes[:, :, 0] = kind[ids] + rng.normal(0, 0.5, ids.shape)
    labels = np.zeros((n, 2, T))
    for s, (o, d) in enumerate(
        ((slice(0, 5), slice(5, 10)), (slice(5, 10), slice(0, 5)))
    ):
        att, dfn = kind[ids[:, o]], kind[ids[:, d]]
        labels[:, s, 2] = (att * np.take_along_axis(dfn, pairing[:, s], axis=1)).sum(1)
    labels += rng.normal(0, 0.3, labels.shape)
    common = dict(no_additive=(2,), absorb_additive=True, delta_penalty=1.0)
    full = _r2(_fit(GNNConfig(**common), data, labels), labels, 2)
    blind = _r2(
        _fit(GNNConfig(relations=("teammate", "opponent"), **common), data, labels),
        labels,
        2,
    )
    assert full > blind + 0.15


def test_gradients_reach_every_block_and_batches_are_isolated():
    rng = np.random.default_rng(7)
    data = _stints(rng, 32)
    data.labels = rng.normal(size=(32, 2, T))
    t, *_ = _tensors(data)
    model = GNN(FULL, 40).train()
    from nba_ou.data_processing.player_graph.fm import stint_loss

    loss = stint_loss(model, t).mean() + model.penalty(32)
    loss.backward()
    missing = [
        name
        for name, p in model.named_parameters()
        if p.grad is None or not torch.any(p.grad != 0)
    ]
    assert missing == []
    model.eval()
    whole = model.forward_batch(t)
    # A stint's prediction depends on its own graph only (no batch effects,
    # nothing from other dates).
    part = model.forward_batch(t.take(slice(5, 9)))
    assert torch.allclose(whole[5:9], part, atol=1e-6)
    changed = t.take(slice(0, 32))
    changed.x[10:] = torch.randn_like(changed.x[10:])
    assert torch.allclose(model.forward_batch(changed)[:10], whole[:10], atol=1e-6)


def test_checkpoint_round_trip_and_warm_start(tmp_path):
    rng = np.random.default_rng(8)
    data = _stints(rng, 50)
    t, vocab, std = _tensors(data)
    model = GNN(NO_GUARD, len(vocab))
    train(model, t, epochs=1, batch_size=16)
    save_checkpoint(tmp_path / "gnn.pt", model, vocab, std, {"fold": "2017-01"})
    loaded, vocab_back, _, extra = load_checkpoint(tmp_path / "gnn.pt")
    assert isinstance(loaded, GNN) and loaded.config == model.config
    assert torch.equal(loaded.forward_batch(t), model.eval().forward_batch(t))
    grown = PlayerVocab(vocab.ids)
    grown.extend(["new"])
    bigger = GNN(NO_GUARD, len(grown))
    warm_start(bigger, loaded, grown, vocab_back)
    assert torch.equal(bigger.eval().forward_batch(t), loaded.forward_batch(t))


def test_exact_centering_fixes_a_short_first_fold():
    """27 steps (the first fold) leave the running mean short of the
    embeddings' mean; the exact training mean centers them, and it reads the
    training stints only."""
    from nba_ou.data_processing.player_graph.fm import finalize_centering

    rng = np.random.default_rng(10)
    data = _stints(rng, 9 * 256)
    data.labels = rng.normal(size=(9 * 256, 2, T))
    t, vocab, _ = _tensors(data)
    torch.manual_seed(0)
    model = GNN(FULL, len(vocab))
    train(model, t, epochs=3, batch_size=256)  # 27 steps
    model.eval()
    with torch.no_grad():
        raw = model.uncentered(t)["embedding"].reshape(-1, FULL.dim)
        stale = (raw - model.embedding_mean).mean(0).abs().max().item()
    finalize_centering(model, t)
    with torch.no_grad():
        exact = (raw - model.embedding_mean).mean(0).abs().max().item()
    assert stale > 0.05
    assert exact < 1e-5
    # Only the given (training) stints define it.
    mean = model.embedding_mean.clone()
    other = _tensors(_stints(np.random.default_rng(11), 64))[0]
    model.forward_batch(other)  # eval-mode prediction never moves it
    assert torch.equal(model.embedding_mean, mean)
    finalize_centering(model, t.take(slice(0, 1024)))
    assert not torch.equal(model.embedding_mean, mean)


def test_exact_centering_for_the_fm_blocks():
    from nba_ou.data_processing.player_graph.fm import (
        FactorizationMachine,
        FMConfig,
        finalize_centering,
    )

    rng = np.random.default_rng(12)
    data = _stints(rng, 512)
    data.labels = rng.normal(size=(512, 2, T))
    t, vocab, _ = _tensors(data)
    torch.manual_seed(0)
    model = FactorizationMachine(
        FMConfig(edges=False, center_interactions=True), len(vocab)
    )
    train(model, t, epochs=1, batch_size=256)
    finalize_centering(model, t)
    with torch.no_grad():
        for name, values in model.uncentered(t).items():
            centered = values.reshape(-1, values.shape[-1]) - model.centering_buffer(
                name
            )
            assert centered.mean(0).abs().max().item() < 1e-5
