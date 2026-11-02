"""Phase 2 smoke-test GNN: tables -> tensors -> model -> loss -> checkpoint."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from nba_ou.data_processing.player_graph.node_profiles import (  # noqa: E402
    PROFILE_COLUMNS,
)
from nba_ou.data_processing.player_graph.smoke_gnn import (  # noqa: E402
    NODE_FEATURES,
    SmokeConfig,
    SmokeGNN,
    Standardizer,
    guard_inputs,
    load_checkpoint,
    node_inputs,
    save_checkpoint,
    to_tensors,
    train,
    weighted_loss,
)
from nba_ou.data_processing.player_graph.stint_graph import (  # noqa: E402
    build_stint_graphs,
)

DATE = pd.Timestamp("2018-11-01")
HOME = [f"h{k}" for k in range(7)]
AWAY = [f"a{k}" for k in range(7)]


def _stints(n=12, seed=0, permute_home=None):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        home = sorted(rng.choice(HOME, 5, replace=False))
        away = sorted(rng.choice(AWAY, 5, replace=False))
        if permute_home is not None:
            home = [home[k] for k in permute_home]
        row = {
            "game_id": "0021800001",
            "seg_idx": i,
            "game_date": DATE,
            "seconds": float(rng.integers(30, 300)),
            "home_lineup": np.array(home),
            "away_lineup": np.array(away),
        }
        for side in ("home", "away"):
            fga = float(rng.integers(0, 8))
            row |= {
                f"{side}_pts": float(rng.integers(0, 12)),
                f"{side}_poss": float(rng.integers(1, 8)),
                f"{side}_tov": float(rng.integers(0, 2)),
                f"{side}_fga": fga,
                f"{side}_fg3a": float(rng.integers(0, fga + 1)),
                f"{side}_fta": float(rng.integers(0, 3)),
                f"{side}_oreb": float(rng.integers(0, 2)),
                f"{side}_dreb": float(rng.integers(0, 4)),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def _expected():
    rng = np.random.default_rng(1)
    rows = [
        {
            "game_id": "0021800001",
            "off_player_id": off,
            "def_player_id": dfn,
            "r_hat": rng.uniform(0.02, 0.4),
            "prior_weight": rng.uniform(0, 1),
            "hist_cofloor_seconds_decayed": rng.uniform(0, 5000),
            "has_pair_history": bool(rng.integers(0, 2)),
        }
        for attackers, defenders in ((HOME, AWAY), (AWAY, HOME))
        for off in attackers
        for dfn in defenders
    ]
    return pd.DataFrame(rows)


def _profiles():
    rng = np.random.default_rng(2)
    players = HOME + AWAY
    frame = pd.DataFrame(
        rng.uniform(0.1, 30, (len(players), len(PROFILE_COLUMNS))),
        columns=list(PROFILE_COLUMNS),
    )
    frame["has_box_history"] = True
    frame.loc[0, "days_since_last_game"] = np.nan  # no appearance in the window
    return frame.assign(as_of_date=DATE, player_id=players)


def _fit_tensors(stints=None):
    graphs = build_stint_graphs(
        stints if stints is not None else _stints(), _expected()
    )
    nodes = node_inputs(graphs, _profiles())
    stats = Standardizer.fit(nodes, guard_inputs(graphs), graphs)
    return graphs, nodes, stats, to_tensors(graphs, nodes, stats)


def test_each_node_reads_its_own_players_profile():
    graphs, nodes, _, data = _fit_tensors()
    profiles = _profiles().set_index("player_id")
    for i in range(len(graphs)):
        for k, player in enumerate(graphs.players[i]):
            expected = profiles.loc[player, "pts_per36"]
            column = NODE_FEATURES.index("pts_per36")
            assert nodes[i, k, column] == pytest.approx(expected)
    assert (nodes[:, :5, -1] == 1).all() and (nodes[:, 5:, -1] == 0).all()
    assert data.nodes.shape == (len(graphs), 10, len(NODE_FEATURES))
    assert torch.isfinite(data.nodes).all() and torch.isfinite(data.guard).all()


def test_a_node_without_a_profile_is_an_error():
    graphs = build_stint_graphs(_stints(), _expected())
    profiles = _profiles()
    with pytest.raises(ValueError, match="no profile"):
        node_inputs(graphs, profiles.loc[profiles.player_id.ne("h0")])


def test_predictions_do_not_depend_on_player_order_within_a_side():
    graphs, nodes, stats, data = _fit_tensors()
    torch.manual_seed(0)
    model = SmokeGNN().eval()
    permuted = build_stint_graphs(_stints(permute_home=[4, 2, 0, 3, 1]), _expected())
    other = to_tensors(permuted, node_inputs(permuted, _profiles()), stats)
    assert not torch.equal(data.nodes, other.nodes)
    with torch.no_grad():
        a = model(data.nodes, data.guard, data.share)
        b = model(other.nodes, other.guard, other.share)
    assert torch.allclose(a, b, atol=1e-5)


def test_missing_labels_never_enter_the_loss():
    stints = _stints()
    stints.loc[0, "home_fga"] = 0.0  # 3PA/FGA and FTA/FGA undefined
    stints.loc[1, "away_poss"] = -0.5  # no possessions: weight 0, all NaN
    _, _, _, data = _fit_tensors(stints)
    assert not data.mask[0, 0, 3] and not data.mask[1, 1].any()
    prediction = torch.zeros_like(data.labels)
    loss, _ = weighted_loss(prediction, data)
    data.labels[0, 0, 3] = 1e6  # a value behind the mask changes nothing
    again, _ = weighted_loss(prediction, data)
    assert torch.isfinite(loss) and again == loss


def test_every_parameter_gets_a_gradient():
    _, _, _, data = _fit_tensors()
    model = SmokeGNN()
    loss, _ = weighted_loss(model(data.nodes, data.guard, data.share), data)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_it_can_overfit_a_tiny_batch():
    _, _, _, data = _fit_tensors(_stints(n=8))
    torch.manual_seed(0)
    model = SmokeGNN(SmokeConfig(dim=32))
    history = train(model, data, epochs=300, batch_size=8, lr=1e-2)
    assert history[-1] < 0.1 * history[0]


def test_checkpoint_round_trip_gives_identical_predictions(tmp_path):
    _, _, stats, data = _fit_tensors()
    torch.manual_seed(0)
    model = SmokeGNN(SmokeConfig(dim=16, layers=1))
    train(model, data, epochs=2, batch_size=4)
    model.eval()
    path = tmp_path / "smoke.pt"
    save_checkpoint(path, model, stats, {"note": "test"})
    loaded, loaded_stats, extra = load_checkpoint(path)
    assert loaded_stats == stats and extra == {"note": "test"}
    with torch.no_grad():
        assert torch.equal(
            model(data.nodes, data.guard, data.share),
            loaded(data.nodes, data.guard, data.share),
        )
