"""Phase 2 smoke-test GNN over the v0 stint graphs (plain PyTorch).

It checks the plumbing end to end, not the signal, and does not fix the phase 6
architecture:

    tables -> tensors -> message passing -> node embeddings -> pooling
    -> prediction head -> loss / backprop -> checkpoint save / reload

**Tables -> tensors.** A stint graph (``stint_graph.build_stint_graphs``) has 10
nodes in a fixed order and fixed edge templates, so a batch is a set of dense
arrays, no graph objects (no PyTorch Geometric):

* nodes ``(n, 10, F)``: the player's as-of profile on the stint's game date
  (``node_profiles``, read strictly before that date) plus ``is_home``;
  count-like columns enter as ``log1p``;
* guard attributes ``(n, 50, A)`` in ``GUARD_EDGES`` order, and the raw guard
  share ``m_ij`` that weights the guard messages;
* labels ``(n, 2, 6)`` per offensive side with their possession weights; a
  missing label (zero denominator) is masked, never zero-filled into the loss.

Everything is standardized with statistics of the training stints only
(labels: possession-weighted), and the statistics travel in the checkpoint.

**Model.** An input MLP, then relational layers: each node adds a mean message
from its 4 teammates, a mean message from its 5 opponents and, as an attacker,
a guard message from each of the 5 defenders conditioned on the edge
attributes and weighted by ``m_ij`` (the shares sum to 1). Mean pooling of
each five gives an (offense, defense) pair per offensive side, and a head
predicts the six labels. The output does not depend on the order of the
players within a side.

**Loss.** Possession-weighted squared error per label on standardized labels,
averaged over labels. Predicting 0 is the constant baseline (the weighted
training mean).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .node_profiles import DEFAULT_PARAMS as PROFILE_PARAMS
from .node_profiles import PROFILE_COLUMNS
from .stint_graph import LABELS, N_NODES, StintGraphs, message_passing_edges

CHECKPOINT_VERSION = 1

#: Profile columns that are counts or durations: they enter as ``log1p``.
LOG_FEATURES = (
    "minutes_window",
    "games_window",
    "minutes_decayed",
    "days_since_last_game",
    "games_in_data",
)
NODE_FEATURES = (*PROFILE_COLUMNS, "is_home")
GUARD_ATTRIBUTES = (
    "guard_share",
    "guard_r_hat",
    "guard_r_hat_known",
    "guard_prior_weight",
    "guard_log_exposure",
    "guard_has_history",
    "guard_fallback",
)


# --------------------------------------------------------------------------
# Tables -> tensors
# --------------------------------------------------------------------------


def node_inputs(graphs: StintGraphs, profiles: pd.DataFrame) -> np.ndarray:
    """Raw node features ``(n, 10, len(NODE_FEATURES))``.

    ``profiles``: node-profile rows keyed ``(as_of_date, player_id)``; each
    stint node reads the row of its own game date. Raises if a node has none.
    """
    n = len(graphs)
    nodes = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(np.repeat(graphs.game_date, N_NODES)),
            "player_id": graphs.players.reshape(-1).astype(str),
        }
    )
    nodes["as_of_date"] = nodes["as_of_date"].dt.normalize()
    table = profiles[["as_of_date", "player_id", *PROFILE_COLUMNS]].copy()
    table["player_id"] = table["player_id"].astype(str)
    table["as_of_date"] = pd.to_datetime(table["as_of_date"]).dt.normalize()
    joined = nodes.merge(
        table, on=["as_of_date", "player_id"], how="left", validate="many_to_one"
    )
    missing = joined["prior_weight"].isna()
    if missing.any():
        example = joined.loc[missing, ["as_of_date", "player_id"]].iloc[0].tolist()
        raise ValueError(
            f"{int(missing.sum())} stint nodes have no profile, e.g. {example}"
        )
    features = joined[list(PROFILE_COLUMNS)].astype(float)
    # No appearance in the window: as long ago as the window allows.
    features["days_since_last_game"] = features["days_since_last_game"].fillna(
        float(PROFILE_PARAMS.window_days)
    )
    for column in LOG_FEATURES:
        features[column] = np.log1p(features[column].clip(lower=0))
    out = features.to_numpy(np.float64).reshape(n, N_NODES, len(PROFILE_COLUMNS))
    is_home = np.broadcast_to(np.repeat([1.0, 0.0], 5)[None, :, None], (n, N_NODES, 1))
    return np.concatenate([out, is_home], axis=-1)


def guard_inputs(graphs: StintGraphs) -> np.ndarray:
    """Raw guard-edge attributes ``(n, 50, len(GUARD_ATTRIBUTES))``."""
    return np.stack(
        [getattr(graphs, name).astype(np.float64) for name in GUARD_ATTRIBUTES],
        axis=-1,
    )


@dataclass(frozen=True)
class Standardizer:
    """Training-set statistics, as plain lists so a checkpoint loads with
    ``torch.load(weights_only=True)``."""

    node_mean: list[float]
    node_std: list[float]
    guard_mean: list[float]
    guard_std: list[float]
    label_mean: list[float]
    label_std: list[float]

    @classmethod
    def fit(
        cls, nodes: np.ndarray, guard: np.ndarray, graphs: StintGraphs
    ) -> Standardizer:
        def moments(values: np.ndarray) -> tuple[list[float], list[float]]:
            flat = values.reshape(-1, values.shape[-1])
            std = flat.std(axis=0)
            return flat.mean(axis=0).tolist(), np.where(std > 1e-9, std, 1.0).tolist()

        node_mean, node_std = moments(nodes)
        guard_mean, guard_std = moments(guard)
        labels = graphs.labels
        weight = np.broadcast_to(graphs.label_weight[..., None], labels.shape)
        weight = np.where(np.isfinite(labels), weight, 0.0)
        values = np.nan_to_num(labels)
        total = weight.sum(axis=(0, 1))
        mean = (weight * values).sum(axis=(0, 1)) / total
        var = (weight * (values - mean) ** 2).sum(axis=(0, 1)) / total
        std = np.sqrt(var)
        return cls(
            node_mean,
            node_std,
            guard_mean,
            guard_std,
            mean.tolist(),
            np.where(std > 1e-9, std, 1.0).tolist(),
        )


@dataclass(frozen=True)
class StintTensors:
    nodes: torch.Tensor  # (n, 10, F) standardized
    guard: torch.Tensor  # (n, 50, A) standardized
    share: torch.Tensor  # (n, 50) raw m_ij, weights of the guard messages
    labels: torch.Tensor  # (n, 2, L) standardized, 0 where missing
    mask: torch.Tensor  # (n, 2, L) True where the label exists
    weight: torch.Tensor  # (n, 2) possessions

    def __len__(self) -> int:
        return self.nodes.shape[0]

    def subset(self, index: torch.Tensor | slice) -> StintTensors:
        return StintTensors(
            *(getattr(self, name)[index] for name in self.__dataclass_fields__)
        )


def to_tensors(
    graphs: StintGraphs, nodes: np.ndarray, stats: Standardizer
) -> StintTensors:
    """Standardized float32 tensors; raises on a non-finite input."""

    def scaled(values: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
        return (values - np.asarray(mean)) / np.asarray(std)

    node_values = scaled(nodes, stats.node_mean, stats.node_std)
    guard_values = scaled(guard_inputs(graphs), stats.guard_mean, stats.guard_std)
    for name, values in (("node", node_values), ("guard", guard_values)):
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {name} inputs")
    mask = np.isfinite(graphs.labels)
    labels = np.where(
        mask, scaled(np.nan_to_num(graphs.labels), stats.label_mean, stats.label_std), 0
    )

    def tensor(values: np.ndarray, dtype=torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(values), dtype=dtype)

    return StintTensors(
        nodes=tensor(node_values),
        guard=tensor(guard_values),
        share=tensor(graphs.guard_share),
        labels=tensor(labels),
        mask=tensor(mask, torch.bool),
        weight=tensor(graphs.label_weight),
    )


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SmokeConfig:
    n_node_features: int = len(NODE_FEATURES)
    n_guard_attributes: int = len(GUARD_ATTRIBUTES)
    n_labels: int = len(LABELS)
    dim: int = 32
    layers: int = 2


DEFAULT_CONFIG = SmokeConfig()


def _scatter_sum(messages: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Sum ``(B, E, d)`` edge messages into ``(B, 10, d)`` at ``dst``."""
    out = messages.new_zeros(messages.shape[0], N_NODES, messages.shape[2])
    return out.index_add(1, dst, messages)


class RelationalLayer(nn.Module):
    """One round of typed message passing with a residual and LayerNorm."""

    def __init__(self, dim: int, n_guard_attributes: int) -> None:
        super().__init__()
        self.self_map = nn.Linear(dim, dim)
        self.teammate = nn.Linear(dim, dim)
        self.opponent = nn.Linear(dim, dim)
        self.guard = nn.Sequential(
            nn.Linear(dim + n_guard_attributes, dim), nn.ReLU(), nn.Linear(dim, dim)
        )
        self.norm = nn.LayerNorm(dim)
        edges = message_passing_edges()
        for relation in ("teammate", "opponent", "guards"):
            index = torch.as_tensor(edges[relation]["index"], dtype=torch.long)
            self.register_buffer(f"{relation}_src", index[0], persistent=False)
            self.register_buffer(f"{relation}_dst", index[1], persistent=False)
        # Every node has 4 teammates and 5 opponents in every stint graph.
        self.teammate_degree = 4.0
        self.opponent_degree = 5.0

    def forward(
        self, h: torch.Tensor, guard: torch.Tensor, share: torch.Tensor
    ) -> torch.Tensor:
        teammates = _scatter_sum(h[:, self.teammate_src], self.teammate_dst)
        opponents = _scatter_sum(h[:, self.opponent_src], self.opponent_dst)
        guard_in = torch.cat([h[:, self.guards_src], guard], dim=-1)
        guarded = _scatter_sum(
            self.guard(guard_in) * share.unsqueeze(-1), self.guards_dst
        )
        update = (
            self.self_map(h)
            + self.teammate(teammates / self.teammate_degree)
            + self.opponent(opponents / self.opponent_degree)
            + guarded
        )
        return self.norm(h + torch.relu(update))


class SmokeGNN(nn.Module):
    def __init__(self, config: SmokeConfig = DEFAULT_CONFIG) -> None:
        super().__init__()
        self.config = config
        dim = config.dim
        self.encoder = nn.Sequential(
            nn.Linear(config.n_node_features, dim), nn.ReLU(), nn.Linear(dim, dim)
        )
        self.layers = nn.ModuleList(
            RelationalLayer(dim, config.n_guard_attributes)
            for _ in range(config.layers)
        )
        self.head = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, config.n_labels)
        )

    def embed(
        self, nodes: torch.Tensor, guard: torch.Tensor, share: torch.Tensor
    ) -> torch.Tensor:
        """Node embeddings ``(B, 10, dim)``."""
        h = self.encoder(nodes)
        for layer in self.layers:
            h = layer(h, guard, share)
        return h

    def forward(
        self, nodes: torch.Tensor, guard: torch.Tensor, share: torch.Tensor
    ) -> torch.Tensor:
        """Standardized label predictions ``(B, 2, n_labels)``; side 0 = home
        on offense."""
        h = self.embed(nodes, guard, share)
        home, away = h[:, :5].mean(dim=1), h[:, 5:].mean(dim=1)
        sides = torch.stack(
            [torch.cat([home, away], -1), torch.cat([away, home], -1)], dim=1
        )
        return self.head(sides)


# --------------------------------------------------------------------------
# Loss, training, checkpoints
# --------------------------------------------------------------------------


def weighted_loss(
    prediction: torch.Tensor, batch: StintTensors
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(mean over labels, per-label)`` possession-weighted squared error."""
    weight = batch.weight.unsqueeze(-1) * batch.mask
    error = (prediction - batch.labels) ** 2 * weight
    per_label = error.sum(dim=(0, 1)) / weight.sum(dim=(0, 1)).clamp_min(1e-9)
    return per_label.mean(), per_label


def predict(
    model: SmokeGNN, data: StintTensors, batch_size: int = 4096
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat(
            [
                model(part.nodes, part.guard, part.share)
                for part in (
                    data.subset(slice(start, start + batch_size))
                    for start in range(0, len(data), batch_size)
                )
            ]
        )


def evaluate(model: SmokeGNN, data: StintTensors) -> dict[str, np.ndarray]:
    """Per-label loss of the model and of the constant baseline (predict 0)."""
    prediction = predict(model, data)
    _, model_loss = weighted_loss(prediction, data)
    _, constant = weighted_loss(torch.zeros_like(prediction), data)
    return {"model": model_loss.numpy(), "constant": constant.numpy()}


def train(
    model: SmokeGNN,
    data: StintTensors,
    *,
    epochs: int,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = 0,
) -> list[float]:
    """Adam over shuffled mini-batches; returns the mean loss of each epoch."""
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(data), generator=generator)
        losses = []
        for start in range(0, len(data), batch_size):
            batch = data.subset(order[start : start + batch_size])
            loss, _ = weighted_loss(model(batch.nodes, batch.guard, batch.share), batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
    return history


def save_checkpoint(
    path: Path,
    model: SmokeGNN,
    stats: Standardizer,
    extra: Mapping[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "stats": asdict(stats),
            "node_features": list(NODE_FEATURES),
            "guard_attributes": list(GUARD_ATTRIBUTES),
            "labels": list(LABELS),
            "extra": dict(extra or {}),
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[SmokeGNN, Standardizer, dict]:
    """Model (eval mode), its standardizer and the stored ``extra``.

    Raises if the checkpoint was written for different inputs or labels.
    """
    payload = torch.load(path, weights_only=True)
    if payload["version"] != CHECKPOINT_VERSION:
        raise ValueError(f"Checkpoint version {payload['version']} is not supported")
    expected = {
        "node_features": list(NODE_FEATURES),
        "guard_attributes": list(GUARD_ATTRIBUTES),
        "labels": list(LABELS),
    }
    for key, value in expected.items():
        if payload[key] != value:
            raise ValueError(f"Checkpoint {key} differ from this code's")
    model = SmokeGNN(SmokeConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, Standardizer(**payload["stats"]), payload["extra"]
