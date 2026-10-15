"""Phase 6, step 6a: the factorization machine (plain PyTorch).

Per stint and offensive side (O attacking, D defending, ``h`` = +1 when O is
home), for every target ``t`` (plan phase 6)::

    y_t = mu_t + h * hca_t + sum_{i in O} off_it - sum_{j in D} def_jt
        + 1/10 sum_{i<k in O} c1_t . (w_i * w_k)        offensive chemistry
        + 1/10 sum_{j<l in D} c2_t . (w'_j * w'_l)      defensive chemistry
        + 1/25 c3_t . (sum_O u_i * sum_D v_j)           team style (uniform)
        + 1/5 sum_{i in O} sum_{j in D} m_ij c4_t . (a_i * b_j)   matchups

``m_ij`` are the **expected** guarding shares of the stint graph (sum to 1 per
attacker; never the observed ones). ``x * y`` is the elementwise product and
``c_t`` a per-target readout over the embedding dimensions, so the embeddings
are shared by all targets and each target reads its own weighted inner
product (``c = 1`` is the plain dot product). The pair terms are means, so an
interaction has the scale of one pair.

**Edge head** (time-attributed edge targets, ``edge_targets``): attacker i
against defender j produces ``f_t(i, j) = mu_t + h * hca_t + oe_it + de_jt +
ce_t . (a_i * b_j)``, and an attacker-game is scored on ``sum_j s_ij f_t(i, j)``
with ``s_ij`` the defender's observed share of the attacker's possessions (a
label weight, never an input). ``a`` and ``b`` are the matchup vectors of the
stint model: the edge targets supervise the same embeddings.

**Prior inside the model.** Every per-player block (the scalars ``off``,
``def``, ``oe``, ``de`` per target; the vectors ``w``, ``w'``, ``u``, ``v``,
``a``, ``b``) is ``A x_i + delta_i``: ``x_i`` the player's standardized as-of
node features (they change with the date), ``A`` shared, ``delta_i`` a
per-player deviation penalized by ``delta_penalty`` pseudo-stints, so it only
grows with data. A player first seen after the training cutoff has no
``delta`` (index 0, fixed at zero) and is described by his profile alone.

Switches in :class:`FMConfig` give the plan's ladder: additive only (no
interaction terms), without the matchup term, with it, with or without edge
supervision.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .pretraining_data import NODE_COLUMNS

CHECKPOINT_VERSION = 1

#: Vectors that enter the stint model only through products.
INTERACTION_BLOCKS = ("w", "w2", "u", "v", "a", "b")
CENTER_MOMENTUM = 0.05

#: Node columns that are counts or durations: they enter as ``log1p``.
LOG_COLUMNS = (
    "minutes_window",
    "games_window",
    "minutes_decayed",
    "days_since_last_game",
    "games_in_data",
    "position_starts",
)
#: A player with no appearance in the profile window: as long ago as it allows.
MISSING_DAYS = 730.0


def node_features(raw: np.ndarray, columns: Iterable[str] = NODE_COLUMNS) -> np.ndarray:
    """Raw node columns (``(..., len(columns))``) -> model features."""
    columns = list(columns)
    out = np.array(raw, dtype=np.float64, copy=True)
    days = columns.index("days_since_last_game")
    out[..., days] = np.where(np.isnan(out[..., days]), MISSING_DAYS, out[..., days])
    for name in LOG_COLUMNS:
        k = columns.index(name)
        out[..., k] = np.log1p(np.clip(out[..., k], 0, None))
    if not np.isfinite(out).all():
        raise ValueError("Non-finite node features")
    return out


# --------------------------------------------------------------------------
# Players and standardization
# --------------------------------------------------------------------------


class PlayerVocab:
    """Player id -> embedding row; row 0 is "no own parameters"."""

    def __init__(self, ids: Iterable[str] = ()) -> None:
        self.ids: list[str] = []
        self.index: dict[str, int] = {}
        self.extend(ids)

    def __len__(self) -> int:
        return len(self.ids)

    def extend(self, ids: Iterable[str]) -> None:
        for pid in sorted({str(p) for p in np.asarray(list(ids)).reshape(-1)}):
            if pid not in self.index:
                self.ids.append(pid)
                self.index[pid] = len(self.ids)

    def lookup(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids).astype(str)
        flat = [self.index.get(p, 0) for p in ids.reshape(-1)]
        return np.asarray(flat, dtype=np.int64).reshape(ids.shape)


@dataclass(frozen=True)
class Scaler:
    """Mean and SD as lists (checkpoint-safe)."""

    mean: list[float]
    std: list[float]

    @classmethod
    def fit(cls, values: np.ndarray, weights: np.ndarray | None = None) -> Scaler:
        """Per last-axis column; ``weights`` (same shape) ignore NaN / zero."""
        flat = values.reshape(-1, values.shape[-1])
        if weights is None:
            mean, std = flat.mean(axis=0), flat.std(axis=0)
        else:
            w = weights.reshape(-1, values.shape[-1])
            w = np.where(np.isfinite(flat), w, 0.0)
            v = np.nan_to_num(flat)
            total = w.sum(axis=0)
            safe = np.where(total > 0, total, 1.0)
            # A column with no weight (no data yet): identity scaling.
            mean = np.where(total > 0, (w * v).sum(axis=0) / safe, 0.0)
            std = np.where(
                total > 0, np.sqrt((w * (v - mean) ** 2).sum(axis=0) / safe), 1.0
            )
        return cls(mean.tolist(), np.where(std > 1e-9, std, 1.0).tolist())

    def apply(self, values: np.ndarray) -> np.ndarray:
        return (values - np.asarray(self.mean)) / np.asarray(self.std)

    def invert(self, values: np.ndarray) -> np.ndarray:
        return values * np.asarray(self.std) + np.asarray(self.mean)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FMConfig:
    """Architecture and penalties (untuned starting values; decided
    2026-10-14: dimension 16, with 8 as a possible predeclared ablation)."""

    n_features: int = len(NODE_COLUMNS)
    n_targets: int = 6
    n_edge_targets: int = 2
    dim: int = 16
    chemistry: bool = True
    style: bool = True
    matchup: bool = True
    edges: bool = True
    edge_task_weight: float = 1.0
    #: Edge head terms beyond the attacker's own ``oe`` (the edge-head
    #: ablation): defenders' ``sum_j s_ij de_j``, and the matchup product.
    edge_defender: bool = True
    edge_matchup: bool = True
    #: Capacity control of the edge-head ablation: ``ce . (a_i * g)`` with
    #: ``g`` one learned vector, the matchup product's attacker capacity
    #: without any defender dependence.
    edge_capacity: bool = False
    #: Target indices whose per-player additive terms (``off``, ``def``) are
    #: switched off: the interaction-only residuals over RAPM, where RAPM
    #: already holds the individual contributions (decided 2026-10-15).
    no_additive: tuple[int, ...] = ()
    #: Center the stint interaction vectors on their running mean over
    #: players (as batch normalization does), so an interaction term's
    #: average over random partners is ~0 and it cannot act as a hidden
    #: additive rating.
    center_interactions: bool = False
    #: False trains on the edge loss alone (edge-only models of the ablation;
    #: ``delta_penalty`` is then in attacker-games).
    stint_loss: bool = True
    #: ``delta`` shrinkage, in training stints: sum ||delta||^2 * this / n.
    delta_penalty: float = 300.0
    #: L2 on the shared prior map ``A``.
    prior_penalty: float = 1e-4

    @property
    def interactions(self) -> bool:
        return self.chemistry or self.style or self.matchup


class FactorizationMachine(nn.Module):
    def __init__(self, config: FMConfig, n_players: int) -> None:
        super().__init__()
        self.config = config
        t, te, d = config.n_targets, config.n_edge_targets, config.dim
        sizes = {"off": t, "def": t}
        if config.chemistry:
            sizes |= {"w": d, "w2": d}
        if config.style:
            sizes |= {"u": d, "v": d}
        if config.matchup or (
            config.edges and (config.edge_matchup or config.edge_capacity)
        ):
            sizes |= {"a": d, "b": d}
        if config.edges:
            sizes |= {"oe": te, "de": te}
        self.sizes = sizes
        offsets = np.cumsum([0, *sizes.values()])
        self.slices = {
            name: slice(int(start), int(stop))
            for name, start, stop in zip(sizes, offsets[:-1], offsets[1:], strict=True)
        }
        width = int(offsets[-1])
        self.prior = nn.Linear(config.n_features, width)
        self.delta = nn.Embedding(n_players + 1, width, padding_idx=0)
        nn.init.zeros_(self.delta.weight)
        with torch.no_grad():
            self.prior.weight.mul_(0.1)
            self.prior.bias.zero_()
        self.mu = nn.Parameter(torch.zeros(t))
        self.hca = nn.Parameter(torch.zeros(t))
        terms = sum((config.chemistry * 2, config.style, config.matchup))
        self.readout = nn.Parameter(torch.ones(max(terms, 1), t, d))
        if config.edges:
            self.mu_edge = nn.Parameter(torch.zeros(te))
            self.hca_edge = nn.Parameter(torch.zeros(te))
            self.readout_edge = nn.Parameter(torch.ones(te, d))
            if config.edge_capacity:
                self.capacity = nn.Parameter(torch.zeros(d))
                self.readout_capacity = nn.Parameter(torch.ones(te, d))
        additive = torch.ones(t)
        additive[list(config.no_additive)] = 0.0
        self.register_buffer("additive_mask", additive, persistent=False)
        for name in INTERACTION_BLOCKS:
            if name in sizes:
                self.register_buffer(f"mean_{name}", torch.zeros(d))

    # Per-player blocks ---------------------------------------------------

    def blocks(self, x: torch.Tensor, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """``A x + delta`` split into named blocks; ``x`` ``(..., F)``."""
        values = self.prior(x) + self.delta(ids)
        return {name: values[..., s] for name, s in self.slices.items()}

    # Stints --------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, ids: torch.Tensor, share: torch.Tensor
    ) -> torch.Tensor:
        """Standardized predictions ``(B, 2, T)``; side 0 = home on offense.

        ``x`` ``(B, 10, F)``, ``ids`` ``(B, 10)``, ``share`` ``(B, 50)`` in
        ``stint_graph.GUARD_EDGES`` order.
        """
        p = self.blocks(x, ids)
        if self.config.center_interactions:
            p = self._centered(p)
        m = share.reshape(-1, 2, 5, 5)  # (side, attacker k, defender l)
        out = []
        for side, (o, d, h) in enumerate(
            ((slice(0, 5), slice(5, 10), 1.0), (slice(5, 10), slice(0, 5), -1.0))
        ):
            additive = p["off"][:, o].sum(1) - p["def"][:, d].sum(1)
            y = self.mu + h * self.hca + additive * self.additive_mask
            term = 0
            if self.config.chemistry:
                y = y + _pair_mean(p["w"][:, o]) @ self.readout[term].T / 10
                y = y + _pair_mean(p["w2"][:, d]) @ self.readout[term + 1].T / 10
                term += 2
            if self.config.style:
                style = p["u"][:, o].sum(1) * p["v"][:, d].sum(1)
                y = y + style @ self.readout[term].T / 25
                term += 1
            if self.config.matchup:
                guarded = torch.einsum("bkl,bld->bkd", m[:, side], p["b"][:, d])
                matchup = (p["a"][:, o] * guarded).sum(1)
                y = y + matchup @ self.readout[term].T / 5
            out.append(y)
        return torch.stack(out, dim=1)

    def _centered(self, p: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Interaction vectors minus their mean over players: the batch's
        while training (updating a running mean), the running one otherwise."""
        out = dict(p)
        for name in INTERACTION_BLOCKS:
            if name not in p:
                continue
            running = getattr(self, f"mean_{name}")
            if self.training:
                mean = p[name].reshape(-1, p[name].shape[-1]).mean(0)
                with torch.no_grad():
                    running.mul_(1 - CENTER_MOMENTUM).add_(CENTER_MOMENTUM * mean)
            else:
                mean = running
            out[name] = p[name] - mean
        return out

    # Edges ---------------------------------------------------------------

    def forward_edges(
        self,
        attacker_x: torch.Tensor,
        attacker_ids: torch.Tensor,
        home: torch.Tensor,
        defender_x: torch.Tensor,
        defender_ids: torch.Tensor,
        share: torch.Tensor,
        pair_index: torch.Tensor,
    ) -> torch.Tensor:
        """Standardized attacker-game predictions ``(G, Te)``:
        ``sum_j s_ij f(i, j)`` (module docstring)."""
        if not self.config.edges:
            raise ValueError("This model has no edge head")
        a = self.blocks(attacker_x, attacker_ids)
        n = attacker_x.shape[0]
        s = share.unsqueeze(-1)
        # Shares sum to 1 per attacker-game, so the constant terms pass through.
        h = (home.float() * 2 - 1).unsqueeze(-1)
        out = self.mu_edge + h * self.hca_edge + a["oe"]
        if self.config.edge_capacity:
            out = out + (a["a"] * self.capacity) @ self.readout_capacity.T
        if self.config.edge_defender or self.config.edge_matchup:
            b = self.blocks(defender_x, defender_ids)
        if self.config.edge_defender:
            de = torch.zeros(n, self.config.n_edge_targets, device=s.device)
            out = out + de.index_add(0, pair_index, s * b["de"])
        if self.config.edge_matchup:
            bv = torch.zeros(n, self.config.dim, device=s.device)
            bv = bv.index_add(0, pair_index, s * b["b"])
            out = out + (a["a"] * bv) @ self.readout_edge.T
        return out

    # Penalty -------------------------------------------------------------

    def penalty(self, n_train: int) -> torch.Tensor:
        delta = self.delta.weight[1:]
        return (
            self.config.delta_penalty * (delta**2).sum() / max(n_train, 1)
            + self.config.prior_penalty * (self.prior.weight**2).sum()
        )


def _pair_mean(v: torch.Tensor) -> torch.Tensor:
    """``sum_{i<k} v_i * v_k`` over the five players (elementwise),
    ``((sum v)^2 - sum v^2) / 2``."""
    return (v.sum(1) ** 2 - (v**2).sum(1)) / 2


# --------------------------------------------------------------------------
# Data -> tensors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StintTensors:
    x: torch.Tensor  # (n, 10, F) standardized features
    ids: torch.Tensor  # (n, 10) vocabulary rows
    share: torch.Tensor  # (n, 50)
    y: torch.Tensor  # (n, 2, T) standardized, 0 where missing
    w: torch.Tensor  # (n, 2, T) exposures, 0 where missing

    def __len__(self) -> int:
        return self.x.shape[0]

    def take(self, index) -> StintTensors:
        return StintTensors(
            *(getattr(self, k)[index] for k in self.__dataclass_fields__)
        )


@dataclass(frozen=True)
class EdgeTensors:
    attacker_x: torch.Tensor  # (g, F)
    attacker_ids: torch.Tensor  # (g,)
    home: torch.Tensor  # (g,) bool
    y: torch.Tensor  # (g, Te) standardized
    w: torch.Tensor  # (g,) attacker partial possessions
    defender_x: torch.Tensor  # (p, F)
    defender_ids: torch.Tensor  # (p,)
    share: torch.Tensor  # (p,)
    pair_index: torch.Tensor  # (p,) contiguous per attacker-game
    offsets: np.ndarray  # (g + 1,) pair rows of each attacker-game

    def __len__(self) -> int:
        return self.attacker_x.shape[0]

    def take(self, games: np.ndarray) -> EdgeTensors:
        """Attacker-games ``games`` (indices) with their pairs."""
        games = np.asarray(games)
        starts, stops = self.offsets[games], self.offsets[games + 1]
        counts = stops - starts
        rows = np.repeat(starts - np.cumsum(np.r_[0, counts[:-1]]), counts) + np.arange(
            counts.sum()
        )
        rows_t = torch.as_tensor(rows)
        return EdgeTensors(
            attacker_x=self.attacker_x[games],
            attacker_ids=self.attacker_ids[games],
            home=self.home[games],
            y=self.y[games],
            w=self.w[games],
            defender_x=self.defender_x[rows_t],
            defender_ids=self.defender_ids[rows_t],
            share=self.share[rows_t],
            pair_index=torch.as_tensor(np.repeat(np.arange(len(games)), counts)),
            offsets=np.r_[0, np.cumsum(counts)],
        )


@dataclass(frozen=True)
class Standardization:
    """Everything fitted on the training rows that a checkpoint must carry."""

    features: Scaler
    targets: Scaler
    edge_targets: Scaler | None

    def to_dict(self) -> dict:
        return {
            "features": asdict(self.features),
            "targets": asdict(self.targets),
            "edge_targets": asdict(self.edge_targets) if self.edge_targets else None,
        }

    @classmethod
    def from_dict(cls, values: Mapping) -> Standardization:
        edge = values["edge_targets"]
        return cls(
            Scaler(**values["features"]),
            Scaler(**values["targets"]),
            Scaler(**edge) if edge else None,
        )


def fit_standardization(stints, edges=None) -> Standardization:
    """From training data: ``stints`` a ``PretrainingStints``, ``edges`` an
    ``EdgeBatch``."""
    features = Scaler.fit(node_features(stints.nodes))
    targets = Scaler.fit(stints.labels, stints.label_weight)
    edge = None
    if edges is not None and len(edges.weight):
        weights = np.broadcast_to(edges.weight[:, None], edges.targets.shape)
        edge = Scaler.fit(edges.targets, np.ascontiguousarray(weights))
    return Standardization(features, targets, edge)


def _tensor(values, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(values), dtype=dtype)


def stint_tensors(stints, vocab: PlayerVocab, std: Standardization) -> StintTensors:
    x = std.features.apply(node_features(stints.nodes))
    mask = np.isfinite(stints.labels) & (stints.label_weight > 0)
    y = np.where(mask, std.targets.apply(np.nan_to_num(stints.labels)), 0.0)
    return StintTensors(
        x=_tensor(x),
        ids=_tensor(vocab.lookup(stints.players), torch.long),
        share=_tensor(stints.guard_share),
        y=_tensor(y),
        w=_tensor(np.where(mask, stints.label_weight, 0.0)),
    )


def edge_tensors(edges, vocab: PlayerVocab, std: Standardization) -> EdgeTensors:
    if std.edge_targets is None:
        raise ValueError("No edge standardization: fit it with the edges")
    counts = np.bincount(edges.pair_game_index, minlength=len(edges.weight))
    if (np.diff(edges.pair_game_index) < 0).any():
        raise ValueError("Pair rows must be grouped by attacker-game in order")
    return EdgeTensors(
        attacker_x=_tensor(std.features.apply(node_features(edges.attacker_nodes))),
        attacker_ids=_tensor(vocab.lookup(edges.attacker), torch.long),
        home=_tensor(edges.attacker_is_home, torch.bool),
        y=_tensor(std.edge_targets.apply(edges.targets)),
        w=_tensor(edges.weight),
        defender_x=_tensor(std.features.apply(node_features(edges.defender_nodes))),
        defender_ids=_tensor(vocab.lookup(edges.defender), torch.long),
        share=_tensor(edges.share),
        pair_index=_tensor(edges.pair_game_index, torch.long),
        offsets=np.r_[0, np.cumsum(counts)],
    )


# --------------------------------------------------------------------------
# Loss, training, prediction
# --------------------------------------------------------------------------


def weighted_mse(
    prediction: torch.Tensor, y: torch.Tensor, w: torch.Tensor
) -> torch.Tensor:
    """Per-target exposure-weighted MSE (last axis = target)."""
    dims = tuple(range(prediction.ndim - 1))
    return ((prediction - y) ** 2 * w).sum(dims) / w.sum(dims).clamp_min(1e-9)


def stint_loss(model: FactorizationMachine, batch: StintTensors) -> torch.Tensor:
    return weighted_mse(model(batch.x, batch.ids, batch.share), batch.y, batch.w)


def edge_loss(model: FactorizationMachine, batch: EdgeTensors) -> torch.Tensor:
    prediction = model.forward_edges(
        batch.attacker_x,
        batch.attacker_ids,
        batch.home,
        batch.defender_x,
        batch.defender_ids,
        batch.share,
        batch.pair_index,
    )
    w = batch.w.unsqueeze(-1).expand_as(prediction)
    return weighted_mse(prediction, batch.y, w)


def train(
    model: FactorizationMachine,
    stints: StintTensors | None,
    edges: EdgeTensors | None = None,
    *,
    epochs: int,
    batch_size: int = 1024,
    lr: float = 3e-3,
    seed: int = 0,
    device: str = "cpu",
) -> list[dict]:
    """Adam over shuffled stint batches, each paired with an edge batch of the
    same share of the attacker-games. Returns per-epoch mean losses.

    A model without the stint loss (``stint_loss=False``) trains on edge
    batches of ``batch_size`` attacker-games alone."""
    model.to(device)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if not model.config.stint_loss:
        return _train_edges_only(
            model, edges, epochs, batch_size, optimizer, generator, device
        )
    use_edges = model.config.edges and edges is not None and len(edges) > 0
    n = len(stints)
    history = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator)
        steps = max(1, -(-n // batch_size))
        edge_order = (
            torch.randperm(len(edges), generator=generator).numpy()
            if use_edges
            else None
        )
        edge_batch = -(-len(edges) // steps) if use_edges else 0
        totals = {"stint": 0.0, "edge": 0.0, "penalty": 0.0}
        for step in range(steps):
            batch = _to(
                stints.take(order[step * batch_size : (step + 1) * batch_size]), device
            )
            stint = stint_loss(model, batch).mean()
            loss = stint + model.penalty(n)
            if use_edges:
                games = edge_order[step * edge_batch : (step + 1) * edge_batch]
                if len(games):
                    edge = edge_loss(model, _to(edges.take(games), device)).mean()
                    loss = loss + model.config.edge_task_weight * edge
                    totals["edge"] += edge.item() / steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals["stint"] += stint.item() / steps
            totals["penalty"] += model.penalty(n).item() / steps
        history.append(totals)
    return history


def _train_edges_only(
    model, edges, epochs, batch_size, optimizer, generator, device
) -> list[dict]:
    if edges is None or len(edges) == 0:
        raise ValueError("An edge-only model needs edges")
    n = len(edges)
    history = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator).numpy()
        steps = max(1, -(-n // batch_size))
        totals = {"stint": 0.0, "edge": 0.0, "penalty": 0.0}
        for step in range(steps):
            games = order[step * batch_size : (step + 1) * batch_size]
            edge = edge_loss(model, _to(edges.take(games), device)).mean()
            loss = edge + model.penalty(n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totals["edge"] += edge.item() / steps
            totals["penalty"] += model.penalty(n).item() / steps
        history.append(totals)
    return history


def _to(batch, device: str):
    if device == "cpu":
        return batch
    return type(batch)(
        **{
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.__dict__.items()
        }
    )


def predict(
    model: FactorizationMachine,
    stints: StintTensors,
    std: Standardization,
    batch_size: int = 8192,
    device: str = "cpu",
) -> np.ndarray:
    """Predictions in target units, ``(n, 2, T)``."""
    model.eval()
    model.to(device)
    parts = []
    with torch.no_grad():
        for start in range(0, len(stints), batch_size):
            b = _to(stints.take(slice(start, start + batch_size)), device)
            parts.append(model(b.x, b.ids, b.share).cpu().numpy())
    return std.targets.invert(np.concatenate(parts).astype(np.float64))


def predict_edges(
    model: FactorizationMachine, edges: EdgeTensors, std: Standardization
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        values = model.forward_edges(
            edges.attacker_x,
            edges.attacker_ids,
            edges.home,
            edges.defender_x,
            edges.defender_ids,
            edges.share,
            edges.pair_index,
        )
    return std.edge_targets.invert(values.numpy().astype(np.float64))


# --------------------------------------------------------------------------
# Warm starts and checkpoints
# --------------------------------------------------------------------------


def warm_start(
    new: FactorizationMachine,
    old: FactorizationMachine,
    new_vocab: PlayerVocab,
    old_vocab: PlayerVocab,
) -> None:
    """Copy ``old``'s parameters into ``new`` (same config), moving each known
    player's ``delta`` to his row in the grown vocabulary."""
    state = old.state_dict()
    delta = state.pop("delta.weight")
    new.load_state_dict(state, strict=False)
    with torch.no_grad():
        rows = new_vocab.lookup(np.asarray(old_vocab.ids))
        new.delta.weight[torch.as_tensor(rows)] = delta[1:]


def save_checkpoint(
    path,
    model: FactorizationMachine,
    vocab: PlayerVocab,
    std: Standardization,
    extra: Mapping | None = None,
) -> None:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version": CHECKPOINT_VERSION,
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "vocab": list(vocab.ids),
            "standardization": std.to_dict(),
            "node_columns": list(NODE_COLUMNS),
            "extra": dict(extra or {}),
        },
        path,
    )


def load_checkpoint(
    path,
) -> tuple[FactorizationMachine, PlayerVocab, Standardization, dict]:
    payload = torch.load(path, weights_only=True)
    if payload["version"] != CHECKPOINT_VERSION:
        raise ValueError(f"Checkpoint version {payload['version']} is not supported")
    if payload["node_columns"] != list(NODE_COLUMNS):
        raise ValueError("Checkpoint node columns differ from this code's")
    vocab = PlayerVocab()
    vocab.ids = list(payload["vocab"])
    vocab.index = {pid: k + 1 for k, pid in enumerate(vocab.ids)}
    model = FactorizationMachine(FMConfig(**payload["config"]), len(vocab))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return (
        model,
        vocab,
        Standardization.from_dict(payload["standardization"]),
        payload["extra"],
    )
