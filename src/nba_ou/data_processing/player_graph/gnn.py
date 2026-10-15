"""Phase 6, step 6b: the relational GNN (plain PyTorch), decided 2026-10-15.

The FM closed with no measurable stint-level signal from learned
interactions beyond RAPM; the GNN asks whether contextual, non-linear message
passing over the stint graph finds any, with the same harness, targets,
residual construction and calibration (``fm.py``, ``pretraining_oof``).

**Encoder.** Every node starts from its standardized as-of profile and a
penalized player-ID embedding (``delta``, zero for a player unseen at the
checkpoint): ``h0 = MLP(x_i) + e_i``. Each of ``layers`` relational layers::

    h <- LayerNorm(h + ReLU(W_self h + B_tm(tm) + B_opp(opp) + B_g(g) + B_gb(gb)))

* ``tm``: mean of the 4 teammates' ``h``; ``opp``: mean of the 5 opponents';
* ``g`` (attacker): ``sum_l m_kl [h_l, e_kl]`` over the 5 defenders, with
  ``m`` the **expected** guarding share (sums to 1) and ``e_kl`` the guard
  edge's attributes (:func:`guard_features`: share, expected rate, prior
  weight, history, provenance); ``gb`` (defender): the same over the attackers
  he is expected to guard, divided by his total share.

Observed guarding of the encoded game never enters: the inputs come from the
pretraining dataset's input tables only (``pretraining_data``).

**Capacity-matched ablations** (``GNNConfig.relations``). A relation switched
off keeps its branch and parameters but receives the node's own state
(``h``, or ``[h, 0]`` for the guard branches) instead of messages, so the
node-only control (no relation), the teammate / opponent GNN (no guards) and
the full GNN have identical parameter counts, inputs, heads and training
budget; only the information that flows between players differs.

**Heads.** As the FM's (``fm.FactorizationMachine``):

* per-player additive scalars (``off``, ``def``) from ``A x + delta``, for
  the four auxiliary targets only: points and pace are residuals over the
  daily RAPM base, which holds the individual contributions
  (``no_additive``);
* an interaction readout over the **centered** final embeddings (running
  mean over players, as batch normalization): offensive and defensive pair
  means over each five and the cross-side product of the two sums, each
  with a per-target readout. Centering keeps an interaction term from acting
  as a hidden additive rating.

So a node-only model is the FM's chemistry / style structure with non-linear
embeddings (a capacity control), and message passing adds what a player's
representation can learn from the rest of the floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .fm import (
    CENTER_MOMENTUM,
    StintTensors,
    _pair_mean,
)
from .pretraining_data import GUARD_SOURCES, NODE_COLUMNS

RELATIONS = ("teammate", "opponent", "guards")
#: share, r_hat x 10, prior weight, log exposure / 10, one-hot source.
N_GUARD_FEATURES = 4 + len(GUARD_SOURCES)


def guard_features(stints) -> np.ndarray:
    """``(n, 50, N_GUARD_FEATURES)`` guard-edge attributes in ``GUARD_EDGES``
    order, from a ``PretrainingStints`` (expected quantities only)."""
    source = np.eye(len(GUARD_SOURCES))[stints.guard_source.astype(int)]
    out = np.concatenate(
        [
            stints.guard_share[..., None],
            10.0 * stints.guard_r_hat[..., None],
            stints.guard_prior_weight[..., None],
            stints.guard_log_exposure[..., None] / 10.0,
            source,
        ],
        axis=-1,
    )
    if not np.isfinite(out).all():
        raise ValueError("Non-finite guard features")
    return out


@dataclass(frozen=True)
class GNNConfig:
    """Selected in advance (2026-10-15): width 32, two layers; penalties as
    the FM's. ``relations`` picks the ablation."""

    n_features: int = len(NODE_COLUMNS)
    n_targets: int = 6
    n_guard_features: int = N_GUARD_FEATURES
    dim: int = 32
    layers: int = 2
    relations: tuple[str, ...] = RELATIONS
    no_additive: tuple[int, ...] = ()
    #: Partial the additive part out instead of forbidding it: the targets of
    #: ``no_additive`` keep per-player additive terms while training (they
    #: absorb individual effects, including the base's drift) and lose them
    #: at prediction, which is then the interaction part alone.
    absorb_additive: bool = False
    center_interactions: bool = True
    #: False drops the interaction readout: the ladder's intercept / home
    #: control in the same code and training (B).
    readout: bool = True
    delta_penalty: float = 300.0
    weight_penalty: float = 1e-4
    #: Read by ``fm.train``: edge supervision is a separate experiment
    #: (decided 2026-10-15), so the GNN has no edge head yet.
    edges: bool = False
    stint_loss: bool = True

    def __post_init__(self) -> None:
        unknown = set(self.relations) - set(RELATIONS)
        if unknown:
            raise ValueError(f"Unknown relations {sorted(unknown)}")
        if self.edges or not self.stint_loss:
            raise ValueError("The GNN has no edge head yet")


class RelationalLayer(nn.Module):
    def __init__(self, dim: int, n_guard: int, relations) -> None:
        super().__init__()
        self.relations = set(relations)
        self.self_map = nn.Linear(dim, dim)
        self.teammate = nn.Linear(dim, dim)
        self.opponent = nn.Linear(dim, dim)
        self.guards = nn.Linear(dim + n_guard, dim)
        self.guarded_by = nn.Linear(dim + n_guard, dim)
        self.norm = nn.LayerNorm(dim)
        self.n_guard = n_guard

    def forward(
        self, h: torch.Tensor, share: torch.Tensor, guard: torch.Tensor
    ) -> torch.Tensor:
        b = h.shape[0]
        sides = h.reshape(b, 2, 5, -1)
        if "teammate" in self.relations:
            tm = (sides.sum(2, keepdim=True) - sides) / 4
            tm = tm.reshape(b, 10, -1)
        else:
            tm = h
        if "opponent" in self.relations:
            other = sides.flip(1).mean(2, keepdim=True).expand_as(sides)
            opp = other.reshape(b, 10, -1)
        else:
            opp = h
        if "guards" in self.relations:
            g, gb = self._guard_messages(sides, share, guard)
        else:
            zeros = h.new_zeros(b, 10, self.n_guard)
            g = gb = torch.cat([h, zeros], dim=-1)
        update = (
            self.self_map(h)
            + self.teammate(tm)
            + self.opponent(opp)
            + self.guards(g)
            + self.guarded_by(gb)
        )
        return self.norm(h + torch.relu(update))

    @staticmethod
    def _guard_messages(sides, share, guard):
        """Attacker and defender messages, ``(B, 10, d + A)`` each, nodes in
        order (home 0-4, away 5-9)."""
        b = sides.shape[0]
        m = share.reshape(b, 2, 5, 5)  # (attacking side, attacker k, defender l)
        e = guard.reshape(b, 2, 5, 5, -1)
        attacking = sides  # side s attacks
        defending = sides.flip(1)  # its defenders are the other side
        # Attacker k of side s: sum_l m [h_l, e_kl].
        h_def = defending.unsqueeze(2).expand(-1, -1, 5, -1, -1)  # (b,2,k,l,d)
        msg = torch.cat([h_def, e], dim=-1)
        g = (m.unsqueeze(-1) * msg).sum(3)  # (b, 2, 5, d + A), indexed by side
        # Defender l of the other side: sum_k m [h_k, e_kl] / sum_k m.
        h_att = attacking.unsqueeze(3).expand(-1, -1, -1, 5, -1)
        msg_back = torch.cat([h_att, e], dim=-1)
        load = m.sum(2).clamp_min(1e-6).unsqueeze(-1)  # (b, 2, l, 1)
        gb = (m.unsqueeze(-1) * msg_back).sum(2) / load  # (b, 2, l, d + A)
        # gb[:, s] belongs to the side defending against s: flip to node order.
        return g.reshape(b, 10, -1), gb.flip(1).reshape(b, 10, -1)


class GNN(nn.Module):
    KIND = "gnn"

    def __init__(self, config: GNNConfig, n_players: int) -> None:
        super().__init__()
        self.config = config
        t, d = config.n_targets, config.dim
        # Per-player additive scalars (off, def) and the ID embedding share
        # one penalized table, as the FM's blocks.
        self.width = 2 * t + d
        self.prior = nn.Linear(config.n_features, 2 * t)
        self.delta = nn.Embedding(n_players + 1, self.width, padding_idx=0)
        nn.init.zeros_(self.delta.weight)
        with torch.no_grad():
            self.prior.weight.mul_(0.1)
            self.prior.bias.zero_()
        self.encoder = nn.Sequential(
            nn.Linear(config.n_features, d), nn.ReLU(), nn.Linear(d, d)
        )
        self.layers = nn.ModuleList(
            RelationalLayer(d, config.n_guard_features, config.relations)
            for _ in range(config.layers)
        )
        self.mu = nn.Parameter(torch.zeros(t))
        self.hca = nn.Parameter(torch.zeros(t))
        # Readouts: offensive pairs, defensive pairs, cross-side product.
        self.readout = nn.Parameter(torch.full((3, t, d), 0.1))
        additive = torch.ones(t)
        additive[list(config.no_additive)] = 0.0
        self.register_buffer("additive_mask", additive, persistent=False)
        self.register_buffer("embedding_mean", torch.zeros(d))

    def embed(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        share: torch.Tensor,
        guard: torch.Tensor | None,
    ) -> torch.Tensor:
        """Final node embeddings ``(B, 10, dim)`` (not centered)."""
        e = self.delta(ids)[..., 2 * self.config.n_targets :]
        h = self.encoder(x) + e
        if guard is None:
            if "guards" in self.config.relations:
                raise ValueError("A GNN with guard edges needs guard features")
            guard = h.new_zeros(*share.shape, self.config.n_guard_features)
        for layer in self.layers:
            h = layer(h, share, guard)
        return h

    def forward(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        share: torch.Tensor,
        guard: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standardized predictions ``(B, 2, T)``; side 0 = home on offense."""
        t = self.config.n_targets
        scalars = self.prior(x) + self.delta(ids)[..., : 2 * t]
        off, dfn = scalars[..., :t], scalars[..., t:]
        h = self.embed(x, ids, share, guard)
        if self.config.center_interactions:
            h = self._centered(h)
        out = []
        for o, d, sign in (
            (slice(0, 5), slice(5, 10), 1.0),
            (slice(5, 10), slice(0, 5), -1.0),
        ):
            additive = off[:, o].sum(1) - dfn[:, d].sum(1)
            mask = (
                1.0
                if (self.training and self.config.absorb_additive)
                else self.additive_mask
            )
            y = self.mu + sign * self.hca + additive * mask
            if self.config.readout:
                y = y + _pair_mean(h[:, o]) @ self.readout[0].T / 10
                y = y + _pair_mean(h[:, d]) @ self.readout[1].T / 10
                y = y + (h[:, o].sum(1) * h[:, d].sum(1)) @ self.readout[2].T / 25
            out.append(y)
        return torch.stack(out, dim=1)

    def forward_batch(self, batch: StintTensors) -> torch.Tensor:
        return self(batch.x, batch.ids, batch.share, batch.guard)

    def _centered(self, h: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = h.reshape(-1, h.shape[-1]).mean(0)
            with torch.no_grad():
                self.embedding_mean.mul_(1 - CENTER_MOMENTUM).add_(
                    CENTER_MOMENTUM * mean
                )
        else:
            mean = self.embedding_mean
        return h - mean

    def penalty(self, n_train: int) -> torch.Tensor:
        weights = sum(
            (p**2).sum()
            for name, p in self.named_parameters()
            if name.endswith("weight")
            and not name.startswith("delta")
            and ".norm." not in name
        )
        delta = self.delta.weight[1:]
        return (
            self.config.delta_penalty * (delta**2).sum() / max(n_train, 1)
            + self.config.weight_penalty * weights
        )
