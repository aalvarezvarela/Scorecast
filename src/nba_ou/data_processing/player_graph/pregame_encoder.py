"""Pregame encoder outputs: a stint-trained encoder applied to a game (phase 7).

**The mismatch to avoid.** The encoders (``fm.py``, ``gnn.py``) are trained
on 10-node stint graphs: five a side, every teammate and opponent mean over
4 and 5 players, guard shares normalized over the 5 defenders on the floor.
A pregame graph (``game_graph``) holds a whole expected rotation: 20-26
players with fractional expected minutes. Feeding it to the encoder would
change node counts, message averages and pooling, a train / inference
mismatch.

**The adapter: the expected rotation as 48 one-minute lineups** (decided
2026-10-15, nothing tuned). For each availability scenario and each team,
:func:`lineup_schedule` lays the team's expected minutes out as ``SLOTS`` =
48 lineups of exactly five players by systematic (Madow) sampling: players in
a fixed order (expected minutes descending, then id), inclusion probability
``minutes / 48`` (summing to 5 because the team's minutes sum to 240), one
grid point ``(k + 1/2) / 48`` per slot. Each lineup has five distinct players
and each player is in ``minutes`` of the 48 slots up to one slot, so the
schedule reproduces the expected minutes to within a minute and the team's
240 exactly. Slot ``k`` of the home team plays slot ``k`` of the away team:
48 ordinary stint graphs, the encoder's own input shape. Their guard shares
are recomputed as in training (expected rates normalized over the five
defenders on the floor); node features are the players' as-of profiles on
the game date.

**Aggregation.** Per scenario, the mean over the 48 slot graphs of the
encoder's outputs: predicted style rates per offensive side (turnovers per
possession, 3PA/FGA, FTA/FGA, offensive rebound rate) and each side's pooled
(mean over its five) centered node embedding. Tonight is the
probability-weighted mean over the availability scenarios (weights renormalized
to 1); the full-health graph is the counterfactual, and the absence shift is
tonight minus full health.

**The game embedding** (16 dimensions): each side's pooled embedding projected
on 8 principal directions fitted, per fold, on that fold's training stints
(:func:`fold_projection`), and rotated onto the previous fold's directions by
orthogonal Procrustes over the previous fold's sample (:func:`align`) so the
coordinates mean the same thing from one checkpoint to the next: home 8 +
away 8.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .pretraining_data import GUARD_SOURCES, NODE_COLUMNS, PretrainingStints
from .stint_graph import LABELS

SLOTS = 48
GAME_MINUTES = 48.0
TEAM_MINUTES = 240.0
STYLE_TARGETS = ("tov_per_poss", "fg3a_per_fga", "fta_per_fga", "oreb_rate")
EMBEDDING_PER_SIDE = 8


# --------------------------------------------------------------------------
# Lineups
# --------------------------------------------------------------------------


def player_order(players: Sequence[str], minutes: Sequence[float]) -> np.ndarray:
    """Indices by expected minutes descending, then player id."""
    frame = pd.DataFrame({"player": [str(p) for p in players], "minutes": minutes})
    return frame.sort_values(
        ["minutes", "player"], ascending=[False, True], kind="mergesort"
    ).index.to_numpy()


def lineup_schedule(
    players: Sequence[str], minutes: Sequence[float], slots: int = SLOTS
) -> np.ndarray:
    """``(slots, 5)`` indices into ``players``: one five per one-minute slot.

    Systematic sampling with inclusion probability ``minutes / 48``; raises
    unless the team's minutes sum to 240 and nobody exceeds 48.
    """
    minutes = np.asarray(minutes, float)
    if not np.isclose(minutes.sum(), TEAM_MINUTES, atol=1e-6):
        raise ValueError(f"Team minutes sum to {minutes.sum():.3f}, not 240")
    if (minutes < 0).any() or (minutes > GAME_MINUTES + 1e-9).any():
        raise ValueError("Every player's minutes must be in [0, 48]")
    order = player_order(players, minutes)
    probability = minutes[order] / GAME_MINUTES
    upper = np.cumsum(probability)
    upper[-1] = 5.0  # absorb rounding so every point is covered
    grid = (np.arange(slots) + 0.5) / slots
    points = grid[:, None] + np.arange(5)[None, :]  # (slots, 5) in [0, 5)
    chosen = np.searchsorted(upper, points, side="right")
    return order[chosen]


def schedule_minutes(schedule: np.ndarray, n_players: int) -> np.ndarray:
    """Minutes each player gets in a schedule (48 x his share of the slots)."""
    counts = np.bincount(schedule.reshape(-1), minlength=n_players)
    return GAME_MINUTES * counts / schedule.shape[0]


# --------------------------------------------------------------------------
# Slot graphs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioInput:
    """One scenario of one game: players per side with expected minutes."""

    game_id: str
    scenario_id: int
    home: pd.DataFrame  # player_id, minutes
    away: pd.DataFrame


def slot_stints(
    scenario: ScenarioInput,
    features: pd.DataFrame,
    guards: pd.DataFrame,
    slots: int = SLOTS,
) -> PretrainingStints:
    """The scenario's ``slots`` stint graphs in the encoder's input format.

    ``features``: raw node rows (``NODE_COLUMNS``) indexed by ``player_id``,
    as of the game date. ``guards``: the game's expected guard rows
    (``off_player_id``, ``def_player_id``, ``r_hat``, ``r_hat_known``,
    ``guard_prior_weight``, ``log_exposure``, ``has_pair_history``). Labels
    are empty: this is prediction.
    """
    home_ids = scenario.home["player_id"].astype(str).to_numpy()
    away_ids = scenario.away["player_id"].astype(str).to_numpy()
    home = home_ids[lineup_schedule(home_ids, scenario.home["minutes"], slots)]
    away = away_ids[lineup_schedule(away_ids, scenario.away["minutes"], slots)]
    players = np.concatenate([home, away], axis=1)  # (slots, 10)
    missing = set(players.reshape(-1)) - set(features.index.astype(str))
    if missing:
        raise ValueError(f"No node features for {sorted(missing)[:3]}")
    nodes = (
        features.reindex(players.reshape(-1))[list(NODE_COLUMNS)]
        .to_numpy(float)
        .reshape(slots, 10, len(NODE_COLUMNS))
    )
    table = guards.assign(
        off_player_id=guards["off_player_id"].astype(str),
        def_player_id=guards["def_player_id"].astype(str),
    ).set_index(["off_player_id", "def_player_id"])
    # (slots, side, attacker k, defender l) in GUARD_EDGES order.
    attackers = np.stack([home, away], axis=1)[:, :, :, None].repeat(5, axis=3)
    defenders = np.stack([away, home], axis=1)[:, :, None, :].repeat(5, axis=2)
    keys = pd.MultiIndex.from_arrays([attackers.reshape(-1), defenders.reshape(-1)])
    rows = table.reindex(keys)
    if rows["r_hat"].isna().any():
        raise ValueError("A pair on the floor has no expected guard row")
    shape = (slots, 2, 5, 5)
    known = rows["r_hat_known"].to_numpy(bool).reshape(shape)
    r_hat = np.where(known, rows["r_hat"].to_numpy(float).reshape(shape), np.nan)
    total = r_hat.sum(axis=3, keepdims=True)
    usable = np.isfinite(total) & (total > 0)
    share = np.where(usable, r_hat / np.where(usable, total, 1.0), 0.2)
    history = rows["has_pair_history"].to_numpy(bool).reshape(shape)
    source = np.where(
        np.broadcast_to(usable, shape),
        np.where(
            history,
            GUARD_SOURCES.index("pair_history"),
            GUARD_SOURCES.index("position_prior"),
        ),
        GUARD_SOURCES.index("uniform"),
    ).astype(np.int8)
    zeros = np.zeros((slots, 2, len(LABELS)))
    return PretrainingStints(
        game_id=np.repeat(scenario.game_id, slots),
        seg_idx=np.arange(slots),
        game_date=np.repeat(np.datetime64("NaT"), slots),
        season=np.zeros(slots, dtype=int),
        players=players,
        seconds=np.full(slots, 60.0),
        nodes=nodes,
        guard_share=share.reshape(slots, 50),
        guard_source=source.reshape(slots, 50),
        guard_r_hat=np.nan_to_num(r_hat, nan=0.0).reshape(slots, 50),
        guard_prior_weight=rows["guard_prior_weight"]
        .to_numpy(float)
        .reshape(slots, 50),
        guard_log_exposure=rows["log_exposure"].to_numpy(float).reshape(slots, 50),
        labels=zeros,
        label_weight=zeros,
    )


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def encode_slots(model, vocab, std, stints: PretrainingStints) -> dict[str, np.ndarray]:
    """Style predictions ``(S, 2, len(STYLE_TARGETS))`` in target units (side
    0 = home on offense) and pooled centered embeddings ``(S, 2, dim)``
    (side 0 = home five)."""
    from .fm import stint_tensors

    graph = hasattr(model, "embed")
    tensors = stint_tensors(stints, vocab, std, guard=graph)
    model.eval()
    with torch.no_grad():
        predictions = std.targets.invert(model.forward_batch(tensors).numpy())
        style = predictions[:, :, [LABELS.index(t) for t in STYLE_TARGETS]]
        if graph:
            h = model.uncentered(tensors)["embedding"] - model.centering_buffer(
                "embedding"
            )
            pooled = torch.stack([h[:, :5].mean(1), h[:, 5:].mean(1)], dim=1).numpy()
        else:
            pooled = np.zeros((len(stints), 2, 0))
    return {"style": style.astype(np.float64), "pooled": pooled.astype(np.float64)}


def aggregate(
    per_scenario: Mapping[int, Mapping[str, np.ndarray]],
    weights: Mapping[int, float],
    full_health: int,
) -> dict[str, np.ndarray]:
    """Tonight (probability-weighted over the availability scenarios), full
    health, and the absence shift, from per-scenario slot means."""
    scenarios = [s for s in per_scenario if s != full_health]
    if not scenarios or full_health not in per_scenario:
        raise ValueError("Need availability scenarios and the full-health graph")
    w = np.array([weights[s] for s in scenarios], float)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("Scenario weights must be non-negative with a positive sum")
    w = w / w.sum()
    out = {}
    for key in per_scenario[full_health]:
        stacked = np.stack([per_scenario[s][key] for s in scenarios])
        tonight = np.tensordot(w, stacked, axes=1)
        healthy = per_scenario[full_health][key]
        out[f"{key}_tonight"] = tonight
        out[f"{key}_full_health"] = healthy
        out[f"{key}_absence_shift"] = tonight - healthy
    return out


def scenario_means(outputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mean over a scenario's slot graphs."""
    return {key: values.mean(axis=0) for key, values in outputs.items()}


# --------------------------------------------------------------------------
# The 16-dimensional game embedding
# --------------------------------------------------------------------------


def fold_projection(pooled: np.ndarray, k: int = EMBEDDING_PER_SIDE) -> np.ndarray:
    """``(dim, k)`` principal directions of side-pooled embeddings ``(n, dim)``
    (a fold's training sample), signs fixed so each direction's
    largest-magnitude loading is positive."""
    centered = pooled - pooled.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:k].T
    signs = np.sign(basis[np.abs(basis).argmax(axis=0), np.arange(k)])
    return basis * np.where(signs == 0, 1.0, signs)


def align(
    basis: np.ndarray,
    anchor_now: np.ndarray,
    previous_basis: np.ndarray,
    anchor_before: np.ndarray,
) -> np.ndarray:
    """Rotate ``basis`` so the anchor stints' coordinates match the previous
    fold's (orthogonal Procrustes). ``anchor_now`` / ``anchor_before``: the
    same stints' side-pooled embeddings under the current and previous
    encoder."""
    current = anchor_now @ basis
    before = anchor_before @ previous_basis
    u, _, vt = np.linalg.svd(current.T @ before)
    return basis @ (u @ vt)


def game_embedding(pooled_tonight: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """``(2 * k,)``: home then away pooled embedding on the fold's directions."""
    return np.concatenate([pooled_tonight[0] @ basis, pooled_tonight[1] @ basis])


# --------------------------------------------------------------------------
# Output columns
# --------------------------------------------------------------------------

STYLE_NAMES = {
    "tov_per_poss": "TOV_RATE",
    "fg3a_per_fga": "FG3A_RATE",
    "fta_per_fga": "FTA_RATE",
    "oreb_rate": "OREB_RATE",
}
SIDE_SUFFIX = ("TEAM_HOME", "TEAM_AWAY")


def style_columns(prefix: str = "GE_STYLE") -> list[str]:
    """Tonight's rates then their absence shifts, per offensive side."""
    return [
        f"{prefix}_{STYLE_NAMES[t]}{kind}_BEFORE_{side}"
        for kind in ("", "_ABSENCE_SHIFT")
        for t in STYLE_TARGETS
        for side in SIDE_SUFFIX
    ]


def embedding_columns(prefix: str) -> list[str]:
    return [
        f"{prefix}_{k}_BEFORE_{side}"
        for side in SIDE_SUFFIX
        for k in range(EMBEDDING_PER_SIDE)
    ]
