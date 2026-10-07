"""Stint graphs (training graphs): who was on the floor, and who guards whom.

Every stint graph has the same shape, so a batch of stints is a set of arrays
with a leading ``n_stints`` axis rather than graph objects (converting to
PyTorch Geometric or plain tensors is a loader detail, decided in phase 6).

**Nodes** (10 per stint, fixed order): the home lineup as nodes 0-4, the away
lineup as nodes 5-9, both in stint-store order. Node weight: stint seconds.

**Edges**, as fixed node-index templates shared by every stint:

* ``TEAMMATE_EDGES`` (2 x 20): every pair on the same side, undirected.
* ``OPPONENT_EDGES`` (2 x 25): every home x away pair, undirected.
* ``GUARD_EDGES`` (2 x 50): directed **defender -> attacker**, for both sides
  attacking. Edge ``e = side * 25 + k * 5 + l`` joins attacker ``k`` of the
  attacking side to defender ``l`` of the other side (side 0 = home attacking).

Teammate and opponent weights are the stint's seconds: on a training graph
every pair shares the whole stint. The guard weight is the **expected**
guarding share among the five defenders on the floor::

    m_ij = r_hat_ij / sum_{l in stint defenders} r_hat_il

from ``expected_guard`` (as of the game's date). Never the shares observed in
the game: those are labels (plan principle 7), and :func:`build_stint_graphs`
rejects any observed or betting column in its input. An attacker whose five
rates are not all known (no history at all, i.e. 2016-17 and the first days of
2017-18) gets uniform shares, flagged in ``guard_fallback``.

Guard-edge attributes kept next to ``m_ij`` so an encoder can tell a share
built on long history from one that is mostly prior: ``guard_r_hat``
(unnormalized), ``guard_prior_weight``, ``guard_log_exposure`` (``log1p`` of
the decayed historical co-floor seconds) and ``guard_has_history``.

Every attribute array is finite: an unknown ``r_hat`` is stored as 0 with
``guard_r_hat_known`` False (and ``prior_weight`` 1, exposure 0), so the arrays
can go into a network as they are.

**Labels** per stint and offensive side (``LABELS``), weighted by that side's
possessions: points, turnovers per possession, possessions per 48 minutes,
3PA/FGA, FTA/FGA and offensive rebound rate OREB / (OREB + opponent DREB). The
possession estimate can be zero or slightly negative in stints of a few seconds
(an offensive rebound of a miss from the previous stint is subtracted): such a
side gets weight 0 and NaN labels, never a negative weight.

**Message passing.** The templates hold each teammate and opponent pair once.
:func:`message_passing_edges` gives every relation in both directions, with
the template edge each directed edge comes from, to gather its weight and
attributes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd

from .pair_game import MATCHUP_COUNT_COLUMNS, NBA_SHARE_COLUMNS

N_NODES = 10
HOME_NODES = tuple(range(5))
AWAY_NODES = tuple(range(5, 10))

TEAMMATE_EDGES = np.array(
    [pair for side in (HOME_NODES, AWAY_NODES) for pair in combinations(side, 2)]
).T
OPPONENT_EDGES = np.array([(h, a) for h in HOME_NODES for a in AWAY_NODES]).T


def _guard_edges() -> np.ndarray:
    defenders, attackers = [], []
    for side in (0, 1):
        attacking = HOME_NODES if side == 0 else AWAY_NODES
        defending = AWAY_NODES if side == 0 else HOME_NODES
        for k in range(5):
            for l in range(5):  # noqa: E741
                defenders.append(defending[l])
                attackers.append(attacking[k])
    return np.array([defenders, attackers])


GUARD_EDGES = _guard_edges()


def message_passing_edges(guard_reverse: bool = False) -> dict[str, dict]:
    """Directed edges per relation, for message passing.

    Returns ``relation -> {"index": (2, E), "source_edge": (E,)}``.
    ``source_edge`` is the column of the template (and of the per-stint weight
    and attribute arrays) each directed edge comes from. Teammate and opponent
    edges are undirected, so both directions are listed. Guard edges stay
    defender -> attacker; ``guard_reverse`` adds ``guarded_by``
    (attacker -> defender) as its own relation.
    """

    def both(index: np.ndarray) -> dict:
        ids = np.arange(index.shape[1])
        return {
            "index": np.concatenate([index, index[::-1]], axis=1),
            "source_edge": np.concatenate([ids, ids]),
        }

    guard_ids = np.arange(GUARD_EDGES.shape[1])
    relations = {
        "teammate": both(TEAMMATE_EDGES),
        "opponent": both(OPPONENT_EDGES),
        "guards": {"index": GUARD_EDGES, "source_edge": guard_ids},
    }
    if guard_reverse:
        relations["guarded_by"] = {
            "index": GUARD_EDGES[::-1].copy(),
            "source_edge": guard_ids,
        }
    return relations


LABELS = (
    "pts_per_poss",
    "tov_per_poss",
    "poss_per_48",
    "fg3a_per_fga",
    "fta_per_fga",
    "oreb_rate",
)

#: Columns that would put the encoded game's own outcome, or the market, into
#: the graph. ``expected_guard`` has none of them; ``pair_game`` has the first.
OBSERVED_COLUMNS = frozenset(
    {*MATCHUP_COUNT_COLUMNS, *NBA_SHARE_COLUMNS, "guard_rate", "has_matchup_row"}
)
BETTING_MARKERS = ("TOTAL_LINE", "SPREAD", "MONEYLINE", "ODDS", "LINE_ERROR")

_EXPECTED_COLUMNS = (
    "r_hat",
    "prior_weight",
    "hist_cofloor_seconds_decayed",
    "has_pair_history",
)


@dataclass(frozen=True)
class StintGraphs:
    """A batch of stint graphs; arrays share the leading ``n_stints`` axis."""

    game_id: np.ndarray  # (n,)
    seg_idx: np.ndarray  # (n,)
    game_date: np.ndarray  # (n,)
    players: np.ndarray  # (n, 10) player ids in node order
    seconds: np.ndarray  # (n,) node and teammate / opponent edge weight
    guard_share: np.ndarray  # (n, 50) m_ij, in GUARD_EDGES order
    guard_r_hat: np.ndarray  # (n, 50), 0 where unknown
    guard_r_hat_known: np.ndarray  # (n, 50) bool
    guard_prior_weight: np.ndarray  # (n, 50)
    guard_log_exposure: np.ndarray  # (n, 50)
    guard_has_history: np.ndarray  # (n, 50) bool
    guard_fallback: np.ndarray  # (n, 50) bool: uniform share
    labels: np.ndarray  # (n, 2, len(LABELS)); side 0 = home on offense
    label_weight: np.ndarray  # (n, 2) possessions of the offensive side

    def __len__(self) -> int:
        return len(self.game_id)

    def graph(self, i: int) -> dict:
        """Stint ``i`` as plain arrays: nodes, typed edges, weights, labels."""
        return {
            "game_id": self.game_id[i],
            "seg_idx": int(self.seg_idx[i]),
            "nodes": {
                "player_id": self.players[i],
                "side": np.repeat(["home", "away"], 5),
                "weight": np.full(N_NODES, self.seconds[i]),
            },
            "teammate": {
                "index": TEAMMATE_EDGES,
                "weight": np.full(TEAMMATE_EDGES.shape[1], self.seconds[i]),
            },
            "opponent": {
                "index": OPPONENT_EDGES,
                "weight": np.full(OPPONENT_EDGES.shape[1], self.seconds[i]),
            },
            "guards": {
                "index": GUARD_EDGES,
                "weight": self.guard_share[i],
                "r_hat": self.guard_r_hat[i],
                "r_hat_known": self.guard_r_hat_known[i],
                "prior_weight": self.guard_prior_weight[i],
                "log_exposure": self.guard_log_exposure[i],
                "has_history": self.guard_has_history[i],
                "fallback": self.guard_fallback[i],
            },
            "labels": dict(zip(LABELS, self.labels[i].T, strict=True)),
            "label_weight": self.label_weight[i],
        }


def check_encoder_inputs(frame: pd.DataFrame) -> None:
    """Raise if ``frame`` carries observed matchup or betting columns."""
    observed = sorted(OBSERVED_COLUMNS.intersection(frame.columns))
    betting = sorted(
        column
        for column in frame.columns
        if any(marker in str(column).upper() for marker in BETTING_MARKERS)
    )
    if observed or betting:
        raise ValueError(
            "Encoder inputs must not hold same-game observed or betting columns: "
            f"{observed + betting}"
        )


def _lineups(stints: pd.DataFrame, column: str) -> np.ndarray:
    lineups = np.stack(stints[column].map(np.asarray).to_numpy()).astype(str)
    if lineups.ndim != 2 or lineups.shape[1] != 5:
        raise ValueError(f"Every stint needs five players in {column}")
    return lineups


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def stint_labels(stints: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``(labels, weights)``: (n, 2, len(LABELS)) and (n, 2) possessions."""
    seconds = stints["seconds"].to_numpy(float)
    sides = []
    for offense, defense in (("home", "away"), ("away", "home")):
        col = {
            stat: stints[f"{offense}_{stat}"].to_numpy(float)
            for stat in ("pts", "poss", "tov", "fga", "fg3a", "fta", "oreb")
        }
        opponent_dreb = stints[f"{defense}_dreb"].to_numpy(float)
        sides.append(
            np.stack(
                [
                    _ratio(col["pts"], col["poss"]),
                    _ratio(col["tov"], col["poss"]),
                    _ratio(col["poss"] * 2880.0, seconds),
                    _ratio(col["fg3a"], col["fga"]),
                    _ratio(col["fta"], col["fga"]),
                    _ratio(col["oreb"], col["oreb"] + opponent_dreb),
                ],
                axis=-1,
            )
        )
    possessions = np.stack(
        [stints["home_poss"].to_numpy(float), stints["away_poss"].to_numpy(float)],
        axis=1,
    )
    labels = np.stack(sides, axis=1)
    # No possessions (or a negative estimate): no information, never a negative
    # weight in a loss.
    empty = ~(possessions > 0)
    labels[empty] = np.nan
    return labels, np.where(empty, 0.0, possessions)


def build_stint_graphs(stints: pd.DataFrame, expected: pd.DataFrame) -> StintGraphs:
    """Stint graphs with v0 guard weights.

    ``stints``: rows of the stint store. ``expected``: ``expected_guard`` rows
    with ``game_id``, covering the pairs of those stints (missing pairs fall
    back to uniform shares).
    """
    check_encoder_inputs(expected)
    stints = stints.reset_index(drop=True)
    n = len(stints)
    home = _lineups(stints, "home_lineup")
    away = _lineups(stints, "away_lineup")

    # (n, side, attacker k, defender l), flattened in GUARD_EDGES order.
    attackers = np.empty((n, 2, 5, 5), dtype=object)
    defenders = np.empty((n, 2, 5, 5), dtype=object)
    attackers[:, 0] = home[:, :, None]
    defenders[:, 0] = away[:, None, :]
    attackers[:, 1] = away[:, :, None]
    defenders[:, 1] = home[:, None, :]
    flat = pd.DataFrame(
        {
            "game_id": np.repeat(stints["game_id"].astype(str).to_numpy(), 50),
            "off_player_id": attackers.reshape(-1),
            "def_player_id": defenders.reshape(-1),
        }
    )
    keys = ["game_id", "off_player_id", "def_player_id"]
    table = expected[[*keys, *_EXPECTED_COLUMNS]].copy()
    for key in keys:
        table[key] = table[key].astype(str)
    if table.duplicated(keys).any():
        raise ValueError("expected_guard has more than one row per game and pair")
    flat = flat.merge(table, on=keys, how="left", validate="many_to_one")

    def grid(column: str, dtype=float) -> np.ndarray:
        return flat[column].to_numpy(dtype).reshape(n, 2, 5, 5)

    r_hat = grid("r_hat")
    totals = r_hat.sum(axis=3, keepdims=True)
    fallback = ~(np.isfinite(totals) & (totals > 0))
    share = np.where(fallback, 0.2, r_hat / np.where(fallback, 1.0, totals))
    fallback = np.broadcast_to(fallback, r_hat.shape)
    exposure = np.log1p(np.nan_to_num(grid("hist_cofloor_seconds_decayed")))
    has_history = flat["has_pair_history"].eq(True).to_numpy(bool)

    labels, weights = stint_labels(stints)
    return StintGraphs(
        game_id=stints["game_id"].astype(str).to_numpy(),
        seg_idx=stints["seg_idx"].to_numpy(),
        game_date=stints["game_date"].to_numpy(),
        players=np.concatenate([home, away], axis=1),
        seconds=stints["seconds"].to_numpy(float),
        guard_share=share.reshape(n, 50),
        guard_r_hat=np.nan_to_num(r_hat, nan=0.0).reshape(n, 50),
        guard_r_hat_known=np.isfinite(r_hat).reshape(n, 50),
        guard_prior_weight=np.nan_to_num(grid("prior_weight"), nan=1.0).reshape(n, 50),
        guard_log_exposure=exposure.reshape(n, 50),
        guard_has_history=has_history.reshape(n, 50),
        guard_fallback=fallback.reshape(n, 50).copy(),
        labels=labels,
        label_weight=weights,
    )
