"""Edge-level targets: what an attacker produced against the defenders who
guarded him (plan phase 6, decided 2026-10-14).

**Labels only.** Built from a game's own observed matchups (``pair_game``), so
never an encoder input (plan principle 7): they live in their own table,
``pretraining/<version>/edges/season=YYYY.parquet``, which the input loader
(``pretraining_data.to_arrays``) does not read. Observed shares enter only as
the exposure weights of these targets.

**Main targets: time-attributed volume** (``TARGETS``). Phase 4B showed that
the NBA credits a shot to whoever is matched when it goes up, so credited
volume mostly measures defensive role (a rim protector is credited the shots
he helps on). The targets take the attacker's whole-game output instead and
attribute it to his defenders by guarding time::

    y_ig  = sum_j count_ijg / sum_j pp_ijg      (FGA, 3PA per partial possession)
    s_ijg = pp_ijg / sum_l pp_ilg               (defender j's share of i's possessions)

A model predicts a rate ``f(i, j)`` per defender and is scored on the
attacker-game, ``y_hat_ig = sum_j s_ijg f(i, j)``, weighted by the attacker's
partial possessions ``pp_ig`` (:func:`attacker_games`). These are noisy
observational proxies, not causal defensive effects: who guards whom reacts
to the game, help defense and switches blur the attribution, and one game's
signal is small (phase 4B: reliability about 0.03 per game, 0.6-0.7 per
season). The exposure weight is the attacker's partial possessions; nothing
else is reweighted.

**Secondary (diagnostics, not a main loss): credited efficiency.** Points per
credited attempt ``player_points / (FGA + 0.44 FTA)`` on the shots the NBA
credits to the pair, weighted by those attempts: the one shot-credited measure
phase 4B found consistent with stint RAPM.

**Observed guarding share** ``observed_share``: the pair's guarding rate
normalized over the defenders the attacker shared the floor with, as the
graphs normalize expected rates: a possible auxiliary label (plan), never an
input.

**Exclusions.** The 69 corrupt matchup games (doubled exposure or no points,
``matchup_defense.doubled_exposure_games``) and the two stint-rejected games
(``pretraining_data.STINT_REJECTED_GAMES``) have no rows. 2016-17 has no
matchups, so no edge labels: edge supervision starts in 2017-18.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

TARGETS = ("fga_per_pp", "fg3a_per_pp")
COUNTS = {"fga_per_pp": "matchup_fga", "fg3a_per_pp": "matchup_fg3a"}
DIAGNOSTICS = ("pts_per_credited_att",)
ATTACKER_GAME = ["game_id", "off_player_id"]

COLUMNS = (
    "game_id",
    "game_date",
    "season_year",
    "off_player_id",
    "def_player_id",
    "off_team_id",
    "def_team_id",
    "attacker_is_home",
    "partial_possessions",
    "matchup_seconds",
    "cofloor_seconds",
    "pp_share",
    "attacker_pp",
    *TARGETS,
    "credited_attempts",
    "pts_per_credited_att",
    "observed_share",
)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator).where(denominator > 0)


def edge_table(pairs: pd.DataFrame, excluded_games: set[str]) -> pd.DataFrame:
    """One row per game, attacker and defender who were matched or shared the
    floor, with the targets above; ``excluded_games`` have no rows."""
    pairs = pairs.assign(game_id=pairs["game_id"].astype(str))
    pairs = pairs.loc[~pairs["game_id"].isin(excluded_games)]
    pairs = pairs.loc[
        (pairs["partial_possessions"] > 0) | (pairs["cofloor_seconds"] > 0)
    ].copy()
    for key in ("off_player_id", "def_player_id"):
        pairs[key] = pairs[key].astype(str)
    pp = pairs["partial_possessions"].clip(lower=0)
    groups = [pairs["game_id"], pairs["off_player_id"]]
    attacker_pp = pp.groupby(groups).transform("sum")
    out = pairs[
        [
            "game_id",
            "game_date",
            "season_year",
            "off_player_id",
            "def_player_id",
            "off_team_id",
            "def_team_id",
            "attacker_is_home",
            "partial_possessions",
            "matchup_seconds",
            "cofloor_seconds",
        ]
    ].copy()
    out["game_date"] = (
        pd.to_datetime(out["game_date"]).dt.normalize().astype("datetime64[ns]")
    )
    out["pp_share"] = _ratio(pp, attacker_pp).fillna(0.0)
    out["attacker_pp"] = attacker_pp
    for target, count in COUNTS.items():
        out[target] = _ratio(pairs[count].groupby(groups).transform("sum"), attacker_pp)
    attempts = pairs["matchup_fga"] + 0.44 * pairs["matchup_fta"]
    out["credited_attempts"] = attempts
    out["pts_per_credited_att"] = _ratio(pairs["player_points"], attempts)
    rate = _ratio(pairs["matchup_seconds"], pairs["cofloor_seconds"]).clip(upper=1.0)
    total = rate.groupby(groups).transform("sum")
    out["observed_share"] = _ratio(rate, total)
    return out[list(COLUMNS)].reset_index(drop=True)


def attacker_games(edges: pd.DataFrame) -> pd.DataFrame:
    """The main targets per attacker-game with their exposure ``attacker_pp``."""
    edges = edges.loc[edges["attacker_pp"] > 0]
    return (
        edges.groupby(ATTACKER_GAME, sort=False)
        .agg(
            game_date=("game_date", "first"),
            season_year=("season_year", "first"),
            attacker_is_home=("attacker_is_home", "first"),
            attacker_pp=("attacker_pp", "first"),
            **{target: (target, "first") for target in TARGETS},
        )
        .reset_index()
    )


def check_edges(edges: pd.DataFrame, excluded_games: set[str]) -> dict:
    """Raises if the table breaks its definition; returns what it checked."""
    failures = []
    present = excluded_games.intersection(edges["game_id"])
    if present:
        failures.append(f"excluded games present: {sorted(present)[:3]}")
    if edges.duplicated(["game_id", "off_player_id", "def_player_id"]).any():
        failures.append("duplicate pairs")
    with_pp = edges.loc[edges["attacker_pp"] > 0]
    share_sum = with_pp.groupby(ATTACKER_GAME)["pp_share"].sum()
    if not np.allclose(share_sum, 1.0):
        failures.append("pp shares do not sum to 1 per attacker-game")
    for target in TARGETS:
        constant = with_pp.groupby(ATTACKER_GAME)[target].nunique(dropna=False)
        if (constant > 1).any():
            failures.append(f"{target} differs within an attacker-game")
        if with_pp[target].isna().any() or (with_pp[target] < 0).any():
            failures.append(f"{target} missing or negative")
    observed = edges.loc[edges["cofloor_seconds"] > 0]
    sums = observed.groupby(ATTACKER_GAME)["observed_share"].sum(min_count=1).dropna()
    if not np.allclose(sums, 1.0):
        failures.append("observed shares do not sum to 1")
    if failures:
        raise ValueError("Edge targets failed checks: " + "; ".join(failures))
    games = attacker_games(edges)
    return {
        "rows": len(edges),
        "games": int(edges["game_id"].nunique()),
        "attacker_games": len(games),
        "attacker_pp_median": float(games["attacker_pp"].median()),
        "target_means": {
            t: float(np.average(games[t], weights=games["attacker_pp"]))
            for t in TARGETS
        },
    }


# --------------------------------------------------------------------------
# Loading: tables -> arrays
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeBatch:
    """Attacker-games and their (defender, share) rows as arrays.

    ``pair_game_index`` maps each pair row to its attacker-game, so a model
    scores ``sum_j s_ij f(i, j)`` with one ``index_add``.
    """

    game_id: np.ndarray  # (g,)
    game_date: np.ndarray  # (g,)
    season: np.ndarray  # (g,)
    attacker: np.ndarray  # (g,) player ids
    attacker_is_home: np.ndarray  # (g,) bool
    attacker_nodes: np.ndarray  # (g, F)
    targets: np.ndarray  # (g, len(TARGETS))
    weight: np.ndarray  # (g,) attacker partial possessions
    pair_game_index: np.ndarray  # (p,) into the attacker-games
    defender: np.ndarray  # (p,) player ids
    defender_nodes: np.ndarray  # (p, F)
    share: np.ndarray  # (p,) s_ij

    def __len__(self) -> int:
        return len(self.game_id)

    def subset(self, games: np.ndarray) -> EdgeBatch:
        """The attacker-games ``games`` (a boolean mask) and their pairs."""
        games = np.asarray(games, bool)
        keep = games[self.pair_game_index]
        remap = np.cumsum(games) - 1
        attacker_fields = {
            f.name: getattr(self, f.name)[games]
            for f in fields(self)
            if f.name not in {"pair_game_index", "defender", "defender_nodes", "share"}
        }
        return EdgeBatch(
            **attacker_fields,
            pair_game_index=remap[self.pair_game_index[keep]],
            defender=self.defender[keep],
            defender_nodes=self.defender_nodes[keep],
            share=self.share[keep],
        )


def _node_rows(
    nodes: pd.DataFrame, dates: pd.Series, players: pd.Series, columns
) -> tuple[np.ndarray, np.ndarray]:
    """Node features of ``(dates, players)`` and whether each was found."""
    slots = pd.DataFrame(
        {
            "as_of_date": pd.to_datetime(dates.to_numpy()).astype("datetime64[ns]"),
            "player_id": players.astype(str).to_numpy(),
        }
    )
    table = nodes.assign(
        as_of_date=pd.to_datetime(nodes["as_of_date"]).astype("datetime64[ns]"),
        player_id=nodes["player_id"].astype(str),
    )
    joined = slots.merge(
        table, on=["as_of_date", "player_id"], how="left", validate="many_to_one"
    )
    found = joined["prior_weight"].notna().to_numpy()
    return joined[list(columns)].to_numpy(np.float64), found


def to_edge_batch(
    edges: pd.DataFrame, nodes: pd.DataFrame, node_columns
) -> tuple[EdgeBatch, dict]:
    """Arrays for the main targets; attacker-games whose attacker or any
    matched defender lacks node features (as of the game date) are dropped and
    counted."""
    edges = edges.loc[(edges["attacker_pp"] > 0) & (edges["pp_share"] > 0)]
    edges = edges.sort_values(
        ["game_date", "game_id", "off_player_id", "def_player_id"], kind="mergesort"
    ).reset_index(drop=True)
    games = attacker_games(edges)
    attacker_nodes, attacker_found = _node_rows(
        nodes, games["game_date"], games["off_player_id"], node_columns
    )
    defender_nodes, defender_found = _node_rows(
        nodes, edges["game_date"], edges["def_player_id"], node_columns
    )
    key = pd.MultiIndex.from_frame(games[ATTACKER_GAME])
    pair_index = key.get_indexer(pd.MultiIndex.from_frame(edges[ATTACKER_GAME]))
    complete = attacker_found.copy()
    np.logical_and.at(complete, pair_index, defender_found)
    batch = EdgeBatch(
        game_id=games["game_id"].to_numpy(),
        game_date=games["game_date"].to_numpy(),
        season=games["season_year"].to_numpy(),
        attacker=games["off_player_id"].to_numpy(),
        attacker_is_home=games["attacker_is_home"].to_numpy(bool),
        attacker_nodes=attacker_nodes,
        targets=games[list(TARGETS)].to_numpy(float),
        weight=games["attacker_pp"].to_numpy(float),
        pair_game_index=pair_index,
        defender=edges["def_player_id"].to_numpy(),
        defender_nodes=defender_nodes,
        share=edges["pp_share"].to_numpy(float),
    )
    dropped = {
        "attacker_games": int((~complete).sum()),
        "attacker_pp": float(batch.weight[~complete].sum()),
    }
    return batch.subset(complete), dropped
