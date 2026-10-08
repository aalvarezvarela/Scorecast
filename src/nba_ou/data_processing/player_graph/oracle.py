"""Oracle diagnostics for the game graph: how much could better weights buy?

**Diagnostic only.** Nothing here produces a feature or trains a model: the
oracle graphs use what actually happened in the game being scored (plan
principle 7). They are compared with the v0 graphs to decide how much effort
phase 4A (minutes / rotation) and phase 4B (guarding) deserve.

The guard weight is a product of two factors, ``m_ij ∝ r_ij * overlap_ij``.
Each oracle replaces one of them:

========================  ===================  ====================  ==========
graph                     node minutes         overlap               rate r_ij
========================  ===================  ====================  ==========
v0                        projected (2_6)      expected (pair lift)  r_hat
oracle rotation (4A)      actual regulation    actual regulation     r_hat
oracle guards (4B)        projected            expected              observed
oracle both               actual regulation    actual regulation     observed
========================  ===================  ====================  ==========

* **Regulation only** (periods 1-4, from the stints): minutes sum to 240 per
  team and overlaps are consistent with them, so an overtime game does not
  reveal its overtime.
* **Observed rate**: the game's ``pair_game.guard_rate`` (matchup seconds /
  co-floor seconds, whole game: matchups have no timestamps). A pair without
  one (did not share the floor, or no tracking) keeps ``r_hat``. With both
  oracles the shares are exactly the observed matchup distribution **in games
  without overtime**; with overtime, whole-game rates times regulation overlap
  only approximate it (the relative shares can shift).
* The **full-health counterfactual** has no actual version, so it keeps
  projected minutes and expected overlap: v0's for the rotation oracle, and
  the guard oracle's (observed rates where a pair has one, ``r_hat`` for an
  absent defender) for both oracles, so each tonight-vs-full-health impact
  changes only the factors its oracle changes.

**Not a strict mathematical ceiling.** Actual minutes and assignments react to
the game (blowouts, foul trouble, a hot scorer drawing a different defender),
so the oracles can overstate what is attainable. And readout R2 uses 2_6's
stint RAPM ``def_j``, which credits the five defenders equally, not the
matchup-adjusted rating of phase 4B, so it can understate what matchups are
worth.

Readouts (:func:`readouts`):

* **R1**: the 2_6 projection, ``sum_i min_i / 48 * rating_i`` per team, then
  possessions and points. Only node minutes move it.
* **R2**: R1 with each team's defense replaced by the defense its attackers
  face, ``5 * sum_i w_i sum_j m_ij def_j`` with ``w_i ∝ min_i * usage_i``
  (as-of usage, identical in every version). With shares proportional to
  floor time it equals R1 exactly, so the difference is who guards whom.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .as_of import AsOfView
from .game_graph import FULL_HEALTH, GameGraphs, guard_shares
from .overlap import build_overlap_game

VERSIONS = ("v0", "oracle_rotation", "oracle_guards", "oracle_both")
REGULATION_PERIODS = 4
ACTUAL_SCENARIO = 0


def usage_as_of(
    view: AsOfView, *, window_days: int = 365, prior_minutes: float = 200.0
) -> pd.Series:
    """Minutes-weighted ``USG_PCT`` per player before the cutoff, shrunk toward
    the league's with ``prior_minutes``. Unknown players get the league value
    from :func:`usage_lookup`."""
    box = view.box_scores(since=view.cutoff - pd.Timedelta(days=window_days))
    if box.empty or "USG_PCT" not in box.columns:
        return pd.Series(dtype=float, name="usage")
    minutes = pd.to_numeric(box["MIN"], errors="coerce").fillna(0.0)
    usage = pd.to_numeric(box["USG_PCT"], errors="coerce")
    keep = (minutes > 0) & usage.notna()
    players = box.loc[keep, "PLAYER_ID"].astype(str)
    weighted = (usage[keep] * minutes[keep]).groupby(players).sum()
    played = minutes[keep].groupby(players).sum()
    league = weighted.sum() / played.sum()
    out = (weighted + prior_minutes * league) / (played + prior_minutes)
    out.attrs["league"] = float(league)
    return out.rename("usage")


def usage_lookup(usage: pd.Series, players: pd.Series) -> np.ndarray:
    league = usage.attrs.get("league", 0.2)
    return players.astype(str).map(usage).fillna(league).to_numpy(float)


def regulation_stints(stints: pd.DataFrame) -> pd.DataFrame:
    return stints.loc[stints["period"] <= REGULATION_PERIODS]


def actual_graphs(
    stints: pd.DataFrame, rates: pd.DataFrame, game_dates: pd.Series
) -> GameGraphs:
    """One scenario per game from its actual regulation rotation.

    ``stints``: the games' regulation stints. ``rates``: ``game_id``,
    ``off_player_id``, ``def_player_id``, ``rate`` (r_hat or observed; NaN if
    unknown) for the opponent pairs. ``game_dates``: ``game_id -> date``.
    """
    nodes = []
    for side in ("home", "away"):
        lineups = np.stack(stints[f"{side}_lineup"].map(np.asarray).to_numpy())
        nodes.append(
            pd.DataFrame(
                {
                    "game_id": np.repeat(stints["game_id"].astype(str).to_numpy(), 5),
                    "player_id": lineups.astype(str).ravel(),
                    "side": side,
                    "minutes": np.repeat(stints["seconds"].to_numpy(float), 5) / 60,
                }
            )
        )
    nodes = (
        pd.concat(nodes, ignore_index=True)
        .groupby(["game_id", "player_id", "side"], as_index=False)["minutes"]
        .sum()
        .assign(scenario_id=ACTUAL_SCENARIO)
    )
    overlap = build_overlap_game(stints)
    pair_edges = pd.DataFrame(
        {
            "game_id": overlap["game_id"],
            "scenario_id": ACTUAL_SCENARIO,
            "relation": overlap["relation"],
            "src": overlap["player_a"],
            "dst": overlap["player_b"],
            "weight": overlap["shared_seconds"] / 60,
            "expected_overlap": overlap["shared_seconds"] / 60,
        }
    )
    opponents = pair_edges.loc[pair_edges["relation"].eq("opponent")]
    directed = pd.concat(
        [
            opponents.rename(columns={"src": "a", "dst": "b"}).assign(
                off=lambda f: f["a"], dfn=lambda f: f["b"]
            ),
            opponents.rename(columns={"src": "a", "dst": "b"}).assign(
                off=lambda f: f["b"], dfn=lambda f: f["a"]
            ),
        ],
        ignore_index=True,
    )
    directed = directed.merge(
        rates.rename(columns={"off_player_id": "off", "def_player_id": "dfn"}),
        on=["game_id", "off", "dfn"],
        how="left",
    )
    shares, fallback = guard_shares(
        [directed["game_id"], directed["off"]],
        directed["rate"].to_numpy(float),
        directed["expected_overlap"].to_numpy(float),
    )
    guard_edges = pd.DataFrame(
        {
            "game_id": directed["game_id"],
            "scenario_id": ACTUAL_SCENARIO,
            "relation": "guards",
            "src": directed["dfn"],
            "dst": directed["off"],
            "weight": shares,
            "expected_overlap": directed["expected_overlap"],
            "r_hat": directed["rate"],
            "fallback": fallback,
        }
    )
    scenarios = (
        nodes[["game_id"]]
        .drop_duplicates()
        .assign(
            game_date=lambda f: f["game_id"].map(game_dates),
            scenario_id=ACTUAL_SCENARIO,
            is_full_health=False,
            weight=1.0,
        )
    )
    return GameGraphs(
        scenarios=scenarios.reset_index(drop=True),
        nodes=nodes[["game_id", "scenario_id", "player_id", "side", "minutes"]],
        edges=pd.concat([pair_edges, guard_edges], ignore_index=True),
    )


def with_observed_rates(graphs: GameGraphs, observed: pd.DataFrame) -> GameGraphs:
    """The guard oracle: r_hat replaced by the observed rate where a pair has
    one; minutes and expected overlaps untouched. ``observed``: ``game_id``,
    ``off_player_id``, ``def_player_id``, ``guard_rate``."""
    edges = graphs.edges
    guards = edges.loc[edges["relation"].eq("guards")]
    known = guards["r_hat_known"].astype(bool) if "r_hat_known" in guards else True
    expected = guards["r_hat"].where(known)
    lookup = observed.set_index(["game_id", "off_player_id", "def_player_id"])[
        "guard_rate"
    ]
    keys = pd.MultiIndex.from_arrays([guards["game_id"], guards["dst"], guards["src"]])
    rate = pd.Series(lookup.reindex(keys).to_numpy(), index=guards.index)
    rate = rate.where(rate.notna(), expected)
    shares, fallback = guard_shares(
        [guards["game_id"], guards["scenario_id"], guards["dst"]],
        rate.to_numpy(float),
        guards["expected_overlap"].to_numpy(float),
    )
    swapped = edges.copy()
    swapped.loc[guards.index, "weight"] = shares
    swapped.loc[guards.index, "fallback"] = fallback
    swapped.loc[guards.index, "r_hat"] = rate.to_numpy()
    return GameGraphs(graphs.scenarios, graphs.nodes, swapped, graphs.metadata)


def scenario_readouts(
    graphs: GameGraphs, ratings: pd.DataFrame, usage: pd.DataFrame
) -> pd.DataFrame:
    """R1 and R2 totals per (game, scenario).

    ``ratings``: ``game_id``, ``player_id``, ``off``, ``def``, ``pace``,
    ``league_ortg``, ``league_pace`` (one row per game and player; a missing
    player is league average). ``usage``: ``game_id``, ``player_id``,
    ``usage``.
    """
    keys = ["game_id", "scenario_id"]
    nodes = graphs.nodes.merge(
        ratings[["game_id", "player_id", "off", "def", "pace"]],
        on=["game_id", "player_id"],
        how="left",
    ).merge(usage, on=["game_id", "player_id"], how="left")
    nodes[["off", "def", "pace"]] = nodes[["off", "def", "pace"]].fillna(0.0)
    share = nodes["minutes"] / 48.0
    team = (
        nodes.assign(
            off=share * nodes["off"],
            dfn=share * nodes["def"],
            pace=share * nodes["pace"],
        )
        .groupby([*keys, "side"])[["off", "dfn", "pace"]]
        .sum()
        .unstack("side")
    )

    # R2: the defense each side's attackers face.
    guards = graphs.edges.loc[graphs.edges["relation"].eq("guards")]
    rated = ratings.set_index(["game_id", "player_id"])["def"]
    defender = rated.reindex(
        pd.MultiIndex.from_arrays([guards["game_id"], guards["src"]])
    ).fillna(0.0)
    faced = (
        guards.assign(x=guards["weight"].to_numpy() * defender.to_numpy())
        .groupby([*keys, "dst"])["x"]
        .sum()
        .rename("x")
        .reset_index()
        .rename(columns={"dst": "player_id"})
        .merge(
            nodes[[*keys, "player_id", "side", "minutes", "usage"]],
            on=[*keys, "player_id"],
        )
    )
    faced["w"] = faced["minutes"] * faced["usage"]
    by_side = faced.groupby([*keys, "side"])
    weighted = (
        (faced["w"] * faced["x"])
        .groupby([faced[k] for k in keys] + [faced["side"]])
        .sum()
    )
    faced_def = (5.0 * weighted / by_side["w"].sum()).unstack("side")

    league = ratings.drop_duplicates("game_id").set_index("game_id")[
        ["league_ortg", "league_pace"]
    ]
    frame = team.copy()
    frame.columns = [f"{stat}_{side}" for stat, side in frame.columns]
    frame = frame.join(faced_def.add_prefix("faced_"))
    frame = frame.reset_index().join(league, on="game_id")
    poss = frame["league_pace"] + frame["pace_home"] + frame["pace_away"]
    ortg = frame["league_ortg"]
    out = frame[keys].copy()
    out["possessions"] = poss
    out["r1_total"] = (
        poss
        / 100
        * (
            2 * ortg
            + frame["off_home"]
            - frame["dfn_away"]
            + frame["off_away"]
            - frame["dfn_home"]
        )
    )
    # Home attackers face faced_home; away attackers face faced_away.
    out["r2_total"] = (
        poss
        / 100
        * (
            2 * ortg
            + frame["off_home"]
            - frame["faced_home"]
            + frame["off_away"]
            - frame["faced_away"]
        )
    )
    return out


def readouts(
    graphs: GameGraphs, ratings: pd.DataFrame, usage: pd.DataFrame
) -> pd.DataFrame:
    """Per game: tonight (scenario-weighted) and full-health R1 / R2 totals."""
    per = scenario_readouts(graphs, ratings, usage).merge(
        graphs.scenarios[["game_id", "scenario_id", "is_full_health", "weight"]],
        on=["game_id", "scenario_id"],
    )
    tonight = per.loc[~per["is_full_health"].astype(bool)]
    weights = tonight["weight"] / tonight.groupby("game_id")["weight"].transform("sum")
    # min_count: a game without ratings (no rating checkpoint) stays NaN
    # rather than summing to a total of 0.
    mean = (
        tonight[["r1_total", "r2_total", "possessions"]]
        .mul(weights, axis=0)
        .groupby(tonight["game_id"])
        .sum(min_count=1)
    )
    healthy = per.loc[per["scenario_id"].eq(FULL_HEALTH)].set_index("game_id")[
        ["r1_total", "r2_total"]
    ]
    return mean.join(healthy.add_suffix("_healthy"), how="left").reset_index()
