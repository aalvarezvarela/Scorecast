"""The phase 4A minutes provider v1: ``q * (b + C)``, reconciled to the
physical constraints.

Per team and availability scenario, the raw minutes of each available player
are ``raw_i = q_i * (b_i + C_i)`` (participation A, baseline B, absence gains C,
``rotation`` and ``participation``); players out in the scenario play 0. The
reconciliation then imposes

    0 <= minutes_i <= 48        sum_i minutes_i = 240

with ``minutes_i = min(48, c * raw_i)`` and the scale ``c`` found by
bisection: the weighted least-squares projection (weights ``1 / raw_i``) onto
the capped simplex. It is proportional rescaling, as ``allocate_minutes``
does, until the 48-minute cap binds, and then the capped players' excess goes
to the others in proportion. ``raw_i >= 0`` keeps every minute non-negative.
Fewer than five players with positive raw minutes cannot fill 240 minutes:
such a team is flagged infeasible and left at its capped values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TEAM_MINUTES = 240.0
PLAYER_CAP = 48.0


@dataclass(frozen=True)
class Reconciled:
    minutes: np.ndarray
    scale: float  # c
    feasible: bool
    capped: int  # players at the 48-minute cap


def reconcile(
    raw: np.ndarray,
    team_minutes: float = TEAM_MINUTES,
    cap: float = PLAYER_CAP,
    tolerance: float = 1e-9,
) -> Reconciled:
    """``min(cap, c * raw)`` summing to ``team_minutes``; see the module."""
    raw = np.clip(np.asarray(raw, dtype=float), 0.0, None)
    positive = raw > 0
    if positive.sum() * cap < team_minutes - tolerance:
        minutes = np.where(positive, cap, 0.0)
        return Reconciled(minutes, np.inf, False, int(positive.sum()))
    low, high = 0.0, team_minutes / raw[positive].min()  # high fills every cap
    for _ in range(200):
        middle = (low + high) / 2
        if np.minimum(cap, middle * raw).sum() < team_minutes:
            low = middle
        else:
            high = middle
        if high - low < tolerance * max(1.0, high):
            break
    minutes = np.minimum(cap, high * raw)
    # Close the last rounding gap on the uncapped players.
    free = minutes < cap - 1e-12
    gap = team_minutes - minutes.sum()
    if free.any() and abs(gap) > 0:
        minutes[free] += gap * raw[free] / raw[free].sum()
    return Reconciled(minutes, float(high), True, int((minutes >= cap - 1e-9).sum()))


# --------------------------------------------------------------------------
# Scenario provider (injury-report scenarios) and joint game graphs
# --------------------------------------------------------------------------

FULL_HEALTH = -1


class ScenarioProvider:
    """v1 minutes for the injury report's scenarios, as an
    ``on_team_game`` callback of ``rotation.walk_forward``.

    For a covered team-game: the expanded roster minus report exclusions, each
    player's ``p_out`` from the report, the scenarios ``enumerate_scenarios``
    makes of them (as v0), and in each scenario the v1 minutes: its sitting
    players are C's absences, ``q`` comes from the month's model (fitted on
    earlier months), ``raw = q * (b + C)``, reconciled. The full-health
    graph is the same provider with nobody sitting. Results accumulate in
    ``team_scenarios`` and ``team_nodes``.
    """

    def __init__(self, p_out, excluded, covered, models):
        self.p_out = p_out
        self.excluded = excluded
        self.covered = covered
        self.models = models
        self.team_scenarios: list[dict] = []
        self.team_nodes: list[dict] = []

    def __call__(self, ctx: dict) -> None:
        from nba_ou.data_processing.lineups.game_projection import (
            PlayerNight,
            enumerate_scenarios,
        )

        game_id, team = ctx["game_id"], ctx["team"]
        model = self.models.get(ctx["date"].to_period("M"))
        if (game_id, team) not in self.covered or model is None:
            return
        roster = [p for p in ctx["roster"] if (game_id, team, p) not in self.excluded]
        nights = [
            PlayerNight(
                p,
                ctx["baseline"].get(p, 0.0),
                self.p_out.get((game_id, team, p), 0.0),
            )
            for p in roster
        ]
        options = [(FULL_HEALTH, frozenset(), 1.0)] + [
            (k, sitting, weight)
            for k, (sitting, weight) in enumerate(enumerate_scenarios(nights))
        ]
        for scenario_id, sitting, weight in options:
            minutes = self._minutes(ctx, model, roster, sitting)
            if not minutes:
                continue
            self.team_scenarios.append(
                {
                    "game_id": game_id,
                    "team_id": team,
                    "scenario_id": scenario_id,
                    "weight": weight,
                    "sitting": sitting,
                }
            )
            self.team_nodes.extend(
                {
                    "game_id": game_id,
                    "team_id": team,
                    "scenario_id": scenario_id,
                    "player_id": p,
                    "minutes": m,
                }
                for p, m in minutes.items()
            )

    def _minutes(self, ctx, model, roster, sitting) -> dict[str, float]:
        from .participation import feature_matrix
        from .rotation import pair_features, shares

        params = ctx["params"]
        baseline, participation = ctx["baseline"], ctx["participation"]
        candidates = [p for p in roster if p not in sitting and p in baseline]
        if not candidates:
            return {}
        absences = {
            x: baseline[x]
            * (
                1.0
                if baseline[x] >= params.rotation_minutes
                else participation.get(x, 0.0)
            )
            for x in sitting
            if x in ctx["current"] and x in baseline
        }
        absences = {x: v for x, v in absences.items() if v >= params.min_vacated}
        gain = dict.fromkeys(candidates, 0.0)
        b = np.array([baseline[y] for y in candidates])
        for x, vacated in absences.items():
            features = pair_features(
                x,
                candidates,
                baseline,
                participation,
                ctx["start_share"],
                ctx["positions"],
            )
            s0 = ctx["structural"](features, b)
            final = shares(x, candidates, ctx["evidence"], s0, params.kappa)
            for y, share in zip(candidates, final, strict=True):
                gain[y] += vacated * share
        history = ctx["history"]
        last = history[-1] if history else None
        rest = (
            (ctx["date"] - last.date).days if last is not None and last.date else np.nan
        )
        ranked = sorted(candidates, key=lambda p: (-baseline[p], p))
        rows = []
        for y in candidates:
            streak = 0
            for g in reversed(history):
                if y in g.played or y not in g.roster:
                    break
                streak += 1
            rows.append(
                {
                    "b": baseline[y],
                    "rank": ranked.index(y) + 1,
                    "gain_c": gain[y],
                    "vacated_others": sum(absences.values()),
                    "n_available": len(candidates),
                    "q": participation.get(y, np.nan),
                    "streak": streak,
                    "last_minutes": last.played.get(y, 0.0) if last else 0.0,
                    "start_share": ctx["start_share"].get(y, 0.0),
                    "rest_days": rest,
                    "season_game": ctx["season_game"],
                }
            )
        frame = pd.DataFrame(rows)
        q = model.predict_proba(feature_matrix(frame))[:, 1]
        raw = q * (b + np.array([gain[y] for y in candidates]))
        result = reconcile(raw)
        return dict(zip(candidates, result.minutes, strict=True))


def team_scenarios_v0(nights) -> tuple[list[dict], list[dict]]:
    """v0's per-team scenarios and minutes (``enumerate_scenarios``,
    ``allocate_minutes``), in the same shape as :class:`ScenarioProvider`."""
    from nba_ou.data_processing.lineups.game_projection import (
        allocate_minutes,
        enumerate_scenarios,
    )

    scenarios, nodes = [], []
    for (game_id, team), players in nights.items():
        options = [(FULL_HEALTH, frozenset(), 1.0)] + [
            (k, sitting, weight)
            for k, (sitting, weight) in enumerate(enumerate_scenarios(players))
        ]
        for scenario_id, sitting, weight in options:
            minutes = allocate_minutes(players, sitting)
            if not minutes:
                continue
            scenarios.append(
                {
                    "game_id": game_id,
                    "team_id": team,
                    "scenario_id": scenario_id,
                    "weight": weight,
                    "sitting": sitting,
                }
            )
            nodes.extend(
                {
                    "game_id": game_id,
                    "team_id": team,
                    "scenario_id": scenario_id,
                    "player_id": p,
                    "minutes": m,
                }
                for p, m in minutes.items()
            )
    return scenarios, nodes


@dataclass
class JointGraphs:
    """The two frames ``game_graph.readout_2_6`` reads: ``scenarios``
    (game_id, scenario_id, weight, is_full_health, as_of_date) and ``nodes``
    (game_id, scenario_id, player_id, side, minutes)."""

    scenarios: pd.DataFrame
    nodes: pd.DataFrame


def joint_graphs(
    team_scenarios: list[dict], team_nodes: list[dict], games: pd.DataFrame
) -> JointGraphs:
    """Home x away scenario products per game (weights multiplied), the
    full-health pair as ``FULL_HEALTH``; games missing a side are dropped.
    ``games``: GAME_ID, GAME_DATE, HOME_TEAM_ID, AWAY_TEAM_ID."""
    scen = pd.DataFrame(team_scenarios)
    nodes = pd.DataFrame(team_nodes)
    by_team = {k: part for k, part in scen.groupby(["game_id", "team_id"])}
    node_by = {
        k: part for k, part in nodes.groupby(["game_id", "team_id", "scenario_id"])
    }
    out_scenarios, out_nodes = [], []
    for game in games.itertuples(index=False):
        home = by_team.get((game.GAME_ID, game.HOME_TEAM_ID))
        away = by_team.get((game.GAME_ID, game.AWAY_TEAM_ID))
        if home is None or away is None:
            continue
        pairs = [(FULL_HEALTH, FULL_HEALTH, 1.0, True)] + [
            (h.scenario_id, a.scenario_id, h.weight * a.weight, False)
            for h in home.loc[home["scenario_id"].ne(FULL_HEALTH)].itertuples()
            for a in away.loc[away["scenario_id"].ne(FULL_HEALTH)].itertuples()
        ]
        for k, (h_id, a_id, weight, full) in enumerate(pairs):
            if (game.GAME_ID, game.HOME_TEAM_ID, h_id) not in node_by or (
                game.GAME_ID,
                game.AWAY_TEAM_ID,
                a_id,
            ) not in node_by:
                continue
            scenario_id = FULL_HEALTH if full else k
            out_scenarios.append(
                {
                    "game_id": game.GAME_ID,
                    "scenario_id": scenario_id,
                    "weight": weight,
                    "is_full_health": full,
                    "as_of_date": game.GAME_DATE,
                }
            )
            for side, team, sid in (
                ("home", game.HOME_TEAM_ID, h_id),
                ("away", game.AWAY_TEAM_ID, a_id),
            ):
                part = node_by[(game.GAME_ID, team, sid)]
                out_nodes.append(
                    pd.DataFrame(
                        {
                            "game_id": game.GAME_ID,
                            "scenario_id": scenario_id,
                            "player_id": part["player_id"].to_numpy(),
                            "side": side,
                            "minutes": part["minutes"].to_numpy(),
                        }
                    )
                )
    return JointGraphs(
        pd.DataFrame(out_scenarios), pd.concat(out_nodes, ignore_index=True)
    )


def expected_minutes(
    team_scenarios: list[dict], team_nodes: list[dict]
) -> pd.DataFrame:
    """Scenario-weighted minutes per (game, team, player), full-health
    excluded, weights renormalized over the non-empty scenarios."""
    scen = pd.DataFrame(team_scenarios)
    scen = scen.loc[scen["scenario_id"].ne(FULL_HEALTH)]
    scen["w"] = scen["weight"] / scen.groupby(["game_id", "team_id"])[
        "weight"
    ].transform("sum")
    nodes = pd.DataFrame(team_nodes).merge(
        scen[["game_id", "team_id", "scenario_id", "w"]],
        on=["game_id", "team_id", "scenario_id"],
    )
    nodes["weighted"] = nodes["w"] * nodes["minutes"]
    return (
        nodes.groupby(["game_id", "team_id", "player_id"], as_index=False)["weighted"]
        .sum()
        .rename(columns={"weighted": "expected"})
    )
