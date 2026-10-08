"""Game graphs (prediction graphs): who is expected to play at T, and against whom.

One graph per game and **availability scenario**, plus the game's
**full-health** graph (nobody out), which is the counterfactual for every
absence feature. Closing only for now: availability is the last injury report
before tip.

**Rosters, availability and minutes come from 2_6, unchanged**: tonight's
roster and recent minutes from ``lineups.availability.build_player_nights``
(strictly earlier dates), play probabilities from ``chance_out``, scenarios
from ``game_projection.enumerate_scenarios`` and minutes from
``allocate_minutes`` (each playing player's recent average rescaled to 240;
inside a scenario nobody's minutes are scaled by a probability). A scenario in
which a team has nobody to play is dropped, as 2_6 drops it.

**Coverage**: as in 2_6, a game is only projected when both teams had filed an
injury report before the cutoff (``report_covered``, required). Otherwise
availability is unknown, and a graph with everybody "available" would be a
falsely certain projection, so the game gets no graphs (its features are NaN).

**Nodes**: the players who play in the scenario with positive minutes,
weighted by minutes. A roster player whose recent average is 0 (only coach's
decision DNPs in the window) is not a node: he adds nothing to 2_6's sums and
would have no defined guard shares.

**Teammate and opponent edges** (undirected, stored once): weight is the
expected overlap in minutes, ``min(lift * min_i * min_j / 48, min_i, min_j)``
with the pair lift of ``overlap.py`` as of the game's date (``independence``
provider: lift 1).

**Guard edges** (defender -> attacker, every opponent pair playing in the
scenario)::

    m_ij = r_hat_ij * E[overlap_ij] / sum_l r_hat_il * E[overlap_il]

over the defenders playing in that scenario, with ``r_hat`` from
``expected_guard`` as of the game's date. An absent defender simply leaves the
denominator, so his attackers are reassigned to tonight's defenders without
any extra rule. An attacker whose rates are not all known gets shares
proportional to the overlap (``fallback``), as in the guard benchmark.

Tables, not graph objects: ``scenarios``, ``nodes`` and ``edges``
(:class:`GameGraphs`). Edge attributes are per relation: ``lift`` and
``lift_prior_weight`` on teammate / opponent edges, ``r_hat``, ``r_hat_known``
and ``guard_prior_weight`` on guard edges; a column that does not apply to a
relation is NaN, while ``weight``, ``expected_overlap`` and ``log_exposure`` are
finite everywhere. The encoder's inputs carry no observed same-game shares
and no betting columns (``stint_graph.check_encoder_inputs``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from nba_ou.data_processing.lineups.game_projection import (
    MAX_ENUMERATED_UNCERTAIN,
    PlayerNight,
    allocate_minutes,
    enumerate_scenarios,
)

from .as_of import PointInTimeData
from .expected_guard import DEFAULT_PARAMS as DEFAULT_GUARD
from .expected_guard import ExpectedGuardParams, expected_guard
from .overlap import DEFAULT_PARAMS as DEFAULT_OVERLAP
from .overlap import OverlapParams, expected_lift, expected_overlap, pair_key
from .positions import positions_as_of
from .stint_graph import check_encoder_inputs

VERSION = "v0"
FULL_HEALTH = -1  # scenario_id of the full-health graph

SCENARIO_COLUMNS = (
    "game_id",
    "game_date",
    "scenario_id",
    "is_full_health",
    "weight",
    "home_sitting",
    "away_sitting",
)
NODE_COLUMNS = ("game_id", "scenario_id", "player_id", "side", "minutes")
EDGE_COLUMNS = (
    "game_id",
    "scenario_id",
    "relation",
    "src",
    "dst",
    "weight",
    "expected_overlap",
    "lift",
    "lift_prior_weight",
    "r_hat",
    "r_hat_known",
    "guard_prior_weight",
    "log_exposure",
    "has_pair_history",
    "fallback",
)


@dataclass(frozen=True)
class GameGraphParams:
    guard: ExpectedGuardParams = DEFAULT_GUARD
    overlap: OverlapParams = DEFAULT_OVERLAP
    game_minutes: float = 48.0
    max_enumerated: int = MAX_ENUMERATED_UNCERTAIN


DEFAULT_PARAMS = GameGraphParams()


@dataclass(frozen=True)
class GameGraphs:
    """Scenario, node and edge tables; one graph per (game_id, scenario_id)."""

    scenarios: pd.DataFrame
    nodes: pd.DataFrame
    edges: pd.DataFrame
    metadata: dict = field(default_factory=dict)

    def graph(self, game_id: str, scenario_id: int) -> dict:
        def pick(frame: pd.DataFrame) -> pd.DataFrame:
            return frame.loc[
                frame["game_id"].eq(game_id) & frame["scenario_id"].eq(scenario_id)
            ].reset_index(drop=True)

        edges = pick(self.edges)
        return {
            "scenario": pick(self.scenarios).iloc[0].to_dict(),
            "nodes": pick(self.nodes),
            **{
                relation: edges.loc[edges["relation"].eq(relation)].reset_index(
                    drop=True
                )
                for relation in ("teammate", "opponent", "guards")
            },
        }


def _playing(minutes: dict[str, float]) -> dict[str, float]:
    """Players with positive minutes; zero-minute roster entries are not nodes."""
    return {player: played for player, played in minutes.items() if played > 0}


def _scenarios(
    home: list[PlayerNight], away: list[PlayerNight], max_enumerated: int
) -> list[dict]:
    """Every scenario with both teams playing, then the full-health graph."""
    out = []
    home_options = [
        (sitting, weight, _playing(allocate_minutes(home, sitting)))
        for sitting, weight in enumerate_scenarios(home, max_enumerated)
    ]
    away_options = [
        (sitting, weight, _playing(allocate_minutes(away, sitting)))
        for sitting, weight in enumerate_scenarios(away, max_enumerated)
    ]
    scenario_id = 0
    for home_sitting, home_weight, home_minutes in home_options:
        for away_sitting, away_weight, away_minutes in away_options:
            if home_minutes and away_minutes:
                out.append(
                    {
                        "scenario_id": scenario_id,
                        "is_full_health": False,
                        "weight": home_weight * away_weight,
                        "home_sitting": tuple(sorted(home_sitting)),
                        "away_sitting": tuple(sorted(away_sitting)),
                        "home": home_minutes,
                        "away": away_minutes,
                    }
                )
            scenario_id += 1
    healthy_home = _playing(allocate_minutes(home))
    healthy_away = _playing(allocate_minutes(away))
    if out and healthy_home and healthy_away:
        out.append(
            {
                "scenario_id": FULL_HEALTH,
                "is_full_health": True,
                "weight": 1.0,
                "home_sitting": (),
                "away_sitting": (),
                "home": healthy_home,
                "away": healthy_away,
            }
        )
    return out


def _roster_pairs(game_id: str, home: list[str], away: list[str]) -> tuple:
    """All teammate / opponent pairs and directed guard pairs of the rosters."""
    undirected = [
        (game_id, a, b, "teammate")
        for side in (home, away)
        for a, b in combinations(sorted(side), 2)
    ] + [(game_id, h, a, "opponent") for h in home for a in away]
    directed = [(game_id, h, a) for h in home for a in away] + [
        (game_id, a, h) for a in away for h in home
    ]
    return undirected, directed


def guard_shares(
    groups: list[pd.Series], rate: np.ndarray, overlap: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(m, fallback)``: ``m`` proportional to ``rate * overlap`` within each
    group (one attacker in one scenario). A group whose rates are not all
    finite, or whose load sums to 0, gets shares proportional to the overlap
    and ``fallback`` True."""
    index = groups[0].index
    load = pd.Series(np.asarray(rate, float) * np.asarray(overlap, float), index=index)
    overlap = pd.Series(np.asarray(overlap, float), index=index)
    finite = np.isfinite(load).groupby(groups).transform("all")
    total = load.where(finite, 0.0).groupby(groups).transform("sum")
    usable = finite & (total > 0)
    constant = overlap / overlap.groupby(groups).transform("sum")
    return (load / total).where(usable, constant).to_numpy(), (~usable).to_numpy()


def _scenario_edges(
    game_id: str,
    scenario: dict,
    lifts: pd.DataFrame,
    guards: pd.DataFrame,
    game_minutes: float,
) -> pd.DataFrame:
    """Teammate, opponent and guard edges of one scenario."""
    minutes = {**scenario["home"], **scenario["away"]}
    home, away = sorted(scenario["home"]), sorted(scenario["away"])
    undirected = [
        (a, b, "teammate") for side in (home, away) for a, b in combinations(side, 2)
    ] + [(h, a, "opponent") for h in home for a in away]
    pairs = pd.DataFrame(undirected, columns=["src", "dst", "relation"])
    pairs["player_a"], pairs["player_b"] = pair_key(pairs["src"], pairs["dst"])
    pairs = pairs.merge(lifts, on=["player_a", "player_b", "relation"], how="left")
    pairs["expected_overlap"] = expected_overlap(
        pairs["src"].map(minutes),
        pairs["dst"].map(minutes),
        pairs["lift"],
        game_minutes,
    )
    pairs["weight"] = pairs["expected_overlap"]
    pairs["log_exposure"] = np.log1p(pairs["hist_independent_seconds_decayed"])

    overlap = pairs.loc[pairs["relation"].eq("opponent")].set_index(
        ["player_a", "player_b"]
    )["expected_overlap"]
    directed = pd.DataFrame(
        [(a, d) for a in home for d in away] + [(a, d) for a in away for d in home],
        columns=["off_player_id", "def_player_id"],
    )
    directed = directed.merge(guards, on=["off_player_id", "def_player_id"], how="left")
    a, b = pair_key(directed["off_player_id"], directed["def_player_id"])
    directed["expected_overlap"] = overlap.reindex(
        pd.MultiIndex.from_arrays([a, b])
    ).to_numpy()
    rate = directed["r_hat"].to_numpy(float)
    shares, fallback = guard_shares(
        [directed["off_player_id"]], rate, directed["expected_overlap"].to_numpy()
    )
    guard_edges = pd.DataFrame(
        {
            "relation": "guards",
            "src": directed["def_player_id"],
            "dst": directed["off_player_id"],
            "weight": shares,
            "expected_overlap": directed["expected_overlap"].to_numpy(),
            "r_hat": np.nan_to_num(rate, nan=0.0),
            "r_hat_known": np.isfinite(rate),
            "guard_prior_weight": directed["prior_weight"].fillna(1.0).to_numpy(),
            "log_exposure": np.log1p(
                directed["hist_cofloor_seconds_decayed"].fillna(0.0).to_numpy()
            ),
            "has_pair_history": directed["has_pair_history"].eq(True).to_numpy(),
            "fallback": fallback,
        }
    )
    pair_edges = pairs[
        [
            "relation",
            "src",
            "dst",
            "weight",
            "expected_overlap",
            "lift",
            "lift_prior_weight",
            "log_exposure",
            "has_pair_history",
        ]
    ].assign(fallback=False)
    edges = pd.concat([pair_edges, guard_edges], ignore_index=True)
    edges.insert(0, "scenario_id", scenario["scenario_id"])
    edges.insert(0, "game_id", game_id)
    return edges.reindex(columns=list(EDGE_COLUMNS))


def build_game_graphs(
    data: PointInTimeData,
    games: pd.DataFrame,
    nights: Mapping[tuple[str, str], list[PlayerNight]],
    *,
    report_covered: set[tuple[str, str]],
    params: GameGraphParams = DEFAULT_PARAMS,
) -> GameGraphs:
    """Game graphs of ``games`` (``GAME_ID``, ``GAME_DATE``, ``HOME_TEAM_ID``,
    ``AWAY_TEAM_ID``), with rosters and availability from ``nights``
    (``lineups.features.game_nights``).

    ``report_covered`` holds the ``(game_id, team_id)`` that had filed an
    injury report before the cutoff (``InjuryReportState.covered``). A game
    without both teams covered, or without a roster on either side, gets no
    graphs; ``metadata`` counts both.
    """
    covered = {(str(game).zfill(10), str(team)) for game, team in report_covered}
    skipped = {"uncovered": 0, "no_roster": 0}
    games = games.assign(
        GAME_ID=games["GAME_ID"].astype(str).str.zfill(10),
        GAME_DATE=pd.to_datetime(games["GAME_DATE"]).dt.normalize(),
        HOME_TEAM_ID=games["HOME_TEAM_ID"].astype(str),
        AWAY_TEAM_ID=games["AWAY_TEAM_ID"].astype(str),
    ).drop_duplicates("GAME_ID")
    scenario_rows, node_frames, edge_frames = [], [], []
    for date, today in games.groupby("GAME_DATE", sort=True):
        rosters = {}
        undirected, directed = [], []
        for game in today.itertuples(index=False):
            if (game.GAME_ID, game.HOME_TEAM_ID) not in covered or (
                game.GAME_ID,
                game.AWAY_TEAM_ID,
            ) not in covered:
                skipped["uncovered"] += 1
                continue
            home = nights.get((game.GAME_ID, game.HOME_TEAM_ID))
            away = nights.get((game.GAME_ID, game.AWAY_TEAM_ID))
            if not home or not away:
                skipped["no_roster"] += 1
                continue
            rosters[game.GAME_ID] = (home, away)
            pairs, guards = _roster_pairs(
                game.GAME_ID,
                [p.player_id for p in home],
                [p.player_id for p in away],
            )
            undirected += pairs
            directed += guards
        if not rosters:
            continue
        view = data.as_of(date)
        lifts = expected_lift(
            view,
            pd.DataFrame(
                undirected, columns=["game_id", "player_a", "player_b", "relation"]
            ),
            params=params.overlap,
        )
        guards = expected_guard(
            view,
            pd.DataFrame(
                directed, columns=["game_id", "off_player_id", "def_player_id"]
            ),
            params=params.guard,
            positions=positions_as_of(view, window_days=params.guard.window_days),
        )
        check_encoder_inputs(lifts)
        check_encoder_inputs(guards)
        lifts_by_game = dict(tuple(lifts.groupby("game_id")))
        guards_by_game = dict(tuple(guards.groupby("game_id")))
        for game_id, (home, away) in rosters.items():
            for scenario in _scenarios(home, away, params.max_enumerated):
                scenario_rows.append(
                    {"game_id": game_id, "game_date": date}
                    | {k: scenario[k] for k in SCENARIO_COLUMNS[2:]}
                )
                node_frames.append(
                    pd.DataFrame(
                        [
                            (game_id, scenario["scenario_id"], player, side, played)
                            for side in ("home", "away")
                            for player, played in sorted(scenario[side].items())
                        ],
                        columns=list(NODE_COLUMNS),
                    )
                )
                edge_frames.append(
                    _scenario_edges(
                        game_id,
                        scenario,
                        lifts_by_game[game_id].drop(columns="game_id"),
                        guards_by_game[game_id].drop(columns="game_id"),
                        params.game_minutes,
                    )
                )
    return GameGraphs(
        scenarios=pd.DataFrame(scenario_rows, columns=list(SCENARIO_COLUMNS)),
        nodes=(
            pd.concat(node_frames, ignore_index=True)
            if node_frames
            else pd.DataFrame(columns=list(NODE_COLUMNS))
        ),
        edges=(
            pd.concat(edge_frames, ignore_index=True)
            if edge_frames
            else pd.DataFrame(columns=list(EDGE_COLUMNS))
        ),
        metadata={
            "version": VERSION,
            "skipped_games": skipped,
            "guard": params.guard.__dict__,
            "overlap": params.overlap.__dict__,
        },
    )


def readout_2_6(graphs: GameGraphs, ratings) -> pd.DataFrame:
    """The 2_6 projection read off the graphs' node minutes (integration check).

    Per scenario: minutes-weighted offense, defense and pace per team
    (``team_aggregate``), then possessions and points (``project_totals``);
    the scenario-weighted mean is tonight's projection and the full-health
    graph is the counterfactual. ``ratings`` is a 2_6 ``RatingBook``. Returns
    ``GAME_ID``, ``raw_total``, ``healthy_total``, ``possessions`` and
    ``impact_points``, which must equal 2_6's ``project_lineup_games``.
    """
    from nba_ou.data_processing.lineups.game_projection import (
        project_totals,
        team_aggregate,
    )

    nodes = {
        key: frame
        for key, frame in graphs.nodes.groupby(["game_id", "scenario_id"], sort=False)
    }
    rows = []
    for game_id, scenarios in graphs.scenarios.groupby("game_id", sort=False):
        rated = ratings.for_date(scenarios["game_date"].iloc[0])
        if rated is None:
            continue
        team_ratings, league_ortg, league_pace = rated
        projections, weights, healthy = [], [], None
        for scenario in scenarios.itertuples(index=False):
            frame = nodes[(game_id, scenario.scenario_id)]
            aggregates = [
                team_aggregate(
                    dict(
                        zip(
                            frame.loc[frame["side"].eq(side), "player_id"],
                            frame.loc[frame["side"].eq(side), "minutes"],
                            strict=True,
                        )
                    ),
                    team_ratings,
                )
                for side in ("home", "away")
            ]
            projection = project_totals(*aggregates, league_ortg, league_pace)
            if scenario.is_full_health:
                healthy = projection
            else:
                projections.append(projection)
                weights.append(scenario.weight)
        if healthy is None or not projections:
            continue
        weights = np.asarray(weights) / np.sum(weights)
        mean = {
            key: float(weights @ np.array([p[key] for p in projections]))
            for key in ("home_points", "away_points", "possessions")
        }
        total = mean["home_points"] + mean["away_points"]
        rows.append(
            {
                "GAME_ID": game_id,
                "raw_total": total,
                "healthy_total": healthy["total"],
                "possessions": mean["possessions"],
                "impact_points": total - healthy["total"],
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "GAME_ID",
            "raw_total",
            "healthy_total",
            "possessions",
            "impact_points",
        ],
    )
