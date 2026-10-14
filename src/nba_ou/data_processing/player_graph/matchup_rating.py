"""Matchup-adjusted defensive rating, v0 (phase 4B).

One row per attacker and game (``pair_game`` through
``matchup_defense.read_matchup_history``): the attacker's points per partial
possession against all his defenders, weighted by his partial possessions,
explained by a walk-forward ridge::

    pts_ig / pp_ig = mu + h * home_ig + a_i + t_T(g) + sum_j s_ijg * d_j

* ``a_i``: the attacker's quality;
* ``t_T``: the defending team's effect (scheme, help, the rest of the floor);
* ``d_j``: the individual defender's effect, entering with ``s_ijg = pp_ij /
  pp_ig``, his share of the attacker's partial possessions (guarding time).
  The attacker's whole output is attributed by who guarded him and for how
  long, never by who was matched when the shot went up (step 1: that credit
  measures defensive role);
* ``h``: home court (practically unpenalized); ``mu``: free intercept.

No defender x attacker-type terms (step 1: not measurable per cell).

Ridge toward 0 with one penalty per block, in partial possessions (the units
of step 1's empirical-Bayes shrinkage). A shift shared by a whole team costs
``lambda_team`` on ``t_T`` but about 13 x ``lambda_defender`` spread over a
roster's defenders, so the team term takes team defense and ``d_j`` is the
defender relative to his team. The normal equations decay with
``half_life_days``; each date is solved on games strictly before it.

Units: ``t_T`` and ``d_j`` are points per partial possession of the attacker.
A defender guards about one partial possession per team possession, so
``-100 * (t_T + d_j)`` is in 2_6's ``d_rating`` units (points per 100
possessions, positive = good defense) and five of them add up to a team's.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from nba_ou.data_processing.lineups.player_ratings import _NormalEquations

from .rapm_sweep import _fit

HOME_PENALTY = 1e-3


@dataclass(frozen=True)
class Penalties:
    attacker: float
    team: float
    defender: float

    @property
    def label(self) -> str:
        return f"a={self.attacker:g} t={self.team:g} d={self.defender:g}"


def attacker_games(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(games, shares)``: one row per attacker and game (points, partial
    possessions, defending team, home) and each defender's share of the
    attacker's partial possessions."""
    pairs = pairs.loc[pairs["partial_possessions"] > 0]
    keys = ["game_id", "off_player_id"]
    games = (
        pairs.groupby(keys, sort=False)
        .agg(
            game_date=("game_date", "first"),
            def_team_id=("def_team_id", "first"),
            home=("attacker_is_home", "first"),
            points=("player_points", "sum"),
            exposure=("partial_possessions", "sum"),
        )
        .reset_index()
    )
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.normalize()
    shares = pairs[[*keys, "def_player_id", "partial_possessions"]].merge(
        games[[*keys, "exposure"]], on=keys
    )
    shares["share"] = shares["partial_possessions"] / shares["exposure"]
    return games, shares[[*keys, "def_player_id", "share"]]


@dataclass
class _Design:
    x: sparse.csr_matrix
    attackers: list[str]
    teams: list[str]
    defenders: list[str]

    @property
    def blocks(self) -> tuple[slice, slice, slice, int]:
        na, nt, nd = len(self.attackers), len(self.teams), len(self.defenders)
        return (
            slice(0, na),
            slice(na, na + nt),
            slice(na + nt, na + nt + nd),
            na + nt + nd,
        )

    def penalty(self, penalties: Penalties) -> np.ndarray:
        return np.r_[
            np.full(len(self.attackers), penalties.attacker),
            np.full(len(self.teams), penalties.team),
            np.full(len(self.defenders), penalties.defender),
            [HOME_PENALTY],
        ]


def _design(games: pd.DataFrame, shares: pd.DataFrame) -> _Design:
    attackers = sorted(games["off_player_id"].astype(str).unique())
    teams = sorted(games["def_team_id"].astype(str).unique())
    defenders = sorted(shares["def_player_id"].astype(str).unique())
    na, nt, nd = len(attackers), len(teams), len(defenders)
    a_index = {p: i for i, p in enumerate(attackers)}
    t_index = {t: na + i for i, t in enumerate(teams)}
    d_index = {p: na + nt + i for i, p in enumerate(defenders)}
    home = na + nt + nd
    row = pd.Series(
        np.arange(len(games)),
        index=pd.MultiIndex.from_frame(games[["game_id", "off_player_id"]]),
    )
    share_rows = row.reindex(
        pd.MultiIndex.from_frame(shares[["game_id", "off_player_id"]])
    ).to_numpy()
    n = len(games)
    is_home = games["home"].astype(bool).to_numpy()
    rows = np.r_[np.arange(n), np.arange(n), share_rows, np.flatnonzero(is_home)]
    cols = np.r_[
        games["off_player_id"].astype(str).map(a_index).to_numpy(),
        games["def_team_id"].astype(str).map(t_index).to_numpy(),
        shares["def_player_id"].astype(str).map(d_index).to_numpy(),
        np.full(is_home.sum(), home),
    ]
    values = np.r_[np.ones(2 * n), shares["share"].to_numpy(), np.ones(is_home.sum())]
    x = sparse.csr_matrix(
        (values, (rows, cols.astype(int))), shape=(n, na + nt + nd + 1)
    )
    return _Design(x, attackers, teams, defenders)


@dataclass
class MatchupRatings:
    """As-of output of :func:`walk_forward_matchup_ratings` for one penalty set.

    ``players``: ``as_of_date, player_id, attack, attack_exposure, defense,
    defense_exposure`` (exposures decayed, in partial possessions);
    ``teams``: ``as_of_date, team_id, team, team_exposure``;
    ``fits``: ``as_of_date, intercept, home, fit_max_game_date``;
    ``predictions``: the attacker-games on the requested dates with
    ``predicted`` and ``predicted_without_defenders`` (rates).
    """

    penalties: Penalties
    players: pd.DataFrame
    teams: pd.DataFrame
    fits: pd.DataFrame
    predictions: pd.DataFrame


def walk_forward_matchup_ratings(
    pairs: pd.DataFrame,
    target_dates: Iterable,
    penalties: Sequence[Penalties],
    *,
    half_life_days: float = 365.0,
    keep_ratings: bool | Iterable = True,
) -> list[MatchupRatings]:
    """Ratings on each target date from attacker-games strictly before it, for
    every penalty set (one shared set of decayed normal equations).
    ``keep_ratings``: keep the player / team ratings on every target date
    (True), on none (False) or on the given dates; predictions and fits are
    always kept."""
    games, shares = attacker_games(pairs)
    design = _design(games, shares)
    att, team, dfn, home = design.blocks
    y = (games["points"] / games["exposure"]).to_numpy(float)
    w = games["exposure"].to_numpy(float)
    by_date = {
        date: idx.to_numpy()
        for date, idx in games.groupby("game_date", sort=False).groups.items()
    }
    requested = {pd.Timestamp(d).normalize() for d in target_dates}
    if isinstance(keep_ratings, bool):
        rating_dates = requested if keep_ratings else set()
    else:
        rating_dates = {pd.Timestamp(d).normalize() for d in keep_ratings}
    equations = _NormalEquations(design.x.shape[1])
    exposure = np.zeros(design.x.shape[1])
    penalty = [design.penalty(p) for p in penalties]
    previous: list[np.ndarray | None] = [None] * len(penalties)
    players: list[list[pd.DataFrame]] = [[] for _ in penalties]
    teams: list[list[pd.DataFrame]] = [[] for _ in penalties]
    fits: list[list[dict]] = [[] for _ in penalties]
    predictions: list[list[pd.DataFrame]] = [[] for _ in penalties]
    last_date, last_fit = None, pd.NaT
    without_defenders = design.x.tolil()
    without_defenders[:, dfn] = 0.0
    without_defenders = without_defenders.tocsr()
    for day in sorted(requested | set(by_date)):
        if last_date is not None:
            factor = 0.5 ** ((day - last_date).days / half_life_days)
            equations.decay(factor)
            exposure *= factor
        last_date = day
        rows = by_date.get(day)
        if day in requested and equations.weight:
            for k in range(len(penalties)):
                beta, intercept = _fit(equations, penalty[k], previous[k])
                previous[k] = beta
                fits[k].append(
                    {
                        "as_of_date": day,
                        "intercept": intercept,
                        "home": beta[home],
                        "fit_max_game_date": last_fit,
                    }
                )
                if rows is not None:
                    part = games.iloc[rows][["game_id", "off_player_id", "game_date"]]
                    predictions[k].append(
                        part.assign(
                            actual=y[rows],
                            weight=w[rows],
                            predicted=design.x[rows] @ beta + intercept,
                            predicted_without_defenders=without_defenders[rows] @ beta
                            + intercept,
                        )
                    )
                if day in rating_dates:
                    seen_a = exposure[att] > 0
                    seen_d = exposure[dfn] > 0
                    attackers = pd.DataFrame(
                        {
                            "player_id": np.asarray(design.attackers)[seen_a],
                            "attack": beta[att][seen_a],
                            "attack_exposure": exposure[att][seen_a],
                        }
                    )
                    defenders = pd.DataFrame(
                        {
                            "player_id": np.asarray(design.defenders)[seen_d],
                            "defense": beta[dfn][seen_d],
                            "defense_exposure": exposure[dfn][seen_d],
                        }
                    )
                    players[k].append(
                        attackers.merge(defenders, on="player_id", how="outer").assign(
                            as_of_date=day
                        )
                    )
                    teams[k].append(
                        pd.DataFrame(
                            {
                                "as_of_date": day,
                                "team_id": design.teams,
                                "team": beta[team],
                                "team_exposure": exposure[team],
                            }
                        )
                    )
        if rows is not None:
            x = design.x[rows]
            equations.add(x, y[rows], w[rows])
            exposure += np.asarray(x.T @ w[rows]).ravel()
            last_fit = day
    out = []
    for k, p in enumerate(penalties):
        out.append(
            MatchupRatings(
                penalties=p,
                players=(
                    pd.concat(players[k], ignore_index=True)
                    if players[k]
                    else pd.DataFrame()
                ),
                teams=(
                    pd.concat(teams[k], ignore_index=True)
                    if teams[k]
                    else pd.DataFrame()
                ),
                fits=pd.DataFrame(fits[k]),
                predictions=(
                    pd.concat(predictions[k], ignore_index=True)
                    if predictions[k]
                    else pd.DataFrame()
                ),
            )
        )
    return out


def player_teams(pairs: pd.DataFrame, dates: Iterable) -> pd.DataFrame:
    """``as_of_date, player_id, team_id``: each player's team on each date,
    the team of his game that day or else of his latest earlier game."""
    played = pd.concat(
        [
            pairs[["game_date", "off_player_id", "off_team_id"]].set_axis(
                ["game_date", "player_id", "team_id"], axis=1
            ),
            pairs[["game_date", "def_player_id", "def_team_id"]].set_axis(
                ["game_date", "player_id", "team_id"], axis=1
            ),
        ]
    ).drop_duplicates(["game_date", "player_id"])
    played["game_date"] = (
        pd.to_datetime(played["game_date"]).dt.normalize().astype("datetime64[ns]")
    )
    grid = pd.MultiIndex.from_product(
        [
            sorted({pd.Timestamp(d).normalize() for d in dates}),
            played["player_id"].unique(),
        ],
        names=["as_of_date", "player_id"],
    ).to_frame(index=False)
    grid["as_of_date"] = grid["as_of_date"].astype("datetime64[ns]")
    merged = pd.merge_asof(
        grid.sort_values("as_of_date"),
        played.sort_values("game_date"),
        left_on="as_of_date",
        right_on="game_date",
        by="player_id",
        direction="backward",
    )
    return merged.dropna(subset=["team_id"])[["as_of_date", "player_id", "team_id"]]


def defensive_ratings(
    ratings: MatchupRatings, teams_on: pd.DataFrame, *, include_team: bool = True
) -> pd.DataFrame:
    """``as_of_date, player_id, d_matchup, d_team, d_individual`` in 2_6's
    ``d_rating`` units (points per 100 possessions, positive = good defense):
    ``d_matchup = d_team + d_individual = -100 * (t_T + d_j)`` with ``T`` the
    player's team that day (``player_teams``). A player without defensive
    history has ``d_individual`` 0; without a team, ``d_team`` 0."""
    frame = teams_on.merge(
        ratings.players[["as_of_date", "player_id", "defense"]],
        on=["as_of_date", "player_id"],
        how="left",
    ).merge(
        ratings.teams[["as_of_date", "team_id", "team"]],
        on=["as_of_date", "team_id"],
        how="left",
    )
    frame["d_individual"] = -100.0 * frame["defense"].fillna(0.0)
    frame["d_team"] = -100.0 * frame["team"].fillna(0.0) if include_team else 0.0
    frame["d_matchup"] = frame["d_team"] + frame["d_individual"]
    return frame[["as_of_date", "player_id", "d_matchup", "d_team", "d_individual"]]
