"""Three-point volume of tonight's matchup, from similar lineup pairs in the past.

Lineups combine additively for scoring, pace, free throws, rebounding and
turnovers -- a matchup plays like the sum of its two sides
(``docs/lineup_projection_plan.md`` section 8.6). **Three-point volume is the
exception**: borrowing from the most similar past offense-five vs defense-five
pairs, from any teams, predicts part of what the additive model misses. As
built here, on tonight's rotations, the interaction correlates +0.065 with the
game's non-additive 3PA rate (2021-2025; positive in four of five seasons).

**What it is not, yet: a totals signal.** Against the closing line's error the
interaction is null (+0.10 per SD, [-0.33, +0.53]), and so is the
absence-driven 3PA shift (+0.17, [-0.27, +0.61]) -- an exploratory probe had
the shift at +0.55, which this walk-forward construction does not reproduce.
Three-point volume is not points. The columns are emitted so the campaign and
the pre-registered 2019-20 / 2020-21 test can judge them with this code fixed
in advance.

The method:

1. **Player style traits**, walk-forward. Each player's decayed, shrunk
   on-court rates in six stats (3PA/FGA, FTA/FGA, offensive rebound %,
   turnovers and points per possession, pace), once with him on offense and
   once on defense, from stints on dates strictly before the one being read.
2. **The pool.** Every past stint direction, described by the joint vector
   [offense five's six offensive traits, defense five's six defensive traits]
   as they stood on that stint's date, and its observed 3PA/FGA.
3. **Monthly refit** on the pool of earlier months: an additive linear model
   of 3PA/FGA on the joint vector, and a nearest-neighbour index over the
   additive model's residuals.
4. **Per game and direction.** Tonight's rotation vectors -- the
   expected-minutes-weighted mean of each side's traits, with tonight's
   absentees sitting -- give the additive expected 3PA rate, and the
   FGA-weighted mean residual of the ``k`` most similar past pairs gives the
   interaction. The additive rate is also computed at full health; the
   difference is the absence-driven shift.

**Temporal contract.** Traits read on date D use stints before D; the pool and
both models used in month M hold only stints before M; the rotation comes from
``availability.build_player_nights`` (box scores before D, last report before
tip). Same-day stints are held back together.

**Cost.** Two neighbour queries per game plus one index build per month: a
few minutes for 2021-2026. The pool vectors are computed once, vectorised.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .game_projection import PlayerNight, allocate_minutes

#: stat -> (numerator, denominator, shrinkage in denominator units). The
#: numerators and denominators are the offense's counts in a stint direction.
STYLE_STATS: dict[str, tuple[str, str, float]] = {
    "fg3a_rate": ("fg3a", "fga", 300.0),
    "fta_rate": ("fta", "fga", 300.0),
    "oreb_pct": ("oreb", "reb_chances", 150.0),
    "tov_rate": ("tov", "poss", 300.0),
    "pts_per_poss": ("pts", "poss", 600.0),
    "pace": ("both_poss", "seconds", 3600.0),
}

#: Half-life of a trait's evidence, in days. A style changes slowly; a season
#: keeps last year's profile relevant at an opener without letting it dominate.
TRAIT_HALF_LIFE_DAYS = 365.0

#: Neighbours behind the interaction. The probe found the effect stable from
#: 200 to 3,000 neighbours and the out-of-sample gain positive from 1,000.
NEIGHBOURS = 1000

#: No model is fitted on fewer past stint directions than this.
MIN_POOL = 20_000

#: No feature when the newest stint evidence is older than this. Longer than an
#: offseason (~130 days), so openers keep last season's styles; shorter than a
#: missing season, so a hole in the stint archive gives NaN instead of styles
#: from years earlier (measured: 2019-20 games were projected from 2018 stints).
MAX_EVIDENCE_GAP_DAYS = 180

STYLE_FEATURE_COLUMNS = (
    "LU_PROJ_FG3A_RATE_BEFORE_TEAM_HOME",
    "LU_PROJ_FG3A_RATE_BEFORE_TEAM_AWAY",
    "LU_MATCHUP_FG3A_INTERACTION_BEFORE",
    "LU_ABSENCE_SHIFT_FG3A_RATE_BEFORE",
)

_COUNTS = (
    "pts",
    "fga",
    "fg3a",
    "fta",
    "oreb",
    "reb_chances",
    "tov",
    "poss",
    "both_poss",
    "seconds",
)


def stint_directions(stints: pd.DataFrame) -> pd.DataFrame:
    """Two rows per stint, one per team on offense, with that offense's counts."""
    frame = stints.loc[stints["seconds"] > 0]
    both = frame["home_poss"].clip(lower=0) + frame["away_poss"].clip(lower=0)
    parts = []
    for off, dfn in (("home", "away"), ("away", "home")):
        part = pd.DataFrame(
            {
                "game_id": frame["game_id"].astype(str).str.zfill(10),
                "game_date": pd.to_datetime(frame["game_date"]).dt.normalize(),
                "o_lineup": frame[f"{off}_lineup"].map(
                    lambda ps: tuple(str(p) for p in ps)
                ),
                "d_lineup": frame[f"{dfn}_lineup"].map(
                    lambda ps: tuple(str(p) for p in ps)
                ),
                "pts": frame[f"{off}_pts"],
                "fga": frame[f"{off}_fga"],
                "fg3a": frame[f"{off}_fg3a"],
                "fta": frame[f"{off}_fta"],
                "oreb": frame[f"{off}_oreb"],
                "reb_chances": frame[f"{off}_oreb"] + frame[f"{dfn}_dreb"],
                "tov": frame[f"{off}_tov"],
                "poss": frame[f"{off}_poss"].clip(lower=0),
                "both_poss": both,
                "seconds": frame["seconds"],
            }
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(
        "game_date", kind="mergesort"
    )


class StyleTraits:
    """Decayed per-player on-court counts, read as shrunk rate deviations.

    Evidence is stored scaled to an epoch (as in ``synergy.SynergyAccumulator``)
    so decay costs nothing per update: a count from date t, read on date T,
    weighs ``2 ** (-(T - t) / half_life)``, which is the stored value divided by
    ``factor(T)``. Reads use the date of the last :meth:`advance`.
    """

    def __init__(self, half_life_days: float = TRAIT_HALF_LIFE_DAYS) -> None:
        self.half_life = half_life_days
        self._epoch: pd.Timestamp | None = None
        self._read_factor = 1.0
        # (side, player) -> count -> epoch-scaled sum; side "o" or "d".
        self._player: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._league: dict[str, float] = defaultdict(float)

    def _factor(self, date: pd.Timestamp) -> float:
        if self._epoch is None:
            self._epoch = pd.Timestamp(date)
        return 2.0 ** ((pd.Timestamp(date) - self._epoch).days / self.half_life)

    def advance(self, date: pd.Timestamp) -> None:
        """Read as of ``date``: evidence is decayed to that day."""
        self._read_factor = self._factor(date)

    def league_rate(self, stat: str) -> float:
        num, den, _ = STYLE_STATS[stat]
        # The epoch scale cancels in a ratio.
        return (
            self._league[num] / self._league[den] if self._league[den] > 0 else np.nan
        )

    def trait(self, player: str, side: str, stat: str) -> float:
        """Shrunk rate minus the league rate; 0 for a player with no evidence."""
        num, den, k = STYLE_STATS[stat]
        mu = self.league_rate(stat)
        counts = self._player.get((side, player))
        if np.isnan(mu) or not counts:
            return 0.0
        n = counts[num] / self._read_factor
        d = counts[den] / self._read_factor
        return (n + k * mu) / (d + k) - mu

    def traits(self, players, side: str) -> pd.DataFrame:
        """``player x stat`` trait table, for vectorised lineup means."""
        return pd.DataFrame(
            {
                stat: [self.trait(p, side, stat) for p in players]
                for stat in STYLE_STATS
            },
            index=pd.Index(list(players), name="player"),
        )

    def vector(self, weights: dict[str, float], side: str) -> np.ndarray:
        """Weighted mean trait vector over ``{player: weight}``."""
        total = sum(weights.values())
        if total <= 0:
            return np.zeros(len(STYLE_STATS))
        table = self.traits(list(weights), side)
        w = np.array([weights[p] for p in table.index]) / total
        return w @ table.to_numpy()

    def update(self, day: pd.DataFrame, date: pd.Timestamp) -> None:
        """Add one date's stint directions (``stint_directions`` rows)."""
        factor = self._factor(date)
        sums = day[list(_COUNTS)].sum()
        for count in _COUNTS:
            self._league[count] += float(sums[count]) * factor
        for side, column in (("o", "o_lineup"), ("d", "d_lineup")):
            per_player = (
                day[[column, *_COUNTS]]
                .explode(column)
                .groupby(column)[list(_COUNTS)]
                .sum()
            )
            for player, values in zip(
                per_player.index, per_player.to_numpy(), strict=True
            ):
                store = self._player[(side, player)]
                for count, value in zip(_COUNTS, values, strict=True):
                    store[count] += float(value) * factor


def _lineup_vectors(day: pd.DataFrame, traits: StyleTraits) -> np.ndarray:
    """Joint vectors [offense five's offensive traits, defense five's defensive]."""
    blocks = []
    for column, side in (("o_lineup", "o"), ("d_lineup", "d")):
        players = pd.unique(np.concatenate(day[column].map(list).to_numpy()))
        table = traits.traits(players, side)
        exploded = day[[column]].reset_index(drop=True).explode(column)
        values = table.reindex(exploded[column].to_numpy()).to_numpy()
        means = (
            pd.DataFrame(values, index=exploded.index)
            .groupby(level=0)
            .mean()
            .to_numpy()
        )
        blocks.append(means)
    return np.hstack(blocks)


class _MonthlyModel:
    """Additive 3PA/FGA model plus a nearest-neighbour index over its residuals."""

    def __init__(self, vectors, rates, weights, neighbours):
        self.mean = vectors.mean(axis=0)
        self.scale = vectors.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        design = np.hstack([np.ones((len(vectors), 1)), vectors])
        root = np.sqrt(weights)
        self.coef = np.linalg.lstsq(design * root[:, None], rates * root, rcond=None)[0]
        self.residuals = rates - design @ self.coef
        self.weights = weights
        self.index = NearestNeighbors(n_neighbors=min(neighbours, len(vectors))).fit(
            (vectors - self.mean) / self.scale
        )

    def additive(self, vector: np.ndarray) -> float:
        return float(self.coef[0] + vector @ self.coef[1:])

    def interaction(self, vector: np.ndarray) -> float:
        """FGA-weighted mean residual of the most similar past pairs."""
        _, idx = self.index.kneighbors(((vector - self.mean) / self.scale)[None, :])
        w = self.weights[idx[0]]
        return float(w @ self.residuals[idx[0]] / w.sum()) if w.sum() > 0 else 0.0


def _rotation(players: list[PlayerNight], healthy: bool) -> dict[str, float]:
    sitting = (
        frozenset()
        if healthy
        else frozenset(p.player_id for p in players if p.p_out >= 0.5)
    )
    return allocate_minutes(players, sitting)


def build_style_matchup_features(
    stints: pd.DataFrame,
    games: pd.DataFrame,
    nights: dict[tuple[str, str], list[PlayerNight]],
    *,
    neighbours: int = NEIGHBOURS,
    half_life_days: float = TRAIT_HALF_LIFE_DAYS,
    min_pool: int = MIN_POOL,
    max_gap_days: int = MAX_EVIDENCE_GAP_DAYS,
) -> pd.DataFrame:
    """One row per game: ``GAME_ID`` and ``STYLE_FEATURE_COLUMNS``.

    ``games`` has ``GAME_ID``, ``GAME_DATE``, ``HOME_TEAM_ID``,
    ``AWAY_TEAM_ID``; ``nights`` is ``availability.build_player_nights``
    output for them. Games before the pool reaches ``min_pool`` directions,
    or without a roster, or more than ``max_gap_days`` after the newest stint,
    are absent from the result.
    """
    empty = pd.DataFrame(columns=["GAME_ID", *STYLE_FEATURE_COLUMNS])
    if stints.empty or games.empty:
        return empty
    directions = stint_directions(stints)
    targets = games.assign(
        GAME_ID=games["GAME_ID"].astype(str).str.zfill(10),
        GAME_DATE=pd.to_datetime(games["GAME_DATE"]).dt.normalize(),
        HOME_TEAM_ID=games["HOME_TEAM_ID"].astype(str),
        AWAY_TEAM_ID=games["AWAY_TEAM_ID"].astype(str),
    )
    stints_by_date = dict(tuple(directions.groupby("game_date", sort=True)))
    targets_by_date = dict(tuple(targets.groupby("GAME_DATE", sort=True)))
    traits = StyleTraits(half_life_days)
    pool_vectors, pool_rates, pool_weights = [], [], []
    model: _MonthlyModel | None = None
    fitted_month = None
    last_evidence: pd.Timestamp | None = None
    rows = []
    for date in sorted(set(stints_by_date) | set(targets_by_date)):
        traits.advance(date)
        month = date.to_period("M")
        if month != fitted_month and pool_vectors:
            vectors = np.vstack(pool_vectors)
            if len(vectors) >= min_pool:
                model = _MonthlyModel(
                    vectors,
                    np.concatenate(pool_rates),
                    np.concatenate(pool_weights),
                    neighbours,
                )
            fitted_month = month
        todays = targets_by_date.get(date)
        fresh = (
            last_evidence is not None and (date - last_evidence).days <= max_gap_days
        )
        if model is not None and todays is not None and fresh:
            for game in todays.itertuples(index=False):
                home = nights.get((game.GAME_ID, game.HOME_TEAM_ID))
                away = nights.get((game.GAME_ID, game.AWAY_TEAM_ID))
                if not home or not away:
                    continue
                vec = {
                    (team, healthy, side): traits.vector(
                        _rotation(roster, healthy), side
                    )
                    for team, roster in (("H", home), ("A", away))
                    for healthy in (False, True)
                    for side in ("o", "d")
                }
                rate, inter, shift = {}, [], 0.0
                for off, dfn in (("H", "A"), ("A", "H")):
                    joint = np.concatenate(
                        [vec[(off, False, "o")], vec[(dfn, False, "d")]]
                    )
                    healthy = np.concatenate(
                        [vec[(off, True, "o")], vec[(dfn, True, "d")]]
                    )
                    i = model.interaction(joint)
                    rate[off] = model.additive(joint) + i
                    inter.append(i)
                    shift += model.additive(joint) - model.additive(healthy)
                rows.append(
                    {
                        "GAME_ID": game.GAME_ID,
                        "LU_PROJ_FG3A_RATE_BEFORE_TEAM_HOME": rate["H"],
                        "LU_PROJ_FG3A_RATE_BEFORE_TEAM_AWAY": rate["A"],
                        "LU_MATCHUP_FG3A_INTERACTION_BEFORE": float(np.mean(inter)),
                        "LU_ABSENCE_SHIFT_FG3A_RATE_BEFORE": shift,
                    }
                )
        # Only now does today's evidence enter the traits and the pool.
        day = stints_by_date.get(date)
        if day is None:
            continue
        day = day.loc[day["fga"] > 0]
        if day.empty:
            traits.update(stints_by_date[date], date)
            continue
        pool_vectors.append(_lineup_vectors(day, traits))
        pool_rates.append((day["fg3a"] / day["fga"]).to_numpy(float))
        pool_weights.append(day["fga"].to_numpy(float))
        traits.update(stints_by_date[date], date)
        last_evidence = date
    return pd.DataFrame(rows, columns=empty.columns) if rows else empty
