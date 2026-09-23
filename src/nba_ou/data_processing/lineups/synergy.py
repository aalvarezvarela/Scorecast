"""Phase D: what five players do together that their ratings do not explain.

The phase-C ratings are additive by construction, so they cannot express that a
particular five plays better or worse than the sum of its parts. This module
measures that leftover directly: for every stint, what the lineup actually
scored per 100 possessions minus what the ratings said it would, accumulated
per pair and per exact five.

**Why this is the remaining lever.** Everything else built so far divides a
fixed 240 minutes among players -- the team aggregate is
``sum_i (min_i / 48) * rating_i`` with the minutes pinned -- so reshuffling
minutes between similarly rated players barely moves the total, which is why
the phase-E minutes model bought almost nothing downstream. Synergy changes the
aggregate itself rather than how it is divided.

**Two scale traps this gets wrong if written naively.** A pair inherits the
residual of every lineup it appeared in, so its mean is a *lineup-level*
quantity; summing a five's ten pairs without dividing by ten inflates the term
tenfold. And the league's mean residual is not synergy at all -- it is the
calibration gap between estimated and real possessions, worth +2.75 points per
100 on this data, which ``project_totals`` already removes with
``total_offset``. Both are handled by reporting every value as a deviation from
the running league mean, divided by the pairs on the floor. Written without
them, the synergy term averaged +14 points per 100 and took the projection's
MAE from 14.5 to 24.7.

**Shrinkage.** A five with twenty possessions together has a residual that is
almost all noise, so every value is shrunk toward zero by ``w = n / (n + k)``
with ``n`` the possessions behind it. ``k`` is an explicit parameter here,
calibrated walk-forward, rather than the expanding empirical-Bayes fit the plan
suggests: the same machinery, one fewer moving part while the idea is being
tested.

**Temporal contract.** The accumulator is fed in date order and read between
dates, so a value read for date D contains only stints from dates < D. Older
evidence decays with a half-life, as the ratings do, so a lineup that has not
played together since last season counts for less than one that plays nightly.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import pandas as pd

#: Possessions at which a pair or five carries half its raw residual.
DEFAULT_SHRINKAGE_POSSESSIONS = 200.0

#: Half-life of the evidence, in days. Matches the ratings' default.
DEFAULT_HALF_LIFE_DAYS = 180.0

#: Pairs on the floor at any instant, for five players.
PAIRS_ON_COURT = 10


def stint_residuals(
    stints: pd.DataFrame,
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    """What each lineup scored per 100 possessions beyond its ratings.

    Two rows per stint, one per team on offense. The prediction uses the
    ratings **as of that stint's own date**, which the walk-forward cache
    already provides, so the residual is the error a same-day observer would
    have made rather than one computed with hindsight.
    """
    if stints.empty or ratings.empty:
        return pd.DataFrame(
            columns=[
                "game_date",
                "offense",
                "defense",
                "residual",
                "possessions",
                "seconds",
            ]
        )
    values = ratings.copy()
    values["as_of_date"] = pd.to_datetime(values["as_of_date"]).dt.normalize()
    values["player_id"] = values["player_id"].astype(str)
    offense = {
        (date, player): value
        for date, player, value in values[
            ["as_of_date", "player_id", "o_rating"]
        ].itertuples(index=False, name=None)
    }
    defense = {
        (date, player): value
        for date, player, value in values[
            ["as_of_date", "player_id", "d_rating"]
        ].itertuples(index=False, name=None)
    }
    intercepts = values.groupby("as_of_date")["league_ortg"].first()

    frame = stints.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.normalize()
    frame = frame.loc[frame["game_date"].isin(set(intercepts.index))]
    rows = []
    for stint in frame.to_dict("records"):
        date = stint["game_date"]
        league = float(intercepts.loc[date])
        home = tuple(str(player) for player in stint["home_lineup"])
        away = tuple(str(player) for player in stint["away_lineup"])
        for attack, defend, side in ((home, away, "home"), (away, home, "away")):
            possessions = float(stint[f"{side}_poss"])
            # Corrections in the play-by-play can leave a segment with zero or
            # negative estimated possessions; there is no rate to measure there.
            if possessions <= 0:
                continue
            predicted = league
            predicted += sum(offense.get((date, player), 0.0) for player in attack)
            predicted -= sum(defense.get((date, player), 0.0) for player in defend)
            observed = 100.0 * float(stint[f"{side}_pts"]) / possessions
            rows.append(
                {
                    "game_date": date,
                    "offense": attack,
                    "defense": defend,
                    "residual": observed - predicted,
                    "possessions": possessions,
                    "seconds": float(stint["seconds"]),
                }
            )
    return pd.DataFrame(rows)


class SynergyAccumulator:
    """Decayed, shrunk pair and five-man residuals, queried between dates.

    Values are held in units scaled to an epoch so decay costs nothing per
    update; see :meth:`_factor` for why that is exact.
    """

    def __init__(
        self,
        shrinkage_possessions: float = DEFAULT_SHRINKAGE_POSSESSIONS,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    ) -> None:
        if shrinkage_possessions <= 0 or half_life_days <= 0:
            raise ValueError("Shrinkage and half-life must both be positive")
        self.k = shrinkage_possessions
        self.half_life = half_life_days
        self._epoch: pd.Timestamp | None = None
        self._pair_weighted: dict[tuple[str, str], float] = defaultdict(float)
        self._pair_possessions: dict[tuple[str, str], float] = defaultdict(float)
        self._pair_seconds: dict[tuple[str, str], float] = defaultdict(float)
        self._five_weighted: dict[frozenset[str], float] = defaultdict(float)
        self._five_possessions: dict[frozenset[str], float] = defaultdict(float)
        self._team_seconds: dict[str, float] = defaultdict(float)
        self._league_weighted = 0.0
        self._league_possessions = 0.0

    def _factor(self, date: pd.Timestamp) -> float:
        """``2 ** (age / half_life)`` for ``date``, relative to the epoch.

        Evidence from ``t`` read at ``T`` must carry ``2 ** (-(T - t) / H)``,
        which factorises as ``factor(t) / factor(T)``. Storing ``value *
        factor(t)`` and reading ``stored / factor(T)`` is therefore exact, and
        costs nothing per update -- the alternative, rescaling every entry
        whenever the date advances, is what this avoids.
        """
        if self._epoch is None:
            self._epoch = date
        return 2.0 ** ((date - self._epoch).days / self.half_life)

    def league_mean(self) -> float:
        """The possession-weighted mean residual across every lineup seen."""
        if self._league_possessions <= 0:
            return 0.0
        return self._league_weighted / self._league_possessions

    @staticmethod
    def _pair(one: str, other: str) -> tuple[str, str]:
        return (one, other) if one <= other else (other, one)

    def observe(
        self,
        lineup: tuple[str, ...],
        residual: float,
        possessions: float,
        seconds: float,
        date: pd.Timestamp,
        team_id: str | None = None,
    ) -> None:
        """Record one lineup's offensive stint."""
        if possessions <= 0:
            return
        factor = self._factor(date)
        five = frozenset(lineup)
        # The league's own mean residual is not synergy: it is the calibration
        # gap between stint-estimated possessions and real ones, which
        # ``project_totals`` already corrects with ``total_offset``. Tracking it
        # here lets every value below be reported as a deviation from it, so
        # the synergy term is mean-zero and cannot double-count that gap.
        self._league_weighted += residual * possessions * factor
        self._league_possessions += possessions * factor
        self._five_weighted[five] += residual * possessions * factor
        self._five_possessions[five] += possessions * factor
        # A pair's residual is the residual of every lineup it appeared in, so
        # a genuinely good duo shows up across all the fives it is part of.
        for one, other in combinations(sorted(five), 2):
            key = self._pair(one, other)
            self._pair_weighted[key] += residual * possessions * factor
            self._pair_possessions[key] += possessions * factor
            self._pair_seconds[key] += seconds * factor
        if team_id is not None:
            self._team_seconds[team_id] += seconds * factor

    def pair_value(self, one: str, other: str, date: pd.Timestamp) -> float:
        """Shrunk residual for a pair, in points per 100 possessions."""
        key = self._pair(one, other)
        stored = self._pair_possessions.get(key, 0.0)
        if stored <= 0:
            return 0.0
        # The mean is scale-free, but the shrinkage weight is not: it must see
        # the possessions as they count *today*, or evidence would appear to
        # grow more trustworthy simply because time has passed.
        possessions = stored / self._factor(date)
        mean = self._pair_weighted[key] / stored
        # A pair inherits the residual of every lineup it played in, so its
        # mean is a lineup-level quantity. Dividing by the ten pairs on the
        # floor turns it into a per-pair contribution, which is what makes
        # summing a five's ten pairs recover that five's own deviation instead
        # of ten times it.
        deviation = (mean - self.league_mean()) / PAIRS_ON_COURT
        return deviation * (possessions / (possessions + self.k))

    def five_value(self, lineup: frozenset[str], date: pd.Timestamp) -> float:
        """Shrunk residual for an exact five."""
        stored = self._five_possessions.get(lineup, 0.0)
        if stored <= 0:
            return 0.0
        possessions = stored / self._factor(date)
        mean = self._five_weighted[lineup] / stored
        deviation = mean - self.league_mean()
        return deviation * (possessions / (possessions + self.k))

    def pair_seconds(self, one: str, other: str) -> float:
        """Seconds the pair has shared, in epoch units.

        Only ever compared against other pairs read at the same moment, where
        the common factor cancels, so it is left unscaled.
        """
        return self._pair_seconds.get(self._pair(one, other), 0.0)

    def five_possessions(self, lineup: frozenset[str], date: pd.Timestamp) -> float:
        """Decayed possessions behind a five, as they count on ``date``."""
        return self._five_possessions.get(lineup, 0.0) / self._factor(date)


def expected_shared_minutes(
    accumulator: SynergyAccumulator,
    available: list[str],
    team_minutes: float = 48.0,
) -> dict[tuple[str, str], float]:
    """How long each available pair is expected to share the floor tonight.

    A pair's past overlap is only informative **among the players who are
    actually going to play**, so the shares are renormalised over the available
    pairs. When a starter is out, the pairs he was part of vanish and everyone
    else's share rises, which is what makes the weighting react to news rather
    than describe an average night.

    The normalisation puts the total at ``PAIRS_ON_COURT * team_minutes``,
    because ten pairs are on the floor at every instant.
    """
    if len(available) < 2:
        return {}
    shares = {
        (one, other): accumulator.pair_seconds(one, other)
        for one, other in combinations(sorted(available), 2)
    }
    total = sum(shares.values())
    if total <= 0:
        # No shared history at all: spread the floor time evenly rather than
        # claiming nobody plays together.
        even = PAIRS_ON_COURT * team_minutes / len(shares)
        return dict.fromkeys(shares, even)
    scale = PAIRS_ON_COURT * team_minutes / total
    return {pair: value * scale for pair, value in shares.items()}


def team_synergy(
    accumulator: SynergyAccumulator,
    available: list[str],
    date: pd.Timestamp,
    team_minutes: float = 48.0,
) -> float:
    """Shared-minutes-weighted synergy for a team, in points per 100.

    ``sum_ij (m_ij / 48) * residual_ij``, which is section 7.1b's weighting and
    the same scale as the ratings: one five playing the whole game gives
    exactly the sum of its ten pair residuals, matching section 6.1's
    definition of a five's value.
    """
    minutes = expected_shared_minutes(accumulator, available, team_minutes)
    if not minutes:
        return 0.0
    return sum(
        shared / team_minutes * accumulator.pair_value(one, other, date)
        for (one, other), shared in minutes.items()
    )
