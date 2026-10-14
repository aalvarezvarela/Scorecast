"""Matchup-adjusted defensive effect: target and signal (phase 4B, step 1).

The ``pair_game`` rows (attacker ``i`` x defender ``j`` x game) are labels and
history only. For a target with numerator ``y`` and exposure ``w`` (e.g.
points per partial possession) each row gets the attacker's **as-of norm**,
his decayed rate over earlier dates shrunk toward his position's league rate,
and a residual::

    r_ijg = y_ijg - w_ijg * norm_i(date)

A defender's effect is how far his attackers land from their norms,
``D_j = sum r / sum w`` (positive = they score more, i.e. worse defense).

How much of ``D_j`` is persistent is measured by **split-half covariance**:
each defender's games alternate between halves A and B, and across defenders
``cov(D_A, D_B)`` estimates the variance of the true effect (the noise of the
two halves is independent). With ``phi`` the noise variance per unit of
exposure, the empirical-Bayes shrinkage is ``k = phi / sigma^2`` pseudo-units
and the reliability of an estimate on ``W`` units is ``W / (W + k)``.

The same holds for any key: an interaction cell (defender x attacker type)
after removing the defender's own effect, a defending team, or a defender
relative to his teammates in the same game.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .pair_game import read_pair_game

HALF_LIFE_DAYS = 365.0  # as expected_guard v0

#: Game-level ``sum matchup_seconds / sum cofloor_seconds`` is 0.075-0.087 in
#: every season 2017-26; a few archive games carry every row's matched seconds
#: and partial possessions at exactly twice their value (about 0.16).
MAX_GAME_GUARD_RATE = 0.12

#: target -> (numerator, exposure); columns derived in ``with_target_columns``.
TARGETS = {
    "pts_per_pp": ("player_points", "partial_possessions"),
    "att_per_pp": ("attempts", "partial_possessions"),
    "fg3a_per_pp": ("matchup_fg3a", "partial_possessions"),
    "fta_per_pp": ("matchup_fta", "partial_possessions"),
    "tov_per_pp": ("matchup_turnovers", "partial_possessions"),
    "pts_per_att": ("player_points", "attempts"),
    "pts_per_min": ("player_points", "matchup_minutes"),
}


def doubled_exposure_games(pairs: pd.DataFrame) -> set[str]:
    """Games whose matched time per co-floor second is about twice the normal
    (``MAX_GAME_GUARD_RATE``): their exposure columns are doubled at the
    source while the counts are not."""
    games = pairs.groupby("game_id")[["matchup_seconds", "cofloor_seconds"]].sum()
    rate = games["matchup_seconds"] / games["cofloor_seconds"].where(
        games["cofloor_seconds"] > 0
    )
    return set(rate.index[rate > MAX_GAME_GUARD_RATE])


def read_matchup_history(
    season_years: Sequence[int], *, local_root: Path = Path("data")
) -> tuple[pd.DataFrame, set[str]]:
    """Phase 4B's ``pair_game`` history: the stored rows of ``season_years``
    without the games of ``doubled_exposure_games``, and those game ids.

    ``read_pair_game`` itself is unchanged: v0 (``expected_guard``) keeps the
    history it was built with.
    """
    pairs = read_pair_game(season_years, local_root=local_root)
    excluded = doubled_exposure_games(pairs)
    return pairs.loc[~pairs["game_id"].isin(excluded)].reset_index(drop=True), excluded


def with_target_columns(pairs: pd.DataFrame) -> pd.DataFrame:
    out = pairs.copy()
    out["attempts"] = out["matchup_fga"] + 0.44 * out["matchup_fta"]
    out["matchup_minutes"] = out["matchup_seconds"] / 60.0
    return out


def as_of_sums(
    frame: pd.DataFrame,
    keys: Sequence[str],
    columns: Sequence[str],
    *,
    date: str = "game_date",
    half_life_days: float = HALF_LIFE_DAYS,
) -> pd.DataFrame:
    """For every row, the decayed sums of ``columns`` over the rows of the same
    ``keys`` on **strictly earlier** dates (same-date rows never count),
    weighted ``0.5 ** (days / half_life_days)``. Indexed like ``frame``."""
    keys = list(keys)
    columns = list(columns)
    group = [*keys, date]
    daily = frame.groupby(group, observed=True, sort=False)[columns].sum().reset_index()
    daily = daily.sort_values(date, kind="stable")
    days = (daily[date] - daily[date].min()).dt.days.to_numpy(dtype=float)
    scale = 2.0 ** (days / half_life_days)
    scaled = daily[columns].mul(scale, axis=0)
    cumulative = (
        scaled.groupby([daily[k] for k in keys], observed=True).cumsum()
        if keys
        else scaled.cumsum()
    )
    daily[columns] = (cumulative - scaled).mul(1.0 / scale, axis=0).clip(lower=0.0)
    merged = frame[group].merge(daily, on=group, how="left")
    merged.index = frame.index
    return merged[columns]


def attacker_norms(
    pairs: pd.DataFrame,
    y: str,
    w: str,
    *,
    prior_weight: float,
    position: str = "position",
    half_life_days: float = HALF_LIFE_DAYS,
) -> pd.Series:
    """Each row's attacker rate as of its date: his decayed ``y / w`` on earlier
    dates, shrunk toward his position's decayed league rate with
    ``prior_weight`` units of ``w``."""
    own = as_of_sums(pairs, ["off_player_id"], [y, w], half_life_days=half_life_days)
    league = as_of_sums(pairs, [position], [y, w], half_life_days=half_life_days)
    overall = as_of_sums(pairs, [], [y, w], half_life_days=half_life_days)
    base = (overall[y] / overall[w]).where(overall[w] > 0)
    prior = (league[y] / league[w]).where(league[w] > 0).fillna(base)
    return (own[y] + prior_weight * prior) / (own[w] + prior_weight)


@dataclass(frozen=True)
class SplitHalf:
    """Split-half signal estimate for one key."""

    keys: int
    signal_var: float  # cov(D_A, D_B) across keys
    signal_se: float
    correlation: float
    noise_per_unit: float  # phi
    median_exposure: float  # per key, both halves together

    @property
    def shrinkage(self) -> float:
        """Empirical-Bayes pseudo-exposure ``phi / sigma^2`` (inf without signal)."""
        return self.noise_per_unit / self.signal_var if self.signal_var > 0 else np.inf

    def reliability(self, exposure: float) -> float:
        k = self.shrinkage
        return 0.0 if not np.isfinite(k) else exposure / (exposure + k)


def row_halves(frame: pd.DataFrame, by: Sequence[str]) -> pd.Series:
    """Half 0 or 1 for every row: each ``by`` group's games alternate between
    the halves in date order (all of a game's rows share its half)."""
    by = list(by)
    games = (
        frame[[*by, "game_id", "game_date"]]
        .drop_duplicates([*by, "game_id"])
        .sort_values(["game_date", "game_id"], kind="stable")
    )
    games["half"] = games.groupby(by, observed=True).cumcount() % 2
    merged = frame[[*by, "game_id"]].merge(
        games[[*by, "game_id", "half"]], on=[*by, "game_id"], how="left"
    )
    return pd.Series(merged["half"].to_numpy(), index=frame.index)


def split_half(
    frame: pd.DataFrame,
    keys: Sequence[str],
    *,
    min_exposure: float,
    center_by: Sequence[str] | None = None,
    r: str = "r",
    w: str = "w",
    half: str = "half",
) -> SplitHalf:
    """Signal variance of ``D_key = sum r / sum w`` by split-half covariance.

    ``frame[half]`` holds each row's half (``row_halves``); keys with less than
    ``min_exposure`` in either half are dropped. ``center_by`` subtracts the
    exposure-weighted mean rate of those groups in each half (e.g. the
    attacker type's mean for interaction cells).
    """
    keys = list(keys)
    halves = frame.groupby([*keys, half], observed=True)[[r, w]].sum().reset_index()
    halves["rate"] = halves[r] / halves[w].where(halves[w] > 0)
    if center_by:
        group = [*center_by, half]
        sums = (
            halves.assign(weighted=halves["rate"] * halves[w])
            .groupby(group, observed=True)[["weighted", w]]
            .transform("sum")
        )
        halves["rate"] -= sums["weighted"] / sums[w]
    wide = halves.pivot_table(index=keys, columns=half, values=["rate", w]).dropna()
    wide = wide.loc[(wide[(w, 0)] >= min_exposure) & (wide[(w, 1)] >= min_exposure)]
    a = wide[("rate", 0)].to_numpy()
    b = wide[("rate", 1)].to_numpy()
    enough = len(wide) > 2
    products = (a - a.mean()) * (b - b.mean()) if enough else np.array([])
    units = frame.groupby([*keys, "game_id"], observed=True)[[r, w]].sum()
    overall = units.groupby(level=list(range(len(keys))), observed=True).transform(
        "sum"
    )
    expected = units[w] * (overall[r] / overall[w].where(overall[w] > 0)).fillna(0.0)
    noise = float(((units[r] - expected) ** 2).sum() / units[w].sum())
    return SplitHalf(
        keys=len(wide),
        signal_var=float(products.mean()) if enough else np.nan,
        signal_se=(
            float(products.std(ddof=1) / np.sqrt(len(wide))) if enough else np.nan
        ),
        correlation=float(np.corrcoef(a, b)[0, 1]) if enough else np.nan,
        noise_per_unit=noise,
        median_exposure=(
            float((wide[(w, 0)] + wide[(w, 1)]).median()) if len(wide) else np.nan
        ),
    )


def shrunk_as_of(
    frame: pd.DataFrame,
    keys: Sequence[str],
    *,
    shrinkage: float,
    prior: pd.Series | float = 0.0,
    r: str = "r",
    w: str = "w",
    half_life_days: float = HALF_LIFE_DAYS,
) -> pd.Series:
    """Each row's ``D_key`` from earlier dates, shrunk toward ``prior`` with
    ``shrinkage`` pseudo-units of exposure (``prior`` when infinite)."""
    if not np.isfinite(shrinkage):
        return pd.Series(prior, index=frame.index, dtype=float)
    sums = as_of_sums(frame, keys, [r, w], half_life_days=half_life_days)
    return (sums[r] + shrinkage * prior) / (sums[w] + shrinkage)


def with_attacker_position(
    pairs: pd.DataFrame, *, local_root: Path = Path("data")
) -> pd.DataFrame:
    """``pairs`` with ``position``: the attacker's most likely soft position
    (G / F / C) in his as-of node profile on the game's date, ``?`` without
    one."""
    profiles = pd.concat(
        [
            pd.read_parquet(
                Path(local_root)
                / "player_graph"
                / "node_profiles"
                / f"season={s}.parquet",
                columns=["as_of_date", "player_id", "p_G", "p_F", "p_C"],
            )
            for s in sorted(pairs["season_year"].unique())
        ],
        ignore_index=True,
    ).rename(columns={"as_of_date": "game_date", "player_id": "off_player_id"})
    profiles["game_date"] = pd.to_datetime(profiles["game_date"]).astype(
        "datetime64[ns]"
    )
    out = pairs.assign(
        game_date=pd.to_datetime(pairs["game_date"]).astype("datetime64[ns]")
    ).merge(profiles, on=["game_date", "off_player_id"], how="left")
    probabilities = out[["p_G", "p_F", "p_C"]]
    out["position"] = (
        probabilities.fillna(-1.0)
        .idxmax(axis=1)
        .str[-1]
        .where(probabilities.notna().any(axis=1), "?")
    )
    return out.drop(columns=["p_G", "p_F", "p_C"])


def matchup_residuals(
    pairs: pd.DataFrame,
    y: str,
    w: str,
    attribution: str,
    *,
    development_season: int,
    burn_in: pd.Timestamp,
    min_games: float,
) -> tuple[pd.DataFrame, float]:
    """Rows with ``w`` / ``r`` / ``r_within`` for one target (``y`` per ``w``)
    and attribution (``shot`` as the NBA credits it, ``time`` by share of the
    attacker's guarding time), and the attacker prior weight.

    The prior weight is empirical Bayes on ``development_season``: split-half
    of the attackers' rates around their position's as-of rate (keys need
    ``min_games`` typical games of exposure in each half). Rows before
    ``burn_in`` (the attacker norms' burn-in) are dropped. ``pairs`` needs
    ``position`` (``with_attacker_position``).
    """
    frame = pairs.assign(w=pairs[w], y=pairs[y])
    position = as_of_sums(frame, ["position"], ["y", "w"])
    overall = as_of_sums(frame, [], ["y", "w"])
    base = (position["y"] / position["w"]).fillna(overall["y"] / overall["w"])
    frame["r"] = frame["y"] - frame["w"] * base
    dev = frame.loc[frame["season_year"].eq(development_season) & frame["r"].notna()]
    dev = dev.copy()
    dev["half"] = row_halves(dev, ["off_player_id"])
    typical = dev.groupby(["off_player_id", "game_id"])["w"].sum().median()
    attacker = split_half(dev, ["off_player_id"], min_exposure=min_games * typical)
    prior_weight = attacker.shrinkage
    frame["norm"] = attacker_norms(frame, "y", "w", prior_weight=prior_weight)
    frame["r"] = frame["y"] - frame["w"] * frame["norm"]
    frame = frame.loc[frame["norm"].notna() & (frame["game_date"] >= burn_in)].copy()
    if attribution == "time":
        basis = "matchup_seconds" if w == "matchup_minutes" else "partial_possessions"
        attacker_game = frame.groupby(["game_id", "off_player_id"])
        share = frame[basis] / attacker_game[basis].transform("sum")
        frame["r"] = attacker_game["r"].transform("sum") * share
        frame["w"] = attacker_game["w"].transform("sum") * share
    team_game = frame.groupby(["game_id", "def_team_id"])[["r", "w"]].transform("sum")
    frame["r_within"] = frame["r"] - frame["w"] * team_game["r"] / team_game["w"]
    return frame, prior_weight


def team_and_within_shrinkage(
    frame: pd.DataFrame, *, development_season: int, min_games: float
) -> tuple[SplitHalf, SplitHalf]:
    """Split-half signal of the defending team (``r``) and of the defender
    relative to his teammates (``r_within``) on ``development_season``: the
    shrinkage of step 1's simple estimator."""
    dev = frame.loc[frame["season_year"].eq(development_season)].copy()
    typical = dev.groupby(["def_player_id", "game_id"])["w"].sum().median()
    dev["half"] = row_halves(dev, ["def_player_id"])
    within = split_half(
        dev, ["def_player_id"], min_exposure=min_games * typical, r="r_within"
    )
    team_rows = dev.copy()
    team_rows["half"] = row_halves(team_rows, ["def_team_id"])
    team_typical = dev.groupby(["def_team_id", "game_id"])["w"].sum().median()
    team = split_half(team_rows, ["def_team_id"], min_exposure=min_games * team_typical)
    return team, within


def shrunk_on_dates(
    frame: pd.DataFrame,
    key: str,
    queries: pd.DataFrame,
    *,
    shrinkage: float,
    r: str = "r",
    w: str = "w",
    half_life_days: float = HALF_LIFE_DAYS,
) -> np.ndarray:
    """``shrunk_as_of`` for arbitrary ``queries`` (``game_date`` and ``key``):
    each query's decayed ``sum r / (sum w + shrinkage)`` over ``frame``'s rows
    of the same key on strictly earlier dates."""
    if not np.isfinite(shrinkage):
        return np.zeros(len(queries))
    probe = queries[["game_date", key]].assign(**{r: 0.0, w: 0.0})
    both = pd.concat([frame[["game_date", key, r, w]], probe], ignore_index=True)
    both["game_date"] = pd.to_datetime(both["game_date"]).astype("datetime64[ns]")
    sums = as_of_sums(both, [key], [r, w], half_life_days=half_life_days)
    tail = sums.iloc[len(frame) :]
    return (tail[r] / (tail[w] + shrinkage)).to_numpy()
