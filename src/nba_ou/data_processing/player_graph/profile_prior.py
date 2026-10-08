"""Profile prior for RAPM (plan phase 3, step 2): the training pairs of ``f``.

``f(profile, position) -> rating`` is fitted on (player, checkpoint) pairs:
the player's as-of node profile at the checkpoint and his walk-forward
zero-prior RAPM rating at the same date (both read strictly before it).
Checkpoints are the first game date of each month, so a pair exists only on
dates that have a node profile and every month gets one ``f``.

**Pseudo-targets.** The ratings come from 2_6's ridge toward 0 at the phase 3
control penalties (10,000 / 10,000), which keeps only a fraction of each
effect: under a **diagonal approximation** of the ridge, ``beta_hat = s * b``
with ``s = e / (e + lambda)`` the data share and ``b`` the unpenalized
estimate. The pseudo-target is ``t = beta_hat / s``, weighted by ``s``
(inverse variance under the same approximation: var(t) ~ tau^2 / s). RAPM's
columns are strongly correlated, so the real shrinkage of a coefficient is
neither independent nor exactly ``s``: ``t`` is a pseudo-target for ``f``, not
the player's true rating, and only out-of-sample checks decide whether ``f``
helps.

Exposure per rating: offense ``poss_weight`` (decayed offensive possessions,
2_6's column), defense ``def_poss_weight`` (decayed possessions defended),
pace ``seconds_weight`` (decayed seconds), each with its own penalty.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nba_ou.data_processing.lineups.player_ratings import walk_forward_player_ratings

from .node_profiles import PROFILE_COLUMNS
from .rating_diagnostics import decayed_exposure

#: The phase 3 zero-prior control (chosen on 2018-19 only, on the rebuilt
#: 2016-17 stints).
LAMBDA_OFFDEF = 3_000.0
LAMBDA_PACE = 10_000.0
HALF_LIFE_DAYS = 180.0

#: rating column -> (exposure column, penalty)
TARGETS = {
    "o_rating": ("poss_weight", LAMBDA_OFFDEF),
    "d_rating": ("def_poss_weight", LAMBDA_OFFDEF),
    "pace_rating": ("seconds_weight", LAMBDA_PACE),
}


def monthly_checkpoints(game_dates: Iterable) -> list[pd.Timestamp]:
    """The first game date of every month with games."""
    dates = pd.Series(pd.to_datetime(list(game_dates))).dt.normalize()
    return sorted(dates.groupby(dates.dt.to_period("M")).min())


def zero_prior_ratings(
    stints: pd.DataFrame,
    dates: Iterable,
    *,
    lambda_offdef: float = LAMBDA_OFFDEF,
    lambda_pace: float = LAMBDA_PACE,
    half_life_days: float = HALF_LIFE_DAYS,
) -> pd.DataFrame:
    """2_6's walk-forward ratings at ``dates`` with the given penalties, plus
    every exposure column of :func:`rating_diagnostics.decayed_exposure`."""
    dates = list(dates)
    ratings = walk_forward_player_ratings(
        stints,
        dates,
        lambda_offdef=lambda_offdef,
        lambda_pace=lambda_pace,
        half_life_days=half_life_days,
    )
    exposure = decayed_exposure(stints, dates, half_life_days=half_life_days)
    return ratings.drop(columns="poss_weight").merge(
        exposure, on=["as_of_date", "player_id"], how="left", validate="one_to_one"
    )


def prior_pairs(ratings: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    """One row per rated player and date with a profile: the profile columns,
    each rating with its exposure ``e_*``, data share ``s_*`` and
    pseudo-target ``t_*`` (``t = rating / s``).

    A rated player without a profile on the date is left out: his last game
    is beyond the profile window (2 seasons), so his rating is a decayed
    remnant (median 10 possessions in 2016-19) with no current profile.
    """
    profiles = profiles.assign(
        as_of_date=pd.to_datetime(profiles["as_of_date"]).dt.normalize(),
        player_id=profiles["player_id"].astype(str),
    )
    pairs = ratings.merge(
        profiles[["as_of_date", "player_id", *PROFILE_COLUMNS]],
        on=["as_of_date", "player_id"],
        how="inner",
        validate="one_to_one",
    )
    for rating, (exposure, penalty) in TARGETS.items():
        name = rating.removesuffix("_rating")
        e = pairs[exposure].fillna(0.0).to_numpy(float)
        s = e / (e + penalty)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(s > 0, pairs[rating].to_numpy(float) / s, np.nan)
        pairs[f"e_{name}"] = e
        pairs[f"s_{name}"] = s
        pairs[f"t_{name}"] = t
    return pairs


# --------------------------------------------------------------------------
# f: weighted linear ridge per rating (v0)
# --------------------------------------------------------------------------

#: Pairs whose data share is below this are left out of ``f`` (their target is
#: amplified more than 50x; 13% of the pairs, 0.4% of the weight).
S_MIN = 0.02

#: Node-profile features of ``f`` (the rating's own exposure is only a weight).
FEATURES = (
    "pts_per36",
    "fga_per36",
    "fg3a_per36",
    "fta_per36",
    "oreb_per36",
    "dreb_per36",
    "ast_per36",
    "tov_per36",
    "stl_per36",
    "blk_per36",
    "pf_per36",
    "ts_pct",
    "fg3_pct",
    "usage",
    "recent_minutes",
    "start_share",
    "p_G",
    "p_F",
    "p_C",
    "log_games_in_data",
    "prior_weight",
    "has_box_history",
)
RATINGS = ("o", "d", "pace")


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """``FEATURES`` from node-profile columns."""
    out = frame.assign(
        log_games_in_data=np.log1p(frame["games_in_data"].astype(float)),
        has_box_history=frame["has_box_history"].astype(float),
    )
    return out[list(FEATURES)].astype(float)


def training_rows(pairs: pd.DataFrame, rating: str) -> pd.DataFrame:
    """Pairs ``f`` learns ``rating`` from: data share at least ``S_MIN``."""
    return pairs.loc[pairs[f"s_{rating}"] >= S_MIN]


@dataclass(frozen=True)
class PriorModel:
    """One rating's ``f``: standardized features, weighted ridge."""

    rating: str
    alpha: float
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = (feature_matrix(frame).to_numpy() - self.mean) / self.scale
        return x @ self.coef + self.intercept


def fit_prior(pairs: pd.DataFrame, rating: str, alpha: float) -> PriorModel:
    """Weighted ridge of ``t_rating`` on the features, weights ``s_rating``.

    The scaler and the model see only ``pairs`` (the caller passes the
    training window).
    """
    from sklearn.linear_model import Ridge

    rows = training_rows(pairs, rating)
    x = feature_matrix(rows).to_numpy()
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    model = Ridge(alpha=alpha).fit(
        (x - mean) / scale, rows[f"t_{rating}"], sample_weight=rows[f"s_{rating}"]
    )
    return PriorModel(rating, alpha, mean, scale, model.coef_, float(model.intercept_))


def weighted_mse(t: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> float:
    return float(np.average((t - prediction) ** 2, weights=weight))


def select_alpha(
    pairs: pd.DataFrame,
    rating: str,
    alphas: Iterable[float],
    *,
    folds: int = 5,
) -> tuple[float, pd.Series]:
    """Grouped-by-player K-fold choice of the ridge strength on ``pairs``
    (the caller passes only pairs before the development season)."""
    from sklearn.model_selection import GroupKFold

    rows = training_rows(pairs, rating).reset_index(drop=True)
    scores = {}
    for alpha in alphas:
        errors, weights = [], []
        for train, test in GroupKFold(n_splits=folds).split(
            rows, groups=rows["player_id"]
        ):
            model = fit_prior(rows.iloc[train], rating, alpha)
            held = rows.iloc[test]
            errors.append(
                weighted_mse(
                    held[f"t_{rating}"].to_numpy(),
                    model.predict(held),
                    held[f"s_{rating}"].to_numpy(),
                )
                * held[f"s_{rating}"].sum()
            )
            weights.append(held[f"s_{rating}"].sum())
        scores[alpha] = sum(errors) / sum(weights)
    table = pd.Series(scores, name="cv_mse").rename_axis("alpha")
    return float(table.idxmin()), table


def debut_prior(pairs: pd.DataFrame, rating: str, debutants: set[str]) -> float:
    """Reliability-weighted mean pseudo-target of earlier debutants in their
    first 20 games: the prior of a player with no box score yet."""
    rows = training_rows(pairs, rating)
    rows = rows.loc[rows["player_id"].isin(debutants) & rows["games_in_data"].le(20)]
    if rows.empty:
        return 0.0
    return float(np.average(rows[f"t_{rating}"], weights=rows[f"s_{rating}"]))


# --------------------------------------------------------------------------
# beta0 for the solver (phase 3, step 3)
# --------------------------------------------------------------------------

#: Where a player's prior came from on a date.
SOURCE_F, SOURCE_DEBUT, SOURCE_ZERO = "f", "debut", "zero"


class PriorProvider:
    """``beta0`` per date for every player of the solver, from the latest
    monthly ``f`` (checkpoints up to the date, all read before it).

    * profile with box-score history on the date -> ``f(profile)``;
    * profile without box scores (a debut) -> the debut prior;
    * no profile on the date (last game beyond the 2-season window) -> 0,
      the exceptional fallback, counted in :attr:`sources`.

    Callable as the ``prior`` of ``rapm_sweep.sweep_predictions``.
    """

    def __init__(
        self,
        pairs: pd.DataFrame,
        profiles: pd.DataFrame,
        debutants: set[str],
        alphas: dict[str, float],
    ) -> None:
        self.pairs = pairs.assign(
            as_of_date=pd.to_datetime(pairs["as_of_date"]).dt.normalize()
        )
        self.checkpoints = sorted(self.pairs["as_of_date"].unique())
        profiles = profiles.assign(
            as_of_date=pd.to_datetime(profiles["as_of_date"]).dt.normalize(),
            player_id=profiles["player_id"].astype(str),
        )
        self.profiles = {
            day: part.set_index("player_id")
            for day, part in profiles.groupby("as_of_date")
        }
        self.debutants = debutants
        self.alphas = alphas
        self._models: dict[pd.Timestamp, tuple[dict, dict]] = {}
        #: day -> {player_id: source}
        self.sources: dict[pd.Timestamp, dict[str, str]] = {}

    def checkpoint(self, day: pd.Timestamp) -> pd.Timestamp:
        eligible = [c for c in self.checkpoints if c <= day]
        if not eligible:
            raise ValueError(f"No profile-prior checkpoint on or before {day.date()}")
        return eligible[-1]

    def models(self, checkpoint: pd.Timestamp) -> tuple[dict, dict]:
        """``({rating: f}, {rating: debut prior})`` fitted on checkpoints up to
        ``checkpoint``."""
        if checkpoint not in self._models:
            train = self.pairs.loc[self.pairs["as_of_date"] <= checkpoint]
            self._models[checkpoint] = (
                {r: fit_prior(train, r, self.alphas[r]) for r in RATINGS},
                {r: debut_prior(train, r, self.debutants) for r in RATINGS},
            )
        return self._models[checkpoint]

    def __call__(self, day, players: list[str]) -> dict[str, np.ndarray]:
        day = pd.Timestamp(day).normalize()
        fs, debut = self.models(self.checkpoint(day))
        profile = self.profiles.get(day, pd.DataFrame(columns=["has_box_history"]))
        present = profile.reindex(players)
        known = present["has_box_history"].notna().to_numpy()
        boxed = present["has_box_history"].eq(True).to_numpy()
        out = {}
        for rating in RATINGS:
            values = np.zeros(len(players))
            if boxed.any():
                values[boxed] = fs[rating].predict(present.loc[boxed])
            values[known & ~boxed] = debut[rating]
            out[rating] = values
        self.sources[day] = dict(
            zip(
                players,
                np.where(boxed, SOURCE_F, np.where(known, SOURCE_DEBUT, SOURCE_ZERO)),
                strict=True,
            )
        )
        return out
