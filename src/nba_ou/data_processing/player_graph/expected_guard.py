"""Expected guarding rates as of a date: the v0 guard-edge weight provider.

For an attacker i and a defender j, from ``pair_game`` rows strictly before
the cutoff D (plan principle 7: a game's own observed shares are never used)::

    r_hat = (sum w * matchup_seconds + k * r_prior) / (sum w * cofloor_seconds + k)

* ``w = 0.5 ** (age_days / half_life_days)`` decays matchup and co-floor
  seconds alike, before they are summed.
* ``r_prior = p_i' R p_j``: ``p`` are the soft G/F/C positions as of D
  (``positions.py``) and ``R[x, y]`` is the decayed rate at which defenders of
  position y guarded attackers of position x in the window, with each history
  row spread over position pairs by ``p_i(x) * p_j(y)``.
* ``k`` is in co-floor seconds: the prior counts as ``k`` seconds of evidence.

``r_hat`` is stored **unnormalized**. A graph turns it into guarding shares
over the defenders on the floor, ``m_ij = r_hat_ij / sum_l r_hat_il``, which
discards magnitude, so the evidence columns are kept next to it:
``n_games``, ``hist_cofloor_seconds_raw``, ``hist_cofloor_seconds_decayed``,
``prior_weight = k / (hist_cofloor_seconds_decayed + k)`` and
``has_pair_history``. ``hist_matchup_seconds_decayed`` and ``r_prior`` let
``r_hat`` be recomputed for another ``k`` without rebuilding (:func:`shrink`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .as_of import AsOfView
from .positions import POSITIONS, PROBABILITY_COLUMNS, positions_as_of

PROVIDER = "expected_guard"
VERSION = "v0"

COLUMNS = (
    "off_player_id",
    "def_player_id",
    "r_hat",
    "r_prior",
    "n_games",
    "hist_cofloor_seconds_raw",
    "hist_cofloor_seconds_decayed",
    "hist_matchup_seconds_decayed",
    "prior_weight",
    "has_pair_history",
)


@dataclass(frozen=True)
class ExpectedGuardParams:
    #: Chosen on 2018-19 only (the 2_7 walk-forward starts in 2019-20): weighted
    #: TVD was flat at 0.2755-0.2756 for k = 200-300 and 0.2786 at 600
    #: (``scripts/player_graph/evaluate_expected_guard.py --seasons 2018``).
    k: float = 300.0
    half_life_days: float = 365.0
    window_days: int = 3 * 365


DEFAULT_PARAMS = ExpectedGuardParams()


def shrink(
    matchup_decayed: pd.Series | np.ndarray,
    cofloor_decayed: pd.Series | np.ndarray,
    r_prior: pd.Series | np.ndarray,
    k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """``(r_hat, prior_weight)`` for shrinkage strength ``k``."""
    matchup = np.asarray(matchup_decayed, float)
    cofloor = np.asarray(cofloor_decayed, float)
    prior = np.asarray(r_prior, float)
    return (matchup + k * prior) / (cofloor + k), k / (cofloor + k)


def _history(view: AsOfView, params: ExpectedGuardParams) -> pd.DataFrame:
    """Decayed history rows with co-floor time, in the window before the cutoff."""
    since = view.cutoff - pd.Timedelta(days=params.window_days)
    rows = view.pair_game(since=since)
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "off_player_id",
                "def_player_id",
                "cofloor",
                "cofloor_w",
                "matchup_w",
            ]
        ).astype({"cofloor": float, "cofloor_w": float, "matchup_w": float})
    rows = rows.loc[rows["cofloor_seconds"] > 0]
    age = (view.cutoff - rows["game_date"]).dt.days.to_numpy(float)
    weight = 0.5 ** (age / params.half_life_days)
    return pd.DataFrame(
        {
            "off_player_id": rows["off_player_id"].astype(str).to_numpy(),
            "def_player_id": rows["def_player_id"].astype(str).to_numpy(),
            "cofloor": rows["cofloor_seconds"].to_numpy(float),
            "cofloor_w": weight * rows["cofloor_seconds"].to_numpy(float),
            "matchup_w": weight * rows["matchup_seconds"].to_numpy(float),
        }
    )


def _position_matrix(players: pd.Series, positions: pd.DataFrame) -> np.ndarray:
    """``len(players) x 3`` soft positions; unknown players get the mean."""
    probs = positions[list(PROBABILITY_COLUMNS)]
    fallback = (
        probs.mean().to_numpy(float)
        if len(probs)
        else np.full(len(POSITIONS), 1.0 / len(POSITIONS))
    )
    matrix = probs.reindex(players.to_numpy()).to_numpy(float)
    missing = np.isnan(matrix).any(axis=1)
    matrix[missing] = fallback
    return matrix


def position_rates(history: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """``R``: decayed guarding rate, attacker position (rows) x defender (columns)."""
    attackers = _position_matrix(history["off_player_id"], positions)
    defenders = _position_matrix(history["def_player_id"], positions)
    matchup = (attackers * history["matchup_w"].to_numpy()[:, None]).T @ defenders
    cofloor = (attackers * history["cofloor_w"].to_numpy()[:, None]).T @ defenders
    with np.errstate(divide="ignore", invalid="ignore"):
        rates = matchup / cofloor
    rates = np.where(cofloor > 0, rates, np.nan)
    # No history at all (the first days of 2017-18): nothing is known, so the
    # prior stays NaN and a graph falls back to uniform shares.
    if not np.isnan(rates).all():
        rates = np.where(np.isnan(rates), np.nanmean(rates), rates)
    return pd.DataFrame(rates, index=list(POSITIONS), columns=list(POSITIONS))


def expected_guard(
    view: AsOfView,
    pairs: pd.DataFrame,
    *,
    params: ExpectedGuardParams = DEFAULT_PARAMS,
    positions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expected guarding rate of each requested pair as of ``view.cutoff``.

    ``pairs`` has ``off_player_id`` and ``def_player_id``; any other columns
    (for example ``game_id``) are carried through. ``positions`` defaults to
    :func:`positions_as_of` over the same window.
    """
    if positions is None:
        positions = positions_as_of(view, window_days=params.window_days)
    history = _history(view, params)
    rates = position_rates(history, positions)

    keys = ["off_player_id", "def_player_id"]
    out = pairs.copy()
    for key in keys:
        out[key] = out[key].astype(str)
    agg = history.groupby(keys).agg(
        n_games=("cofloor", "size"),
        hist_cofloor_seconds_raw=("cofloor", "sum"),
        hist_cofloor_seconds_decayed=("cofloor_w", "sum"),
        hist_matchup_seconds_decayed=("matchup_w", "sum"),
    )
    out = out.merge(agg, left_on=keys, right_index=True, how="left")
    evidence = [
        "n_games",
        "hist_cofloor_seconds_raw",
        "hist_cofloor_seconds_decayed",
        "hist_matchup_seconds_decayed",
    ]
    out[evidence] = out[evidence].fillna(0)
    out["n_games"] = out["n_games"].astype(int)

    attackers = _position_matrix(out["off_player_id"], positions)
    defenders = _position_matrix(out["def_player_id"], positions)
    out["r_prior"] = np.einsum("ix,xy,iy->i", attackers, rates.to_numpy(), defenders)
    out["r_hat"], out["prior_weight"] = shrink(
        out["hist_matchup_seconds_decayed"],
        out["hist_cofloor_seconds_decayed"],
        out["r_prior"],
        params.k,
    )
    out["has_pair_history"] = out["n_games"] > 0
    extra = [column for column in pairs.columns if column not in keys]
    return out[[*extra, *COLUMNS]]


def provider_metadata(params: ExpectedGuardParams) -> dict:
    """What a stored table records about the provider that built it."""
    return {"provider": PROVIDER, "version": VERSION, **asdict(params)}
