"""Where walk-forward RAPM ratings fail on future stints (plan phase 3, step 1).

Scores any rating table with 2_6's columns (``as_of_date``, ``player_id``,
``o_rating``, ``d_rating``, ``pace_rating``, ``league_ortg``, ``league_pace``)
on future stints, exactly as 2_6 projects them
(``lineups.rating_cv.score_stint_predictions``):

* efficiency, per stint and offensive side, weighted by its possessions:
  ``pts/100 = league_ortg + sum off(offense) - sum def(defense)``;
* pace, per stint, weighted by its seconds:
  ``poss/48 = league_pace + sum pace(all ten)``.

A player without a rating on the date (no stint before it) counts 0, as in 2_6.
Each row also carries the as-of **exposure** of the players on the floor, the
sample their ratings were fitted on, never anything from the scored stint:

* ``poss_weight``: decayed offensive possessions, 2_6's own exposure column
  (the diagonal of the offensive block of the ridge's normal matrix);
* decayed on-court seconds, the same for the pace model;
* the rating's share of data, ``s = e / (e + lambda)``: 0 for an unseen player
  (the rating is the ridge's 0), 1/2 when the data weigh as much as the
  penalty.

Two "no ratings" references:

* ``baseline``: the fitted intercepts alone (``league_ortg``, ``league_pace``),
  i.e. the same fit with every rating set to 0;
* ``baseline_inf``: the lambda -> infinity limit of the same ridge, where every
  rating is 0 and the free intercept becomes the decayed, weighted historical
  mean (:func:`decayed_league_means`). This is the model with no player
  information at all; what the ratings add is the drop in squared error from
  it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from nba_ou.data_processing.lineups.player_ratings import _lineup, _poss

#: Fixed exposure ranges in decayed offensive possessions. With 2_6's
#: lambda_offdef = 1000, the data share s at the upper edges is 0, 0.09, 0.23,
#: 0.5, 0.75 and 0.83.
EXPOSURE_EDGES = (0.0, 1e-9, 100.0, 300.0, 1000.0, 3000.0, 5000.0, np.inf)
EXPOSURE_LABELS = (
    "0 (unrated)",
    "(0, 100]",
    "(100, 300]",
    "(300, 1k]",
    "(1k, 3k]",
    "(3k, 5k]",
    "> 5k",
)


def decayed_exposure(
    stints: pd.DataFrame,
    target_dates: Iterable,
    *,
    half_life_days: float = 180.0,
) -> pd.DataFrame:
    """Per ``(as_of_date, player_id)``: ``poss_weight`` (decayed offensive
    possessions) and ``seconds_weight`` (decayed seconds on court in stints
    that enter the pace fit), from stints strictly before each date.

    Mirrors the accumulation in ``walk_forward_player_ratings``.
    """
    frame = stints.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.normalize()
    rows = []
    for row in frame.to_dict("records"):
        home, away = _lineup(row["home_lineup"]), _lineup(row["away_lineup"])
        home_poss, away_poss = _poss(row, "home"), _poss(row, "away")
        for lineup, poss in ((home, home_poss), (away, away_poss)):
            if poss > 0:
                rows.extend((row["game_date"], pid, poss, 0.0) for pid in lineup)
        seconds = (row["end_ds"] - row["start_ds"]) / 10
        if (home_poss + away_poss) / 2 > 0 and seconds > 0:
            rows.extend((row["game_date"], pid, 0.0, seconds) for pid in home + away)
    daily = (
        pd.DataFrame(rows, columns=["game_date", "player_id", "poss", "seconds"])
        .groupby(["game_date", "player_id"], sort=True)
        .sum()
    )
    players = sorted(daily.index.get_level_values("player_id").unique())
    index = {pid: i for i, pid in enumerate(players)}
    by_day = {day: part.droplevel(0) for day, part in daily.groupby(level=0)}
    targets = {pd.Timestamp(date).normalize() for date in target_dates}
    poss = np.zeros(len(players))
    seconds = np.zeros(len(players))
    output = []
    previous = None
    for day in sorted(targets | set(by_day)):
        if previous is not None:
            factor = 0.5 ** ((day - previous).days / half_life_days)
            poss *= factor
            seconds *= factor
        previous = day
        if day in targets:
            seen = (poss > 0) | (seconds > 0)
            output.append(
                pd.DataFrame(
                    {
                        "as_of_date": day,
                        "player_id": np.asarray(players)[seen],
                        "poss_weight": poss[seen],
                        "seconds_weight": seconds[seen],
                    }
                )
            )
        if day in by_day:
            part = by_day[day]
            ids = part.index.map(index).to_numpy()
            poss[ids] += part["poss"].to_numpy()
            seconds[ids] += part["seconds"].to_numpy()
    columns = ["as_of_date", "player_id", "poss_weight", "seconds_weight"]
    return (
        pd.concat(output, ignore_index=True)
        if output
        else pd.DataFrame(columns=columns)
    )


def decayed_league_means(
    stints: pd.DataFrame,
    target_dates: Iterable,
    *,
    half_life_days: float = 180.0,
) -> pd.DataFrame:
    """Per ``as_of_date``: ``ortg_inf`` and ``pace_inf``, the decayed weighted
    means of the ridge targets (pts/100 weighted by possessions, poss/48 by
    seconds) over stints strictly before the date: 2_6's intercepts when lambda
    goes to infinity."""
    frame = stints.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.normalize()
    rows = []
    for row in frame.to_dict("records"):
        home_poss, away_poss = _poss(row, "home"), _poss(row, "away")
        for side, poss in (("home", home_poss), ("away", away_poss)):
            if poss > 0:
                rows.append(
                    (row["game_date"], 100 * float(row[f"{side}_pts"]), poss, 0.0, 0.0)
                )
        seconds = (row["end_ds"] - row["start_ds"]) / 10
        game_poss = (home_poss + away_poss) / 2
        if game_poss > 0 and seconds > 0:
            rows.append((row["game_date"], 0.0, 0.0, game_poss * 2880, seconds))
    # sums of w*y and w per day: pts/100 * poss = 100 * pts; poss/48 * sec = 2880 * poss
    daily = (
        pd.DataFrame(
            rows, columns=["game_date", "off_wy", "off_w", "pace_wy", "pace_w"]
        )
        .groupby("game_date", sort=True)
        .sum()
    )
    targets = sorted({pd.Timestamp(date).normalize() for date in target_dates})
    totals = np.zeros(4)
    output = []
    previous = None
    for day in sorted(set(targets) | set(daily.index)):
        if previous is not None:
            totals *= 0.5 ** ((day - previous).days / half_life_days)
        previous = day
        if day in targets:
            output.append(
                (
                    day,
                    totals[0] / totals[1] if totals[1] else np.nan,
                    totals[2] / totals[3] if totals[3] else np.nan,
                )
            )
        if day in daily.index:
            totals += daily.loc[day].to_numpy(float)
    return pd.DataFrame(output, columns=["as_of_date", "ortg_inf", "pace_inf"])


def scored_rows(
    stints: pd.DataFrame,
    ratings: pd.DataFrame,
    exposure: pd.DataFrame,
    *,
    lambda_offdef: float,
    lambda_pace: float,
    league_means: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(efficiency rows, pace rows)`` for the stints on rated dates.

    ``exposure`` comes from :func:`decayed_exposure` and ``league_means`` from
    :func:`decayed_league_means` on the same dates (``baseline_inf`` is NaN
    without it).
    """
    frame = stints.copy()
    frame["game_date"] = pd.to_datetime(frame["game_date"]).dt.normalize()
    ratings = ratings.assign(
        as_of_date=pd.to_datetime(ratings["as_of_date"]).dt.normalize(),
        player_id=ratings["player_id"].astype(str),
    )
    frame = frame.loc[frame["game_date"].isin(set(ratings["as_of_date"]))]
    keyed = ratings.set_index(["as_of_date", "player_id"])
    off = keyed["o_rating"].to_dict()
    dfn = keyed["d_rating"].to_dict()
    pace = keyed["pace_rating"].to_dict()
    intercepts = ratings.groupby("as_of_date")[["league_ortg", "league_pace"]].first()
    exposure = exposure.assign(
        as_of_date=pd.to_datetime(exposure["as_of_date"]).dt.normalize(),
        player_id=exposure["player_id"].astype(str),
    ).set_index(["as_of_date", "player_id"])
    e_poss = exposure["poss_weight"].to_dict()
    e_sec = exposure["seconds_weight"].to_dict()
    means = (
        league_means.assign(
            as_of_date=pd.to_datetime(league_means["as_of_date"]).dt.normalize()
        ).set_index("as_of_date")
        if league_means is not None
        else pd.DataFrame(columns=["ortg_inf", "pace_inf"], dtype=float)
    )

    def mean_on(date: pd.Timestamp, column: str) -> float:
        return float(means[column].get(date, np.nan))

    def share(values: Sequence[float], lam: float) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return values / (values + lam)

    efficiency, pace_rows = [], []
    for row in frame.to_dict("records"):
        date = row["game_date"]
        home, away = _lineup(row["home_lineup"]), _lineup(row["away_lineup"])
        exp = {pid: e_poss.get((date, pid), 0.0) for pid in home + away}
        ten = np.array([exp[pid] for pid in home + away])
        base = {
            "game_id": str(row["game_id"]),
            "game_date": date,
            "seg_idx": row.get("seg_idx"),
            "min_exposure_10": ten.min(),
            "confidence_10": share(ten, lambda_offdef).mean(),
            "n_below_300": int((ten <= 300).sum()),
            "n_below_1000": int((ten <= 1000).sum()),
        }
        for attacking, defending, side in ((home, away, "home"), (away, home, "away")):
            poss = _poss(row, side)
            if poss <= 0:
                continue
            league = float(intercepts.loc[date, "league_ortg"])
            o = np.array([exp[pid] for pid in attacking])
            d = np.array([exp[pid] for pid in defending])
            efficiency.append(
                base
                | {
                    "offense": side,
                    "weight": poss,
                    "actual": 100 * float(row[f"{side}_pts"]) / poss,
                    "predicted": league
                    + sum(off.get((date, pid), 0.0) for pid in attacking)
                    - sum(dfn.get((date, pid), 0.0) for pid in defending),
                    "baseline": league,
                    "baseline_inf": mean_on(date, "ortg_inf"),
                    "min_exposure_off": o.min(),
                    "min_exposure_def": d.min(),
                    "confidence_off": share(o, lambda_offdef).mean(),
                    "confidence_def": share(d, lambda_offdef).mean(),
                }
            )
        seconds = (float(row["end_ds"]) - float(row["start_ds"])) / 10
        game_poss = (_poss(row, "home") + _poss(row, "away")) / 2
        if seconds > 0 and game_poss > 0:
            sec = [e_sec.get((date, pid), 0.0) for pid in home + away]
            league = float(intercepts.loc[date, "league_pace"])
            pace_rows.append(
                base
                | {
                    "weight": seconds,
                    "actual": game_poss * 2880 / seconds,
                    "predicted": league
                    + sum(pace.get((date, pid), 0.0) for pid in home + away),
                    "baseline": league,
                    "baseline_inf": mean_on(date, "pace_inf"),
                    "confidence_pace_10": share(sec, lambda_pace).mean(),
                }
            )
    return pd.DataFrame(efficiency), pd.DataFrame(pace_rows)


def _clustered_mean(
    values: np.ndarray, weights: np.ndarray, clusters: np.ndarray
) -> tuple[float, float]:
    """Weighted mean and its standard error clustered by ``clusters``."""
    total = weights.sum()
    mean = float((weights * values).sum() / total)
    scores = pd.Series(weights * (values - mean)).groupby(clusters).sum().to_numpy()
    g = len(scores)
    se = float(np.sqrt((scores**2).sum() * g / max(g - 1, 1)) / total)
    return mean, se


def error_summary(
    rows: pd.DataFrame, reference: str = "baseline_inf"
) -> dict[str, float]:
    """Bias, MAE, MSE of the ratings and of a no-rating ``reference`` column
    (the ``*_zero`` statistics), and the ratings' drop in squared error."""
    w = rows["weight"].to_numpy(float)
    games = rows["game_id"].to_numpy()
    error = (rows["actual"] - rows["predicted"]).to_numpy(float)
    zero = (rows["actual"] - rows[reference]).to_numpy(float)
    bias, bias_se = _clustered_mean(error, w, games)
    gain, gain_se = _clustered_mean(zero**2 - error**2, w, games)
    mse_zero = float(np.average(zero**2, weights=w))
    return {
        "rows": len(rows),
        "weight_share": float(w.sum()),
        "bias": bias,
        "bias_se": bias_se,
        "bias_zero": float(np.average(zero, weights=w)),
        "mae": float(np.average(np.abs(error), weights=w)),
        "mae_zero": float(np.average(np.abs(zero), weights=w)),
        "mse": float(np.average(error**2, weights=w)),
        "mse_zero": mse_zero,
        "mse_gain": gain,
        "mse_gain_se": gain_se,
        "rel_gain": gain / mse_zero if mse_zero else np.nan,
    }


def bucket_report(
    rows: pd.DataFrame, buckets: pd.Series, reference: str = "baseline_inf"
) -> pd.DataFrame:
    """:func:`error_summary` per bucket plus an ``all`` row; ``weight_share``
    becomes the bucket's share of the total weight."""
    table = {
        str(key): error_summary(part, reference)
        for key, part in rows.groupby(buckets, observed=True, sort=True)
    }
    table["all"] = error_summary(rows, reference)
    out = pd.DataFrame(table).T
    out["weight_share"] = out["weight_share"] / out.loc["all", "weight_share"]
    return out


def exposure_buckets(values: pd.Series) -> pd.Series:
    """Fixed ranges of decayed offensive possessions (``EXPOSURE_EDGES``)."""
    return pd.cut(
        values,
        bins=list(EXPOSURE_EDGES),
        labels=list(EXPOSURE_LABELS),
        right=True,
        include_lowest=True,
    )


def quantile_buckets(values: pd.Series, weights: pd.Series, q: int = 10) -> pd.Series:
    """Weight-quantile buckets (equal weight each), labelled by their range."""
    order = np.argsort(values.to_numpy(), kind="mergesort")
    cumulative = np.cumsum(weights.to_numpy()[order]) / weights.sum()
    rank = np.empty(len(values), dtype=int)
    rank[order] = np.minimum((cumulative * q).astype(int), q - 1)
    ranks = pd.Series(rank, index=values.index)
    edges = values.groupby(ranks).agg(["min", "max"])
    fmt = "{:,.0f}" if values.abs().max() > 10 else "{:.2f}"
    labels = {
        k: f"q{k + 1} [{fmt.format(lo)}, {fmt.format(hi)}]"
        for k, (lo, hi) in edges.iterrows()
    }
    return pd.Series(
        pd.Categorical(
            ranks.map(labels),
            categories=[labels[k] for k in sorted(labels)],
            ordered=True,
        ),
        index=values.index,
    )
