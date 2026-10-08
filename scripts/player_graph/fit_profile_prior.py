"""Fit and check the profile prior ``f`` on its own (phase 3, step 2).

Nothing is connected to the RAPM solver here. On the (profile, rating) pairs
of ``build_prior_pairs.py``:

1. the ridge strength of each rating's ``f`` is chosen by player-grouped
   5-fold CV on pairs **before 2018-10-01** only, then frozen;
2. each 2018-19 checkpoint is predicted by an ``f`` fitted on the checkpoints
   strictly before it (out of time), against the pseudo-targets ``t``,
   weighted by ``s``, pairs with ``s >= S_MIN``:

   * weighted MSE of ``f`` vs zero (the ridge's current prior), the training
     mean (an average player) and a position-only model, overall and for thin
     players (exposure <= 1,000);
   * calibration: weighted slope of ``t`` on ``f(x)`` (equivalently of the
     rating on ``s * f(x)``), 1 if the prior has the right scale;
   * centering: weighted mean of ``t - f(x)``;
3. the debut prior (earlier debutants in their first 20 games) against the
   2018-19 debutants' pseudo-targets.

    python scripts/player_graph/fit_profile_prior.py

Development data only. Reads ``data/player_graph/profile_prior/pairs.parquet``
and the stint store (first appearances); writes the frozen alphas to
``data/player_graph/profile_prior/alphas.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.lineups.stint_store import read_stints
from nba_ou.data_processing.player_graph.profile_prior import (
    FEATURES,
    LAMBDA_OFFDEF,
    LAMBDA_PACE,
    RATINGS,
    S_MIN,
    PriorModel,
    debut_prior,
    fit_prior,
    select_alpha,
    training_rows,
    weighted_mse,
)

DEVELOPMENT_START = pd.Timestamp("2018-10-01")
FIRST_CHECKPOINT = pd.Timestamp("2016-12-01")
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)
POSITION = ("p_G", "p_F", "p_C")
THIN_EXPOSURE = 1_000.0


def position_only(train: pd.DataFrame, test: pd.DataFrame, rating: str) -> np.ndarray:
    rows = training_rows(train, rating)
    x = rows[list(POSITION)].to_numpy()
    w = rows[f"s_{rating}"].to_numpy()
    design = np.c_[x[:, :2], np.ones(len(x))]  # p_C is implied
    coef, *_ = np.linalg.lstsq(
        design * np.sqrt(w)[:, None], rows[f"t_{rating}"] * np.sqrt(w), rcond=None
    )
    return np.c_[test[list(POSITION[:2])].to_numpy(), np.ones(len(test))] @ coef


def summary(rows: pd.DataFrame, rating: str, predictions: dict) -> dict:
    t = rows[f"t_{rating}"].to_numpy()
    w = rows[f"s_{rating}"].to_numpy()
    out = {"pairs": len(rows), "weight": w.sum()}
    for name, prediction in predictions.items():
        out[f"mse {name}"] = weighted_mse(t, prediction, w)
    f = predictions["f"]
    centered = f - np.average(f, weights=w)
    slope = np.sum(w * centered * (t - np.average(t, weights=w))) / np.sum(
        w * centered**2
    )
    out["calibration slope"] = slope
    out["mean t - f"] = float(np.average(t - f, weights=w))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.local_root / "player_graph" / "profile_prior"
    pairs = pd.read_parquet(root / "pairs.parquet")
    pairs["as_of_date"] = pd.to_datetime(pairs["as_of_date"])
    if pairs["as_of_date"].max() >= pd.Timestamp("2019-07-01"):
        raise SystemExit("The prior is developed on seasons up to 2018-19 only")
    stints = read_stints([2016, 2017, 2018], local_root=args.local_root)
    first = {}
    for row in stints[["game_date", "home_lineup", "away_lineup"]].itertuples(
        index=False
    ):
        for player in (*row.home_lineup, *row.away_lineup):
            first.setdefault(str(int(player)), row.game_date)
    debutants = {p for p, d in first.items() if d >= FIRST_CHECKPOINT}

    before = pairs.loc[pairs["as_of_date"] < DEVELOPMENT_START]
    alphas = {}
    print(f"1. ridge strength, grouped 5-fold CV on {len(before):,} pairs before "
          f"{DEVELOPMENT_START.date()} (s >= {S_MIN})")  # fmt: skip
    for rating in RATINGS:
        alpha, table = select_alpha(before, rating, ALPHAS)
        alphas[rating] = alpha
        print(f"  {rating}: " + "  ".join(f"{a:g}: {v:.3f}" for a, v in table.items())
              + f"  -> {alpha:g}")  # fmt: skip

    print("\n2. out of time on 2018-19 checkpoints (each fitted on the ones before)")
    checkpoints = sorted(
        pairs.loc[pairs["as_of_date"] >= DEVELOPMENT_START, "as_of_date"].unique()
    )
    for rating in RATINGS:
        rows, preds = [], {"f": [], "zero": [], "mean": [], "position": []}
        for checkpoint in checkpoints:
            train = pairs.loc[pairs["as_of_date"] < checkpoint]
            test = training_rows(pairs.loc[pairs["as_of_date"] == checkpoint], rating)
            model = fit_prior(train, rating, alphas[rating])
            fitted = training_rows(train, rating)
            preds["f"].append(model.predict(test))
            preds["zero"].append(np.zeros(len(test)))
            preds["mean"].append(
                np.full(
                    len(test),
                    np.average(fitted[f"t_{rating}"], weights=fitted[f"s_{rating}"]),
                )
            )
            preds["position"].append(position_only(train, test, rating))
            rows.append(test)
        rows = pd.concat(rows, ignore_index=True)
        preds = {k: np.concatenate(v) for k, v in preds.items()}
        exposure = rows[f"e_{rating}"].to_numpy()
        thin = exposure <= (THIN_EXPOSURE if rating != "pace" else np.inf)
        table = {"all": summary(rows, rating, preds)}
        if rating != "pace":
            for name, mask in (("e <= 1000", thin), ("e > 1000", ~thin)):
                table[name] = summary(
                    rows.loc[mask], rating, {k: v[mask] for k, v in preds.items()}
                )
        out = pd.DataFrame(table).T
        for name in ("zero", "mean", "position"):
            out[f"f vs {name}"] = 1 - out["mse f"] / out[f"mse {name}"]
        print(f"\n  {rating} (alpha {alphas[rating]:g}):")
        print(out.round(3).to_string())

    print("\n3. standardized coefficients of f, fitted on all pairs before 2018-10-01")
    coefs = {}
    for rating in RATINGS:
        model: PriorModel = fit_prior(before, rating, alphas[rating])
        coefs[rating] = pd.Series(model.coef, index=list(FEATURES))
    print(pd.DataFrame(coefs).round(3).to_string())

    print("\n4. debut prior (debutants after 2016-12-01, first 20 games)")
    for rating in RATINGS:
        prior = debut_prior(before, rating, debutants)
        later = training_rows(
            pairs.loc[pairs["as_of_date"] >= DEVELOPMENT_START], rating
        )
        later = later.loc[
            later["player_id"].isin(debutants)
            & ~later["player_id"].isin(set(before["player_id"]))
            & later["games_in_data"].le(20)
        ]
        realized = (
            np.average(later[f"t_{rating}"], weights=later[f"s_{rating}"])
            if len(later)
            else np.nan
        )
        print(
            f"  {rating}: prior as of 2018-10-01 {prior:+.2f}; 2018-19 debutants "
            f"(first 20 games, {later['player_id'].nunique()} players): "
            f"{realized:+.2f}"
        )

    (root / "alphas.json").write_text(
        json.dumps(
            {
                "selected_on": f"pairs before {DEVELOPMENT_START.date()}",
                "cv": "player-grouped 5-fold, weighted MSE of t",
                "s_min": S_MIN,
                "lambda_offdef": LAMBDA_OFFDEF,
                "lambda_pace": LAMBDA_PACE,
                "alphas": alphas,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwritten {root / 'alphas.json'}")


if __name__ == "__main__":
    main()
