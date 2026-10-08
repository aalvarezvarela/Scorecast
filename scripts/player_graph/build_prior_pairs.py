"""Build the (profile, rating) pairs of the profile prior and inspect the
pseudo-targets (phase 3, step 2). No model is fitted here.

Pairs: every rated player at the first game date of each month from
2016-12 to the end of the last season, with the zero-prior ratings at the
phase 3 control penalties and the as-of node profile
(``nba_ou.data_processing.player_graph.profile_prior``).

The inspection shows the pseudo-targets ``t = rating / s`` by data share
``s``. Under the diagonal approximation var(t) ~ tau^2 / s, so ``s * var(t)``
should be roughly flat across buckets; where it grows, the approximation
fails and amplified targets would carry noise into ``f``.

    python scripts/player_graph/build_prior_pairs.py

Development data only (through 2018-19). Reads the stint store and the node
profiles; no database. Writes ``data/player_graph/profile_prior/pairs.parquet``.

``--evaluation --last-season 2025`` builds the pairs of the frozen 2019-25
evaluation into ``pairs_evaluation.parquet`` (each pair still reads only its
own past); development files are never overwritten by it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.lineups.stint_store import read_stints
from nba_ou.data_processing.player_graph.profile_prior import (
    TARGETS,
    monthly_checkpoints,
    prior_pairs,
    zero_prior_ratings,
)

LAST_DEVELOPMENT_SEASON = 2018
FIRST_CHECKPOINT = pd.Timestamp("2016-12-01")
S_EDGES = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def weighted_var(values: np.ndarray, weights: np.ndarray) -> float:
    mean = np.average(values, weights=weights)
    return float(np.average((values - mean) ** 2, weights=weights))


def inspect(pairs: pd.DataFrame, name: str) -> None:
    s, t = pairs[f"s_{name}"], pairs[f"t_{name}"]
    rating = pairs[f"{name}_rating"]
    buckets = pd.cut(s, list(S_EDGES), right=False)
    rows = {}
    for bucket, part in pairs.groupby(buckets, observed=True):
        ss, tt = part[f"s_{name}"].to_numpy(), part[f"t_{name}"].to_numpy()
        rows[str(bucket)] = {
            "pairs": len(part),
            "weight": ss.sum() / s.sum(),
            "rating sd": part[f"{name}_rating"].std(),
            "t p1": np.percentile(tt, 1),
            "t p50": np.percentile(tt, 50),
            "t p99": np.percentile(tt, 99),
            "max |t|": np.abs(tt).max(),
            "t sd (w)": np.sqrt(weighted_var(tt, ss)),
            "s * var(t)": float(np.mean(ss) * weighted_var(tt, ss)),
        }
    table = pd.DataFrame(rows).T
    table["pairs"] = table["pairs"].map("{:,.0f}".format)
    table["weight"] = table["weight"].map("{:.1%}".format)
    print(f"\n{name}: pseudo-target t = rating / s by data share s")
    print(table.round(2).to_string())
    z = (t * np.sqrt(s)).abs()
    top = pairs.assign(z=z).nlargest(5, "z")
    print(f"  largest |t| * sqrt(s) (overall rating sd {rating.std():.2f}):")
    print(
        top[
            [
                "as_of_date",
                "player_id",
                f"e_{name}",
                f"s_{name}",
                f"{name}_rating",
                f"t_{name}",
                "games_in_data",
                "recent_minutes",
            ]
        ]  # fmt: skip
        .round(3)
        .to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--last-season", type=int, default=LAST_DEVELOPMENT_SEASON)
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="Evaluation only: pairs past 2018-19 for the frozen 2019-25 check",
    )
    args = parser.parse_args()
    if args.last_season > LAST_DEVELOPMENT_SEASON and not args.evaluation:
        raise SystemExit("The prior is developed on seasons up to 2018-19 only")
    if args.evaluation:
        print("EVALUATION ONLY: nothing here may be used to choose anything")
    seasons = list(range(2016, args.last_season + 1))
    started = time.time()
    stints = read_stints(seasons, local_root=args.local_root)
    checkpoints = [
        date
        for date in monthly_checkpoints(stints["game_date"])
        if date >= FIRST_CHECKPOINT
    ]
    ratings = zero_prior_ratings(stints, checkpoints)
    profiles = pd.concat(
        [
            pd.read_parquet(
                args.local_root
                / "player_graph"
                / "node_profiles"
                / f"season={s}.parquet"
            )
            for s in seasons
        ],
        ignore_index=True,
    )
    profiles = profiles.loc[profiles["as_of_date"].isin(checkpoints)]
    pairs = prior_pairs(ratings, profiles)
    print(
        f"{len(checkpoints)} checkpoints {checkpoints[0].date()} -> "
        f"{checkpoints[-1].date()}, {len(pairs):,} pairs, "
        f"{pairs['player_id'].nunique()} players ({time.time() - started:.0f}s); "
        f"{len(ratings) - len(pairs)} rated player-dates left out (no profile: "
        "last game beyond the 2-season profile window)"
    )
    for name in (rating.removesuffix("_rating") for rating in TARGETS):
        inspect(pairs, name)
    out = args.local_root / "player_graph" / "profile_prior"
    out.mkdir(parents=True, exist_ok=True)
    name = "pairs_evaluation.parquet" if args.evaluation else "pairs.parquet"
    pairs.to_parquet(out / name, index=False)
    print(f"\nwritten to {out / name}")


if __name__ == "__main__":
    main()
