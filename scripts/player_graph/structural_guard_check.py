"""Score the structural guard rule against observed guarding shares (2017-18, 2018-19).

2016-17 has no matchup tracking, so its guard edges in the pretraining dataset
are the same-position rule ``r = sum_x p_i(x) p_j(x)`` on as-of positions
built from box-score starts (``pretraining_data``). It cannot be scored in
2016-17; this scores it where shares are observed, against the weighted TVD
of ``guard_benchmark`` (lower is better):

* constant rate (uniform over the floor, the fallback it replaces);
* structural rule (what 2016-17 gets);
* learned position prior (``expected_guard``'s ``r_prior``);
* ``expected_guard`` v0 (``r_hat``, k = 300, corrected history).

Reads the stored pretraining guard tables and ``pair_game`` without the
corrupt games; no database. Development seasons only (2_7 evaluation starts in
2019-20); the rule has no parameter, so nothing is tuned here.

    python scripts/player_graph/structural_guard_check.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.player_graph.guard_benchmark import (
    ATTACKER,
    benchmark_frame,
    implied_shares,
    share_tvd,
)
from nba_ou.data_processing.player_graph.matchup_defense import read_matchup_history
from nba_ou.data_processing.player_graph.pretraining_data import (
    ROOT,
    VERSION,
    season_path,
)

SEASONS = (2017, 2018)
ESTIMATORS = {
    "constant rate (uniform)": None,
    "structural rule (2016-17's edges)": "r_structural",
    "learned position prior": "r_prior",
    "expected_guard v0 (k=300)": "r_hat",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.local_root / ROOT / VERSION
    guard = pd.concat(
        [pd.read_parquet(season_path(root, "guard", s)) for s in SEASONS],
        ignore_index=True,
    )
    pairs, excluded = read_matchup_history(SEASONS, local_root=args.local_root)
    frame = benchmark_frame(pairs, guard)
    dates = guard.drop_duplicates("game_id").set_index("game_id")["game_date"]
    frame["month"] = pd.to_datetime(frame["game_id"].map(dates)).dt.to_period("M")
    weights = frame.groupby(ATTACKER)["matchup_seconds"].sum()
    attackers = frame.groupby(ATTACKER).agg(
        season=("season_year", "first"), month=("month", "first")
    )
    print(
        f"{len(attackers):,} attacker-games, {len(frame):,} pairs "
        f"({len(excluded)} corrupt games out of the history)\n"
    )
    rows, monthly = {}, {}
    for name, column in ESTIMATORS.items():
        rate = np.ones(len(frame)) if column is None else frame[column]
        error = share_tvd(frame, rate)
        _, fallback = implied_shares(frame, rate)
        w = weights.loc[error.index]
        row = {"fallback": float(fallback.loc[error.index].mean())}
        for season in SEASONS:
            mask = attackers.loc[error.index, "season"].eq(season)
            row[f"{season}-{(season + 1) % 100:02d}"] = float(
                np.average(error[mask], weights=w[mask])
            )
        rows[name] = row
        months = attackers.loc[error.index, "month"]
        monthly[name] = (
            pd.DataFrame({"e": error * w, "w": w, "m": months})
            .groupby("m")
            .sum()
            .pipe(lambda t: t["e"] / t["w"])
        )
    pd.set_option("display.width", 200)
    print("Weighted TVD, observed vs implied shares (lower is better):")
    print(pd.DataFrame(rows).T.round(4).to_string())
    print("\nBy month, 2017-18 (positions build up from 2016-17 box starts):")
    table = pd.DataFrame(monthly)
    print(
        table.loc[table.index.year * 100 + table.index.month < 201807]
        .round(4)
        .to_string()
    )


if __name__ == "__main__":
    main()
