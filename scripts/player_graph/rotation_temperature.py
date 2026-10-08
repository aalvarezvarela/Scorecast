"""Concentration calibration of the redistribution shares (phase 4A, 2018-19).

``s'(X -> Y) = s^tau / sum_Y s^tau`` over the same eligible teammates, for
``tau in TAUS``, on the v1 shares (the frozen baseline) and the v2 shares.
Only the final shares change: baselines, events and targets are those of the
frozen run (``rotation_diagnostic.py``, which saved them), so every
configuration is scored on the same rows.

Selection criterion, fixed before running: MSE of the minutes (``b`` + the
gains of that game's absences) of every player who played in 2018-19.
Decision rules, fixed before running: if v1 and v2 tie, v1; ``tau > 1`` only
for a clear gain, not a few tenths; no ``tau`` that buys a small global gain
by clearly worsening games with several absences. The diagnostics below do not
change the criterion.

    python scripts/player_graph/rotation_temperature.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TAUS = (1.0, 1.5, 2.0, 3.0)
RULES = ("s", "s_v2")
SEASON = 2018
STACK_MINUTES = 5.0  # a bag "lands" on a player when it gives him this much


def sharpened(absorption: pd.DataFrame, rule: str, tau: float) -> pd.Series:
    powered = absorption[rule].clip(lower=0.0) ** tau
    total = powered.groupby(
        [absorption["game_id"], absorption["team_id"], absorption["absent"]]
    ).transform("sum")
    return (powered / total.where(total > 0)).fillna(absorption[rule])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.local_root / "player_graph" / "rotation"
    players = pd.read_parquet(root / f"players_season={SEASON}.parquet")
    absorption = pd.read_parquet(root / f"absorption_season={SEASON}.parquet")
    everyone = players.loc[(players["minutes"] > 0) & players["b"].notna()]
    keys = ["game_id", "team_id", "player_id"]
    single = absorption.loc[absorption["n_absent"].eq(1) & absorption["vacated"].ge(20)]

    rows = {}
    for rule in RULES:
        for tau in TAUS:
            work = absorption.assign(share=sharpened(absorption, rule, tau))
            work["pred"] = work["vacated"] * work["share"]
            per_player = work.groupby(keys).agg(
                pred=("pred", "sum"),
                gain=("gain", "first"),
                played=("played", "first"),
                n_absent=("n_absent", "first"),
                bags=("pred", lambda p: int((p >= STACK_MINUTES).sum())),
            )
            joined = everyone.join(per_player["pred"], on=keys)
            error = joined["b"] + joined["pred"].fillna(0.0) - joined["minutes"]
            on = per_player.loc[per_player["played"]]
            line = {
                "minutes MSE": float((error**2).mean()),
                "minutes MAE": float(error.abs().mean()),
                "gain MSE, one absence": float(
                    ((on["pred"] - on["gain"])[on["n_absent"].eq(1)] ** 2).mean()
                ),
                "gain MSE, 2+ absences": float(
                    ((on["pred"] - on["gain"])[on["n_absent"].ge(2)] ** 2).mean()
                ),
            }
            # Concentration on single absences >= 20 min.
            shares_to_top, effective = [], []
            one = work.loc[work.index.isin(single.index)]
            for _, event in one.groupby(["game_id", "team_id"]):
                actual = event.loc[event["played"]]
                if actual.empty or actual["gain"].max() <= 0:
                    continue
                top = actual.loc[actual["gain"].idxmax(), "player_id"]
                share = event.set_index("player_id")["share"]
                share = share / share.sum()
                shares_to_top.append(share[top])
                q = share[share > 0]
                effective.append(np.exp(-(q * np.log(q)).sum()))
            line["share to real top (median)"] = float(np.median(shares_to_top))
            line["effective absorbers (median)"] = float(np.median(effective))
            # Stacking: several bags landing on one player in multi-absence games.
            multi = per_player.loc[per_player["n_absent"].ge(2)]
            games = multi.groupby(level=["game_id", "team_id"])
            line["multi-absence games with a stacked player"] = float(
                games["bags"].max().ge(2).mean()
            )
            biggest = multi.loc[
                multi.groupby(level=["game_id", "team_id"])["pred"].idxmax()
            ]
            biggest = biggest.loc[biggest["played"]]
            line["largest predicted gain, mean"] = float(biggest["pred"].mean())
            line["... its actual gain, mean"] = float(biggest["gain"].mean())
            rows[(rule, tau)] = line
    table = pd.DataFrame(rows).T
    table.index.names = ["rule", "tau"]
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(
        f"{len(everyone):,} player-games (minutes criterion); {single.groupby(['game_id', 'team_id']).ngroups} single absences >= 20 min"
    )
    print(table.round(3).to_string())
    print("\nRanking (top-1 / top-3) does not depend on tau: a power keeps the order.")


if __name__ == "__main__":
    main()
