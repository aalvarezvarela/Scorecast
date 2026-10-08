"""Participation A: does a medically available player enter the rotation?
(phase 4A, v1 logistic; development season 2018-19.)

Runs the frozen rotation engine (B, C) over 2016-17 -> 2018-19, labels the
roster rows (``participation.labelled_rows``: injury-report labels where the
team filed, the heuristic otherwise), fits the logistic ``q`` month by month
on earlier months only and scores 2018-19:

1. Brier (primary), log-loss and reliability against the recent
   participation rate and v0 (q = 1 for every available player), overall and
   by slice;
2. raw minutes before any reconciliation: per team-game, the sum of
   ``b + C`` and of ``q * (b + C)`` over tonight's available players against
   240, and how much ``q`` takes off.

The scenario is the engine's realized absences (as for B and C); in the
provider it will be the injury report's.

    python scripts/player_graph/participation_diagnostic.py
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.lineups.availability import player_out_probabilities
from nba_ou.data_processing.lineups.stint_store import read_stints
from nba_ou.data_processing.player_graph.participation import (
    brier,
    labelled_rows,
    log_loss,
    walk_forward_q,
)
from nba_ou.data_processing.player_graph.positions import PROBABILITY_COLUMNS
from nba_ou.data_processing.player_graph.rotation import team_game_minutes, walk_forward

SEASON = 2018
FIRST = 2016


def scores(rows: pd.DataFrame) -> dict[str, float]:
    out = {"rows": len(rows), "played": rows["played"].mean()}
    for name in ("logistic", "recent", "v0"):
        out[f"Brier {name}"] = brier(rows[f"q_{name}"], rows["played"])
    for name in ("logistic", "recent"):
        out[f"log-loss {name}"] = log_loss(rows[f"q_{name}"], rows["played"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    from nba_ou.data_processing.injury_status.report_state import (
        load_injury_report_state,
    )

    started = time.time()
    state = load_injury_report_state()
    statuses = state.statuses
    p_out = player_out_probabilities(statuses)
    covered = {(str(g).zfill(10), str(t)) for g, t in state.covered}
    listed: dict[tuple[str, str], set[str]] = {}
    for g, t, p in statuses[["game_id", "team_id", "player_id"]].itertuples(
        index=False
    ):
        listed.setdefault((str(g).zfill(10), str(t)), set()).add(str(p))
    minutes = team_game_minutes(
        read_stints(list(range(FIRST, SEASON + 1)), local_root=args.local_root)
    )
    profiles = pd.concat(
        [
            pd.read_parquet(
                args.local_root
                / "player_graph"
                / "node_profiles"
                / f"season={s}.parquet",
                columns=["as_of_date", "player_id", *PROBABILITY_COLUMNS],
            )
            for s in range(FIRST, SEASON + 1)
        ],
        ignore_index=True,
    )
    lookup = {(d, p): np.array(v) for d, p, *v in profiles.itertuples(index=False)}
    players, _ = walk_forward(
        minutes, lambda d, p: lookup.get((d, p), np.full(3, 1 / 3)), listed
    )
    print(f"engine in {time.time() - started:.0f}s")

    rows = labelled_rows(players, p_out, covered)
    available = players.loc[
        players["on_roster"] & ~players["absent_event"] & players["b"].notna()
    ]
    q_rows = walk_forward_q(rows, rows)
    q_available = walk_forward_q(rows, available)
    rows = rows.assign(
        q_logistic=q_rows, q_recent=rows["q"].fillna(0.0).clip(0, 1), q_v0=1.0
    )
    season = set(minutes.loc[minutes["season"].eq(SEASON), "game_id"])
    test = rows.loc[rows["game_id"].isin(season) & rows["q_logistic"].notna()]
    print(
        f"labelled rows: {len(rows):,} ({rows['availability_source'].value_counts().to_dict()}); "
        f"2018-19 scored: {len(test):,}"
    )

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\n1. CALIBRATION AND ACCURACY OF q (2018-19)")
    slices = {
        "all": test,
        "rotation (b >= 15)": test.loc[test["b"] >= 15],
        "bench (b < 15)": test.loc[test["b"] < 15],
        "no baseline": test.loc[test["b"].isna()],
        "nobody out": test.loc[test["n_absent"].eq(0)],
        ">= 1 out": test.loc[test["n_absent"].gt(0)],
        "returning (>= 5 games out)": test.loc[test["streak"] >= 5],
        "C gain >= 3 min": test.loc[test["gain_c"] >= 3],
        "rank 1-5": test.loc[test["rank"] <= 5],
        "rank 6-8": test.loc[test["rank"].between(6, 8)],
        "rank 9-10": test.loc[test["rank"].between(9, 10)],
        "rank 11+": test.loc[test["rank"] >= 11],
        "injury-report labels": test.loc[
            test["availability_source"].eq("injury_report")
        ],
        "heuristic labels": test.loc[test["availability_source"].eq("heuristic")],
    }
    print(
        pd.DataFrame({k: scores(v) for k, v in slices.items() if len(v)})
        .T.round(4)
        .to_string()
    )
    bins = pd.cut(
        test["q_logistic"],
        [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
        include_lowest=True,
    )
    reliability = test.groupby(bins, observed=True).agg(
        rows=("played", "size"),
        mean_q=("q_logistic", "mean"),
        observed=("played", "mean"),
    )
    print("\nreliability of the logistic q (2018-19):")
    print(reliability.round(3).to_string())
    for source in ("injury_report", "heuristic"):
        part = test.loc[test["availability_source"].eq(source)]
        rel = part.groupby(
            pd.cut(
                part["q_logistic"], [0, 0.25, 0.5, 0.75, 0.9, 1.0], include_lowest=True
            ),
            observed=True,
        ).agg(
            rows=("played", "size"),
            mean_q=("q_logistic", "mean"),
            observed=("played", "mean"),
        )
        print(f"  {source}:\n{rel.round(3).to_string()}")

    print(
        "\n2. RAW MINUTES BEFORE RECONCILIATION (2018-19 team-games, available players)"
    )
    avail = available.assign(q_logistic=q_available)
    avail = avail.loc[avail["game_id"].isin(season) & avail["q_logistic"].notna()]
    avail["bc"] = avail["b"] + avail["gain_c"]
    avail["abc"] = avail["q_logistic"] * avail["bc"]
    team = avail.groupby(["game_id", "team_id"]).agg(
        bc=("bc", "sum"),
        abc=("abc", "sum"),
        actual=("minutes", "sum"),
        players=("player_id", "size"),
    )
    team["q removes"] = team["bc"] - team["abc"]
    print(
        team[["bc", "abc", "actual", "q removes", "players"]]
        .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        .round(2)
        .to_string()
    )
    print(
        f"  scale 240 / sum needed: B+C {(240 / team['bc']).median():.3f} (median), "
        f"A+B+C {(240 / team['abc']).median():.3f}; actual minutes of these players "
        f"{team['actual'].mean():.1f} of 240 (the rest: players off the roster or absent)"
    )
    out = args.local_root / "player_graph" / "rotation"
    rows.to_parquet(out / f"participation_rows_season={SEASON}.parquet", index=False)
    avail.to_parquet(
        out / f"participation_available_season={SEASON}.parquet", index=False
    )


if __name__ == "__main__":
    main()
