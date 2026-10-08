"""Phase 3 out-of-sample evaluation, 2019-25: one walk-forward pass.

Everything was frozen on 2018-19 before this ran (see "Phase 3: frozen before
the 2019-25 evaluation" in ``docs/player_graph/frozen_decisions_and_future_checks.md``):
penalties 3,000 / 10,000 for both arms, ``f``'s ridge strengths from
``alphas.json``, its features, ``s_min``, the debut prior and the monthly
expanding refit. For a date D every rating, ``f_C``, profile and prior reads
only information before D; earlier evaluation seasons join the training as
the pass advances, nothing else changes.

Main comparison: clean zero-prior RAPM vs clean profile-prior RAPM. Lambda ->
infinity and 2_6 as stored are context. Reported pooled and per season,
offense / defense (prior on one block at a time) and pace, squared error
(primary) and MAE, and the slices of steps 1-3. Standard errors are clustered
by game.

    python scripts/player_graph/evaluate_profile_prior.py

Needs ``build_prior_pairs.py --evaluation --last-season 2025`` first. Reads the
stint store, node profiles and the 2_6 cache (dates, context); no database.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.lineups.stint_store import read_stints
from nba_ou.data_processing.player_graph.profile_prior import (
    LAMBDA_OFFDEF,
    LAMBDA_PACE,
    PriorProvider,
)
from nba_ou.data_processing.player_graph.rapm_sweep import (
    squared_error_gain,
    sweep_predictions,
)
from nba_ou.data_processing.player_graph.rating_diagnostics import (
    decayed_exposure,
    decayed_league_means,
    scored_rows,
)

SEASONS = range(2019, 2026)
HISTORY_FROM = 2016
INF = float("inf")
FIRST_CHECKPOINT = pd.Timestamp("2016-12-01")
DEBUT_GAMES = 10
EFFICIENCY = {
    "no info": (INF, INF, False, False),
    "zero prior": (LAMBDA_OFFDEF, LAMBDA_OFFDEF, False, False),
    "profile prior": (LAMBDA_OFFDEF, LAMBDA_OFFDEF, True, True),
    "prior on offense": (LAMBDA_OFFDEF, LAMBDA_OFFDEF, True, False),
    "prior on defense": (LAMBDA_OFFDEF, LAMBDA_OFFDEF, False, True),
}
PACE = {
    "no info": (INF, False),
    "zero prior": (LAMBDA_PACE, False),
    "profile prior": (LAMBDA_PACE, True),
}


def slices(rows: pd.DataFrame) -> dict[str, pd.Series]:
    n = rows["n_below_1000"]
    return {
        "all": pd.Series(True, index=rows.index),
        ">= 1 player <= 1000": n >= 1,
        ">= 1 player <= 300": rows["n_below_300"] >= 1,
        "0 players <= 1000": n == 0,
        "1 player <= 1000": n == 1,
        "2 players <= 1000": n == 2,
        "3 players <= 1000": n == 3,
        "4+ players <= 1000": n >= 4,
        ">= 1 unrated": rows["n_unrated"] >= 1,
        f">= 1 debut (<= {DEBUT_GAMES} games)": rows["n_debut"] >= 1,
    }


def gain(rows, better, worse) -> str:
    value, se = squared_error_gain(rows, better.to_numpy(), worse.to_numpy())
    return f"{value:+7.2f} ± {se:5.2f}"


def slice_table(rows, preds, context, paired) -> pd.DataFrame:
    out = {}
    for name, mask in slices(rows).items():
        part, p = rows.loc[mask], preds.loc[mask]
        line = {
            "share": f"{part['weight'].sum() / rows['weight'].sum():.1%}",
            "rows": f"{len(part):,}",
            "zero vs inf": gain(part, p["zero prior"], p["no info"]),
            "profile vs inf": gain(part, p["profile prior"], p["no info"]),
            "2_6 stored vs inf": gain(part, part[context], p["no info"]),
        }
        for better, worse in paired:
            line[f"{better} - {worse}"] = gain(part, p[better], p[worse])
        out[name] = line
    return pd.DataFrame(out).T


def mae(rows, prediction) -> float:
    return float(
        np.average((rows["actual"] - prediction).abs(), weights=rows["weight"])
    )


def season_table(rows, preds, paired) -> pd.DataFrame:
    out = {}
    for season, part in rows.groupby("season"):
        p = preds.loc[part.index]
        line = {"rows": f"{len(part):,}"}
        for better, worse in paired:
            line[f"{better} - {worse}"] = gain(part, p[better], p[worse])
        line["MAE zero"] = round(mae(part, p["zero prior"]), 3)
        line["MAE profile"] = round(mae(part, p["profile prior"]), 3)
        out[f"{season}-{(season + 1) % 100:02d}"] = line
    return pd.DataFrame(out).T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--ratings",
        type=Path,
        default=Path("data/lineup_ratings/player_ratings.parquet"),
        help="2_6 as stored: the evaluation dates and a context arm",
    )
    args = parser.parse_args()
    root = args.local_root / "player_graph"
    print("EVALUATION: frozen phase 3 configuration, one pass over 2019-25")
    history = read_stints(
        list(range(HISTORY_FROM, max(SEASONS) + 1)), local_root=args.local_root
    )
    stints = history.loc[history["season_year"].isin(list(SEASONS))]
    dates = sorted(pd.to_datetime(stints["game_date"]).dt.normalize().unique())
    stored = pd.read_parquet(args.ratings)
    stored = stored.loc[stored["as_of_date"].isin(dates)]
    exposure = decayed_exposure(history, dates)
    efficiency, pace = scored_rows(
        stints,
        stored,
        exposure,
        lambda_offdef=LAMBDA_OFFDEF,
        lambda_pace=LAMBDA_PACE,
        league_means=decayed_league_means(history, dates),
    )
    season_of = stints.drop_duplicates("game_id").set_index("game_id")["season_year"]
    for rows in (efficiency, pace):
        rows["season"] = rows["game_id"].map(season_of)
        rows["two_six"] = rows["predicted"]

    profiles = pd.concat(
        [
            pd.read_parquet(root / "node_profiles" / f"season={s}.parquet")
            for s in range(HISTORY_FROM, max(SEASONS) + 1)
        ],
        ignore_index=True,
    )
    games = profiles.set_index(["as_of_date", "player_id"])["games_in_data"].to_dict()
    rated = exposure.set_index(["as_of_date", "player_id"])["poss_weight"].to_dict()
    for rows, players in (
        (efficiency, efficiency["offense_players"] + efficiency["defense_players"]),
        (pace, pace["players"]),
    ):
        rows["n_unrated"] = [
            sum(rated.get((d, p), 0.0) <= 0 for p in ten)
            for d, ten in zip(rows["game_date"], players, strict=True)
        ]
        rows["n_debut"] = [
            sum(games.get((d, p), 0) <= DEBUT_GAMES for p in ten)
            for d, ten in zip(rows["game_date"], players, strict=True)
        ]

    first = {}
    for row in history[["game_date", "home_lineup", "away_lineup"]].itertuples(
        index=False
    ):
        for player in (*row.home_lineup, *row.away_lineup):
            first.setdefault(str(int(player)), row.game_date)
    debutants = {p for p, d in first.items() if d >= FIRST_CHECKPOINT}
    pairs = pd.read_parquet(root / "profile_prior" / "pairs_evaluation.parquet")
    frozen = json.loads((root / "profile_prior" / "alphas.json").read_text())
    provider = PriorProvider(pairs, profiles, debutants, frozen["alphas"])

    started = time.time()
    eff, pac = sweep_predictions(
        history,
        efficiency,
        pace,
        offdef_grid=list(EFFICIENCY.values()),
        pace_grid=list(PACE.values()),
        prior=provider,
    )
    eff.columns, pac.columns = list(EFFICIENCY), list(PACE)
    print(
        f"{len(efficiency):,} offensive stint-sides and {len(pace):,} pace stints on "
        f"{len(dates)} dates; penalties {LAMBDA_OFFDEF:g} / {LAMBDA_PACE:g}; f "
        f"alphas {frozen['alphas']} ({frozen['selected_on']}); "
        f"{time.time() - started:.0f}s"
    )
    on_floor = Counter(
        provider.sources[d][p]
        for d, ten in zip(
            efficiency["game_date"],
            efficiency["offense_players"] + efficiency["defense_players"],
            strict=True,
        )
        for p in ten
    )
    print(
        "beta0 of players on the floor: "
        + ", ".join(f"{k} {v:,}" for k, v in on_floor.items())
    )

    pd.set_option("display.width", 250, "display.max_columns", 30)
    eff_paired = [
        ("profile prior", "zero prior"),
        ("prior on offense", "zero prior"),
        ("prior on defense", "zero prior"),
    ]
    print("\nEFFICIENCY, pooled 2019-25: drop in squared error, (pts/100)^2")
    print(slice_table(efficiency, eff, "two_six", eff_paired).to_string())
    print("\nPACE, pooled 2019-25: drop in squared error, (poss/48)^2")
    print(
        slice_table(pace, pac, "two_six", [("profile prior", "zero prior")]).to_string()
    )
    print("\nMAE, pooled:")
    for label, rows, preds in (("efficiency", efficiency, eff), ("pace", pace, pac)):
        values = {name: mae(rows, preds[name]) for name in preds.columns}
        values["2_6 stored"] = mae(rows, rows["two_six"])
        print(f"  {label}: " + ", ".join(f"{k} {v:.3f}" for k, v in values.items()))
    print("\nEFFICIENCY by season:")
    print(season_table(efficiency, eff, eff_paired).to_string())
    print("\nPACE by season:")
    print(season_table(pace, pac, [("profile prior", "zero prior")]).to_string())

    out = root / "rating_diagnostics" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    eff.join(
        efficiency[["game_id", "season", "weight", "actual", "two_six"]]
    ).to_parquet(out / "profile_prior_efficiency_2019_2025.parquet")
    pac.join(pace[["game_id", "season", "weight", "actual", "two_six"]]).to_parquet(
        out / "profile_prior_pace_2019_2025.parquet"
    )
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
