"""Phase 3, step 3: RAPM shrunk toward the profile prior instead of zero.

Same walk-forward ridge, same penalties as the frozen zero-prior control
(3,000 / 10,000), only the target of the shrinkage changes:

    (X'X + lambda I) beta = X'y + lambda beta0

with ``beta0`` from ``profile_prior.PriorProvider`` (the latest monthly ``f``;
the debut prior for players without box scores; 0 only as a counted
fallback). Scored on the next stints exactly as steps 1 and 2a, per slice,
for:

* no information (lambda -> inf), the zero-prior control, the profile prior;
* the prior on offense only and on defense only (attribution);
* the profile alone (``f`` with no RAPM data: infinite penalty, prior).

Paired differences have standard errors clustered by game.

    python scripts/player_graph/rapm_profile_prior.py

Development season only (2018-19). Reads the stint store, the node profiles,
the prior pairs and alphas (``build_prior_pairs.py``, ``fit_profile_prior.py``)
and, for the dates, the clean 2_6-penalty cache of step 1.
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

SEASON = 2018
INF = float("inf")
FIRST_CHECKPOINT = pd.Timestamp("2016-12-01")
DEBUT_GAMES = 10
EFFICIENCY = {
    "no info": (INF, INF, False, False),
    "zero prior": (LAMBDA_OFFDEF, LAMBDA_OFFDEF, False, False),
    "profile prior": (LAMBDA_OFFDEF, LAMBDA_OFFDEF, True, True),
    "prior on offense": (LAMBDA_OFFDEF, LAMBDA_OFFDEF, True, False),
    "prior on defense": (LAMBDA_OFFDEF, LAMBDA_OFFDEF, False, True),
    "profile only": (INF, INF, True, True),
}
PACE = {
    "no info": (INF, False),
    "zero prior": (LAMBDA_PACE, False),
    "profile prior": (LAMBDA_PACE, True),
    "profile only": (INF, True),
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


def report(rows, predictions, names, paired) -> pd.DataFrame:
    """Gain vs no information per config, and paired gains of each
    ``(better, worse)`` in ``paired``, per slice."""
    out = {}
    for name, mask in slices(rows).items():
        part = rows.loc[mask]
        line = {
            "share": part["weight"].sum() / rows["weight"].sum(),
            "rows": len(part),
        }
        reference = predictions.loc[mask, names["no info"]].to_numpy()
        for label in ("zero prior", "profile prior", "profile only"):
            gain, se = squared_error_gain(
                part, predictions.loc[mask, names[label]].to_numpy(), reference
            )
            line[f"{label} vs inf"] = f"{gain:+7.2f} ± {se:5.2f}"
        for better, worse in paired:
            gain, se = squared_error_gain(
                part,
                predictions.loc[mask, names[better]].to_numpy(),
                predictions.loc[mask, names[worse]].to_numpy(),
            )
            line[f"{better} - {worse}"] = f"{gain:+6.2f} ± {se:4.2f}"
        out[name] = line
    table = pd.DataFrame(out).T
    table["share"] = table["share"].map("{:.1%}".format)
    table["rows"] = table["rows"].map("{:,}".format)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--dates-from",
        type=Path,
        default=Path("data/player_graph/rating_diagnostics/clean_2_6_lambdas.parquet"),
    )
    args = parser.parse_args()
    root = args.local_root / "player_graph"
    history = read_stints([2016, 2017, SEASON], local_root=args.local_root)
    stints = history.loc[history["season_year"].eq(SEASON)]
    dates = sorted(pd.to_datetime(stints["game_date"]).dt.normalize().unique())
    if max(dates) >= pd.Timestamp("2019-07-01"):
        raise SystemExit("Development season only")
    reference = pd.read_parquet(args.dates_from)
    reference = reference.loc[reference["as_of_date"].isin(dates)]
    exposure = decayed_exposure(history, dates)
    efficiency, pace = scored_rows(
        stints,
        reference,
        exposure,
        lambda_offdef=LAMBDA_OFFDEF,
        lambda_pace=LAMBDA_PACE,
        league_means=decayed_league_means(history, dates),
    )
    profiles = pd.concat(
        [
            pd.read_parquet(root / "node_profiles" / f"season={s}.parquet")
            for s in (2016, 2017, SEASON)
        ],
        ignore_index=True,
    )
    games = profiles.set_index(["as_of_date", "player_id"])["games_in_data"].to_dict()
    rated = exposure.set_index(["as_of_date", "player_id"])["poss_weight"].to_dict()

    def counts(rows: pd.DataFrame, players: pd.Series) -> None:
        unrated, debut = [], []
        for day, ten in zip(rows["game_date"], players, strict=True):
            unrated.append(sum(rated.get((day, p), 0.0) <= 0 for p in ten))
            debut.append(sum(games.get((day, p), 0) <= DEBUT_GAMES for p in ten))
        rows["n_unrated"], rows["n_debut"] = unrated, debut

    counts(efficiency, efficiency["offense_players"] + efficiency["defense_players"])
    counts(pace, pace["players"])

    first = {}
    for row in history[["game_date", "home_lineup", "away_lineup"]].itertuples(
        index=False
    ):
        for player in (*row.home_lineup, *row.away_lineup):
            first.setdefault(str(int(player)), row.game_date)
    debutants = {p for p, d in first.items() if d >= FIRST_CHECKPOINT}
    pairs = pd.read_parquet(root / "profile_prior" / "pairs.parquet")
    alphas = json.loads((root / "profile_prior" / "alphas.json").read_text())["alphas"]
    provider = PriorProvider(pairs, profiles, debutants, alphas)

    started = time.time()
    eff, pac = sweep_predictions(
        history,
        efficiency,
        pace,
        offdef_grid=list(EFFICIENCY.values()),
        pace_grid=list(PACE.values()),
        prior=provider,
    )
    print(
        f"season {SEASON}: {len(efficiency):,} offensive stint-sides, "
        f"{len(pace):,} pace stints; lambda {LAMBDA_OFFDEF:g} / {LAMBDA_PACE:g}; "
        f"alphas {alphas}; {time.time() - started:.0f}s"
    )
    design = Counter(s for day in provider.sources.values() for s in day.values())
    on_floor = Counter(
        provider.sources[day][p]
        for day, ten in zip(
            efficiency["game_date"],
            efficiency["offense_players"] + efficiency["defense_players"],
            strict=True,
        )
        for p in ten
    )
    print(
        "beta0 sources, solver players x dates: "
        + ", ".join(f"{k} {v / sum(design.values()):.2%}" for k, v in design.items())
        + "; players on the floor (scored rows): "
        + ", ".join(f"{k} {v:,}" for k, v in on_floor.items())
    )
    assert np.allclose(eff[EFFICIENCY["no info"]], efficiency["baseline_inf"])

    pd.set_option("display.width", 250, "display.max_columns", 30)
    names = dict(EFFICIENCY)
    print("\nEFFICIENCY: drop in squared error, (pts/100)^2")
    print(
        report(
            efficiency,
            eff,
            names,
            [
                ("profile prior", "zero prior"),
                ("prior on offense", "zero prior"),
                ("prior on defense", "zero prior"),
            ],
        ).to_string()
    )
    print("\nPACE: drop in squared error, (poss/48)^2")
    print(report(pace, pac, dict(PACE), [("profile prior", "zero prior")]).to_string())
    for label, rows, preds, configs in (
        ("efficiency", efficiency, eff, EFFICIENCY),
        ("pace", pace, pac, PACE),
    ):
        mae = {
            name: np.average(
                (rows["actual"] - preds[config]).abs(), weights=rows["weight"]
            )
            for name, config in configs.items()
        }
        print(f"\n{label} MAE: " + ", ".join(f"{k} {v:.3f}" for k, v in mae.items()))

    out = root / "rating_diagnostics"
    eff.set_axis(list(EFFICIENCY), axis=1).to_parquet(
        out / f"profile_prior_efficiency_season={SEASON}.parquet"
    )
    pac.set_axis(list(PACE), axis=1).to_parquet(
        out / f"profile_prior_pace_season={SEASON}.parquet"
    )


if __name__ == "__main__":
    main()
