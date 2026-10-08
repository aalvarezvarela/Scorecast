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

``--sweep`` retunes the penalty toward the prior on 2018-19 only (the best
penalty toward an informative prior need not be the best toward zero): the
zero-prior sweep's log grid, shared offense / defense plus pace, and a
secondary grid of separate offense / defense penalties; selection by
weighted squared error on all rows, compared with the zero-prior control.

    python scripts/player_graph/rapm_profile_prior.py
    python scripts/player_graph/rapm_profile_prior.py --sweep

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


GRID = (100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, INF)
PACE_GRID = (1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, INF)


def label(value: float) -> str:
    return "inf" if np.isinf(value) else f"{value:g}"


def load(args):
    """Season rows (with unrated / debut counts), history and the provider."""
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
    return history, efficiency, pace, provider, alphas


def same_penalties(args) -> None:
    root = args.local_root / "player_graph"
    history, efficiency, pace, provider, alphas = load(args)
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


def curve(rows, predictions, configs, reference, control) -> pd.DataFrame:
    """Per penalty toward the prior: gain vs no information (all and two
    low-sample slices), paired gain vs the zero-prior control, MAE."""
    masks = slices(rows)
    out = {}
    for name, config in configs.items():
        line = {}
        for slice_name in ("all", ">= 1 player <= 1000", ">= 1 player <= 300"):
            mask = masks[slice_name]
            gain, se = squared_error_gain(
                rows.loc[mask],
                predictions.loc[mask, config].to_numpy(),
                predictions.loc[mask, reference].to_numpy(),
            )
            line[f"{slice_name} vs inf"] = f"{gain:+7.2f} ± {se:5.2f}"
        gain, se = squared_error_gain(
            rows, predictions[config].to_numpy(), predictions[control].to_numpy()
        )
        line["vs zero-prior control"] = f"{gain:+6.2f} ± {se:4.2f}"
        line["_gain"] = squared_error_gain(
            rows, predictions[config].to_numpy(), predictions[reference].to_numpy()
        )[0]
        line["MAE"] = np.average(
            (rows["actual"] - predictions[config]).abs(), weights=rows["weight"]
        )
        out[name] = line
    return pd.DataFrame(out).T


def sweep(args) -> None:
    root = args.local_root / "player_graph"
    history, efficiency, pace, provider, alphas = load(args)
    control, no_info = EFFICIENCY["zero prior"], EFFICIENCY["no info"]
    diagonal = {label(lam): (lam, lam, True, True) for lam in GRID}
    secondary = [(lo, ld, True, True) for lo in GRID for ld in GRID if lo != ld]
    pace_control, pace_none = PACE["zero prior"], PACE["no info"]
    pace_configs = {label(lam): (lam, True) for lam in PACE_GRID}
    started = time.time()
    eff, pac = sweep_predictions(
        history,
        efficiency,
        pace,
        offdef_grid=[control, no_info, *diagonal.values(), *secondary],
        pace_grid=[pace_control, pace_none, *pace_configs.values()],
        prior=provider,
    )
    print(
        f"season {SEASON}: {len(efficiency):,} offensive stint-sides, "
        f"{len(pace):,} pace stints; {2 + len(diagonal) + len(secondary)} "
        f"offense/defense and {2 + len(pace_configs)} pace configurations in "
        f"{time.time() - started:.0f}s; f alphas {alphas}"
    )
    pd.set_option("display.width", 250, "display.max_columns", 30)

    table = curve(efficiency, eff, diagonal, no_info, control)
    print("\nEFFICIENCY, profile prior, shared offense/defense penalty (inf = the")
    print("profile alone): drop in squared error, (pts/100)^2")
    print(table.drop(columns="_gain").round(3).to_string())
    best = diagonal[table["_gain"].astype(float).idxmax()]
    pace_table = curve(pace, pac, pace_configs, pace_none, pace_control)
    print("\nPACE, profile prior: drop in squared error, (poss/48)^2")
    print(pace_table.drop(columns="_gain").round(3).to_string())
    best_pace = pace_configs[pace_table["_gain"].astype(float).idxmax()]

    print(
        f"\nbest profile-prior penalties: efficiency {label(best[0])}, pace "
        f"{label(best_pace[0])} (zero-prior control: {label(control[0])}, "
        f"{label(pace_control[0])})"
    )
    named = {"best profile": best, "zero prior": control, "no info": no_info}
    print(f"\nEFFICIENCY, best profile prior ({label(best[0])}) vs the control:")
    print(
        report(
            efficiency,
            eff,
            {**named, "profile prior": best, "profile only": diagonal["inf"]},
            [("best profile", "zero prior")],
        ).to_string()
    )
    pace_named = {
        "best profile": best_pace,
        "zero prior": pace_control,
        "no info": pace_none,
        "profile prior": best_pace,
        "profile only": pace_configs["inf"],
    }
    print(f"\nPACE, best profile prior ({label(best_pace[0])}) vs the control:")
    print(report(pace, pac, pace_named, [("best profile", "zero prior")]).to_string())

    names = [label(lam) for lam in GRID]
    grid = pd.DataFrame(index=pd.Index(names, name="off \\ def"), columns=names)
    reference = eff[no_info].to_numpy()
    for lo in GRID:
        for ld in GRID:
            grid.loc[label(lo), label(ld)] = squared_error_gain(
                efficiency, eff[(lo, ld, True, True)].to_numpy(), reference
            )[0]
    grid = grid.astype(float)
    print("\nSECONDARY: separate penalties toward the prior, drop in squared error")
    print("vs no information (rows: offense, columns: defense; inf = profile alone)")
    print(grid.round(2).to_string())
    pair = grid.stack().idxmax()
    pair_config = tuple(INF if v == "inf" else float(v) for v in pair) + (True, True)
    gain, se = squared_error_gain(
        efficiency, eff[pair_config].to_numpy(), eff[best].to_numpy()
    )
    print(
        f"best pair (off {pair[0]}, def {pair[1]}): {grid.stack().max():+.2f}; vs "
        f"best shared {label(best[0])}: {gain:+.2f} ± {se:.2f}"
    )

    (root / "profile_prior" / "lambda_sweep.json").write_text(
        json.dumps(
            {
                "season": SEASON,
                "selection": "weighted squared error on the season's stints, all rows",
                "lambda_offdef_profile_prior": best[0],
                "lambda_pace_profile_prior": best_pace[0],
                "zero_prior_control": {
                    "lambda_offdef": control[0],
                    "lambda_pace": pace_control[0],
                },
                "secondary_best_pair": {
                    "lambda_off": pair_config[0],
                    "lambda_def": pair_config[1],
                },
                "f_alphas": alphas,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--dates-from",
        type=Path,
        default=Path("data/player_graph/rating_diagnostics/clean_2_6_lambdas.parquet"),
    )
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()
    sweep(args) if args.sweep else same_penalties(args)


if __name__ == "__main__":
    main()
