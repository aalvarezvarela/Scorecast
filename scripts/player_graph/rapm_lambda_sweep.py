"""Retune 2_6's ridge penalties on 2018-19 only (phase 3, step 2: the control).

The baseline chain phase 3 has to beat, each fitted walk-forward and scored on
the next stints exactly as in the step 1 diagnostic
(``rapm_exposure_diagnostic.py``):

    no player information (lambda -> inf)  ->  best zero-prior ridge
    ->  profile-prior ridge (phase 3)

Primary: 2_6's own structure, one penalty shared by offense and defense plus
one for pace, over a log grid far wider than 2_6's (10 to 1,000), including
infinity. Secondary: separate offense and defense penalties (a 2-D grid; an
infinite penalty drops that block), to see where the gain comes from.

Selection: weighted squared error on the season's stints (efficiency per
offensive side weighted by possessions, pace per stint weighted by seconds),
reported as the drop from lambda -> inf. Paired differences have standard
errors clustered by game. Slices use the as-of exposure (low sample = decayed
offensive possessions <= 1,000, where the data weigh less than the 2_6
penalty).

    python scripts/player_graph/rapm_lambda_sweep.py

Development season only (2018-19). Reads the stint store and the 2_6 rating
cache (for the reproduction check); no database. Writes the predictions and
the chosen penalties to ``data/player_graph/rating_diagnostics/``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.lineups.stint_store import read_stints
from nba_ou.data_processing.player_graph.rapm_sweep import (
    squared_error_gain,
    sweep_predictions,
)
from nba_ou.data_processing.player_graph.rating_diagnostics import (
    decayed_exposure,
    decayed_league_means,
    scored_rows,
)

LAST_DEVELOPMENT_SEASON = 2018
INF = float("inf")
GRID = (100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, INF)
PACE_GRID = (1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7, INF)


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
    }


def label(value: float) -> str:
    return "inf" if np.isinf(value) else f"{value:g}"


def gains(rows: pd.DataFrame, predictions: pd.DataFrame, reference) -> pd.DataFrame:
    """Drop in squared error from ``reference`` for every column, per slice."""
    out = {}
    worse = predictions[reference].to_numpy()
    for name, mask in slices(rows).items():
        part = rows.loc[mask]
        base = worse[mask.to_numpy()]
        mse_ref = float(
            np.average((part["actual"] - base) ** 2, weights=part["weight"])
        )
        for column in predictions.columns:
            gain, se = squared_error_gain(
                part, predictions.loc[mask, column].to_numpy(), base
            )
            out[(column, name)] = {"gain": gain, "se": se, "rel": gain / mse_ref}
    return pd.DataFrame(out).T


def weighted_mae(rows: pd.DataFrame, prediction: pd.Series) -> float:
    return float(
        np.average((rows["actual"] - prediction).abs(), weights=rows["weight"])
    )


def curve(rows, predictions, columns, names) -> pd.DataFrame:
    """Gain vs lambda -> inf per penalty, globally and on the two slices."""
    reference = columns[-1]
    table = gains(rows, predictions[columns], reference)
    out = pd.DataFrame(index=names)
    for name in ("all", ">= 1 player <= 1000", ">= 1 player <= 300"):
        part = table.xs(name, level=1)
        part.index = names
        out[f"{name}: gain"] = part["gain"]
        out[f"{name}: se"] = part["se"]
        out[f"{name}: rel"] = part["rel"].map("{:+.2%}".format)
    out["MAE"] = [weighted_mae(rows, predictions[c]) for c in columns]
    return out


def compare(rows, predictions, best, others: dict) -> pd.DataFrame:
    """Paired drop in squared error of ``best`` against each other column."""
    out = {}
    for name, mask in slices(rows).items():
        part = rows.loc[mask]
        better = predictions.loc[mask, best].to_numpy()
        row = {"share": part["weight"].sum() / rows["weight"].sum()}
        for other_name, other in others.items():
            gain, se = squared_error_gain(
                part, better, predictions.loc[mask, other].to_numpy()
            )
            row[f"vs {other_name}"] = f"{gain:+7.2f} ± {se:5.2f}"
        out[name] = row
    table = pd.DataFrame(out).T
    table["share"] = table["share"].map("{:.1%}".format)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--season", type=int, default=LAST_DEVELOPMENT_SEASON)
    parser.add_argument(
        "--ratings",
        type=Path,
        default=Path("data/lineup_ratings/player_ratings.parquet"),
    )
    parser.add_argument(
        "--diagonal-only",
        action="store_true",
        help="Skip the secondary grid of separate offense / defense penalties",
    )
    args = parser.parse_args()
    if args.season > LAST_DEVELOPMENT_SEASON:
        raise SystemExit("Penalties are chosen on development seasons only")
    metadata = json.loads(args.ratings.with_suffix(".metadata.json").read_text())
    half_life = metadata["half_life_days"]
    history = read_stints(
        list(range(metadata["first_season"], args.season + 1)),
        local_root=args.local_root,
    )
    stints = history.loc[history["season_year"].eq(args.season)]
    dates = sorted(pd.to_datetime(stints["game_date"]).dt.normalize().unique())
    stored = pd.read_parquet(args.ratings)
    stored = stored.loc[stored["as_of_date"].isin(dates)]
    efficiency, pace = scored_rows(
        stints,
        stored,
        decayed_exposure(history, dates, half_life_days=half_life),
        lambda_offdef=metadata["lambda_offdef"],
        lambda_pace=metadata["lambda_pace"],
        league_means=decayed_league_means(history, dates, half_life_days=half_life),
    )
    offdef_grid = [(lam, lam) for lam in GRID]
    if not args.diagonal_only:
        offdef_grid += [(lo, ld) for lo in GRID for ld in GRID if lo != ld]
    started = time.time()
    eff, pac = sweep_predictions(
        history,
        efficiency,
        pace,
        offdef_grid=offdef_grid,
        pace_grid=list(PACE_GRID),
        half_life_days=half_life,
    )
    print(
        f"season {args.season}: {len(efficiency):,} offensive stint-sides, "
        f"{len(pace):,} pace stints, {len(dates)} dates; {len(offdef_grid)} "
        f"offense/defense and {len(PACE_GRID)} pace penalties in "
        f"{time.time() - started:.0f}s"
    )
    current = (metadata["lambda_offdef"], metadata["lambda_offdef"])
    print(
        "reproduces the stored 2_6 ratings: efficiency max |diff| "
        f"{(eff[current] - efficiency['predicted']).abs().max():.1e}, pace "
        f"{(pac[metadata['lambda_pace']] - pace['predicted']).abs().max():.1e}; "
        "lambda -> inf equals the decayed mean: "
        f"{np.allclose(eff[(INF, INF)], efficiency['baseline_inf'])}, "
        f"{np.allclose(pac[INF], pace['baseline_inf'])}"
    )

    pd.set_option("display.width", 250, "display.max_columns", 30)
    diagonal = [(lam, lam) for lam in GRID]
    names = [label(lam) for lam in GRID]
    table = curve(efficiency, eff, diagonal, names)
    print("\nEFFICIENCY, shared offense/defense lambda (2_6 structure): drop in")
    print("squared error vs lambda -> inf, (pts/100)^2")
    print(table.round(2).to_string())
    best = diagonal[int(np.argmax(table["all: gain"].to_numpy()))]
    pace_table = curve(pace, pac, list(PACE_GRID), [label(v) for v in PACE_GRID])
    print("\nPACE: drop in squared error vs lambda -> inf, (poss/48)^2")
    print(pace_table.round(2).to_string())
    best_pace = PACE_GRID[int(np.argmax(pace_table["all: gain"].to_numpy()))]
    print(
        f"\nbest shared lambda_offdef {label(best[0])}, best lambda_pace "
        f"{label(best_pace)} (2_6: {label(current[0])}, "
        f"{label(metadata['lambda_pace'])})"
    )
    print(f"\nEFFICIENCY, best shared ({label(best[0])}): paired drop in squared error")
    print(
        compare(
            efficiency, eff, best, {"2_6 (1000)": current, "lambda inf": (INF, INF)}
        ).to_string()
    )
    print(f"\nPACE, best ({label(best_pace)}): paired drop in squared error")
    print(
        compare(
            pace,
            pac,
            best_pace,
            {f"2_6 ({label(metadata['lambda_pace'])})": metadata["lambda_pace"],
             "lambda inf": INF},  # fmt: skip
        ).to_string()
    )

    chosen = {
        "season": args.season,
        "selection": "weighted squared error on the season's stints, all rows",
        "lambda_offdef": best[0],
        "lambda_pace": best_pace,
        "half_life_days": half_life,
        "first_season": metadata["first_season"],
    }
    if not args.diagonal_only:
        grid = pd.DataFrame(
            index=pd.Index(names, name="lambda_off \\ lambda_def"), columns=names
        )
        reference = eff[(INF, INF)].to_numpy()
        for lo, ld in eff.columns:
            gain, _ = squared_error_gain(
                efficiency, eff[(lo, ld)].to_numpy(), reference
            )
            grid.loc[label(lo), label(ld)] = gain
        grid = grid.astype(float)
        print("\nSECONDARY: separate penalties, drop in squared error vs lambda -> inf")
        print("(rows: offense penalty, columns: defense penalty; inf drops the block)")
        print(grid.round(2).to_string())
        flat = grid.stack()
        best_pair = flat.idxmax()
        offense_only = grid["inf"].drop("inf").idxmax()
        defense_only = grid.loc["inf"].drop("inf").idxmax()
        print(
            f"best pair (off {best_pair[0]}, def {best_pair[1]}): "
            f"{flat.max():+.2f}; offense only (def inf), best off {offense_only}: "
            f"{grid.loc[offense_only, 'inf']:+.2f}; defense only (off inf), best def "
            f"{defense_only}: {grid.loc['inf', defense_only]:+.2f}; best shared "
            f"{label(best[0])}: {grid.loc[label(best[0]), label(best[1])]:+.2f}"
        )
        pair = tuple(INF if v == "inf" else float(v) for v in best_pair)
        others = {
            f"shared {label(best[0])}": best,
            f"offense only ({offense_only})": (
                float(offense_only) if offense_only != "inf" else INF,
                INF,
            ),
            "lambda inf": (INF, INF),
        }
        print(
            f"\nSECONDARY, best pair (off {best_pair[0]}, def {best_pair[1]}): paired"
        )
        print("drop in squared error")
        print(compare(efficiency, eff, pair, others).to_string())
        chosen["secondary_best_pair"] = {"lambda_off": pair[0], "lambda_def": pair[1]}

    out = args.local_root / "player_graph" / "rating_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    flat_columns = [f"off={label(lo)}|def={label(ld)}" for lo, ld in eff.columns]
    eff.set_axis(flat_columns, axis=1).to_parquet(
        out / f"lambda_sweep_efficiency_season={args.season}.parquet"
    )
    pac.set_axis([f"pace={label(v)}" for v in pac.columns], axis=1).to_parquet(
        out / f"lambda_sweep_pace_season={args.season}.parquet"
    )
    (out / f"lambda_sweep_season={args.season}.json").write_text(
        json.dumps(chosen, indent=2, default=str) + "\n"
    )
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
