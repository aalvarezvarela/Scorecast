"""Where 2_6's walk-forward RAPM fails for lack of sample (phase 3, step 1).

Diagnostic only: scores the stored 2_6 ratings on the next stints of one
development season, as 2_6 projects them, and breaks the error down by the
as-of exposure of the players on the floor (the sample their ratings were
fitted on, never the scored stint): the least-exposed of the ten, of the
offensive five and of the defensive five, how many of the ten are thin, and
the mean data share of their ratings ``e / (e + lambda)``. See
``nba_ou.data_processing.player_graph.rating_diagnostics``.

Errors are actual - predicted (efficiency in points per 100 possessions,
possession-weighted; pace in possessions per 48, seconds-weighted). The
reference (``*_zero`` columns) is 2_6's ridge with lambda -> infinity: no player
information, the intercept being the decayed historical mean; ``mse_gain`` is
what the ratings remove from its squared error. Standard errors are clustered
by game.

    python scripts/player_graph/rapm_exposure_diagnostic.py

Development season only (2018-19): the 2_7 evaluation starts in 2019-20. Reads
the stint store and ``data/lineup_ratings/player_ratings.parquet``; no
database. Writes the scored rows to ``data/player_graph/rating_diagnostics/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.lineups.rating_cv import score_stint_predictions
from nba_ou.data_processing.lineups.stint_store import read_stints
from nba_ou.data_processing.player_graph.rating_diagnostics import (
    bucket_report,
    decayed_exposure,
    decayed_league_means,
    error_summary,
    exposure_buckets,
    quantile_buckets,
    scored_rows,
)

LAST_DEVELOPMENT_SEASON = 2018
COLUMNS = ["rows", "weight_share", "bias", "bias_se", "bias_zero", "mae", "mae_zero",
           "mse_gain", "mse_gain_se", "rel_gain"]  # fmt: skip


def show(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    out = table[COLUMNS].astype(float)
    out["rows"] = out["rows"].map("{:,.0f}".format)
    out["weight_share"] = out["weight_share"].map("{:.1%}".format)
    out["rel_gain"] = out["rel_gain"].map("{:.2%}".format)
    print(out.round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--season", type=int, default=LAST_DEVELOPMENT_SEASON)
    parser.add_argument(
        "--ratings",
        type=Path,
        default=Path("data/lineup_ratings/player_ratings.parquet"),
    )
    args = parser.parse_args()
    if args.season > LAST_DEVELOPMENT_SEASON:
        raise SystemExit("Step 1 diagnoses development seasons only (<= 2018-19)")
    metadata = json.loads(args.ratings.with_suffix(".metadata.json").read_text())
    history = read_stints(
        list(range(metadata["first_season"], args.season + 1)),
        local_root=args.local_root,
    )
    stints = history.loc[history["season_year"].eq(args.season)]
    dates = sorted(pd.to_datetime(stints["game_date"]).dt.normalize().unique())
    ratings = pd.read_parquet(args.ratings)
    ratings = ratings.loc[ratings["as_of_date"].isin(dates)]
    print(
        f"2_6 ratings (solver v{metadata['solver_version']}, lambda_offdef "
        f"{metadata['lambda_offdef']:g}, lambda_pace {metadata['lambda_pace']:g}, "
        f"half-life {metadata['half_life_days']:g} d, stints from "
        f"{metadata['first_season']}); season {args.season}: {len(stints):,} stints "
        f"on {len(dates)} dates"
    )

    exposure = decayed_exposure(
        history, dates, half_life_days=metadata["half_life_days"]
    )
    check = ratings.merge(exposure, on=["as_of_date", "player_id"], how="outer")
    print(
        "exposure reproduces 2_6's poss_weight: max |diff| "
        f"{(check['poss_weight_x'] - check['poss_weight_y']).abs().max():.2e}, "
        f"rated without exposure {check['poss_weight_y'].isna().sum()}, "
        f"exposure without rating {check['poss_weight_x'].isna().sum()}"
    )
    means = decayed_league_means(
        history, dates, half_life_days=metadata["half_life_days"]
    )
    efficiency, pace = scored_rows(
        stints,
        ratings,
        exposure,
        lambda_offdef=metadata["lambda_offdef"],
        lambda_pace=metadata["lambda_pace"],
        league_means=means,
    )
    reference = score_stint_predictions(stints, ratings)
    print(
        f"MAE vs 2_6's score_stint_predictions: efficiency "
        f"{efficiency['actual'].sub(efficiency['predicted']).abs().mul(efficiency['weight']).sum() / efficiency['weight'].sum():.4f} "
        f"/ {reference['offdef_mae']:.4f}, pace "
        f"{pace['actual'].sub(pace['predicted']).abs().mul(pace['weight']).sum() / pace['weight'].sum():.4f} "
        f"/ {reference['pace_mae']:.4f}"
    )
    unrated = efficiency["min_exposure_10"].eq(0)
    print(
        f"efficiency rows {len(efficiency):,}; with an unrated player on the floor "
        f"{unrated.mean():.1%} of rows, {efficiency.loc[unrated, 'weight'].sum() / efficiency['weight'].sum():.1%} of possessions"
    )

    for name, rows in (("efficiency", efficiency), ("pace", pace)):
        for reference in ("baseline", "baseline_inf"):
            summary = error_summary(rows, reference)
            print(
                f"{name} vs {reference}: bias of the reference "
                f"{summary['bias_zero']:+.3f}, MSE gain of the ratings "
                f"{summary['mse_gain']:+.2f} ± {summary['mse_gain_se']:.2f} "
                f"({summary['rel_gain']:+.2%})"
            )
    eff = efficiency
    show(
        "EFFICIENCY by the least-exposed player of the OFFENSIVE five",
        bucket_report(eff, exposure_buckets(eff["min_exposure_off"])),
    )
    show(
        "EFFICIENCY by the least-exposed player of the DEFENSIVE five",
        bucket_report(eff, exposure_buckets(eff["min_exposure_def"])),
    )
    show(
        "EFFICIENCY by the least-exposed of the ten (fixed ranges)",
        bucket_report(eff, exposure_buckets(eff["min_exposure_10"])),
    )
    show(
        "EFFICIENCY by the least-exposed of the ten (possession deciles)",
        bucket_report(eff, quantile_buckets(eff["min_exposure_10"], eff["weight"])),
    )
    show(
        "EFFICIENCY by how many of the ten have exposure <= 300",
        bucket_report(eff, eff["n_below_300"].clip(upper=3)),
    )
    show(
        "EFFICIENCY by how many of the ten have exposure <= 1000 (data share <= 1/2)",
        bucket_report(eff, eff["n_below_1000"].clip(upper=4)),
    )
    show(
        "EFFICIENCY by the mean data share of the ten ratings (possession quintiles)",
        bucket_report(eff, quantile_buckets(eff["confidence_10"], eff["weight"], 5)),
    )
    show(
        "PACE by the least-exposed of the ten (fixed ranges)",
        bucket_report(pace, exposure_buckets(pace["min_exposure_10"])),
    )
    show(
        "PACE by the mean data share of the ten pace ratings (seconds quintiles)",
        bucket_report(
            pace, quantile_buckets(pace["confidence_pace_10"], pace["weight"], 5)
        ),
    )

    out = args.local_root / "player_graph" / "rating_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    stem = args.ratings.stem
    efficiency.to_parquet(out / f"{stem}_efficiency_season={args.season}.parquet")
    pace.to_parquet(out / f"{stem}_pace_season={args.season}.parquet")
    print(f"\nrows written to {out}")


if __name__ == "__main__":
    main()
