"""Matchup-adjusted defensive rating vs stint RAPM on 2018-19 stints (phase 4B).

Rating books as of every 2018-19 date, all with the offense, pace and league
level of the clean RAPM book (2_6's penalties refit on clean stints), so only
the defense changes:

* ``rapm``: the clean book's ``d_rating`` (reference);
* ``matchup``: the joint ridge's ``d_team + d_individual``
  (``matchup_rating.defensive_ratings``; penalties from
  ``lambda_sweep.json``);
* ``matchup_team``: its team component alone;
* ``rapm_plus_individual``: RAPM ``d`` plus the ridge's defender-within-team
  component;
* ``simple``: step 1's estimator (empirical-Bayes team effect + defender
  within team, time-attributed), same units.

Each book scores the 2018-19 stint-sides (``rating_diagnostics.scored_rows``:
``league + sum off - sum def`` against points per 100 possessions): drop in
squared error against the decayed league mean, and each book's paired
difference with ``rapm`` (clustered by game), overall and by the defense's
lowest RAPM exposure. Calibration: the slope of RAPM's stint residual on each
book's shift from RAPM (1 = the shift is right, 0 = no information), and a
joint regression of ``actual - league - sum off`` on the defense's summed
RAPM ``d``, matchup team and matchup individual components (all in
``d_rating`` units, so a calibrated component has coefficient -1).
Development season only.

The books are written in the rating-cache format for
``oracle_ceiling.py build --ratings NAME=PATH`` (R2 rerun).

    python scripts/player_graph/evaluate_matchup_rating.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.lineups.stint_store import read_stints
from nba_ou.data_processing.player_graph.matchup_defense import (
    matchup_residuals,
    read_matchup_history,
    shrunk_on_dates,
    team_and_within_shrinkage,
    with_attacker_position,
)
from nba_ou.data_processing.player_graph.matchup_rating import (
    Penalties,
    defensive_ratings,
    player_teams,
    walk_forward_matchup_ratings,
)
from nba_ou.data_processing.player_graph.rating_diagnostics import (
    _clustered_mean,
    _lineup,
    decayed_exposure,
    decayed_league_means,
    error_summary,
    scored_rows,
)

FIRST, SEASON = 2017, 2018
BURN_IN = pd.Timestamp("2017-12-01")
MIN_GAMES = 7
CLEAN_RATINGS = Path("data/player_graph/rating_diagnostics/clean_2_6_lambdas.parquet")
EXPOSURE_BANDS = [-1, 300, 1000, 3000, np.inf]


def wls(y, x, weights, clusters):
    """Weighted least squares with an intercept; SE clustered by ``clusters``."""
    x = np.column_stack([np.ones(len(y)), np.asarray(x, dtype=float)])
    w = np.asarray(weights, dtype=float)
    bread = np.linalg.inv(x.T @ (x * w[:, None]))
    beta = bread @ (x.T @ (w * y))
    score = x * (w * (y - x @ beta))[:, None]
    meat = pd.DataFrame(score).groupby(np.asarray(clusters)).sum().to_numpy()
    cov = bread @ (meat.T @ meat) @ bread
    return beta[1:], np.sqrt(np.diag(cov))[1:]


def simple_ratings(
    pairs: pd.DataFrame, queries: pd.DataFrame, local_root: Path
) -> np.ndarray:
    """Step 1's estimator in ``d_rating`` units for ``queries``
    (``as_of_date, player_id, team_id``)."""
    frame, _ = matchup_residuals(
        with_attacker_position(pairs, local_root=local_root),
        "player_points",
        "partial_possessions",
        "time",
        development_season=FIRST,
        burn_in=BURN_IN,
        min_games=MIN_GAMES,
    )
    team, within = team_and_within_shrinkage(
        frame, development_season=FIRST, min_games=MIN_GAMES
    )
    print(
        f"simple estimator shrinkage (2017-18 split-half): team {team.shrinkage:,.1f}, "
        f"defender within team {within.shrinkage:,.1f}"
    )
    asked = queries.rename(
        columns={
            "as_of_date": "game_date",
            "player_id": "def_player_id",
            "team_id": "def_team_id",
        }
    )
    t_hat = shrunk_on_dates(frame, "def_team_id", asked, shrinkage=team.shrinkage)
    w_hat = shrunk_on_dates(
        frame, "def_player_id", asked, shrinkage=within.shrinkage, r="r_within"
    )
    return -100.0 * (t_hat + w_hat)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.local_root / "player_graph" / "matchup_rating"
    sweep = json.loads((root / "lambda_sweep.json").read_text())
    penalties = Penalties(**{k: float(v) for k, v in sweep["chosen"].items()})
    print(f"matchup ridge penalties (2018-19 sweep): {penalties.label}")

    metadata = json.loads(CLEAN_RATINGS.with_suffix(".metadata.json").read_text())
    clean = pd.read_parquet(CLEAN_RATINGS)
    clean["as_of_date"] = pd.to_datetime(clean["as_of_date"]).astype("datetime64[ns]")
    clean["player_id"] = clean["player_id"].astype(str)
    dates = sorted(clean["as_of_date"].unique())

    pairs, excluded = read_matchup_history(
        range(FIRST, SEASON + 1), local_root=args.local_root
    )
    print(f"{len(excluded)} corrupt matchup games excluded from the history")
    (ratings,) = walk_forward_matchup_ratings(pairs, dates, [penalties])
    teams_on = player_teams(pairs, dates)
    teams_on["player_id"] = teams_on["player_id"].astype(str)
    matchup = defensive_ratings(ratings, teams_on)
    keyed = clean[["as_of_date", "player_id"]].merge(
        teams_on, on=["as_of_date", "player_id"], how="left"
    )
    keyed = keyed.merge(matchup, on=["as_of_date", "player_id"], how="left")
    known = keyed["team_id"].notna()
    keyed["d_simple"] = 0.0
    keyed.loc[known, "d_simple"] = simple_ratings(
        pairs, keyed.loc[known, ["as_of_date", "player_id", "team_id"]], args.local_root
    )
    keyed = keyed.fillna({"d_matchup": 0.0, "d_team": 0.0, "d_individual": 0.0})
    books = {
        "rapm": clean,
        "matchup": clean.assign(d_rating=keyed["d_matchup"].to_numpy()),
        "matchup_team": clean.assign(d_rating=keyed["d_team"].to_numpy()),
        "rapm_plus_individual": clean.assign(
            d_rating=clean["d_rating"].to_numpy() + keyed["d_individual"].to_numpy()
        ),
        "simple": clean.assign(d_rating=keyed["d_simple"].to_numpy()),
    }
    season_rows = keyed["as_of_date"].isin(dates)
    print(
        f"{len(clean):,} book rows on {len(dates)} dates; with a team that day "
        f"{known.mean():.1%}; d_rating SD (points per 100): "
        + ", ".join(
            f"{name} {book.loc[season_rows, 'd_rating'].std():.2f}"
            for name, book in books.items()
        )
    )
    print(
        "correlation with RAPM d: "
        + ", ".join(
            f"{name} {np.corrcoef(book['d_rating'], clean['d_rating'])[0, 1]:.3f}"
            for name, book in books.items()
            if name != "rapm"
        )
    )
    out = root / "books"
    out.mkdir(parents=True, exist_ok=True)
    for name, book in books.items():
        path = out / f"{name}.parquet"
        book.to_parquet(path, index=False)
        path.with_suffix(".metadata.json").write_text(
            json.dumps(
                {
                    **metadata,
                    "variant": name,
                    "matchup_penalties": penalties.__dict__,
                    "matchup_half_life_days": 365.0,
                },
                indent=2,
            )
        )

    history = read_stints(
        list(range(metadata["first_season"], SEASON + 1)), local_root=args.local_root
    )
    stints = history.loc[history["season_year"].eq(SEASON)]
    exposure = decayed_exposure(
        history, dates, half_life_days=metadata["half_life_days"]
    )
    means = decayed_league_means(
        history, dates, half_life_days=metadata["half_life_days"]
    )
    scored = {}
    for name, book in books.items():
        efficiency, _ = scored_rows(
            stints,
            book,
            exposure,
            lambda_offdef=metadata["lambda_offdef"],
            lambda_pace=metadata["lambda_pace"],
            league_means=means,
        )
        scored[name] = efficiency
    reference = scored["rapm"]
    band = pd.cut(
        reference["min_exposure_def"],
        EXPOSURE_BANDS,
        labels=["<= 300", "300-1,000", "1,000-3,000", "> 3,000"],
    )
    rows = []
    w = reference["weight"].to_numpy(float)
    games = reference["game_id"].to_numpy()
    rapm_sq = ((reference["actual"] - reference["predicted"]) ** 2).to_numpy()
    for name, efficiency in scored.items():
        assert efficiency["game_id"].equals(reference["game_id"])
        summary = error_summary(efficiency)
        sq = ((efficiency["actual"] - efficiency["predicted"]) ** 2).to_numpy()
        diff, diff_se = _clustered_mean(rapm_sq - sq, w, games)
        line = {
            "book": name,
            "bias": summary["bias"],
            "mae": summary["mae"],
            "gain vs league": summary["mse_gain"],
            "SE": summary["mse_gain_se"],
            "vs rapm": diff,
            "vs rapm SE": diff_se,
        }
        for label in band.cat.categories:
            mask = (band == label).to_numpy()
            d, s = _clustered_mean((rapm_sq - sq)[mask], w[mask], games[mask])
            line[f"vs rapm, def min exp {label}"] = f"{d:+.2f} ± {s:.2f}"
        rows.append(line)
    pd.set_option("display.width", 250, "display.max_columns", 20)
    print(
        f"\n2018-19 stint-sides: {len(reference):,} ({reference['game_id'].nunique():,} "
        "games). Squared error in (points per 100 possessions)^2, weighted by "
        "possessions; 'vs rapm' > 0 = better than RAPM's defense."
    )
    print(
        "share of possessions by the defense's lowest RAPM exposure: "
        + ", ".join(
            f"{label} {w[(band == label).to_numpy()].sum() / w.sum():.1%}"
            for label in band.cat.categories
        )
    )
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    residual = (reference["actual"] - reference["predicted"]).to_numpy(float)
    slopes = []
    for name, efficiency in scored.items():
        if name == "rapm":
            continue
        shift = (efficiency["predicted"] - reference["predicted"]).to_numpy(float)
        (slope,), (slope_se,) = wls(residual, shift[:, None], w, games)
        slopes.append({"book": name, "slope": slope, "SE": slope_se})
    print("\ncalibration: slope of RAPM's residual on each book's shift from RAPM")
    print(pd.DataFrame(slopes).round(3).to_string(index=False))
    components = keyed.assign(d_rapm=clean["d_rating"].to_numpy()).set_index(
        ["as_of_date", "player_id"]
    )[["d_rapm", "d_team", "d_individual"]]
    lookup = {c: components[c].to_dict() for c in components.columns}
    dates_of = pd.to_datetime(reference["game_date"]).dt.normalize()
    sums = {
        c: np.array(
            [
                sum(lookup[c].get((d, p), 0.0) for p in _lineup(players))
                for d, players in zip(
                    dates_of, reference["defense_players"], strict=True
                )
            ]
        )
        for c in components.columns
    }
    offense = reference["predicted"].to_numpy(float) + sums["d_rapm"]
    target = reference["actual"].to_numpy(float) - offense
    x = np.column_stack([sums[c] for c in components.columns])
    beta, beta_se = wls(target, x, w, games)
    print(
        "joint: actual - league - sum off on the defense's sums "
        "(calibrated component = -1): "
        + ", ".join(
            f"{c} {b:+.3f} ± {e:.3f}"
            for c, b, e in zip(components.columns, beta, beta_se, strict=True)
        )
    )


if __name__ == "__main__":
    main()
