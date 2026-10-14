"""Penalties of the matchup-adjusted defensive rating, walk-forward on 2018-19.

``matchup_rating.walk_forward_matchup_ratings`` (attacker + defending team +
guarding-share-weighted defenders + home, half-life 365 days) on the
``pair_game`` history from 2017-18 (``read_matchup_history``: corrupt games
dropped), every penalty set of the grid solved on the same decayed normal
equations.

**Criterion:** weighted MSE (weights = the attacker's partial possessions) of
each 2018-19 attacker-game's points per partial possession, predicted from
games strictly before its date. **Rule:** the set with the lowest MSE; among
the sets within one paired SE (clustered by game) of it, the largest defender
penalty, then the **smallest team penalty**, then the largest attacker
penalty.

The team penalty barely moves the criterion: in an attacker-game, a shift of
the defending team is the same as shifting all its defenders (their guarding
shares add up to 1), so only the penalties decide who takes team defense. The
design wants the team term to take it (the individual rating must not learn
team defense), hence the smallest team penalty among ties; ``team share of
d`` measures what is left: the between-team share of the variance of the
defenders' ``d_j`` on the last date (exposure-weighted, defenders with at
least 1,000 partial possessions).

First run (2026-10-09): grid attacker {100, 300, 1000} x team {3000, 10000,
30000, inf} x defender {1000, 3000, 10000, 30000, inf} and a tie-break toward
the largest team penalty. The best set sat at the lowest attacker and defender
penalties, and that tie-break let the defenders absorb team defense, so the
grid was extended downward and the team tie-break reversed (both still on
2018-19).

Also reported: the gain over the intercept-and-home model, the out-of-sample
contribution of the defender terms (the same prediction with them zeroed),
and step 1's simple estimator (as-of attacker norm + empirical-Bayes team
effect + defender within team) on the same attacker-games.

    python scripts/player_graph/matchup_rating_sweep.py

Writes ``data/player_graph/matchup_rating/lambda_sweep.json``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.player_graph.matchup_defense import (
    matchup_residuals,
    read_matchup_history,
    shrunk_as_of,
    team_and_within_shrinkage,
    with_attacker_position,
)
from nba_ou.data_processing.player_graph.matchup_rating import (
    Penalties,
    player_teams,
    walk_forward_matchup_ratings,
)
from nba_ou.data_processing.player_graph.rating_diagnostics import _clustered_mean

FIRST, SEASON = 2017, 2018
BURN_IN = pd.Timestamp("2017-12-01")
MIN_GAMES = 7
INF = float("inf")
GRID = {
    "attacker": (10.0, 30.0, 100.0, 300.0),
    "team": (100.0, 300.0, 1000.0, 3000.0, 30000.0),
    "defender": (100.0, 300.0, 1000.0, 3000.0, INF),
}
MIN_TEAM_SHARE_EXPOSURE = 1000.0
NOTHING = Penalties(INF, INF, INF)


def simple_estimator(pairs: pd.DataFrame, local_root: Path) -> pd.DataFrame:
    """Step 1's estimator on 2018-19 attacker-games: as-of attacker norm +
    defending team + sum of guarding share x defender within team, team and
    within-team shrinkage by split-half on 2017-18."""
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
    frame["t_hat"] = shrunk_as_of(frame, ["def_team_id"], shrinkage=team.shrinkage)
    frame["w_hat"] = shrunk_as_of(
        frame, ["def_player_id"], shrinkage=within.shrinkage, r="r_within"
    )
    frame["share_w_hat"] = frame["w"] * frame["w_hat"]
    games = frame.loc[frame["season_year"].eq(SEASON)].groupby(
        ["game_id", "off_player_id"]
    )
    out = games.agg(
        points=("y", "sum"),
        weight=("w", "sum"),
        norm=("norm", "first"),
        t_hat=("t_hat", "first"),
        share_w_hat=("share_w_hat", "sum"),
    ).reset_index()
    out["actual"] = out["points"] / out["weight"]
    out["predicted"] = out["norm"] + out["t_hat"] + out["share_w_hat"] / out["weight"]
    out.attrs["shrinkage"] = {"team": team.shrinkage, "within": within.shrinkage}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    started = time.time()
    pairs, excluded = read_matchup_history(
        range(FIRST, SEASON + 1), local_root=args.local_root
    )
    print(f"{len(excluded)} corrupt games excluded")
    dates = sorted(
        pd.to_datetime(pairs.loc[pairs["season_year"].eq(SEASON), "game_date"])
        .dt.normalize()
        .unique()
    )
    grid = [
        Penalties(a, t, d)
        for a, t, d in itertools.product(
            GRID["attacker"], GRID["team"], GRID["defender"]
        )
    ]
    results = walk_forward_matchup_ratings(
        pairs, dates, [*grid, NOTHING], keep_ratings=[dates[-1]]
    )
    teams_on = player_teams(pairs, [dates[-1]])
    print(
        f"{len(grid)} penalty sets + reference on {len(dates)} dates in {time.time() - started:.0f}s"
    )
    reference = results[-1].predictions
    keys = ["game_id", "off_player_id"]

    def squared(p: pd.DataFrame, column: str = "predicted") -> np.ndarray:
        return ((p["actual"] - p[column]) ** 2).to_numpy(float)

    w = reference["weight"].to_numpy(float)
    games = reference["game_id"].to_numpy()
    ref_sq = squared(reference)
    rows = []
    for result in results[:-1]:
        p = result.predictions
        assert p[keys].equals(reference[keys])
        sq = squared(p)
        gain, gain_se = _clustered_mean(ref_sq - sq, w, games)
        dfn, dfn_se = _clustered_mean(
            squared(p, "predicted_without_defenders") - sq, w, games
        )
        last = result.players.dropna(subset=["defense"]).merge(
            teams_on, on=["as_of_date", "player_id"]
        )
        last = last.loc[last["defense_exposure"] >= MIN_TEAM_SHARE_EXPOSURE]
        weights = last["defense_exposure"]
        mean = np.average(last["defense"], weights=weights)
        team_mean = (last["defense"] * weights).groupby(last["team_id"]).transform(
            "sum"
        ) / weights.groupby(last["team_id"]).transform("sum")
        total = np.average((last["defense"] - mean) ** 2, weights=weights)
        between = np.average((team_mean - mean) ** 2, weights=weights)
        rows.append(
            {
                "attacker": result.penalties.attacker,
                "team": result.penalties.team,
                "defender": result.penalties.defender,
                "mse": float(np.average(sq, weights=w)),
                "gain_vs_nothing": gain,
                "gain_se": gain_se,
                "defender_terms_gain": dfn,
                "defender_terms_se": dfn_se,
                "team_share_of_d": between / total if total > 0 else np.nan,
                "_sq": sq,
            }
        )
    table = pd.DataFrame(rows)
    best = table["mse"].idxmin()
    paired = [
        _clustered_mean(row["_sq"] - table.loc[best, "_sq"], w, games)
        for _, row in table.iterrows()
    ]
    table["vs_best"] = [m for m, _ in paired]
    table["vs_best_se"] = [s for _, s in paired]
    tied = table.loc[table["vs_best"] <= table["vs_best_se"]]
    chosen = tied.sort_values(
        ["defender", "team", "attacker"], ascending=[False, True, False]
    ).index[0]
    table = table.drop(columns="_sq")

    simple = simple_estimator(pairs, args.local_root)
    simple_shrinkage = simple.attrs.get("shrinkage")
    simple = simple.merge(
        reference[[*keys, "actual", "weight"]], on=keys, suffixes=("", "_ref")
    )
    simple_mse = float(
        np.average(
            (simple["actual_ref"] - simple["predicted"]) ** 2,
            weights=simple["weight_ref"],
        )
    )
    ridge_on_same = table.loc[chosen, "mse"]

    pd.set_option("display.width", 250, "display.max_columns", 20)
    scale = 1e4  # MSE in (points per 100 partial possessions)^2
    shown = table.copy()
    for column in (
        "mse",
        "gain_vs_nothing",
        "gain_se",
        "defender_terms_gain",
        "defender_terms_se",
        "vs_best",
        "vs_best_se",
    ):
        shown[column] *= scale
    print(
        f"\n{len(reference):,} attacker-games in 2018-19; MSE etc. in (points per 100 "
        "partial possessions)^2, weighted by partial possessions"
    )
    print(shown.round(3).to_string())
    print(
        f"\nbest: {table.loc[best, ['attacker', 'team', 'defender']].to_dict()}; "
        f"chosen (within 1 SE: largest defender, smallest team, largest attacker "
        f"penalty): "
        f"{table.loc[chosen, ['attacker', 'team', 'defender']].to_dict()}"
    )
    print(
        f"step-1 simple estimator on {len(simple):,} of the same attacker-games: "
        f"MSE {scale * simple_mse:.3f} (chosen ridge on all rows {scale * ridge_on_same:.3f}); "
        f"its shrinkage {simple_shrinkage}"
    )
    out = args.local_root / "player_graph" / "matchup_rating"
    out.mkdir(parents=True, exist_ok=True)
    (out / "lambda_sweep.json").write_text(
        json.dumps(
            {
                "criterion": "walk-forward weighted MSE, attacker-game points per pp, 2018-19",
                "half_life_days": 365.0,
                "chosen": table.loc[chosen, ["attacker", "team", "defender"]].to_dict(),
                "best": table.loc[best, ["attacker", "team", "defender"]].to_dict(),
                "excluded_games": sorted(excluded),
                "table": table.replace({INF: "inf"}).to_dict(orient="records"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
