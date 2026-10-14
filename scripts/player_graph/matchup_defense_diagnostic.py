"""Phase 4B, step 1: is there a matchup-adjusted defensive effect, and does
it depend on whom the defender guards? (2017-18 and 2018-19 only.)

Rows are ``pair_game`` (attacker x defender x game), labels and history only.
For each target (numerator ``y`` per exposure ``w``: points, scoring attempts,
3PA, FTA and turnovers per partial possession, points per attempt, points per
matched minute) every row gets the attacker's as-of norm (decayed, shrunk
toward his position's rate) and a residual ``r = y - w * norm``. A defender's
effect is ``D_j = sum r / sum w``; positive = his attackers beat their norms.

Two attributions of the attacker's output to defenders:

* ``shot``: as the NBA credits it, to whoever was matched at the time of each
  shot (so help defenders and rim protectors collect drives they barely
  guarded);
* ``time``: the attacker's whole-game residual spread over his defenders by
  their share of his partial possessions (seconds for the per-minute target),
  immune to who was nearest when the shot went up.

0. Attribution check: credited attempts per unit of guarding time, the
   residual by defender position x attacker position, and how ``D_j`` relates
   to blocks and to 2_6's stint RAPM ``d_rating``.
1. Signal by split-half (each defender's games alternate between halves):
   defender, defender relative to his teammates in the same game, defending
   team; 2017-18 (from December, after the attacker norms' burn-in) and
   2018-19; persistence of ``D_j`` from one season to the next.
2. Dependence on the attacker: interaction cells defender x attacker type
   (position, usage tier, 3PA-share tier) after removing the defender's own
   effect and the type's mean, by split-half.
3. Walk-forward on 2018-19: as-of estimates (decayed, empirical-Bayes
   shrinkage from 2017-18's split-half, nothing fitted on 2018-19) predicting
   each defender-game's residual; the interaction cells' gain over the
   defender's main effect; 2_6's stint RAPM ``d_rating`` (clean refit, as of
   the date) on the same rows, alone and jointly.

    python scripts/player_graph/matchup_defense_diagnostic.py

Reads local stores only (``pair_game``, node profiles, the clean rating book).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.player_graph.matchup_defense import (
    TARGETS,
    as_of_sums,
    matchup_residuals,
    read_matchup_history,
    row_halves,
    shrunk_as_of,
    split_half,
    with_target_columns,
)

FIRST, SEASON = 2017, 2018
BURN_IN = pd.Timestamp("2017-12-01")
CLEAN_RATINGS = Path("data/player_graph/rating_diagnostics/clean_2_6_lambdas.parquet")
MIN_GAMES = 7  # per half, in typical games of exposure
DIMENSIONS = ("position", "usage_tier", "three_tier")
PER_100 = {t for t, (_, w) in TARGETS.items() if w == "partial_possessions"}
ATTRIBUTIONS = ("shot", "time")


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


def load(root: Path) -> pd.DataFrame:
    pairs, doubled = read_matchup_history(range(FIRST, SEASON + 1), local_root=root)
    print(
        f"  excluded {len(doubled)} games with doubled matchup exposure: "
        f"{sorted(doubled)[:2]} ... {sorted(doubled)[-2:]}"
    )
    pairs = with_target_columns(pairs.loc[pairs["partial_possessions"] > 0])
    pairs["game_date"] = pd.to_datetime(pairs["game_date"]).astype("datetime64[ns]")
    profiles = pd.concat(
        [
            pd.read_parquet(
                root / "player_graph" / "node_profiles" / f"season={s}.parquet",
                columns=[
                    "as_of_date",
                    "player_id",
                    "p_G",
                    "p_F",
                    "p_C",
                    "usage",
                    "fg3a_per36",
                    "fga_per36",
                ],
            )
            for s in range(FIRST, SEASON + 1)
        ],
        ignore_index=True,
    ).rename(columns={"as_of_date": "game_date", "player_id": "off_player_id"})
    profiles["game_date"] = pd.to_datetime(profiles["game_date"]).astype(
        "datetime64[ns]"
    )
    defenders = profiles[["game_date", "off_player_id", "p_G", "p_F", "p_C"]].rename(
        columns={
            "off_player_id": "def_player_id",
            "p_G": "d_G",
            "p_F": "d_F",
            "p_C": "d_C",
        }
    )
    pairs = pairs.merge(profiles, on=["game_date", "off_player_id"], how="left")
    pairs = pairs.merge(defenders, on=["game_date", "def_player_id"], how="left")
    for prefix, column in (("d_", "def_position"), ("p_", "position")):
        probabilities = pairs[[f"{prefix}{p}" for p in "GFC"]]
        pairs[column] = (
            probabilities.fillna(-1.0)
            .idxmax(axis=1)
            .str[-1]
            .where(probabilities.notna().any(axis=1), "?")
        )
    pairs["three_share"] = (pairs["fg3a_per36"] / pairs["fga_per36"]).where(
        pairs["fga_per36"] > 0
    )
    development = pairs["season_year"].eq(FIRST)
    for column, tier in (("usage", "usage_tier"), ("three_share", "three_tier")):
        part = pairs.loc[development & pairs[column].notna()]
        order = part.sort_values(column)
        cumulative = (
            order["partial_possessions"].cumsum() / order["partial_possessions"].sum()
        )
        cuts = [order.loc[cumulative >= q, column].iloc[0] for q in (1 / 3, 2 / 3)]
        pairs[tier] = pd.cut(
            pairs[column], [-np.inf, *cuts, np.inf], labels=["low", "mid", "high"]
        ).astype(str)
        print(f"  {tier} cut points (2017-18, exposure terciles): {np.round(cuts, 3)}")
    return pairs


def residuals(
    pairs: pd.DataFrame, y: str, w: str, attribution: str
) -> tuple[pd.DataFrame, float]:
    return matchup_residuals(
        pairs,
        y,
        w,
        attribution,
        development_season=FIRST,
        burn_in=BURN_IN,
        min_games=MIN_GAMES,
    )


def signal_rows(frame: pd.DataFrame, season: int) -> dict:
    part = frame.loc[frame["season_year"].eq(season)].copy()
    out = {}
    typical = part.groupby(["def_player_id", "game_id"])["w"].sum().median()
    part["half"] = row_halves(part, ["def_player_id"])
    out["defender"] = split_half(
        part, ["def_player_id"], min_exposure=MIN_GAMES * typical
    )
    out["within team"] = split_half(
        part, ["def_player_id"], min_exposure=MIN_GAMES * typical, r="r_within"
    )
    team_part = part.copy()
    team_part["half"] = row_halves(team_part, ["def_team_id"])
    team_typical = part.groupby(["def_team_id", "game_id"])["w"].sum().median()
    out["team"] = split_half(
        team_part, ["def_team_id"], min_exposure=MIN_GAMES * team_typical
    )
    # Interaction: the defender's residual by attacker type, net of his own
    # effect in that half and of the type's mean.
    per_half = part.groupby(["def_player_id", "half"])[["r", "w"]].transform("sum")
    part["r_int"] = part["r"] - part["w"] * per_half["r"] / per_half["w"]
    for dim in DIMENSIONS:
        cells = part.loc[part[dim].ne("?") & part[dim].ne("nan")]
        cell_typical = (
            cells.groupby(["def_player_id", dim, "game_id"])["w"].sum().median()
        )
        out[f"x {dim}"] = split_half(
            cells,
            ["def_player_id", dim],
            min_exposure=MIN_GAMES * cell_typical,
            center_by=[dim],
            r="r_int",
        )
    out["_typical"] = typical
    return out


def persistence(frame: pd.DataFrame) -> tuple[int, float, float]:
    seasons = frame.groupby(["def_player_id", "season_year"])[["r", "w"]].sum()
    typical = frame.groupby(["def_player_id", "game_id"])["w"].sum().median()
    seasons = seasons.loc[seasons["w"] >= 2 * MIN_GAMES * typical]
    rate = (seasons["r"] / seasons["w"]).unstack("season_year").dropna()
    a, b = rate[FIRST], rate[SEASON]
    return len(rate), float(np.cov(a, b)[0, 1]), float(np.corrcoef(a, b)[0, 1])


def walk_forward(frame: pd.DataFrame, dev: dict) -> pd.DataFrame:
    """As-of predictors for 2018-19 (shrinkage from 2017-18), summed per
    defender-game: ``R`` the residual, ``W`` the exposure, ``P_<name>`` each
    predictor times exposure, every predictor centered on its exposure-weighted
    mean over the date's rows (pre-game information, removes level drift)."""
    frame = frame.copy()
    predictors = {}
    predictors["defender"] = shrunk_as_of(
        frame, ["def_player_id"], shrinkage=dev["defender"].shrinkage
    )
    predictors["team"] = shrunk_as_of(
        frame, ["def_team_id"], shrinkage=dev["team"].shrinkage
    )
    predictors["team + within"] = predictors["team"] + shrunk_as_of(
        frame, ["def_player_id"], shrinkage=dev["within team"].shrinkage, r="r_within"
    )
    # Attacker-type terms, centered within the defender on his as-of exposure
    # mix so they only redistribute his level by tonight's attackers: the
    # type's mean deviation (fixed from 2017-18), then his own cells' deviation
    # from his as-of rate, shrunk with the cell's exposure.
    own = as_of_sums(frame, ["def_player_id"], ["r", "w"])
    d_raw = (own["r"] / own["w"]).fillna(0.0)
    previous = frame.loc[frame["season_year"].eq(FIRST)]
    season_d = previous.groupby("def_player_id")[["r", "w"]].transform("sum")
    previous = previous.assign(
        dev=previous["r"] - previous["w"] * season_d["r"] / season_d["w"]
    )
    for dim in DIMENSIONS:
        levels = sorted(set(frame[dim]) - {"?", "nan"})
        sums = previous.groupby(dim)[["dev", "w"]].sum()
        c = (sums["dev"] / sums["w"]).reindex(levels).fillna(0.0)
        k = dev[f"x {dim}"].shrinkage
        parts = frame[["def_player_id", "game_date"]].copy()
        for level in levels:
            mine = frame[dim].eq(level)
            parts[f"r {level}"] = frame["r"].where(mine, 0.0)
            parts[f"w {level}"] = frame["w"].where(mine, 0.0)
        cells = as_of_sums(
            parts, ["def_player_id"], [f"{v} {level}" for v in "rw" for level in levels]
        )
        exposure = sum(cells[f"w {level}"] for level in levels)
        type_center = pd.Series(0.0, index=frame.index)
        cell_center = pd.Series(0.0, index=frame.index)
        own_type = pd.Series(0.0, index=frame.index)
        own_cell = pd.Series(0.0, index=frame.index)
        for level in levels:
            share = (cells[f"w {level}"] / exposure).where(
                exposure > 0, 1.0 / len(levels)
            )
            deviation = (
                (cells[f"r {level}"] - cells[f"w {level}"] * (d_raw + c[level]))
                / (cells[f"w {level}"] + k)
                if np.isfinite(k)
                else 0.0
            )
            term = c[level] + deviation
            type_center += share * c[level]
            cell_center += share * term
            mine = frame[dim].eq(level)
            own_type = own_type.where(~mine, c[level])
            own_cell = own_cell.where(~mine, term)
        known = frame[dim].isin(levels)
        predictors[f"+ type x {dim}"] = predictors["defender"] + (
            own_type - type_center
        ).where(known, 0.0)
        predictors[f"+ cells x {dim}"] = predictors["defender"] + (
            own_cell - cell_center
        ).where(known, 0.0)
    test = frame.loc[frame["season_year"].eq(SEASON)].copy()
    book = pd.read_parquet(
        CLEAN_RATINGS, columns=["as_of_date", "player_id", "d_rating"]
    ).rename(columns={"as_of_date": "game_date", "player_id": "def_player_id"})
    book["game_date"] = pd.to_datetime(book["game_date"]).astype("datetime64[ns]")
    rating = test[["game_date", "def_player_id"]].merge(
        book, on=["game_date", "def_player_id"], how="left"
    )
    # Positive = attackers score more, like D_j; per possession, not per 100.
    predictors["stint RAPM -d"] = pd.Series(
        -rating["d_rating"].fillna(0.0).to_numpy() / 100.0, index=test.index
    )
    total_w = test["w"].groupby(test["game_date"]).transform("sum")
    for name, values in predictors.items():
        x = values.loc[test.index]
        mean = (x * test["w"]).groupby(test["game_date"]).transform("sum") / total_w
        test[f"P_{name}"] = test["w"] * (x - mean)
    units = test.groupby(["game_id", "def_player_id"]).agg(
        R=("r", "sum"),
        W=("w", "sum"),
        **{f"P_{name}": (f"P_{name}", "sum") for name in predictors},
    )
    return units.reset_index().loc[lambda u: u["W"] > 0]


def gain(units: pd.DataFrame, prediction: str, baseline: str | None = None):
    """Drop in squared error of the defender-game residual (slope 1, nothing
    fitted), as a % of the residual's sum of squares, ± SE clustered by game."""
    base = 0.0 if baseline is None else units[baseline]
    per = (units["R"] - base) ** 2 - (units["R"] - units[prediction]) ** 2
    per_game = per.groupby(units["game_id"]).sum()
    total = (units["R"] ** 2).groupby(units["game_id"]).sum().mean()
    return (
        100 * per_game.mean() / total,
        100 * per_game.std(ddof=1) / np.sqrt(len(per_game)) / total,
    )


def defender_seasons(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    """2018-19 ``D_j`` of defenders with two-half exposure, with their position,
    blocks per 36 and stint RAPM ``d_rating`` at the end of the season."""
    part = frame.loc[frame["season_year"].eq(SEASON)]
    typical = part.groupby(["def_player_id", "game_id"])["w"].sum().median()
    table = part.groupby("def_player_id")[["r", "w"]].sum()
    table = table.loc[table["w"] >= 2 * MIN_GAMES * typical]
    table["D"] = table["r"] / table["w"]
    profiles = pd.read_parquet(
        root / "player_graph" / "node_profiles" / f"season={SEASON}.parquet",
        columns=["as_of_date", "player_id", "p_G", "p_F", "p_C", "blk_per36"],
    )
    last = profiles.sort_values("as_of_date").groupby("player_id").tail(1)
    last = last.set_index("player_id")
    last["def_position"] = last[["p_G", "p_F", "p_C"]].idxmax(axis=1).str[-1]
    book = pd.read_parquet(
        CLEAN_RATINGS, columns=["as_of_date", "player_id", "d_rating"]
    )
    rating = book.sort_values("as_of_date").groupby("player_id").tail(1)
    return table.join(last[["def_position", "blk_per36"]]).join(
        rating.set_index("player_id")["d_rating"]
    )


def attribution_check(pairs: pd.DataFrame, frames: dict, root: Path) -> None:
    season = pairs.loc[pairs["season_year"].eq(SEASON) & pairs["def_position"].ne("?")]
    totals = season.groupby("def_position")[
        ["attempts", "partial_possessions", "player_points"]
    ].sum()
    shares = totals / totals.sum()
    print(
        "  credited per unit of guarding time, relative to all defenders (2018-19): "
        + "; ".join(
            f"{pos} attempts {shares.loc[pos, 'attempts'] / shares.loc[pos, 'partial_possessions']:.2f}, "
            f"points {shares.loc[pos, 'player_points'] / shares.loc[pos, 'partial_possessions']:.2f}"
            for pos in shares.index
        )
    )
    rows = []
    for (attribution, target), frame in frames.items():
        part = frame.loc[
            frame["season_year"].eq(SEASON) & frame["def_position"].ne("?")
        ]
        cells = part.groupby(["def_position", "position"])[["r", "w"]].sum()
        grid = (cells["r"] / cells["w"] * 100).unstack().round(1)
        print(
            f"\n  {attribution} / {target}: residual per 100 pp, defender position "
            "(rows) x attacker position (columns), 2018-19"
        )
        print(grid.to_string())
        table = defender_seasons(frame, root)
        line = {
            "attribution": attribution,
            "target": target,
            "defenders": len(table),
            "corr(D, blocks/36)": table["D"].corr(table["blk_per36"]),
            "corr(D, RAPM d)": table["D"].corr(table["d_rating"]),
        }
        for pos, group in table.groupby("def_position"):
            line[f"mean D {pos} x100"] = 100 * group["D"].mean()
            line[f"corr(D, RAPM d) {pos}"] = group["D"].corr(group["d_rating"])
        rows.append(line)
    print(
        "\n  defender season D_j (2018-19) vs blocks and stint RAPM d_rating "
        "(d > 0 = good defense, so a real effect correlates negatively)"
    )
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("Loading pair_game 2017-18 and 2018-19 ...")
    pairs = load(args.local_root)
    print(
        f"  {len(pairs):,} pair rows with exposure, {pairs['game_id'].nunique():,} games, "
        f"{pairs['def_player_id'].nunique():,} defenders"
    )

    signal_table, interaction_table, wf_table, joint_table = [], [], [], []
    check = {}
    for attribution in ATTRIBUTIONS:
        for target, (y, w) in TARGETS.items():
            scale = 100.0 if target in PER_100 else 1.0
            frame, prior_weight = residuals(pairs, y, w, attribution)
            if target in ("pts_per_pp", "att_per_pp"):
                check[(attribution, target)] = frame
            label = {"attribution": attribution, "target": target}
            per_season = {s: signal_rows(frame, s) for s in (FIRST, SEASON)}
            n, cov, corr = persistence(frame)
            for season, result in per_season.items():
                typical = result["_typical"]
                for name in ("defender", "within team", "team"):
                    s = result[name]
                    signal_table.append(
                        {
                            **label,
                            "season": f"{season}-{str(season + 1)[2:]}",
                            "key": name,
                            "keys": s.keys,
                            "signal SD": np.sign(s.signal_var)
                            * np.sqrt(abs(s.signal_var))
                            * scale,
                            "t": s.signal_var / s.signal_se,
                            "split corr": s.correlation,
                            "k (units)": s.shrinkage,
                            "rel. at median": s.reliability(s.median_exposure),
                            "rel. one game": (
                                s.reliability(typical) if name != "team" else np.nan
                            ),
                        }
                    )
                for dim in DIMENSIONS:
                    s = result[f"x {dim}"]
                    main_var = result["defender"].signal_var
                    interaction_table.append(
                        {
                            **label,
                            "season": f"{season}-{str(season + 1)[2:]}",
                            "attacker type": dim,
                            "cells": s.keys,
                            "interaction SD": np.sign(s.signal_var)
                            * np.sqrt(abs(s.signal_var))
                            * scale,
                            "t": s.signal_var / s.signal_se,
                            "split corr": s.correlation,
                            "var / main var": (
                                s.signal_var / main_var if main_var > 0 else np.nan
                            ),
                            "rel. at median": s.reliability(s.median_exposure),
                        }
                    )
            signal_table.append(
                {
                    **label,
                    "season": "17-18 -> 18-19",
                    "key": "defender, season to season",
                    "keys": n,
                    "signal SD": np.sign(cov) * np.sqrt(abs(cov)) * scale,
                    "split corr": corr,
                }
            )
            units = walk_forward(frame, per_season[FIRST])
            y = (units["R"] / units["W"]).to_numpy()
            for name in [c[2:] for c in units.columns if c.startswith("P_")]:
                x = (units[f"P_{name}"] / units["W"]).to_numpy()
                slope, slope_se = wls(y, x[:, None], units["W"], units["game_id"])
                weights = units["W"].to_numpy()
                xc = x - np.average(x, weights=weights)
                yc = y - np.average(y, weights=weights)
                corr = np.sum(weights * xc * yc) / np.sqrt(
                    np.sum(weights * xc**2) * np.sum(weights * yc**2)
                )
                line = {**label, "predictor": name}
                if name.startswith("+ cells"):
                    baseline = "P_+ type" + name[len("+ cells") :]
                    line["vs"] = "+ type"
                elif name.startswith("+ type"):
                    baseline = "P_defender"
                    line["vs"] = "defender"
                else:
                    baseline = None
                    line["vs"] = "0"
                if name != "stint RAPM -d":
                    line["gain %"], line["SE %"] = gain(units, f"P_{name}", baseline)
                line.update({"slope": slope[0], "slope SE": slope_se[0], "corr": corr})
                wf_table.append(line)
            x = np.column_stack(
                [
                    units["P_defender"] / units["W"],
                    units["P_stint RAPM -d"] / units["W"],
                ]
            )
            x = (x - x.mean(axis=0)) / x.std(axis=0)
            beta, beta_se = wls(
                (units["R"] / units["W"]).to_numpy(), x, units["W"], units["game_id"]
            )
            joint_table.append(
                {
                    **label,
                    "t D_j (joint)": beta[0] / beta_se[0],
                    "t RAPM (joint)": beta[1] / beta_se[1],
                    "corr(D_j, RAPM)": np.corrcoef(x[:, 0], x[:, 1])[0, 1],
                    "attacker prior weight": prior_weight,
                    "defender-games": len(units),
                }
            )

    print("\n0. ATTRIBUTION CHECK")
    attribution_check(pairs, check, args.local_root)
    print(
        "\n1. SIGNAL (split-half covariance; SD per 100 units for per-possession "
        "targets; t = cov / SE; k = empirical-Bayes shrinkage in exposure units)"
    )
    print(pd.DataFrame(signal_table).round(3).to_string(index=False))
    print("\n2. DOES THE EFFECT DEPEND ON THE ATTACKER? (defender x attacker type)")
    print(pd.DataFrame(interaction_table).round(3).to_string(index=False))
    print(
        "\n3. WALK-FORWARD 2018-19: defender-game residual, as-of predictors centered "
        "by date (shrinkage from 2017-18); gain at slope 1 vs the 'vs' column; "
        "RAPM slope in its own units"
    )
    print(pd.DataFrame(wf_table).round(4).to_string(index=False))
    print("\n   joint regression (standardized), clustered by game")
    print(pd.DataFrame(joint_table).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
