"""Phase 5 report: the pre-registered reading of the paired walk-forward.

Reads ``run_pi_walk_forward.py``'s output for both arms and every seed and
writes ``<out>/report/{report.md,metrics.json,games.parquet}``. Everything
here was fixed before any evaluation run (see
``experiments/pi_block_v1_closing_2026_10/README.md``):

* **Completeness**: every arm and seed covers the same games; refuses
  otherwise (``--allow-incomplete`` for smoke tests).
* **Seed band of the baseline** (arm A): MAE per seed, its spread, and the 45
  pairwise MAE differences between A's seeds.
* **Primary**: each arm's prediction is the mean of its seeds; per game
  ``delta = |error_A| - |error_B|`` (positive = the PI block helps); mean,
  SE clustered by game date, 95% interval.
* **Training randomness**: the per-seed paired differences (seed s of A vs
  seed s of B): mean, SD, range, how many are positive.
* **Decision rule**: improves if the 95% interval is above 0 and at least
  ``MIN_AGREEING`` of the per-seed differences are positive; hurts if the
  interval is below 0 and at least ``MIN_AGREEING`` are negative; else null.
* **Secondary (no decisions)**: squared error, calibration slope (actual on
  predicted line error), per season, fixed slices (|``LU_ABSENCE_IMPACT_PTS_BEFORE``|
  > 3; no PI coverage, a placebo; the first 30 days of each season; after the
  trade deadline), and betting at the spec's edge threshold (win rate with a
  Wilson interval, ROI). The closing dataset carries lines but no over /
  under prices, so bets settle at the stored prices when present and at -110
  (the spec's ``flat_decimal_odds``) otherwise, which today is always.

    python scripts/player_graph/report_pi_walk_forward.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ARMS = ("a_baseline", "b_pi_primary")
SEEDS = tuple(range(101, 1011, 101))
MIN_AGREEING = 8
EDGE_THRESHOLD = 0.1  # the spec's betting.primary_edge_threshold
#: Fixed slice definitions (trade deadlines of the evaluated seasons).
TRADE_DEADLINES = {
    2020: "2021-03-25",
    2021: "2022-02-10",
    2022: "2023-02-09",
    2023: "2024-02-08",
    2024: "2025-02-06",
    2025: "2026-02-05",
}
EARLY_DAYS = 30
ABSENCE_SLICE = 3.0


def clustered_mean(values: np.ndarray, clusters: np.ndarray) -> tuple[float, float]:
    """Mean and its standard error clustered by ``clusters`` (small-sample
    factor G / (G - 1))."""
    values = np.asarray(values, float)
    mean = float(values.mean())
    sums = pd.Series(values - mean).groupby(np.asarray(clusters)).sum().to_numpy()
    g = len(sums)
    se = float(np.sqrt((sums**2).sum() * g / max(g - 1, 1)) / len(values))
    return mean, se


def decide(low: float, high: float, per_seed: np.ndarray) -> str:
    if low > 0 and (per_seed > 0).sum() >= MIN_AGREEING:
        return "improves"
    if high < 0 and (per_seed < 0).sum() >= MIN_AGREEING:
        return "hurts"
    return "null"


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def betting(games: pd.DataFrame, prediction: str) -> dict:
    edge = games[prediction]
    margin = games["actual_outcome"] - games["target_line"]
    bet = edge.abs() > EDGE_THRESHOLD
    over = edge > 0
    decided = bet & margin.ne(0)
    won = decided & (np.sign(margin) == np.sign(edge))
    price = np.where(over, games["decimal_odds_over"], games["decimal_odds_under"])
    price = pd.Series(price, index=games.index).fillna(1.0 + 100 / 110)
    profit = np.where(won, price - 1.0, -1.0)[decided.to_numpy()]
    n = int(decided.sum())
    low, high = wilson(int(won.sum()), n)
    return {
        "bets": n,
        "bet_rate": float(bet.mean()),
        "win_rate": float(won.sum() / n) if n else float("nan"),
        "win_rate_95": [low, high],
        "roi": float(profit.mean()) if n else float("nan"),
    }


def seed_band(args, arm: str, devices: set) -> None:
    """Arm A alone: MAE per seed and the pairwise differences between seeds."""
    frame = load(args.out, arm, args.seeds)
    games = {seed: frozenset(part["GAME_ID"]) for seed, part in frame.groupby("seed")}
    if len(set(games.values())) != 1 and not args.allow_incomplete:
        raise SystemExit("seeds do not cover the same games")
    common = frozenset.intersection(*games.values())
    frame = frame.loc[frame["GAME_ID"].isin(common)]
    mae = (
        frame.assign(err=(frame["y_true"] - frame["y_pred"]).abs())
        .groupby("seed")["err"]
        .mean()
    )
    seeds = list(mae.index)
    pairs = [abs(mae[s] - mae[t]) for i, s in enumerate(seeds) for t in seeds[i + 1 :]]
    band = {
        "arm": arm,
        "devices": sorted(devices),
        "games": len(common),
        "mae_per_seed": mae.round(5).to_dict(),
        "mae_range": float(mae.max() - mae.min()),
        "mae_sd": float(mae.std(ddof=1)) if len(mae) > 1 else None,
        "pair_abs_diff_median": float(np.median(pairs)) if pairs else None,
        "pair_abs_diff_max": float(max(pairs)) if pairs else None,
    }
    report = args.out / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "seed_band.json").write_text(json.dumps(band, indent=2) + "\n")
    print(json.dumps(band, indent=2))


def load(out: Path, arm: str, seeds) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        paths = sorted((out / arm / f"seed={seed}").glob("month=*.parquet"))
        frames += [pd.read_parquet(p) for p in paths]
    if not frames:
        raise SystemExit(f"no output for {arm} under {out}")
    return pd.concat(frames, ignore_index=True)


def season_of(dates: pd.Series) -> pd.Series:
    return dates.dt.year.where(dates.dt.month >= 9, dates.dt.year - 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/experiments/pi_block_v1")
    )
    parser.add_argument("--arms", nargs=2, default=list(ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--research-file", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument(
        "--seed-band-only",
        action="store_true",
        help="Arm A's seed band only (read before arm B runs)",
    )
    args = parser.parse_args()
    arm_a, arm_b = args.arms
    devices = {
        json.loads(path.read_text())["device"]
        for arm in ((arm_a,) if args.seed_band_only else (arm_a, arm_b))
        for path in (args.out / arm).glob("seed=*/run.json")
    }
    if len(devices) > 1:
        raise SystemExit(f"runs used different devices: {devices}")
    if args.seed_band_only:
        seed_band(args, arm_a, devices)
        return
    rows = {arm: load(args.out, arm, args.seeds) for arm in (arm_a, arm_b)}
    keys = {
        (arm, seed): frozenset(part["GAME_ID"])
        for arm, frame in rows.items()
        for seed, part in frame.groupby("seed")
    }
    expected = {(arm, seed) for arm in (arm_a, arm_b) for seed in args.seeds}
    complete = set(keys) == expected and len(set(keys.values())) == 1
    if not complete and not args.allow_incomplete:
        raise SystemExit(
            "arms / seeds do not cover the same games; see --allow-incomplete"
        )
    common = frozenset.intersection(*keys.values())
    # Same 2_5 features in both arms: B is A plus its PI columns, nothing else.
    feature_sets = {}
    for arm in (arm_a, arm_b):
        runs = sorted((args.out / arm).glob("seed=*/run.json"))
        names = {frozenset(json.loads(p.read_text())["feature_names"]) for p in runs}
        if len(names) != 1:
            raise SystemExit(f"{arm}: seeds ran with different feature sets")
        feature_sets[arm] = next(iter(names))
    base_b = {f for f in feature_sets[arm_b] if not f.startswith("PI_")}
    if base_b != set(feature_sets[arm_a]) or any(
        f.startswith("PI_") for f in feature_sets[arm_a]
    ):
        raise SystemExit("arm B is not arm A plus PI columns")
    added = sorted(feature_sets[arm_b] - feature_sets[arm_a])

    wide = {}
    for arm, frame in rows.items():
        frame = frame.loc[frame["GAME_ID"].isin(common)]
        wide[arm] = frame.pivot_table(index="GAME_ID", columns="seed", values="y_pred")
    info = (
        rows[arm_a]
        .loc[rows[arm_a]["GAME_ID"].isin(common)]
        .drop_duplicates("GAME_ID")
        .set_index("GAME_ID")[
            [
                "date",
                "y_true",
                "target_line",
                "actual_outcome",
                "decimal_odds_over",
                "decimal_odds_under",
            ]
        ]
    )
    seeds = sorted(set(wide[arm_a].columns) & set(wide[arm_b].columns))
    games = info.loc[wide[arm_a].index].copy()
    games["date"] = pd.to_datetime(games["date"]).dt.normalize()
    games["pred_a"] = wide[arm_a][seeds].mean(axis=1)
    games["pred_b"] = wide[arm_b][seeds].mean(axis=1)
    games["abs_a"] = (games["y_true"] - games["pred_a"]).abs()
    games["abs_b"] = (games["y_true"] - games["pred_b"]).abs()
    games["delta"] = games["abs_a"] - games["abs_b"]
    games["sq_delta"] = (games["y_true"] - games["pred_a"]) ** 2 - (
        games["y_true"] - games["pred_b"]
    ) ** 2
    research = pd.read_parquet(args.research_file)
    research["GAME_ID"] = research["GAME_ID"].astype(str).str.zfill(10)
    research = research.set_index("GAME_ID")
    games["lu_impact"] = research["LU_ABSENCE_IMPACT_PTS_BEFORE"].reindex(games.index)
    games["pi_covered"] = (
        research["PI_ABSENCE_IMPACT_PRIOR_PTS_BEFORE"].reindex(games.index).notna()
    )
    games["season"] = season_of(games["date"])
    clusters = games["date"].to_numpy()

    # Baseline seed band.
    errors = {
        arm: (wide[arm][seeds].sub(games["y_true"], axis=0)).abs() for arm in wide
    }
    mae = {arm: errors[arm].mean() for arm in errors}
    band = [
        abs(mae[arm_a][s] - mae[arm_a][t])
        for i, s in enumerate(seeds)
        for t in seeds[i + 1 :]
    ]
    per_seed = np.array([mae[arm_a][s] - mae[arm_b][s] for s in seeds])
    mean, se = clustered_mean(games["delta"].to_numpy(), clusters)
    low, high = mean - 1.96 * se, mean + 1.96 * se
    sq_mean, sq_se = clustered_mean(games["sq_delta"].to_numpy(), clusters)

    def slope(prediction: str) -> float:
        x = games[prediction].to_numpy()
        y = games["y_true"].to_numpy()
        return float(np.polyfit(x, y, 1)[0])

    def slice_line(mask: pd.Series) -> dict:
        part = games.loc[mask]
        if len(part) < 2:
            return {"games": int(len(part))}
        m, s = clustered_mean(part["delta"].to_numpy(), part["date"].to_numpy())
        return {"games": int(len(part)), "delta": m, "se": s}

    first_day = games.groupby("season")["date"].transform("min")
    deadline = games["season"].map(
        lambda s: pd.Timestamp(TRADE_DEADLINES.get(s, "2100-01-01"))
    )
    slices = {
        f"|LU_ABSENCE_IMPACT_PTS_BEFORE| > {ABSENCE_SLICE:g}": games["lu_impact"].abs()
        > ABSENCE_SLICE,
        "no PI coverage (placebo)": ~games["pi_covered"],
        f"first {EARLY_DAYS} days of the season": (games["date"] - first_day).dt.days
        < EARLY_DAYS,
        "after the trade deadline": games["date"] > deadline,
    }
    metrics = {
        "complete": complete,
        "games": int(len(games)),
        "game_days": int(games["date"].nunique()),
        "window": [str(games["date"].min().date()), str(games["date"].max().date())],
        "seeds": seeds,
        "devices": sorted(devices),
        "features": {arm_a: len(feature_sets[arm_a]), "added_in_" + arm_b: added},
        "mae_per_seed": {arm: mae[arm].round(5).to_dict() for arm in mae},
        "baseline_seed_band": {
            "mae_sd": float(mae[arm_a].std(ddof=1)) if len(seeds) > 1 else None,
            "mae_range": float(mae[arm_a].max() - mae[arm_a].min()),
            "pair_abs_diff_median": float(np.median(band)) if band else None,
            "pair_abs_diff_max": float(max(band)) if band else None,
        },
        "ensemble_mae": {
            arm_a: float(games["abs_a"].mean()),
            arm_b: float(games["abs_b"].mean()),
        },
        "primary": {"delta": mean, "se_date_clustered": se, "ci95": [low, high]},
        "per_seed_delta": {
            "values": dict(zip(seeds, per_seed.round(5).tolist(), strict=True)),
            "mean": float(per_seed.mean()),
            "sd": float(per_seed.std(ddof=1)) if len(seeds) > 1 else None,
            "min": float(per_seed.min()),
            "max": float(per_seed.max()),
            "positive": int((per_seed > 0).sum()),
        },
        "decision": decide(low, high, per_seed),
        "secondary": {
            "squared_error_delta": {"delta": sq_mean, "se_date_clustered": sq_se},
            "calibration_slope": {arm_a: slope("pred_a"), arm_b: slope("pred_b")},
            "by_season": {
                int(s): slice_line(games["season"].eq(s))
                for s in sorted(games["season"].unique())
            },
            "slices": {name: slice_line(mask) for name, mask in slices.items()},
            "betting": {
                arm_a: betting(games, "pred_a"),
                arm_b: betting(games, "pred_b"),
            },
        },
    }
    report = args.out / "report"
    report.mkdir(parents=True, exist_ok=True)
    (report / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str) + "\n"
    )
    games.reset_index().to_parquet(report / "games.parquet", index=False)
    p = metrics["primary"]
    lines = [
        "# Phase 5: 2_5 vs 2_5 + PI block v1 (closing LINE_ERROR)",
        "",
        f"{metrics['games']:,} games on {metrics['game_days']} days, "
        f"{metrics['window'][0]} to {metrics['window'][1]}; seeds {seeds}"
        + ("" if complete else " (INCOMPLETE: smoke or partial run)"),
        "",
        f"**Decision: {metrics['decision']}.** Ensemble MAE {arm_a} "
        f"{metrics['ensemble_mae'][arm_a]:.4f}, {arm_b} {metrics['ensemble_mae'][arm_b]:.4f}; "
        f"paired delta {p['delta']:+.4f} (date-clustered SE {p['se_date_clustered']:.4f}, "
        f"95% [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]); per-seed deltas positive "
        f"{metrics['per_seed_delta']['positive']} of {len(seeds)} "
        f"(range {metrics['per_seed_delta']['min']:+.4f} to {metrics['per_seed_delta']['max']:+.4f}).",
        "",
        f"Baseline seed band: MAE range {metrics['baseline_seed_band']['mae_range']:.4f}, "
        f"median pairwise difference {metrics['baseline_seed_band']['pair_abs_diff_median']}.",
        "",
        "Secondary (no decisions):",
        "",
        "```",
        json.dumps(metrics["secondary"], indent=2, default=str),
        "```",
    ]
    (report / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:9]))
    print(f"written to {report}")


if __name__ == "__main__":
    main()
