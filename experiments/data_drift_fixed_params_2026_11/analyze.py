"""Read data_drift_fixed_params_2026_11 as pre-registered (README.md).

For each horizon and each of the five seeds (the main cell's primary seed plus
the four seed-replicate cells), pair the old and new arms game by game, on CV
and on the holdout separately, and report new - old for MAE and for the
directional win rate at each minimum edge. Two intervals:

- seed: a t interval over the five per-seed paired differences (how much the
  difference moves with the seed, on these games);
- game: the per-game |error| difference averaged over the five seeds, with its
  game-level standard error (how much it would move with other games).

The seed noise floor is the spread of each arm's own metric across the five
seeds. Also checks that each replicate's holdout reproduces the main cell's
evaluation seed, and compares the arms' feature lists.

    .venv/bin/python experiments/data_drift_fixed_params_2026_11/analyze.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = "data_drift_fixed_params_2026_11"
ROOT = REPO / "artifacts" / "experiments" / CAMPAIGN
OUT = Path(__file__).resolve().parent / "results"
CELLS = ["close", "t480", "t720"]
PRIMARY = {"close": 17, "t480": 16, "t720": 16}
SEEDS = [101, 202, 303, 404]
EDGES = [0.0, 0.1, 0.5, 1.0, 2.0]
PRIMARY_EDGE = 0.1
T975_DF4 = 2.7764451051977987


def run_dir(name: str) -> Path | None:
    pattern = re.compile(rf"^{re.escape(name)}_\d{{8}}_\d{{6}}$")
    found = sorted(p for p in ROOT.iterdir() if pattern.match(p.name)) if ROOT.exists() else []
    return found[-1] if found else None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def betting(df: pd.DataFrame, edge: float) -> dict:
    """Win rate excludes pushes; ROI at -110 over every bet, pushes included
    (as training_pipeline's betting_sweep.csv computes them)."""
    bet = df["y_pred"].abs() >= edge
    push = df["y_true"] == 0
    win = bet & ~push & (np.sign(df["y_pred"]) == np.sign(df["y_true"]))
    loss = bet & ~push & ~win
    n_bets, k, l = int(bet.sum()), int(win.sum()), int(loss.sum())
    profit = k * (100 / 110) - l
    return dict(
        n_bets=n_bets,
        wins=k,
        losses=l,
        win_rate=k / (k + l) if k + l else math.nan,
        roi=profit / n_bets if n_bets else math.nan,
    )


def keyed(df: pd.DataFrame, date_col: str, by_fold: bool) -> pd.DataFrame:
    """Key each game by (fold,) date and its position within the date: both
    files keep the same order, and positions in the dev frame shift when the
    NaN budget keeps a different early game."""
    df = df.copy()
    groups = (["fold"] if by_fold else []) + [date_col]
    df["_pos"] = df.groupby(groups).cumcount()
    df["_key"] = df[groups + ["_pos"]].astype(str).agg("|".join, axis=1)
    return df.set_index("_key")


def pair(old: pd.DataFrame, new: pd.DataFrame, date_col: str, by_fold: bool) -> pd.DataFrame:
    o, n = keyed(old, date_col, by_fold), keyed(new, date_col, by_fold)
    if set(o.index) != set(n.index):
        raise SystemExit(f"arms hold different games: {len(set(o.index) ^ set(n.index))}")
    n = n.loc[o.index]
    if not (o["TOTAL_POINTS"].to_numpy() == n["TOTAL_POINTS"].to_numpy()).all():
        raise SystemExit("paired rows disagree on TOTAL_POINTS")
    return pd.DataFrame(
        {
            "y_true_old": o["y_true"].to_numpy(),
            "y_pred_old": o["y_pred"].to_numpy(),
            "y_true_new": n["y_true"].to_numpy(),
            "y_pred_new": n["y_pred"].to_numpy(),
            "line_old": o["target_line"].to_numpy(),
            "line_new": n["target_line"].to_numpy(),
        },
        index=o.index,
    )


def metrics(df: pd.DataFrame) -> dict:
    out = {"mae": float((df["y_true"] - df["y_pred"]).abs().mean())}
    for edge in EDGES:
        out[f"edge_{edge}"] = betting(df, edge)
    return out


def summarize(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    sd = float(a.std(ddof=1))
    half = T975_DF4 * sd / math.sqrt(len(a))
    return dict(
        mean=float(a.mean()),
        sd=sd,
        ci95=[float(a.mean() - half), float(a.mean() + half)],
        min=float(a.min()),
        max=float(a.max()),
        values=[float(v) for v in a],
    )


def analyze_cell(cell: str) -> dict:
    runs = {}
    for arm in ("old", "new"):
        runs[(arm, PRIMARY[cell])] = run_dir(f"drift_{cell}_{arm}")
        for seed in SEEDS:
            runs[(arm, seed)] = run_dir(f"drift_{cell}_{arm}_s{seed}")
    missing = [f"{arm}/{seed}" for (arm, seed), d in runs.items() if d is None]
    if missing:
        return {"missing_runs": missing}

    seeds = [PRIMARY[cell], *SEEDS]
    res: dict = {"seeds": seeds, "runs": {f"{a}_{s}": str(d.relative_to(REPO)) for (a, s), d in runs.items()}}

    # Reproducibility: a replicate's holdout must equal the main cell's
    # evaluation seed, which ran the same fit.
    repro = {}
    for arm in ("old", "new"):
        stab = pd.read_csv(runs[(arm, PRIMARY[cell])] / "seed_stability.csv").set_index("random_state")
        for seed in SEEDS:
            # No evaluation seeds, so no seed_stability.csv: its primary seed's
            # holdout predictions are that seed's fit.
            rep = pd.read_parquet(runs[(arm, seed)] / "final_test_predictions.parquet")
            rep_mae = float((rep["y_true"] - rep["y_pred"]).abs().mean())
            repro[f"{arm}_{seed}"] = float(abs(rep_mae - stab.loc[seed, "mae"]))
    res["replicate_vs_eval_seed_max_abs_mae_diff"] = max(repro.values())

    # Feature lists, from each run's own feature_schema.json.
    feats = {
        arm: json.load(open(runs[(arm, PRIMARY[cell])] / "feature_schema.json"))["feature_names"]
        for arm in ("old", "new")
    }
    res["features"] = {
        "n_old": len(feats["old"]),
        "n_new": len(feats["new"]),
        "identical": feats["old"] == feats["new"],
        "only_old": sorted(set(feats["old"]) - set(feats["new"])),
        "only_new": sorted(set(feats["new"]) - set(feats["old"])),
    }

    for split, file, date_col, by_fold in (
        ("holdout", "final_test_predictions.parquet", "date", False),
        ("cv", "cv_predictions.parquet", "GAME_DATE", True),
    ):
        per_seed = {"old": [], "new": [], "diff": []}
        game_diffs = []
        side_changed = []
        n_line_changed = None
        for seed in seeds:
            old = pd.read_parquet(runs[("old", seed)] / file)
            new = pd.read_parquet(runs[("new", seed)] / file)
            p = pair(old, new, date_col, by_fold)
            n_line_changed = int((p["line_old"] != p["line_new"]).sum())
            mo = metrics(old)
            mn = metrics(new)
            per_seed["old"].append(mo)
            per_seed["new"].append(mn)
            game_diffs.append(
                (p["y_true_new"] - p["y_pred_new"]).abs() - (p["y_true_old"] - p["y_pred_old"]).abs()
            )
            side_changed.append(float((np.sign(p["y_pred_new"]) != np.sign(p["y_pred_old"])).mean()))
        n_games = len(game_diffs[0])
        out = {"n_games": n_games, "n_games_line_changed": n_line_changed}
        out["mae_old"] = summarize([m["mae"] for m in per_seed["old"]])
        out["mae_new"] = summarize([m["mae"] for m in per_seed["new"]])
        out["mae_diff_by_seed"] = summarize(
            [n["mae"] - o["mae"] for o, n in zip(per_seed["old"], per_seed["new"], strict=True)]
        )
        g = pd.concat(game_diffs, axis=1).mean(axis=1)
        out["mae_diff_game_level"] = dict(
            mean=float(g.mean()),
            se=float(g.std(ddof=1) / math.sqrt(len(g))),
        )
        out["side_changed_rate"] = summarize(side_changed)
        for edge in EDGES:
            k = f"edge_{edge}"
            wo = [m[k]["win_rate"] for m in per_seed["old"]]
            wn = [m[k]["win_rate"] for m in per_seed["new"]]
            ro = [m[k]["roi"] for m in per_seed["old"]]
            rn = [m[k]["roi"] for m in per_seed["new"]]
            out[k] = dict(
                win_old=summarize(wo),
                win_new=summarize(wn),
                win_diff_by_seed=summarize([b - a for a, b in zip(wo, wn, strict=True)]),
                roi_old=summarize(ro),
                roi_new=summarize(rn),
                roi_diff_by_seed=summarize([b - a for a, b in zip(ro, rn, strict=True)]),
                bets_old=[m[k]["n_bets"] for m in per_seed["old"]],
                bets_new=[m[k]["n_bets"] for m in per_seed["new"]],
                wilson_old_primary=wilson(per_seed["old"][0][k]["wins"], per_seed["old"][0][k]["wins"] + per_seed["old"][0][k]["losses"]),
                wilson_new_primary=wilson(per_seed["new"][0][k]["wins"], per_seed["new"][0][k]["wins"] + per_seed["new"][0][k]["losses"]),
            )
        res[split] = out

    # The framework's own betting_sweep.csv must agree with this module's
    # betting() on the primary seed, or the edge definitions differ.
    sweep = pd.read_csv(runs[("old", PRIMARY[cell])] / "betting_sweep.csv").set_index("min_edge")
    mine = res["holdout"][f"edge_{PRIMARY_EDGE}"]["win_old"]["values"][0]
    res["sweep_check"] = float(abs(sweep.loc[PRIMARY_EDGE, "win_rate"] - mine))
    return res


def fmt(s: dict, scale: float = 1.0, digits: int = 3) -> str:
    return (
        f"{s['mean'] * scale:+.{digits}f} [{s['ci95'][0] * scale:+.{digits}f}, "
        f"{s['ci95'][1] * scale:+.{digits}f}]"
    )


def table(results: dict) -> str:
    lines = [
        "| slot | split | games | MAE old (seed range) | MAE new (seed range) | MAE new-old, seeds [95% CI] | MAE new-old, games (SE) | win% @0.1 old (range) | win% @0.1 new (range) | win pp new-old [95% CI] | side changed |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cell, r in results.items():
        if "missing_runs" in r:
            lines.append(f"| {cell} | missing: {', '.join(r['missing_runs'])} |")
            continue
        for split in ("cv", "holdout"):
            s = r[split]
            e = s[f"edge_{PRIMARY_EDGE}"]
            lines.append(
                f"| {cell} | {split} | {s['n_games']} "
                f"| {s['mae_old']['mean']:.3f} ({s['mae_old']['min']:.3f}-{s['mae_old']['max']:.3f}) "
                f"| {s['mae_new']['mean']:.3f} ({s['mae_new']['min']:.3f}-{s['mae_new']['max']:.3f}) "
                f"| {fmt(s['mae_diff_by_seed'])} "
                f"| {s['mae_diff_game_level']['mean']:+.3f} ({s['mae_diff_game_level']['se']:.3f}) "
                f"| {100 * e['win_old']['mean']:.1f} ({100 * e['win_old']['min']:.1f}-{100 * e['win_old']['max']:.1f}) "
                f"| {100 * e['win_new']['mean']:.1f} ({100 * e['win_new']['min']:.1f}-{100 * e['win_new']['max']:.1f}) "
                f"| {fmt(e['win_diff_by_seed'], 100, 1)} "
                f"| {100 * s['side_changed_rate']['mean']:.0f}% |"
            )
    return "\n".join(lines)


def edge_table(results: dict) -> str:
    lines = [
        "| slot | split | min edge | bets old/new (primary seed) | win% old (5-seed mean) | win% new | win pp new-old [95% CI] | ROI pp new-old [95% CI] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cell, r in results.items():
        if "missing_runs" in r:
            continue
        for split in ("cv", "holdout"):
            for edge in EDGES:
                e = r[split][f"edge_{edge}"]
                lines.append(
                    f"| {cell} | {split} | {edge} | {e['bets_old'][0]}/{e['bets_new'][0]} "
                    f"| {100 * e['win_old']['mean']:.1f} | {100 * e['win_new']['mean']:.1f} "
                    f"| {fmt(e['win_diff_by_seed'], 100, 1)} | {fmt(e['roi_diff_by_seed'], 100, 1)} |"
                )
    return "\n".join(lines)


def main() -> int:
    results = {cell: analyze_cell(cell) for cell in CELLS}
    OUT.mkdir(exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(results, indent=1, default=float))
    text = (
        "# data_drift_fixed_params_2026_11: results\n\n"
        "Generated by `analyze.py`. new - old: negative MAE is better for new, "
        "positive win pp is better for new. Seeds: primary (17 close / 16 "
        "intermediate) + 101, 202, 303, 404; CIs are t intervals over the five "
        "paired seed differences.\n\n## Primary threshold (0.1 points)\n\n"
        + table(results)
        + "\n\n## Minimum-edge variants\n\n"
        + edge_table(results)
        + "\n\n## Checks\n\n"
    )
    for cell, r in results.items():
        if "missing_runs" in r:
            continue
        f = r["features"]
        text += (
            f"- {cell}: features old {f['n_old']}, new {f['n_new']}, identical {f['identical']}; "
            f"only old {f['only_old']}; only new {f['only_new']}. "
            f"Replicate vs evaluation seed max |MAE diff| {r['replicate_vs_eval_seed_max_abs_mae_diff']:.2e}; "
            f"betting_sweep check {r['sweep_check']:.2e}; games with a changed line: "
            f"holdout {r['holdout']['n_games_line_changed']}, CV {r['cv']['n_games_line_changed']}.\n"
        )
    (OUT / "summary.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
