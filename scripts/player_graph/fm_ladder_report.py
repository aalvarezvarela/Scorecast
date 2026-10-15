"""Report of the phase 6 FM ladder (development folds 2016-12 -> 2019-06 only).

Reads the out-of-fold predictions of ``run_fm_ladder.sh`` (four variants x
three seeds), the constant and RAPM baselines (``pretraining_baselines.py``)
and the pretraining labels, and writes
``artifacts/experiments/fm_ladder_dev/{report.md,metrics.json,folds.csv}``.

Declared before the runs:

* every model is scored on the same stint rows (those every model predicts);
* each variant's prediction is the mean over its seeds; the spread across
  seeds is the training-noise band (per-seed comparisons are paired by seed);
* gains are exposure-weighted drops in squared error with SEs clustered by
  game date (``pretraining_eval``), per target, overall, by season and by fold;
* comparisons: every model against the constant and RAPM; chemistry / style
  (``no_matchup`` vs ``additive``); matchups (``matchup`` vs ``no_matchup``);
  edge supervision (``matchup_edges`` vs ``matchup``); ``additive`` vs RAPM;
* edge targets (2017-11 -> 2019-06 folds) against a constant (pp-weighted
  mean, 365-day half-life) and an attacker norm (his decayed rate shrunk
  toward the constant with ``ATTACKER_PRIOR_PP`` partial possessions, a fixed
  reference value, not tuned).

Nothing here selects a model or a hyperparameter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nba_ou.data_processing.player_graph.edge_targets import TARGETS as EDGE_TARGETS
from nba_ou.data_processing.player_graph.pretraining_data import (
    ROOT,
    VERSION,
    load_stints,
)
from nba_ou.data_processing.player_graph.pretraining_eval import (
    paired_gain,
    target_errors,
)
from nba_ou.data_processing.player_graph.pretraining_oof import (
    PredictionStore,
    monthly_folds,
)
from nba_ou.data_processing.player_graph.stint_graph import LABELS

UNTIL = pd.Timestamp("2019-07-01")
VARIANTS = ("additive", "no_matchup", "matchup", "matchup_edges")
SEEDS = (0, 1, 2)
EDGE_HALF_LIFE = 365.0
ATTACKER_PRIOR_PP = 100.0
COMPARISONS = (
    ("rapm", "constant", "RAPM over constant"),
    ("additive", "constant", "additive FM over constant"),
    ("additive", "rapm", "additive FM over RAPM"),
    ("no_matchup", "rapm", "chemistry/style FM over RAPM"),
    ("matchup", "rapm", "matchup FM over RAPM"),
    ("matchup_edges", "rapm", "matchup + edges FM over RAPM"),
    ("no_matchup", "additive", "chemistry/style increment"),
    ("matchup", "no_matchup", "matchup increment"),
    ("matchup_edges", "matchup", "edge supervision increment"),
)
KEY_TARGETS = ("pts_per_poss", "poss_per_48")


def gains(labels, weights, dates, better, worse, mask=None) -> pd.DataFrame:
    if mask is not None:
        labels, weights, dates = labels[mask], weights[mask], dates[mask]
        better, worse = better[mask], worse[mask]
    return paired_gain(labels, weights, better, worse, dates, LABELS)


def stint_section(root: Path, out: dict, lines: list[str]) -> None:
    data = load_stints(root, range(2016, 2019), until=UNTIL)
    store = PredictionStore(root / "oof")
    read = lambda name: store.read(name, data.game_id, data.seg_idx)  # noqa: E731
    per_seed = {v: [read(f"fm_{v}_s{s}") for s in SEEDS] for v in VARIANTS}
    models = {"constant": read("constant"), "rapm": read("rapm")}
    models |= {v: np.mean(per_seed[v], axis=0) for v in VARIANTS}
    common = np.ones(data.labels.shape, bool)
    for values in [*models.values(), *(p for ps in per_seed.values() for p in ps)]:
        common &= np.isfinite(values)
    rows = common.any(axis=(1, 2))
    labels = data.labels[rows]
    weights = np.where(common, data.label_weight, 0.0)[rows]
    dates = data.game_date[rows]
    seasons = data.season[rows]
    models = {k: v[rows] for k, v in models.items()}
    per_seed = {k: [p[rows] for p in v] for k, v in per_seed.items()}
    lines.append(
        f"Stints scored: {int(rows.sum()):,} (2016-12 -> 2019-06), "
        "every model on the same rows.\n"
    )

    # 1. Errors per target.
    errors = {m: target_errors(labels, weights, v, LABELS) for m, v in models.items()}
    mse = pd.DataFrame({m: e["mse"] for m, e in errors.items()})
    relative = mse.div(mse["constant"], axis=0)
    out["mse"] = mse.to_dict()
    lines.append("## Out-of-fold exposure-weighted MSE per target\n")
    lines.append(mse.to_markdown(floatfmt=".5g") + "\n")
    lines.append("Relative to the constant baseline (1 = no better):\n")
    pooled = relative.mean().rename("mean over targets")
    lines.append(
        pd.concat([relative, pooled.to_frame().T]).to_markdown(floatfmt=".5f") + "\n"
    )

    # 2. Paired gains overall and by season.
    out["gains"] = {}
    lines.append("## Paired gains (% of the reference's MSE, SE clustered by date)\n")
    table = []
    for better, worse, label in COMPARISONS:
        for scope, mask in [("all", None)] + [
            (f"{s}-{(s + 1) % 100:02d}", seasons == s) for s in (2016, 2017, 2018)
        ]:
            g = gains(labels, weights, dates, models[better], models[worse], mask)
            out["gains"][f"{label} | {scope}"] = g.to_dict()
            for target, row in g.iterrows():
                table.append(
                    {
                        "comparison": label,
                        "scope": scope,
                        "target": target,
                        "gain %": row["gain_pct"],
                        "se %": row["se_pct"],
                        "t": row["t"],
                    }
                )
    table = pd.DataFrame(table)
    overall = (
        table.loc[table["scope"].eq("all")]
        .pivot(index="comparison", columns="target", values="gain %")
        .reindex([c[2] for c in COMPARISONS])[list(LABELS)]
    )
    tstat = (
        table.loc[table["scope"].eq("all")]
        .pivot(index="comparison", columns="target", values="t")
        .reindex([c[2] for c in COMPARISONS])[list(LABELS)]
    )
    lines.append("Gain %, all development folds:\n")
    lines.append(overall.to_markdown(floatfmt="+.3f") + "\n")
    lines.append("t statistics:\n")
    lines.append(tstat.to_markdown(floatfmt="+.1f") + "\n")
    lines.append("By season, points per possession and pace (gain % ± SE %):\n")
    key = table.loc[table["target"].isin(KEY_TARGETS)].copy()
    key["value"] = key.apply(lambda r: f"{r['gain %']:+.3f} ± {r['se %']:.3f}", axis=1)
    lines.append(
        key.pivot_table(
            index=["comparison", "target"],
            columns="scope",
            values="value",
            aggfunc="first",
        )
        .reindex(columns=["all", "2016-17", "2017-18", "2018-19"])
        .to_markdown()
        + "\n"
    )

    # 3. By fold, the key targets.
    folds = monthly_folds(dates, first="2016-12-01")
    fold_rows = []
    for fold in folds:
        _, mask = fold.masks(dates)
        for better, worse, label in COMPARISONS:
            g = gains(labels, weights, dates, models[better], models[worse], mask)
            for target in KEY_TARGETS:
                if target in g.index:
                    fold_rows.append(
                        {
                            "fold": fold.label,
                            "comparison": label,
                            "target": target,
                            "gain_pct": g.loc[target, "gain_pct"],
                            "se_pct": g.loc[target, "se_pct"],
                        }
                    )
    fold_table = pd.DataFrame(fold_rows)
    out["folds"] = fold_table.to_dict(orient="records")
    positive = (
        fold_table.assign(up=fold_table["gain_pct"] > 0)
        .groupby(["comparison", "target"])["up"]
        .agg(["sum", "count"])
    )
    lines.append("## By fold: folds with a positive gain (key targets)\n")
    lines.append(
        positive.reindex([c[2] for c in COMPARISONS], level=0).to_markdown() + "\n"
    )

    # 4. Seed band.
    lines.append("## Training noise: seeds\n")
    seed_rows, paired_rows = [], []
    for v in VARIANTS:
        per = [target_errors(labels, weights, p, LABELS)["mse"] for p in per_seed[v]]
        spread = pd.concat(per, axis=1).max(axis=1) - pd.concat(per, axis=1).min(axis=1)
        seed_rows.append((spread / mse["constant"] * 100).rename(v))
    band = pd.DataFrame(seed_rows)
    out["seed_spread_pct_of_constant_mse"] = band.to_dict()
    lines.append(
        "Spread of MSE across the three seeds of a variant, % of the constant's MSE:\n"
    )
    lines.append(band.to_markdown(floatfmt=".3f") + "\n")
    for better, worse, label in COMPARISONS:
        if better not in VARIANTS or worse not in VARIANTS:
            continue
        for s in SEEDS:
            g = gains(labels, weights, dates, per_seed[better][s], per_seed[worse][s])
            paired_rows.append(
                {"comparison": label, "seed": s, **g["gain_pct"].to_dict()}
            )
    paired = pd.DataFrame(paired_rows)
    out["per_seed_increments"] = paired.to_dict(orient="records")
    lines.append("Increments seed by seed (gain %, same seed in both variants):\n")
    lines.append(paired.to_markdown(index=False, floatfmt="+.3f") + "\n")


def _decayed(frame, cutoff, keys, columns):
    age = (cutoff - frame["game_date"]).dt.days.to_numpy(float)
    w = 0.5 ** (age / EDGE_HALF_LIFE)
    parts = frame[columns].mul(w, axis=0)
    return parts.groupby([frame[k] for k in keys]).sum() if keys else parts.sum()


def edge_report(root: Path, out: dict, lines: list[str], load_edges) -> None:
    edges = load_edges(root, (2017, 2018))
    frame = pd.DataFrame(
        {
            "game_id": edges.game_id,
            "off_player_id": edges.attacker,
            "game_date": pd.to_datetime(edges.game_date),
            "season": edges.season,
            "pp": edges.weight,
            **{t: edges.targets[:, k] for k, t in enumerate(EDGE_TARGETS)},
        }
    )
    frame = frame.loc[frame["game_date"] < UNTIL].reset_index(drop=True)
    for t in EDGE_TARGETS:
        frame[f"{t}_count"] = frame[t] * frame["pp"]
    folds = [f for f in monthly_folds(frame["game_date"], first="2016-12-01")]
    predictions = {"constant": {}, "attacker norm": {}}
    for fold in folds:
        train, predict = fold.masks(frame["game_date"])
        if not train.any() or not predict.any():
            continue
        history = frame.loc[train]
        counts = [f"{t}_count" for t in EDGE_TARGETS]
        league = _decayed(history, fold.checkpoint, [], [*counts, "pp"])
        own = _decayed(history, fold.checkpoint, ["off_player_id"], [*counts, "pp"])
        target_rows = frame.loc[predict]
        players = own.reindex(target_rows["off_player_id"]).fillna(0.0)
        for k, t in enumerate(EDGE_TARGETS):
            rate = league[counts[k]] / league["pp"]
            predictions["constant"].setdefault(t, {}).update(
                dict.fromkeys(target_rows.index, rate)
            )
            norm = (players[counts[k]].to_numpy() + ATTACKER_PRIOR_PP * rate) / (
                players["pp"].to_numpy() + ATTACKER_PRIOR_PP
            )
            predictions["attacker norm"].setdefault(t, {}).update(
                dict(zip(target_rows.index, norm, strict=True))
            )
    models = {
        name: np.column_stack(
            [pd.Series(values[t]).reindex(frame.index).to_numpy() for t in EDGE_TARGETS]
        )
        for name, values in predictions.items()
    }
    seeds = []
    for s in SEEDS:
        files = sorted(
            (root / "edges_oof" / f"fm_matchup_edges_s{s}").glob("fold=*.parquet")
        )
        fm = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        merged = frame[["game_id", "off_player_id"]].merge(
            fm, on=["game_id", "off_player_id"], how="left", validate="one_to_one"
        )
        seeds.append(merged[list(EDGE_TARGETS)].to_numpy(float))
    models["FM matchup + edges"] = np.mean(seeds, axis=0)
    common = np.ones(len(frame), bool)
    for values in [*models.values(), *seeds]:
        common &= np.isfinite(values).all(axis=1)
    y = frame[list(EDGE_TARGETS)].to_numpy(float)[common][:, None, :]
    w = np.broadcast_to(frame["pp"].to_numpy(float)[common][:, None, None], y.shape)
    w = np.ascontiguousarray(w)
    dates = frame["game_date"].to_numpy()[common]
    seasons = frame["season"].to_numpy()[common]
    pm = {k: v[common][:, None, :] for k, v in models.items()}
    lines.append(
        "## Edge targets (time-attributed, per attacker-game, weight = partial possessions)\n"
    )
    lines.append(
        f"Attacker-games scored: {int(common.sum()):,} (2017-11 -> 2019-06).\n"
    )
    mse = pd.DataFrame(
        {m: target_errors(y, w, v, EDGE_TARGETS)["mse"] for m, v in pm.items()}
    )
    out["edge_mse"] = mse.to_dict()
    lines.append(mse.to_markdown(floatfmt=".6g") + "\n")
    rows = []
    for better, worse in (
        ("attacker norm", "constant"),
        ("FM matchup + edges", "constant"),
        ("FM matchup + edges", "attacker norm"),
    ):
        for scope, mask in [("all", np.ones(len(y), bool))] + [
            (f"{s}-{(s + 1) % 100:02d}", seasons == s) for s in (2017, 2018)
        ]:
            g = paired_gain(
                y[mask],
                w[mask],
                pm[better][mask],
                pm[worse][mask],
                dates[mask],
                EDGE_TARGETS,
            )
            for t, r in g.iterrows():
                rows.append(
                    {
                        "comparison": f"{better} over {worse}",
                        "scope": scope,
                        "target": t,
                        "gain %": r["gain_pct"],
                        "se %": r["se_pct"],
                        "t": r["t"],
                    }
                )
    edge_table = pd.DataFrame(rows)
    out["edge_gains"] = edge_table.to_dict(orient="records")
    lines.append(edge_table.to_markdown(index=False, floatfmt="+.3f") + "\n")


def training_section(root: Path, out: dict, lines: list[str]) -> None:
    rows, path_rows = [], []
    for v in VARIANTS:
        for s in SEEDS:
            logs = sorted((root / "fm" / f"fm_{v}_s{s}").glob("fold=*.json"))
            for path in logs:
                log = json.loads(path.read_text())
                hist = log["history"]
                stint = [h["stint"] for h in hist]
                rows.append(
                    {
                        "variant": v,
                        "seed": s,
                        "fold": log["fold"],
                        "seconds": log["seconds"],
                        "train_stints": log["train_stints"],
                        "first_epoch": stint[0],
                        "last_epoch": stint[-1],
                        "edge_last": hist[-1]["edge"],
                        "finite": bool(np.isfinite(stint).all()),
                        "rising": bool(stint[-1] > stint[0] + 1e-3),
                    }
                )
    table = pd.DataFrame(rows)
    out["training"] = table.to_dict(orient="records")
    summary = (
        table.groupby("variant")
        .agg(
            minutes=("seconds", lambda x: x.sum() / 60 / len(SEEDS)),
            folds=("fold", "nunique"),
            non_finite=("finite", lambda x: int((~x).sum())),
            rising_folds=("rising", "sum"),
            last_epoch_loss_last_fold=("last_epoch", "last"),
        )
        .reindex(list(VARIANTS))
    )
    lines.append("## Training\n")
    lines.append(
        "Minutes per seed, folds, folds with a non-finite loss, folds whose "
        "training loss rose over the epochs:\n"
    )
    lines.append(summary.to_markdown(floatfmt=".3f") + "\n")
    path_rows = table.loc[table["seed"].eq(0)].pivot(
        index="fold", columns="variant", values="last_epoch"
    )[list(VARIANTS)]
    out["train_loss_by_fold_seed0"] = path_rows.to_dict()
    lines.append(
        "Standardized training loss (stint targets, last epoch, seed 0) by fold:\n"
    )
    lines.append(path_rows.to_markdown(floatfmt=".4f") + "\n")


def overfit_section(root: Path, out: dict, lines: list[str]) -> None:
    """Training loss vs the same fold's out-of-fold loss, both standardized
    with the fold's own training scaler (seed 0)."""
    data = load_stints(root, range(2016, 2019), until=UNTIL)
    store = PredictionStore(root / "oof")
    rows = []
    for v in VARIANTS:
        name = f"fm_{v}_s0"
        predictions = store.read(name, data.game_id, data.seg_idx)
        for path in sorted((root / "fm" / name).glob("fold=*.pt")):
            payload = torch.load(path, weights_only=True)
            label = payload["extra"]["fold"]
            fold = next(f for f in monthly_folds(data.game_date) if f.label == label)
            _, mask = fold.masks(data.game_date)
            std = np.asarray(payload["standardization"]["targets"]["std"])
            e = (data.labels[mask] - predictions[mask]) / std
            w = np.where(np.isfinite(e), data.label_weight[mask], 0.0)
            per_target = (np.nan_to_num(e) ** 2 * w).sum((0, 1)) / w.sum((0, 1))
            rows.append(
                {
                    "variant": v,
                    "fold": label,
                    "train": payload["extra"]["history"][-1]["stint"],
                    "out_of_fold": float(per_target.mean()),
                }
            )
    table = pd.DataFrame(rows)
    table["gap"] = table["out_of_fold"] - table["train"]
    out["overfit"] = table.to_dict(orient="records")
    lines.append(
        "## Training vs out-of-fold loss (standardized, mean over targets, seed 0)\n"
    )
    lines.append(
        table.groupby("variant")[["train", "out_of_fold", "gap"]]
        .mean()
        .reindex(list(VARIANTS))
        .to_markdown(floatfmt=".4f")
        + "\n"
    )
    lines.append(
        table.pivot(index="fold", columns="variant", values="gap")[
            list(VARIANTS)
        ].to_markdown(floatfmt="+.4f")
        + "\n"
    )


def main() -> None:
    import importlib.util

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/experiments/fm_ladder_dev")
    )
    args = parser.parse_args()
    root = args.local_root / ROOT / VERSION
    spec = importlib.util.spec_from_file_location(
        "fm_oof", Path(__file__).with_name("fm_oof.py")
    )
    fm_oof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fm_oof)
    out: dict = {
        "until": str(UNTIL.date()),
        "variants": VARIANTS,
        "seeds": SEEDS,
        "attacker_prior_pp": ATTACKER_PRIOR_PP,
    }
    lines = ["# Phase 6 FM ladder, development folds 2016-12 -> 2019-06\n"]
    stint_section(root, out, lines)
    edge_report(root, out, lines, fm_oof.load_edges)
    training_section(root, out, lines)
    overfit_section(root, out, lines)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text("\n".join(lines))
    (args.out / "metrics.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n"
    )
    pd.DataFrame(out["folds"]).to_csv(args.out / "folds.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
