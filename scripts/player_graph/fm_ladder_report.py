"""Reports of the phase 6 FM ladders (development folds 2016-12 -> 2019-06 only).

``--ladder v1`` (``run_fm_ladder.sh``): the four FM variants trained from
scratch against the constant and the monthly RAPM baselines.
``--ladder residual`` (``run_fm_residual_ladder.sh``, decided 2026-10-15): the
same variants learning residuals over the daily as-of RAPM, against RAPM
alone, and the edge-head ablation (edge-only models: attacker; + defenders;
+ matchup product). Each ladder is declared in ``LADDERS``.

Reads the out-of-fold predictions (three seeds per variant), the baselines
(``pretraining_baselines.py``, ``pretraining_daily_rapm.py``) and the
pretraining labels; writes ``<out>/{report.md,metrics.json,folds.csv}``.

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
from dataclasses import dataclass, field
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
SEEDS = (0, 1, 2)
EDGE_HALF_LIFE = 365.0
ATTACKER_PRIOR_PP = 100.0
KEY_TARGETS = ("pts_per_poss", "poss_per_48")


@dataclass(frozen=True)
class Ladder:
    """Variants (run ``fm_<variant>_s<seed>``), baselines (label -> stored
    model, and whether its NaN targets take the constant), seed-averaged
    references from other ladders, comparisons ``(better, worse, label)``,
    and the edge models (label -> run prefix) with their comparisons."""

    title: str
    out: str
    variants: tuple[str, ...]
    baselines: dict[str, tuple[str, bool]]
    comparisons: tuple[tuple[str, str, str], ...]
    edge_models: dict[str, str]
    edge_comparisons: tuple[tuple[str, str], ...]
    references: tuple[str, ...] = field(default=())


LADDERS = {
    "v1": Ladder(
        title="Phase 6 FM ladder, development folds 2016-12 -> 2019-06",
        out="artifacts/experiments/fm_ladder_dev",
        variants=("additive", "no_matchup", "matchup", "matchup_edges"),
        baselines={"constant": ("constant", False), "rapm": ("rapm", False)},
        comparisons=(
            ("rapm", "constant", "RAPM over constant"),
            ("additive", "constant", "additive FM over constant"),
            ("additive", "rapm", "additive FM over RAPM"),
            ("no_matchup", "rapm", "chemistry/style FM over RAPM"),
            ("matchup", "rapm", "matchup FM over RAPM"),
            ("matchup_edges", "rapm", "matchup + edges FM over RAPM"),
            ("no_matchup", "additive", "chemistry/style increment"),
            ("matchup", "no_matchup", "matchup increment"),
            ("matchup_edges", "matchup", "edge supervision increment"),
        ),
        edge_models={"FM matchup + edges": "fm_matchup_edges"},
        edge_comparisons=(
            ("attacker norm", "constant"),
            ("FM matchup + edges", "constant"),
            ("FM matchup + edges", "attacker norm"),
        ),
    ),
    "residual": Ladder(
        title="Phase 6 FM residual ladder over daily RAPM, development folds "
        "2016-12 -> 2019-06",
        out="artifacts/experiments/fm_residual_ladder_dev",
        variants=("res_additive", "res_no_matchup", "res_matchup", "res_matchup_edges"),
        baselines={"constant": ("constant", False), "rapm": ("rapm_daily", True)},
        references=("additive", "matchup"),
        comparisons=(
            ("rapm", "constant", "1. RAPM (daily) over constant"),
            ("res_additive", "rapm", "2. RAPM + additive residual over RAPM"),
            ("res_no_matchup", "rapm", "3. RAPM + chemistry/style over RAPM"),
            ("res_matchup", "rapm", "4. RAPM + matchup over RAPM"),
            ("res_matchup_edges", "rapm", "5. RAPM + matchup + edges over RAPM"),
            ("res_no_matchup", "res_additive", "chemistry/style increment (3 vs 2)"),
            ("res_matchup", "res_no_matchup", "matchup increment (4 vs 3)"),
            ("res_matchup", "res_additive", "all interactions (4 vs 2)"),
            ("res_matchup_edges", "res_matchup", "edge supervision increment (5 vs 4)"),
            ("res_additive", "additive", "residual vs from scratch, additive"),
            ("res_matchup", "matchup", "residual vs from scratch, matchup"),
        ),
        edge_models={
            "edge-only: attacker": "fm_edge_attacker",
            "edge-only: + defenders": "fm_edge_defender",
            "edge-only: + matchup": "fm_edge_matchup",
            "joint: residual matchup + edges": "fm_res_matchup_edges",
        },
        edge_comparisons=(
            ("attacker norm", "constant"),
            ("edge-only: attacker", "attacker norm"),
            ("edge-only: + defenders", "edge-only: attacker"),
            ("edge-only: + matchup", "edge-only: + defenders"),
            ("edge-only: + matchup", "edge-only: attacker"),
            ("joint: residual matchup + edges", "edge-only: + matchup"),
        ),
    ),
    "interaction": Ladder(
        title="Phase 6 interaction-only FM residuals over daily RAPM (raw and "
        "calibrated), development folds 2016-12 -> 2019-06",
        out="artifacts/experiments/fm_interaction_ladder_dev",
        variants=(
            "int_control",
            "int_chemistry_style",
            "int_matchup",
            "int_control_cal",
            "int_chemistry_style_cal",
            "int_matchup_cal",
        ),
        baselines={"constant": ("constant", False), "rapm": ("rapm_daily", True)},
        references=("res_additive", "res_additive_cal"),
        comparisons=(
            ("rapm", "constant", "1. RAPM (daily) over constant"),
            ("int_control", "rapm", "2. + intercept / home residual, raw, over RAPM"),
            ("int_chemistry_style", "rapm", "3. + chemistry/style, raw, over RAPM"),
            ("int_matchup", "rapm", "4. + matchup, raw, over RAPM"),
            ("int_control_cal", "rapm", "2c. calibrated, over RAPM"),
            ("int_chemistry_style_cal", "rapm", "3c. calibrated, over RAPM"),
            ("int_matchup_cal", "rapm", "4c. calibrated, over RAPM"),
            ("int_chemistry_style", "int_control", "chemistry/style increment, raw"),
            ("int_matchup", "int_chemistry_style", "matchup increment, raw"),
            ("int_matchup", "int_control", "all interactions, raw"),
            (
                "int_chemistry_style_cal",
                "int_control_cal",
                "chemistry/style increment, calibrated",
            ),
            (
                "int_matchup_cal",
                "int_chemistry_style_cal",
                "matchup increment, calibrated",
            ),
            ("int_matchup_cal", "int_control_cal", "all interactions, calibrated"),
            ("int_matchup_cal", "int_matchup", "calibration effect (matchup)"),
            (
                "res_additive_cal",
                "rapm",
                "step 4's additive residual, calibrated, over RAPM",
            ),
        ),
        edge_models={
            "attacker": "fm_edge_attacker",
            "capacity control": "fm_edge_capacity",
            "capacity + defenders": "fm_edge_capacity_defender",
            "capacity + defenders + matchup": "fm_edge_capacity_matchup",
            "defenders + matchup (no control)": "fm_edge_matchup",
        },
        edge_comparisons=(
            ("attacker", "attacker norm"),
            ("capacity control", "attacker"),
            ("capacity + defenders", "capacity control"),
            ("capacity + defenders + matchup", "capacity + defenders"),
            ("capacity + defenders + matchup", "capacity control"),
            ("defenders + matchup (no control)", "capacity + defenders + matchup"),
        ),
    ),
    "gnn": Ladder(
        title="Phase 6 GNN ladder: interaction-only residuals over daily RAPM "
        "(raw and calibrated), development folds 2016-12 -> 2019-06",
        out="artifacts/experiments/gnn_ladder_dev",
        variants=(
            "gnn_control",
            "gnn_node",
            "gnn_no_guard",
            "gnn_full",
            "gnn_control_cal",
            "gnn_node_cal",
            "gnn_no_guard_cal",
            "gnn_full_cal",
        ),
        baselines={"constant": ("constant", False), "rapm": ("rapm_daily", True)},
        references=("int_matchup_cal",),
        comparisons=(
            ("rapm", "constant", "A. RAPM (daily) over constant"),
            ("gnn_control", "rapm", "B. + intercept / home, raw, over RAPM"),
            ("gnn_node", "rapm", "C. + node-only neural, raw, over RAPM"),
            ("gnn_no_guard", "rapm", "D. + GNN teammate/opponent, raw, over RAPM"),
            ("gnn_full", "rapm", "E. + full GNN, raw, over RAPM"),
            ("gnn_control_cal", "rapm", "Bc. calibrated, over RAPM"),
            ("gnn_node_cal", "rapm", "Cc. calibrated, over RAPM"),
            ("gnn_no_guard_cal", "rapm", "Dc. calibrated, over RAPM"),
            ("gnn_full_cal", "rapm", "Ec. calibrated, over RAPM"),
            ("gnn_node", "gnn_control", "neural capacity (C vs B), raw"),
            ("gnn_no_guard", "gnn_node", "teammate/opponent messages (D vs C), raw"),
            ("gnn_full", "gnn_no_guard", "guard edges (E vs D), raw"),
            ("gnn_node_cal", "gnn_control_cal", "neural capacity (C vs B), calibrated"),
            (
                "gnn_no_guard_cal",
                "gnn_node_cal",
                "teammate/opponent messages (D vs C), calibrated",
            ),
            ("gnn_full_cal", "gnn_no_guard_cal", "guard edges (E vs D), calibrated"),
            (
                "gnn_full_cal",
                "gnn_node_cal",
                "all message passing (E vs C), calibrated",
            ),
            ("gnn_full_cal", "int_matchup_cal", "full GNN vs FM matchup, calibrated"),
        ),
        edge_models={},
        edge_comparisons=(),
    ),
}


def gains(labels, weights, dates, better, worse, mask=None) -> pd.DataFrame:
    if mask is not None:
        labels, weights, dates = labels[mask], weights[mask], dates[mask]
        better, worse = better[mask], worse[mask]
    return paired_gain(labels, weights, better, worse, dates, LABELS)


def stint_section(ladder: Ladder, root: Path, out: dict, lines: list[str]) -> None:
    VARIANTS, COMPARISONS = ladder.variants, ladder.comparisons  # noqa: N806
    data = load_stints(root, range(2016, 2019), until=UNTIL)
    store = PredictionStore(root / "oof")
    read = lambda name: store.read(name, data.game_id, data.seg_idx)  # noqa: E731
    per_seed = {v: [read(f"fm_{v}_s{s}") for s in SEEDS] for v in VARIANTS}
    constant = read("constant")
    models = {}
    for label, (stored, fill) in ladder.baselines.items():
        values = read(stored)
        models[label] = np.where(np.isnan(values), constant, values) if fill else values
    models |= {v: np.mean(per_seed[v], axis=0) for v in VARIANTS}
    models |= {
        r: np.mean([read(f"fm_{r}_s{s}") for s in SEEDS], axis=0)
        for r in ladder.references
    }
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


def edge_report(
    ladder: Ladder, root: Path, out: dict, lines: list[str], load_edges
) -> None:
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
    per_seed = {}
    for label, prefix in ladder.edge_models.items():
        per_seed[label] = []
        for s in SEEDS:
            files = sorted(
                (root / "edges_oof" / f"{prefix}_s{s}").glob("fold=*.parquet")
            )
            fm = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            merged = frame[["game_id", "off_player_id"]].merge(
                fm, on=["game_id", "off_player_id"], how="left", validate="one_to_one"
            )
            per_seed[label].append(merged[list(EDGE_TARGETS)].to_numpy(float))
        models[label] = np.mean(per_seed[label], axis=0)
    seeds = [v for values in per_seed.values() for v in values]
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
    for better, worse in ladder.edge_comparisons:
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
    paired = []
    for better, worse in ladder.edge_comparisons:
        if better not in per_seed or worse not in per_seed:
            continue
        for s in SEEDS:
            g = paired_gain(
                y,
                w,
                per_seed[better][s][common][:, None, :],
                per_seed[worse][s][common][:, None, :],
                dates,
                EDGE_TARGETS,
            )
            paired.append(
                {"comparison": f"{better} over {worse}", "seed": s}
                | g["gain_pct"].to_dict()
            )
    if paired:
        out["edge_per_seed"] = paired
        lines.append("Edge increments seed by seed (gain %, paired seeds):\n")
        lines.append(
            pd.DataFrame(paired).to_markdown(index=False, floatfmt="+.3f") + "\n"
        )


def training_section(ladder: Ladder, root: Path, out: dict, lines: list[str]) -> None:
    VARIANTS = (  # noqa: N806
        *ladder.variants,
        *(p.removeprefix("fm_") for p in ladder.edge_models.values()),
    )
    VARIANTS = tuple(dict.fromkeys(VARIANTS))  # noqa: N806
    rows, path_rows = [], []
    for v in VARIANTS:
        for s in SEEDS:
            logs = sorted((root / "fm" / f"fm_{v}_s{s}").glob("fold=*.json"))
            for path in logs:
                log = json.loads(path.read_text())
                hist = log["history"]
                stint = [h["stint"] or h["edge"] for h in hist]
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
    path_rows = (
        table.loc[table["seed"].eq(0)]
        .pivot(index="fold", columns="variant", values="last_epoch")
        .reindex(columns=[v for v in VARIANTS if v in set(table["variant"])])
    )
    out["train_loss_by_fold_seed0"] = path_rows.to_dict()
    lines.append(
        "Standardized training loss (stint targets, last epoch, seed 0) by fold:\n"
    )
    lines.append(path_rows.to_markdown(floatfmt=".4f") + "\n")


def overfit_section(ladder: Ladder, root: Path, out: dict, lines: list[str]) -> None:
    """Training loss vs the same fold's out-of-fold loss, both standardized
    with the fold's own training scaler (seed 0)."""
    VARIANTS = ladder.variants  # noqa: N806
    data = load_stints(root, range(2016, 2019), until=UNTIL)
    store = PredictionStore(root / "oof")
    rows = []
    for v in VARIANTS:
        name = f"fm_{v}_s0"
        if not (root / "fm" / name).exists():  # calibrated: no training of its own
            continue
        predictions = store.read(name, data.game_id, data.seg_idx)
        for path in sorted((root / "fm" / name).glob("fold=*.pt")):
            payload = torch.load(path, weights_only=True)
            label = payload["extra"]["fold"]
            if payload["extra"].get("uncorrected_residual_targets"):
                # No training residual yet: its scaler is the identity, so the
                # standardized loss is not comparable.
                continue
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
        "Folds whose residual targets had no training data (the first) are left "
        "out.\n"
    )
    lines.append(
        table.groupby("variant")[["train", "out_of_fold", "gap"]]
        .mean()
        .reindex([v for v in VARIANTS if v in set(table["variant"])])
        .to_markdown(floatfmt=".4f")
        + "\n"
    )
    lines.append(
        table.pivot(index="fold", columns="variant", values="gap")
        .reindex(columns=[v for v in VARIANTS if v in set(table["variant"])])
        .to_markdown(floatfmt="+.4f")
        + "\n"
    )


def main() -> None:
    import importlib.util

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--ladder", choices=list(LADDERS), default="v1")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    ladder = LADDERS[args.ladder]
    args.out = args.out or Path(ladder.out)
    root = args.local_root / ROOT / VERSION
    spec = importlib.util.spec_from_file_location(
        "fm_oof", Path(__file__).with_name("fm_oof.py")
    )
    fm_oof = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fm_oof)
    out: dict = {
        "until": str(UNTIL.date()),
        "ladder": args.ladder,
        "variants": ladder.variants,
        "edge_models": ladder.edge_models,
        "seeds": SEEDS,
        "attacker_prior_pp": ATTACKER_PRIOR_PP,
    }
    lines = [f"# {ladder.title}\n"]
    stint_section(ladder, root, out, lines)
    if ladder.edge_models:
        edge_report(ladder, root, out, lines, fm_oof.load_edges)
    training_section(ladder, root, out, lines)
    overfit_section(ladder, root, out, lines)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text("\n".join(lines))
    (args.out / "metrics.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n"
    )
    pd.DataFrame(out["folds"]).to_csv(args.out / "folds.csv", index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
