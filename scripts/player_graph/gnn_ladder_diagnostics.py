"""Diagnostics of the GNN ladder's residual corrections (development folds only).

Complements ``fm_ladder_report.py --ladder gnn``:

* **Size and calibration of the corrections** per variant (mean over seeds,
  raw): the exposure-weighted SD of ``prediction - base`` for points and pace,
  and the slope of the realized residual ``y - base`` on it (1 = calibrated,
  0 = noise), over the folds after the first.
* **Calibration factors**: the walk-forward shrinkage per fold
  (``shrinkage.json`` of each calibrated run), and a recomputation of each
  fold's factor from the rows dated before its checkpoint only.
* **The absorber at prediction**: for a fold checkpoint of each variant, the
  stored out-of-fold predictions are recomputed with the model in prediction
  mode (additive terms of points and pace dropped) and compared with the
  same model's output with those terms included.

    python scripts/player_graph/gnn_ladder_diagnostics.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from nba_ou.data_processing.player_graph.fm import load_checkpoint, stint_tensors
from nba_ou.data_processing.player_graph.pretraining_data import (
    ROOT,
    VERSION,
    load_stints,
)
from nba_ou.data_processing.player_graph.pretraining_oof import (
    PredictionStore,
    monthly_folds,
)
from nba_ou.data_processing.player_graph.stint_graph import LABELS

UNTIL = pd.Timestamp("2019-07-01")
VARIANTS = ("gnn_control", "gnn_node", "gnn_no_guard", "gnn_full")
SEEDS = (0, 1, 2)
TARGETS = ("pts_per_poss", "poss_per_48")
CHECK_FOLD = "2018-12"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/experiments/gnn_ladder_dev")
    )
    args = parser.parse_args()
    root = args.local_root / ROOT / VERSION
    store = PredictionStore(root / "oof")
    data = load_stints(root, range(2016, 2019), until=UNTIL)
    base = store.read("rapm_daily", data.game_id, data.seg_idx)
    folds = monthly_folds(data.game_date, first="2016-12-01")
    after_first = pd.to_datetime(data.game_date) >= folds[1].checkpoint
    out: dict = {}

    # 1. Size and calibration of the raw corrections.
    rows = []
    for variant in (*VARIANTS, "int_matchup"):
        pred = np.mean(
            [
                store.read(f"fm_{variant}_s{s}", data.game_id, data.seg_idx)
                for s in SEEDS
            ],
            axis=0,
        )
        for target in TARGETS:
            k = LABELS.index(target)
            c = (pred - base)[after_first][:, :, k].ravel()
            r = (data.labels - base)[after_first][:, :, k].ravel()
            w = data.label_weight[after_first][:, :, k].ravel()
            ok = np.isfinite(c) & np.isfinite(r) & (w > 0)
            c, r, w = c[ok], r[ok], w[ok]
            cm, rm = np.average(c, weights=w), np.average(r, weights=w)
            sd_c = np.sqrt(np.average((c - cm) ** 2, weights=w))
            sd_r = np.sqrt(np.average((r - rm) ** 2, weights=w))
            slope = np.sum(w * (c - cm) * (r - rm)) / np.sum(w * (c - cm) ** 2)
            rows.append(
                {
                    "variant": variant,
                    "target": target,
                    "correction_sd": sd_c,
                    "correction_mean": cm,
                    "residual_sd": sd_r,
                    "slope_realized_on_correction": slope,
                }
            )
    corrections = pd.DataFrame(rows)
    out["corrections"] = corrections.to_dict(orient="records")
    print("Raw corrections (seed mean, folds after the first):")
    print(corrections.round(4).to_string(index=False))

    # 2. Calibration factors, and their recomputation from earlier rows only.
    factor_rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            name = f"fm_{variant}_cal_s{seed}"
            factors = pd.DataFrame(
                json.loads((store.root / name / "shrinkage.json").read_text())
            )
            raw = store.read(f"fm_{variant}_s{seed}", data.game_id, data.seg_idx)
            for fold in folds:
                before = pd.to_datetime(data.game_date) < fold.checkpoint
                row = {"variant": variant, "seed": seed, "fold": fold.label}
                for target in TARGETS:
                    k = LABELS.index(target)
                    c = raw[before][:, :, k] - base[before][:, :, k]
                    r = data.labels[before][:, :, k] - base[before][:, :, k]
                    w = data.label_weight[before][:, :, k]
                    ok = np.isfinite(c) & np.isfinite(r) & (w > 0)
                    den = float((w[ok] * c[ok] ** 2).sum())
                    again = (
                        float(np.clip((w[ok] * c[ok] * r[ok]).sum() / den, 0, 1))
                        if den > 0
                        else 0.0
                    )
                    stored = float(
                        factors.loc[factors["fold"].eq(fold.label), target].iloc[0]
                    )
                    row[target] = stored
                    row[f"{target}_recomputed_matches"] = abs(stored - again) < 1e-9
                factor_rows.append(row)
    factors = pd.DataFrame(factor_rows)
    out["calibration_factors"] = factors.to_dict(orient="records")
    summary = factors.groupby("variant")[list(TARGETS)].agg(["mean", "min", "max"])
    print("\nWalk-forward calibration factors (all folds, seeds):")
    print(summary.round(3).to_string())
    matches = factors[[f"{t}_recomputed_matches" for t in TARGETS]].all().all()
    first = factors.loc[factors["fold"].eq(folds[0].label), list(TARGETS)]
    print(
        f"factors recomputed from rows before each checkpoint: "
        f"{'all match' if matches else 'MISMATCH'}; first fold factors "
        f"{first.to_numpy().ravel().tolist()}"
    )
    out["calibration_recomputed_from_earlier_rows"] = bool(matches)

    # 3. The absorber: in training, not at prediction.
    fold = next(f for f in folds if f.label == CHECK_FOLD)
    _, mask = fold.masks(data.game_date)
    target = data.subset(mask)
    checks = []
    for variant in VARIANTS:
        name = f"fm_{variant}_s0"
        model, vocab, std, extra = load_checkpoint(
            root / "fm" / name / f"fold={CHECK_FOLD}.pt"
        )
        tensors = stint_tensors(target, vocab, std, guard=True)
        with torch.no_grad():
            model.eval()
            served = std.targets.invert(model.forward_batch(tensors).numpy())
            model.additive_mask.fill_(1.0)  # put the absorbed terms back
            with_absorber = std.targets.invert(model.forward_batch(tensors).numpy())
        stored = store.read(name, target.game_id, target.seg_idx)
        row = {
            "variant": variant,
            "absorb_additive": extra["config"]["absorb_additive"],
        }
        for t in TARGETS:
            k = LABELS.index(t)
            served_k = served[:, :, k] + base[mask][:, :, k]
            row[f"{t}_stored_equals_served"] = bool(
                np.allclose(stored[:, :, k], served_k, atol=1e-5)
            )
            row[f"{t}_absorber_sd"] = float(
                np.std(with_absorber[:, :, k] - served[:, :, k])
            )
        checks.append(row)
    absorber = pd.DataFrame(checks)
    out["absorber"] = absorber.to_dict(orient="records")
    print(f"\nAbsorber at prediction (fold {CHECK_FOLD}, seed 0):")
    print(absorber.round(4).to_string(index=False))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "diagnostics.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
