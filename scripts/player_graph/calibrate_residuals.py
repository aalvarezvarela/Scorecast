"""Walk-forward calibration of residual FM corrections (phase 6, decided 2026-10-15).

For each run ``fm_<variant>_s<seed>`` given, shrinks its points and pace
corrections over the daily RAPM base by a factor in ``[0, 1]`` fitted before
each fold on the run's own earlier out-of-fold predictions
(``pretraining_oof.calibrated_residuals``: least squares through the origin of
the realized residual on the correction; 0 where nothing earlier exists).
Writes the calibrated predictions as ``fm_<variant>_cal_s<seed>`` in the
prediction store (the other four targets unchanged) and the factors to
``oof/fm_<variant>_cal_s<seed>/shrinkage.json``. No parameter is chosen here.

    python scripts/player_graph/calibrate_residuals.py --variants int_matchup --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.player_graph.pretraining_data import (
    ROOT,
    VERSION,
    load_stints,
)
from nba_ou.data_processing.player_graph.pretraining_oof import (
    PredictionStore,
    calibrated_residuals,
    monthly_folds,
)

UNTIL = pd.Timestamp("2019-07-01")
TARGETS = ("pts_per_poss", "poss_per_48")
BASE = "rapm_daily"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    root = args.local_root / ROOT / VERSION
    store = PredictionStore(root / "oof")
    data = load_stints(root, range(2016, 2019), until=UNTIL)
    base = store.read(BASE, data.game_id, data.seg_idx)
    folds = monthly_folds(data.game_date, first="2016-12-01")
    for variant in args.variants:
        for seed in args.seeds:
            raw = store.read(f"fm_{variant}_s{seed}", data.game_id, data.seg_idx)
            calibrated, factors = calibrated_residuals(
                raw,
                base,
                data.labels,
                data.label_weight,
                data.game_date,
                folds,
                TARGETS,
            )
            name = f"fm_{variant}_cal_s{seed}"
            meta = {
                "calibrated_from": f"fm_{variant}_s{seed}",
                "base": BASE,
                "targets": list(TARGETS),
                "rule": "per fold, slope through the origin of (y - base) on "
                "(prediction - base) over earlier out-of-fold rows, clipped to [0, 1]",
            }
            for k, fold in enumerate(folds):
                _, predict = fold.masks(data.game_date)
                store.write(
                    name,
                    fold,
                    data.game_id[predict],
                    data.seg_idx[predict],
                    calibrated[predict],
                    meta if k == len(folds) - 1 else None,
                )
            (store.root / name / "shrinkage.json").write_text(
                json.dumps(factors.to_dict(orient="records"), indent=2) + "\n"
            )
            last = factors.iloc[-1]
            print(
                f"{name}: last fold factors "
                + ", ".join(f"{t} {last[t]:.3f}" for t in TARGETS)
                + f"; mean {factors[list(TARGETS)].mean().round(3).to_dict()}",
                flush=True,
            )


if __name__ == "__main__":
    main()
