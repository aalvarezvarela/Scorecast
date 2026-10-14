"""Out-of-fold baselines of phase 6 on the monthly folds: constant and RAPM.

Writes ``data/player_graph/pretraining/<version>/oof/{constant,rapm}/
fold=YYYY-MM.parquet`` for every fold from 2016-12 on
(``nba_ou.data_processing.player_graph.pretraining_oof``) and prints, per
target, RAPM's gain over the constant baseline on the development seasons
(folds before ``--report-until``, default 2019-07-01: 2_7's evaluation starts
in 2019-20, so phase 6 design reads nothing after it).

RAPM is the frozen phase 3 pipeline (profile prior, lambda 3,000 / 10,000)
fitted as of each checkpoint on the stint store; targets it has no rating for
take the constant baseline in ``rapm`` (marked in ``model.json``).

    python scripts/player_graph/pretraining_baselines.py

Reads the pretraining tables, the stint store and phase 3's stored prior
inputs; no database.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.data_processing.lineups.stint_store import read_stints
from nba_ou.data_processing.player_graph.pretraining_data import (
    ROOT,
    VERSION,
    load_stints,
)
from nba_ou.data_processing.player_graph.pretraining_eval import ladder, target_errors
from nba_ou.data_processing.player_graph.pretraining_oof import (
    FROZEN,
    PredictionStore,
    constant_predictions,
    monthly_folds,
    rapm_predictions,
    with_constant,
)
from nba_ou.data_processing.player_graph.rating_book import frozen_prior_provider
from nba_ou.data_processing.player_graph.stint_graph import LABELS

SEASONS = range(2016, 2026)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--report-until", default="2019-07-01")
    args = parser.parse_args()
    root = args.local_root / ROOT / VERSION
    store = PredictionStore(root / "oof")

    started = time.time()
    data = load_stints(root, SEASONS)
    dates = data.game_date
    folds = monthly_folds(dates)
    print(f"{len(data):,} stints, {len(folds)} folds ({time.time() - started:.0f}s)")

    constant = constant_predictions(data.labels, data.label_weight, dates, folds)
    history = read_stints(list(SEASONS), local_root=args.local_root)
    started = time.time()
    rapm, book = rapm_predictions(
        data.players,
        dates,
        folds,
        history,
        prior=frozen_prior_provider(history, local_root=args.local_root),
        spec=FROZEN,
    )
    print(
        f"RAPM book at {book['as_of_date'].nunique()} checkpoints "
        f"({time.time() - started:.0f}s)"
    )
    rapm = with_constant(rapm, constant)
    for name, values, extra in (
        ("constant", constant, {"model": "decayed exposure-weighted mean"}),
        (
            "rapm",
            rapm,
            {
                "model": "phase 3 frozen ratings as of the checkpoint",
                "spec": FROZEN.__dict__,
                "constant_targets": [
                    t for t in LABELS if t not in ("pts_per_poss", "poss_per_48")
                ],
            },
        ),
    ):
        for k, fold in enumerate(folds):
            _, predict = fold.masks(dates)
            store.write(
                name,
                fold,
                data.game_id[predict],
                data.seg_idx[predict],
                values[predict],
                extra if k == len(folds) - 1 else None,
            )

    report = pd.to_datetime(dates) < pd.Timestamp(args.report_until)
    report &= np.isfinite(constant).all(axis=(1, 2))
    print(
        f"\nDevelopment folds (before {args.report_until}): {int(report.sum()):,} stints"
    )
    print(
        target_errors(
            data.labels[report], data.label_weight[report], rapm[report], LABELS
        )
        .round(5)
        .to_string()
    )
    table = ladder(
        data.labels[report],
        data.label_weight[report],
        {"constant": constant[report], "rapm": rapm[report]},
        dates[report],
        LABELS,
        reference="constant",
        masks={f"season {s}": data.season[report] == s for s in (2016, 2017, 2018)},
    )
    pd.set_option("display.width", 200)
    print("\nRAPM gain over constant (squared error, SE clustered by date):")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
