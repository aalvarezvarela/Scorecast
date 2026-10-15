"""Daily as-of RAPM for the stints: the residual models' base (phase 6).

The frozen phase 3 ratings (profile prior, lambda 3,000 / 10,000) refitted on
every game date from 2016-12-01; each stint gets the points-per-possession
and pace predictions of the ratings fitted on stints strictly before its own
date (``pretraining_oof.daily_rapm_predictions``). The same values are the
base of the residual training targets and of the evaluation. The other four
targets are NaN (RAPM has no rating for them).

Writes ``data/player_graph/pretraining/<version>/oof/rapm_daily/
fold=YYYY-MM.parquet`` (one file per month, for the prediction store). Stops
at ``--until`` (default 2019-07-01: the development period; extending it is
not tuning, but phase 6 design reads nothing later).

    python scripts/player_graph/pretraining_daily_rapm.py

Reads the pretraining tables, the stint store and phase 3's stored prior
inputs; no database.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
from nba_ou.data_processing.lineups.stint_store import read_stints
from nba_ou.data_processing.player_graph.pretraining_data import (
    ROOT,
    VERSION,
    load_stints,
)
from nba_ou.data_processing.player_graph.pretraining_oof import (
    FROZEN,
    PredictionStore,
    daily_rapm_predictions,
    monthly_folds,
)
from nba_ou.data_processing.player_graph.rating_book import frozen_prior_provider

NAME = "rapm_daily"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-root", type=Path, default=Path("data"))
    parser.add_argument("--until", default="2019-07-01")
    args = parser.parse_args()
    until = pd.Timestamp(args.until)
    root = args.local_root / ROOT / VERSION
    seasons = range(2016, until.year + 1)
    data = load_stints(root, seasons, until=until)
    history = read_stints(list(seasons), local_root=args.local_root)
    history = history.loc[pd.to_datetime(history["game_date"]) < until]
    started = time.time()
    base, book = daily_rapm_predictions(
        data.players,
        data.game_date,
        history,
        prior=frozen_prior_provider(history, local_root=args.local_root),
        until=until,
        spec=FROZEN,
    )
    print(
        f"{book['as_of_date'].nunique()} daily fits for {len(data):,} stints "
        f"({time.time() - started:.0f}s)",
        flush=True,
    )
    store = PredictionStore(root / "oof")
    folds = monthly_folds(data.game_date)
    meta = {
        "model": "phase 3 frozen ratings refitted daily, as of each stint's date",
        "spec": FROZEN.__dict__,
        "targets": ["pts_per_poss", "poss_per_48"],
        "until": str(until.date()),
    }
    for k, fold in enumerate(folds):
        _, predict = fold.masks(data.game_date)
        store.write(
            NAME,
            fold,
            data.game_id[predict],
            data.seg_idx[predict],
            base[predict],
            meta if k == len(folds) - 1 else None,
        )
    print(f"stored {len(folds)} months under {store.root / NAME}")


if __name__ == "__main__":
    main()
