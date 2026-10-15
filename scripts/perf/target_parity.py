#!/usr/bin/env python3
"""Targets for the tonight-only parity test, and the reference rows they must equal.

    python scripts/perf/target_parity.py REFERENCE.parquet OUT_DIR [--random 30 --seed 0]

Writes ``OUT_DIR/targets.txt`` (the GAME_IDs of the chosen dates) and
``OUT_DIR/reference.parquet`` (the reference build's rows for them, in its
order). A build with ``reference_build.py --targets OUT_DIR/targets.txt``
replayed from the same tape must then equal it under
``compare_builds.py --exact``.

The dates cover the edges a history-dependent stage can get wrong: the first
loaded date (no history at all), season openers (roster fallback to the prior
season), the 2020 bubble restart, play-in and playoff dates, the last date;
plus a seeded random sample of the rest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EDGE_DATES = (
    "2017-10-17",
    "2017-12-25",
    "2018-04-11",
    "2018-10-16",
    "2019-02-21",
    "2019-10-22",
    "2020-07-30",
    "2020-08-17",
    "2020-12-22",
    "2021-05-18",
    "2022-04-16",
    "2023-02-15",
    "2024-04-16",
    "2025-10-21",
    "2025-10-22",
    "2026-01-15",
    "2026-04-12",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reference", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--random", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    reference = pd.read_parquet(args.reference)
    dates = pd.to_datetime(reference["GAME_DATE"]).dt.normalize()
    available = pd.DatetimeIndex(dates.unique()).sort_values()
    chosen = {pd.Timestamp(d) for d in EDGE_DATES if pd.Timestamp(d) in set(available)}
    chosen |= {available[0], available[-1]}
    rest = [d for d in available if d not in chosen]
    rng = np.random.default_rng(args.seed)
    chosen |= {rest[i] for i in rng.choice(len(rest), args.random, replace=False)}

    rows = reference[dates.isin(chosen)]
    from nba_ou.create_training_data.schema_layers import registry as layers

    wanted = layers.required_input_columns(
        layers.BASE_SCHEMA_VERSION, layers.latest_version(), layers.CLOSING_LINE
    )
    keys = reference[[c for c in wanted if c in reference.columns]]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "targets.txt").write_text("\n".join(rows["GAME_ID"].astype(str)) + "\n")
    rows.to_parquet(args.out / "reference.parquet")
    # What the layers read, for every row: a target build's layer inputs
    # (reference_build.py writes them) must equal these.
    keys.reset_index(drop=True).to_parquet(args.out / "layer_inputs_reference.parquet")
    print(f"{len(chosen)} dates, {len(rows)} games -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
