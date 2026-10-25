#!/usr/bin/env python3
"""Rebuild one season in memory and compare it with its stored partition.

Read-only: nothing is written to S3. It answers "would a rebuild change this
season, and where?" before paying for one (about 40 minutes for an
intermediate season, a few for a closing one)::

    # The season at the cutoff it was stored with: only data and code changes
    # since then show up (a line-history fallback, a database correction)
    python scripts/create_train_data/compare_season_partition.py \\
        --dataset-type intermediate_line --season 2019

    # At another cutoff, e.g. the partitions' current one: the games added
    # since the stored cutoff enter the global fits too
    python scripts/create_train_data/compare_season_partition.py \\
        --dataset-type intermediate_line --season 2019 --limit-date 2026-10-25

``--save`` writes the rebuilt rows to a Parquet file for a closer look.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nba_ou.config.dataset_versions import BASE_SCHEMA_VERSION
from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.season_partitions import (
    CLOSING_LINE,
    DATE_COLUMN,
    GAME_ID_COLUMN,
    INTERMEDIATE_LINE,
    OVERLAP_RTOL,
    ROW_ORDER,
    SEASON_COLUMN,
    SeasonPartitions,
    _match_stored_schema,
    build_rows,
    season_game_ids,
)
from nba_ou.utils.s3_models import make_s3_client


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Per row: equal, as the daily overlap check counts it."""
    if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
        return np.isclose(
            a.to_numpy(dtype=float),
            b.to_numpy(dtype=float),
            rtol=OVERLAP_RTOL,
            atol=0.0,
            equal_nan=True,
        )
    a_na, b_na = a.isna().to_numpy(), b.isna().to_numpy()
    equal = a.astype(object).to_numpy() == b.astype(object).to_numpy()
    return (a_na & b_na) | (~a_na & ~b_na & equal)


def compare(stored: pd.DataFrame, rebuilt: pd.DataFrame, keys: list[str]) -> int:
    """Print where ``rebuilt`` differs from ``stored``; 1 if it does."""
    stored_ids = set(stored[GAME_ID_COLUMN].astype(str))
    rebuilt_ids = set(rebuilt[GAME_ID_COLUMN].astype(str))
    only_stored = sorted(stored_ids - rebuilt_ids)
    only_rebuilt = sorted(rebuilt_ids - stored_ids)
    common = stored_ids & rebuilt_ids
    print(
        f"games: {len(stored_ids)} stored, {len(rebuilt_ids)} rebuilt, "
        f"{len(common)} in both"
    )
    if only_stored:
        print(f"  only stored ({len(only_stored)}): {only_stored[:10]}")
    if only_rebuilt:
        print(f"  only rebuilt ({len(only_rebuilt)}): {only_rebuilt[:10]}")

    old = stored[stored[GAME_ID_COLUMN].astype(str).isin(common)]
    new = rebuilt[rebuilt[GAME_ID_COLUMN].astype(str).isin(common)]
    old = old.sort_values(keys, kind="mergesort").reset_index(drop=True)
    new = new.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if len(old) != len(new) or not old[keys].astype(str).equals(new[keys].astype(str)):
        print(f"rows: {len(old)} stored vs {len(new)} rebuilt for the common games")
        return 1

    rows_differ = np.zeros(len(old), dtype=bool)
    shares = {}
    for column in old.columns:
        same = _same(old[column], new[column])
        if not same.all():
            shares[column] = float((~same).mean())
            rows_differ |= ~same
    if not shares:
        print(f"identical: {len(old)} rows, {len(old.columns)} columns")
        return 1 if only_stored or only_rebuilt else 0

    games = old.loc[rows_differ, [DATE_COLUMN, GAME_ID_COLUMN]].drop_duplicates(
        GAME_ID_COLUMN
    )
    dates = pd.to_datetime(games[DATE_COLUMN])
    print(
        f"{len(games)} of {len(common)} game(s) differ "
        f"({rows_differ.sum()} of {len(old)} rows), {dates.min().date()} .. "
        f"{dates.max().date()}; first: "
        f"{games.sort_values(DATE_COLUMN)[GAME_ID_COLUMN].astype(str).head(10).tolist()}"
    )
    print("games differing per month:")
    for month, count in dates.dt.to_period("M").value_counts().sort_index().items():
        print(f"  {month}: {count}")
    print(f"{len(shares)} column(s) differ, by share of rows:")
    for column, share in sorted(shares.items(), key=lambda kv: -kv[1])[:40]:
        print(f"  {share:7.1%}  {column}")
    if len(shares) > 40:
        print(f"  ... and {len(shares) - 40} more")
    return 1


def main() -> int:
    # Roster continuity's last bits depend on the hash seed, as in a rebuild.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schema-version", default=BASE_SCHEMA_VERSION)
    parser.add_argument(
        "--dataset-type", choices=(CLOSING_LINE, INTERMEDIATE_LINE), required=True
    )
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument(
        "--limit-date", help="default: the cutoff the season was stored with"
    )
    parser.add_argument("--save", type=Path, help="write the rebuilt rows here")
    args = parser.parse_args()

    partitions = SeasonPartitions(
        s3_client=make_s3_client(
            profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region
        ),
        bucket=SETTINGS.s3_bucket,
        dataset_type=args.dataset_type,
        schema_version=args.schema_version,
    )
    manifests = partitions.manifests()
    if args.season not in manifests:
        raise SystemExit(f"No stored {args.dataset_type} partition for {args.season}.")
    manifest = manifests[args.season]
    limit = pd.Timestamp(args.limit_date or manifest.limit_date).date()
    print(
        f"{args.dataset_type} {args.season}: stored with cutoff "
        f"{manifest.limit_date} ({manifest.source}); rebuilding with {limit}"
    )

    stored = partitions.read(args.season, manifest=manifest)
    game_ids = season_game_ids(args.season, limit)
    print(f"building {len(game_ids)} game(s)")
    rows = build_rows(args.dataset_type, game_ids, limit)
    rows = rows.loc[pd.to_numeric(rows[SEASON_COLUMN]).astype(int) == args.season]
    rows = _match_stored_schema(rows.copy(), stored)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(args.save, index=False)
        print(f"rebuilt rows written to {args.save}")
    return compare(stored, rows, list(ROW_ORDER[args.dataset_type]))


if __name__ == "__main__":
    raise SystemExit(main())
