"""Export the current 2_5 S3 season partitions as full-build Parquet files.

Read-only against S3 and the line-history database. Writes, under
``data/train_data/``:

- ``closing_line_data_2_5_<stamp>.parquet`` (+ ``.manifest.json``)
- ``intermediate_line_data_2_5_<stamp>.parquet`` (+ ``.manifest.json``)
- ``intermediate_line_data_2_5_<stamp>_scoring.parquet``

and, with ``--restrict-to-reference``, ``<dataset>_data_2_5_<stamp>_rows<ref>.parquet``
copies (+ sidecar) that keep only the reference file's row keys.

The partitions are read with ``SeasonPartitions.load()``, the reader the daily
refit and TabPFN use, so each file is exactly the frame they fit on. The
partitions do not keep the intermediate scoring sidecar, so it is rebuilt the
way ``create_intermediate_line_df._build_scoring_frame`` builds it: the
intermediate rows' keys, the closing pipeline's per-game line columns renamed
``ODDS_<x>`` -> ``ODDS_CLOSING_<x>`` (taken here from the closing partitions),
``TIPOFF_UTC`` from ``line_history_aiven.fetch.fetch_games`` and
``SNAPSHOT_TS_UTC = TIPOFF_UTC - TIME_TO_MATCH_MIN``. ``--check-recipe`` first
applies that recipe to the 20261016 files and compares it with their real
sidecar.

    PYTHONPATH=src:. .venv/bin/python \\
        experiments/data_drift_fixed_params_2026_11/export_partitions.py \\
        --stamp 20261105 [--check-recipe]
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from nba_ou.config.odds_columns import total_line_col
from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.season_partitions import (
    CLOSING_LINE,
    INTERMEDIATE_LINE,
    SeasonPartitions,
)
from nba_ou.postgre_db.line_history_aiven.fetch import fetch_games
from nba_ou.utils.s3_models import make_s3_client
from training_pipeline.data import compute_file_checksum
from training_pipeline.layered_dataset import write_base_manifest
from training_pipeline.parquet_dataset import write_training_dataset

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "data" / "train_data"
SCHEMA = "2_5"
KEYS = ["GAME_ID", "TIME_TO_MATCH_MIN"]


def _partitions(dataset_type: str) -> SeasonPartitions:
    return SeasonPartitions(
        s3_client=make_s3_client(
            profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region
        ),
        bucket=SETTINGS.s3_bucket,
        dataset_type=dataset_type,
        schema_version=SCHEMA,
    )


def export(dataset_type: str, stamp: str) -> Path:
    partitions = _partitions(dataset_type)
    manifests = partitions.manifests()
    frame = partitions.load()
    df = frame.df
    expected = sum(m.n_rows for m in manifests.values())
    if len(df) != expected:
        raise SystemExit(f"{dataset_type}: loaded {len(df)} rows, manifests say {expected}")
    for season, manifest in manifests.items():
        n = int((pd.to_numeric(df["SEASON_YEAR"]).astype(int) == season).sum())
        if n != manifest.n_rows:
            raise SystemExit(f"{dataset_type} {season}: {n} rows vs manifest {manifest.n_rows}")
    path = OUT / f"{dataset_type}_data_{SCHEMA}_{stamp}.parquet"
    write_training_dataset(df, path)
    build_args = {
        "source": "s3 season partitions (SeasonPartitions.load)",
        "bucket": SETTINGS.s3_bucket,
        "prefix": frame.prefix,
        "partitions_checksum": frame.checksum,
        "limit_date": str(frame.limit_date),
        "seasons": {
            str(season): {
                "n_rows": m.n_rows,
                "n_games": m.n_games,
                "first_game_date": str(m.first_game_date),
                "last_game_date": str(m.last_game_date),
                "limit_date": str(m.limit_date),
                "checksum": m.checksum,
                "source": m.source,
                "git_commit": m.git_commit,
                "written_at": m.written_at.isoformat(),
            }
            for season, m in manifests.items()
        },
        "exported_by": "experiments/data_drift_fixed_params_2026_11/export_partitions.py",
    }
    write_base_manifest(
        path,
        schema_version=SCHEMA,
        dataset_type=dataset_type,
        n_rows=len(df),
        n_columns=df.shape[1],
        build_args=build_args,
    )
    print(f"{dataset_type}: {len(df):,} rows x {df.shape[1]:,} columns -> {path}")
    del df, frame
    gc.collect()
    return path


def scoring_frame(intermediate: Path, closing: Path, closing_columns: list[str]) -> pd.DataFrame:
    """The sidecar ``_build_scoring_frame`` would write for these files."""
    keys = pq.read_table(intermediate, columns=[*KEYS, "SEASON_YEAR"]).to_pandas()
    keys["GAME_ID"] = keys["GAME_ID"].astype(str)
    sources = {c: "ODDS_" + c.removeprefix("ODDS_CLOSING_") for c in closing_columns}
    lines = pq.read_table(closing, columns=["GAME_ID", *sources.values()]).to_pandas()
    lines["GAME_ID"] = lines["GAME_ID"].astype(str)
    lines = lines.rename(columns={v: k for k, v in sources.items()})
    missing = set(keys["GAME_ID"]) - set(lines["GAME_ID"])
    if missing:
        raise SystemExit(f"{len(missing)} intermediate games not in {closing.name}")
    seasons = sorted(pd.to_numeric(keys["SEASON_YEAR"]).astype(int).unique().tolist())
    games = fetch_games(seasons)[["game_id", "tipoff_utc"]].rename(
        columns={"game_id": "GAME_ID", "tipoff_utc": "TIPOFF_UTC"}
    )
    games["GAME_ID"] = games["GAME_ID"].astype(str)
    scoring = (
        keys[KEYS]
        .merge(lines, on="GAME_ID", how="left", validate="many_to_one")
        .merge(games, on="GAME_ID", how="left", validate="many_to_one")
    )
    scoring["SNAPSHOT_TS_UTC"] = scoring["TIPOFF_UTC"] - pd.to_timedelta(
        scoring["TIME_TO_MATCH_MIN"], unit="m"
    )
    scoring = scoring[[*KEYS, *closing_columns, "TIPOFF_UTC", "SNAPSHOT_TS_UTC"]]
    return scoring.sort_values(KEYS).reset_index(drop=True)


def restrict(dataset_type: str, stamp: str, ref: str) -> None:
    """``<stamp>`` rows the ``<ref>`` file could train on, in ``<stamp>`` order:
    the key is in ``<ref>`` and ``<ref>`` has the main total line there.

    The arms of a paired comparison must hold the same games: a fixed
    ``train_games`` window counts games back from each fold, so one extra game
    day shifts every window. A game the reference holds without a line (the
    2024-03-29 slate, since filled from Covers) is dropped by cleaning there
    and kept here, so it is removed here too.
    """
    keys = KEYS if dataset_type == INTERMEDIATE_LINE else KEYS[:1]
    source = OUT / f"{dataset_type}_data_{SCHEMA}_{stamp}.parquet"
    reference = OUT / f"{dataset_type}_data_{SCHEMA}_{ref}.parquet"
    target = OUT / f"{dataset_type}_data_{SCHEMA}_{stamp}_rows{ref}.parquet"
    line = total_line_col()
    wanted = pq.read_table(reference, columns=[*keys, line]).to_pandas()
    wanted["GAME_ID"] = wanted["GAME_ID"].astype(str)
    no_line = int(wanted[line].isna().sum())
    wanted = wanted.loc[wanted[line].notna(), keys]
    df = pd.read_parquet(source)
    present = df[keys].astype({"GAME_ID": str}).merge(
        wanted.assign(_keep=True), on=keys, how="left"
    )["_keep"].fillna(False).to_numpy(dtype=bool)
    kept = df.loc[present].reset_index(drop=True)
    dropped = df.loc[~present]
    del df
    gc.collect()
    if len(kept) != len(wanted):
        raise SystemExit(
            f"{dataset_type}: {len(wanted) - len(kept)} reference rows are not in {source.name}"
        )
    write_training_dataset(kept, target)
    write_base_manifest(
        target,
        schema_version=SCHEMA,
        dataset_type=dataset_type,
        n_rows=len(kept),
        n_columns=kept.shape[1],
        build_args={
            "source": source.name,
            "source_checksum": compute_file_checksum(source),
            "rows_restricted_to": reference.name,
            "reference_rows_without_line_dropped": no_line,
            "line_column": line,
            "reference_checksum": compute_file_checksum(reference),
            "rows_dropped": len(dropped),
            "dropped_game_dates": sorted(
                str(d.date()) for d in pd.to_datetime(dropped["GAME_DATE"]).unique()
            ),
            "exported_by": "experiments/data_drift_fixed_params_2026_11/export_partitions.py",
        },
    )
    print(f"{dataset_type}: kept {len(kept):,}, dropped {len(dropped):,} -> {target}")
    if dataset_type == INTERMEDIATE_LINE:
        sidecar = pd.read_parquet(source.with_name(f"{source.stem}_scoring.parquet"))
        sidecar["GAME_ID"] = sidecar["GAME_ID"].astype(str)
        k = kept[KEYS].astype({"GAME_ID": str})
        sidecar = k.merge(sidecar, on=KEYS, how="left", validate="one_to_one")
        sidecar = sidecar.sort_values(KEYS).reset_index(drop=True)
        if sidecar.drop(columns=KEYS).isna().all(axis=1).any():
            raise SystemExit("a kept row has no sidecar entry")
        path = target.with_name(f"{target.stem}_scoring.parquet")
        write_training_dataset(sidecar, path)
        print(f"scoring sidecar: {len(sidecar):,} rows -> {path}")


def compare(built: pd.DataFrame, real: pd.DataFrame) -> bool:
    real = real.copy()
    real["GAME_ID"] = real["GAME_ID"].astype(str)
    real = real.sort_values(KEYS).reset_index(drop=True)
    ok = list(built.columns) == list(real.columns) and len(built) == len(real)
    print(f"columns equal: {list(built.columns) == list(real.columns)}; rows {len(built)} vs {len(real)}")
    if len(built) != len(real):
        return False
    ok &= bool((built[KEYS].values == real[KEYS].values).all())
    for column in built.columns[2:]:
        a, b = built[column], real[column]
        if pd.api.types.is_numeric_dtype(a):
            same = np.isclose(a.astype(float), b.astype(float), equal_nan=True)
        else:
            same = (a == b) | (a.isna() & b.isna())
        if not same.all():
            ok = False
            ids = built.loc[~same, "GAME_ID"].unique()
            print(f"  {column}: {(~same).sum()} rows differ, {len(ids)} games, e.g. {list(ids[:5])}")
    print("recipe reproduces the sidecar" if ok else "recipe DIFFERS")
    return ok


def main() -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stamp", required=True)
    parser.add_argument("--check-recipe", action="store_true")
    parser.add_argument("--reference-stamp", default="20261016")
    parser.add_argument(
        "--restrict-to-reference",
        action="store_true",
        help="write <stamp>_rows<ref> copies holding only the reference file's row keys",
    )
    args = parser.parse_args()

    ref = args.reference_stamp
    ref_scoring_path = OUT / f"intermediate_line_data_{SCHEMA}_{ref}_scoring.parquet"
    closing_columns = [
        c
        for c in pq.read_schema(ref_scoring_path).names
        if c.startswith("ODDS_CLOSING_")
    ]
    if args.check_recipe:
        built = scoring_frame(
            OUT / f"intermediate_line_data_{SCHEMA}_{ref}.parquet",
            OUT / f"closing_line_data_{SCHEMA}_{ref}.parquet",
            closing_columns,
        )
        compare(built, pd.read_parquet(ref_scoring_path))
        return 0

    if args.restrict_to_reference:
        for dataset_type in (CLOSING_LINE, INTERMEDIATE_LINE):
            restrict(dataset_type, args.stamp, ref)
        return 0

    closing = export(CLOSING_LINE, args.stamp)
    intermediate = export(INTERMEDIATE_LINE, args.stamp)
    scoring = scoring_frame(intermediate, closing, closing_columns)
    n_rows = pq.ParquetFile(intermediate).metadata.num_rows
    if len(scoring) != n_rows:
        raise SystemExit(f"sidecar {len(scoring)} rows vs dataset {n_rows}")
    if scoring.duplicated(KEYS).any():
        raise SystemExit("sidecar repeats a (GAME_ID, TIME_TO_MATCH_MIN) key")
    path = intermediate.with_name(f"{intermediate.stem}_scoring.parquet")
    write_training_dataset(scoring, path)
    print(f"scoring sidecar: {len(scoring):,} rows -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
