#!/usr/bin/env python3
"""Create, update and inspect the season partitions the daily refit fits on.

Each training dataset is kept in S3 one season per file
(``nba_ou.create_training_data.season_partitions``)::

    # Where they stand
    python scripts/create_train_data/build_season_partitions.py status

    # Start them from a full build (a local file or an S3 key); the cutoff is
    # the one the file was built with
    python scripts/create_train_data/build_season_partitions.py split \\
        data/train_data/intermediate_line_data_2_5_20261016.parquet \\
        --limit-date 2026-10-16

    # Bring them up to a date, as the daily refit does first
    python scripts/create_train_data/build_season_partitions.py update \\
        --dataset-type closing_line [--limit-date 2026-10-22]

    # Rebuild one season from the database (after a correction the daily
    # overlap check reported, or a code change that moved values)
    python scripts/create_train_data/build_season_partitions.py rebuild \\
        --dataset-type intermediate_line --season 2025

The dataset type and schema version of ``split`` come from the filename
(``<dataset_type>_data_<schema>_<date>.parquet``). A rebuild builds every game
of the season in target mode with today's cutoff: about 40 minutes for an
intermediate season, a few for a closing one.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from nba_ou.config.dataset_versions import BASE_SCHEMA_VERSION
from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.season_partitions import (
    CLOSING_LINE,
    DATE_COLUMN,
    INTERMEDIATE_LINE,
    SEASON_COLUMN,
    SeasonPartitions,
    build_rows,
    manifest_summary,
    season_game_ids,
    split_into_partitions,
    update_from_database,
)
from nba_ou.utils.s3_models import make_s3_client, read_s3_object_bytes

DATASET_TYPES = (CLOSING_LINE, INTERMEDIATE_LINE)
_FILENAME = re.compile(r"^(?P<dataset_type>[a-z_]+?)_data_(?P<schema>\d+_\d+)_\d{8}")


def _today() -> str:
    return datetime.now(ZoneInfo("Europe/Madrid")).strftime("%Y-%m-%d")


def _partitions(dataset_type: str, schema_version: str) -> SeasonPartitions:
    return SeasonPartitions(
        s3_client=make_s3_client(
            profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region
        ),
        bucket=SETTINGS.s3_bucket,
        dataset_type=dataset_type,
        schema_version=schema_version,
    )


def _read_full_build(source: str) -> pd.DataFrame:
    path = Path(source)
    if path.exists():
        return pd.read_parquet(path)
    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    import io

    return pd.read_parquet(
        io.BytesIO(
            read_s3_object_bytes(s3_client=s3, bucket=SETTINGS.s3_bucket, key=source)
        )
    )


def cmd_status(args) -> int:
    for dataset_type in DATASET_TYPES:
        manifests = _partitions(dataset_type, args.schema_version).manifests()
        print(f"{dataset_type} {args.schema_version}:")
        print(manifest_summary(manifests) if manifests else "  none")
    return 0


def cmd_split(args) -> int:
    match = _FILENAME.match(Path(args.source).name)
    if match is None:
        raise SystemExit(
            f"Cannot read the dataset type and schema from {args.source!r}."
        )
    dataset_type, schema_version = match["dataset_type"], match["schema"]
    partitions = _partitions(dataset_type, schema_version)
    if partitions.manifests() and not args.replace:
        raise SystemExit(
            f"{dataset_type} {schema_version} partitions exist; --replace to "
            "overwrite them."
        )
    df = _read_full_build(args.source)
    written = split_into_partitions(
        df,
        partitions,
        limit_date=args.limit_date,
        source=f"split {Path(args.source).name}",
    )
    print(manifest_summary(written))
    return 0


def cmd_update(args) -> int:
    report = update_from_database(
        _partitions(args.dataset_type, args.schema_version),
        limit_date=args.limit_date or _today(),
    )
    print(report.describe())
    return 1 if report.drifted else 0


def cmd_rebuild(args) -> int:
    partitions = _partitions(args.dataset_type, args.schema_version)
    manifests = partitions.manifests()
    limit = args.limit_date or _today()
    current = max((m.limit_date for m in manifests.values()), default=None)
    if current is not None and pd.Timestamp(limit).date() != current:
        raise SystemExit(
            f"The partitions are current to {current}; rebuild with that cutoff "
            f"(--limit-date {current}) so every season matches, or update first."
        )
    game_ids = season_game_ids(args.season, pd.Timestamp(limit).date())
    print(f"Rebuilding {len(game_ids)} game(s) of {args.season} with cutoff {limit}")
    rows = build_rows(args.dataset_type, game_ids, pd.Timestamp(limit).date())
    rows = rows.loc[pd.to_numeric(rows[SEASON_COLUMN]).astype(int) == args.season]
    if args.season in manifests:
        stored = partitions.read(args.season, manifest=manifests[args.season])
        lost = set(stored["GAME_ID"].astype(str)) - set(rows["GAME_ID"].astype(str))
        if lost:
            print(f"{len(lost)} stored game(s) not rebuilt: {sorted(lost)[:10]}")
    manifest = partitions.write(
        args.season, rows, limit_date=limit, source=f"rebuild {args.season}"
    )
    print(
        f"season {args.season}: {manifest.n_games} games, {manifest.n_rows} rows, "
        f"{pd.to_datetime(rows[DATE_COLUMN]).min().date()} .. "
        f"{manifest.last_game_date} ({manifest.checksum})"
    )
    return 0


def main() -> int:
    # Roster continuity's last bits depend on the hash seed; the overlap check
    # tolerates it, but a rebuild should be reproducible.
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schema-version", default=BASE_SCHEMA_VERSION)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Print every partition's manifest.")

    split = sub.add_parser("split", help="Start the partitions from a full build.")
    split.add_argument("source", help="local path or S3 key of a full build")
    split.add_argument("--limit-date", required=True, help="the file's cutoff")
    split.add_argument("--replace", action="store_true")

    for name, text in (
        ("update", "Bring the partitions up to a date."),
        ("rebuild", "Rebuild one season from the database."),
    ):
        command = sub.add_parser(name, help=text)
        command.add_argument("--dataset-type", choices=DATASET_TYPES, required=True)
        command.add_argument("--limit-date", help="default: today (Europe/Madrid)")
        if name == "rebuild":
            command.add_argument("--season", type=int, required=True)

    args = parser.parse_args()
    return {
        "status": cmd_status,
        "split": cmd_split,
        "update": cmd_update,
        "rebuild": cmd_rebuild,
    }[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
