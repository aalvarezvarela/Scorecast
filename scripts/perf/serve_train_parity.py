#!/usr/bin/env python3
"""A date's archived prediction rows against the training rows built afterwards.

    python scripts/perf/serve_train_parity.py --date 2026-10-30 [--impact] \
        [--dataset intermediate_line|closing_line|both] [--out DIR]

Prediction rows come from the S3 run archive
(``prediction_snapshots/v2/game_date=D/``):

* ``intermediate_line``: the date's last past-snapshot (``own``) run, every
  passed snapshot of every game -- the exact row each slot was trained to read;
* ``closing_line``: each game's closing row from the last daily run before its
  tip. It is a few minutes early by design, so line moves, public percentages
  and report ages differ legitimately (see the ``serve-train-parity`` skill).

Training rows are the current ``2_5`` season partition. Rows are matched on
``GAME_ID`` (plus ``TIME_TO_MATCH_MIN``); every shared column is compared
(numeric within 1e-9, NaN equal to NaN). Outcome columns and columns only one
side carries are listed apart, as they differ by construction.

``--impact`` serves every enabled slot of the dataset twice -- on the
prediction rows, and on the same rows with the training values of every
shared non-outcome column -- and reports how far each slot's prediction moved
and how many sides flipped. That separates mismatches that matter from ones no
model reads. Read-only: nothing is written but the ``--out`` CSVs.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src")]

from nba_ou.config.dataset_versions import BASE_SCHEMA_VERSION  # noqa: E402
from nba_ou.config.settings import SETTINGS  # noqa: E402
from nba_ou.create_training_data.season_partitions import SeasonPartitions  # noqa: E402
from nba_ou.create_training_data.select_intermediate_columns import (  # noqa: E402
    TARGET_COLUMNS,
)
from nba_ou.utils.general_utils import get_season_year_from_date  # noqa: E402
from nba_ou.utils.s3_models import make_s3_client  # noqa: E402

ARCHIVE = "prediction_snapshots/v2/game_date={date}/"
#: Outcomes and post-game columns: missing before the game by construction.
OUTCOMES = frozenset(
    {
        *TARGET_COLUMNS,
        "IS_OVERTIME",
        "PTS_TEAM_HOME",
        "PTS_TEAM_AWAY",
        "MATCHUP_TEAM_HOME",
        "MATCHUP_TEAM_AWAY",
    }
)


def archived_runs(s3, date: str) -> dict[str, list[str]]:
    """Run folder -> object names, in run start order."""
    prefix = ARCHIVE.format(date=date)
    runs: dict[str, list[str]] = {}
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=SETTINGS.s3_bucket, Prefix=prefix
    ):
        for item in page.get("Contents", []):
            run, _, name = item["Key"][len(prefix) :].partition("/")
            runs.setdefault(run, []).append(name)
    return dict(sorted(runs.items()))


def read_archived(s3, date: str, run: str, name: str) -> pd.DataFrame:
    body = s3.get_object(
        Bucket=SETTINGS.s3_bucket, Key=f"{ARCHIVE.format(date=date)}{run}/{name}"
    )["Body"].read()
    return pd.read_parquet(io.BytesIO(body))


def prediction_rows(s3, date: str, dataset: str) -> pd.DataFrame:
    name = f"inputs_{dataset}.parquet"
    runs = {
        run: names for run, names in archived_runs(s3, date).items() if name in names
    }
    if dataset == "intermediate_line":
        own = [run for run in runs if "_past_snapshots_" in run]
        if not own:
            raise SystemExit(f"No past-snapshot run archived for {date}")
        print(f"intermediate rows: {own[-1]}")
        return read_archived(s3, date, own[-1], name)
    frames = []
    for run in (run for run in runs if "_daily_" in run):
        frame = read_archived(s3, date, run, name)
        frame["_RUN_AT"] = pd.to_datetime(run[:16], format="%Y%m%dT%H%M%SZ", utc=True)
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No daily run archived a closing frame for {date}")
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[rows["_RUN_AT"] < pd.to_datetime(rows["GAME_TIME"], utc=True)]
    rows = rows.sort_values("_RUN_AT").groupby("GAME_ID").tail(1)
    lead = pd.to_datetime(rows["GAME_TIME"], utc=True) - rows["_RUN_AT"]
    print(
        "closing rows: last daily run before each tip, "
        f"{lead.dt.total_seconds().div(60).round().astype(int).tolist()} min early"
    )
    return rows.drop(columns="_RUN_AT").reset_index(drop=True)


def training_rows(s3, date: str, dataset: str) -> pd.DataFrame:
    store = SeasonPartitions(
        s3_client=s3,
        bucket=SETTINGS.s3_bucket,
        dataset_type=dataset,
        schema_version=BASE_SCHEMA_VERSION,
    )
    season = int(get_season_year_from_date(date))
    manifest = store.manifests()[season]
    print(
        f"{dataset} partition {manifest.file}, written {manifest.written_at}, "
        f"limit {manifest.limit_date}"
    )
    rows = store.read(season, manifest=manifest)
    return rows[pd.to_datetime(rows["GAME_DATE"]).dt.date.astype(str).eq(date)]


def compare(pred: pd.DataFrame, train: pd.DataFrame, keys: list[str]):
    """Matched rows and one line per differing shared column."""
    pred, train = pred.copy(), train.copy()
    for key in keys:
        pred[key], train[key] = pred[key].astype(str), train[key].astype(str)
    merged = pred.merge(train, on=keys, suffixes=("__pred", "__train"))
    report = []
    for column in [c for c in pred if c in train and c not in keys]:
        a, b = merged[f"{column}__pred"], merged[f"{column}__train"]
        an, bn = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
        numeric = (an.notna() | a.isna()).all() and (bn.notna() | b.isna()).all()
        if numeric:
            x, y = an.to_numpy(float), bn.to_numpy(float)
            both_nan = np.isnan(x) & np.isnan(y)
            differ = ~(np.isclose(x, y, rtol=1e-9, atol=1e-9) | both_nan)
            gap = np.abs(x - y)[differ & ~np.isnan(x) & ~np.isnan(y)]
        else:
            differ = (a.astype(str) != b.astype(str)).to_numpy() & ~(
                a.isna() & b.isna()
            ).to_numpy()
            gap = np.array([])
        if differ.any():
            report.append(
                dict(
                    column=column,
                    rows=int(differ.sum()),
                    nan_in_pred=int((a.isna() & b.notna()).sum()),
                    nan_in_train=int((a.notna() & b.isna()).sum()),
                    max_abs=float(gap.max()) if gap.size else np.nan,
                    outcome=column in OUTCOMES,
                )
            )
    columns = ["column", "rows", "nan_in_pred", "nan_in_train", "max_abs", "outcome"]
    return merged, pd.DataFrame(report, columns=columns)


def with_training_values(pred, train, keys, columns) -> pd.DataFrame:
    """``pred`` with ``columns`` taken from the training row of the same keys."""
    key = pred[keys].astype(str).agg("|".join, axis=1)
    source = train.assign(_k=train[keys].astype(str).agg("|".join, axis=1))
    source = source.drop_duplicates("_k").set_index("_k")
    out = pred.copy()
    for column in columns:
        out[column] = key.map(source[column]).to_numpy()
    return out


def impact(s3, dataset, pred, swapped) -> pd.DataFrame:
    from nba_ou.prediction.prediction import (
        load_registry_model_and_predict,
        read_slot_builds,
    )

    slots = [
        slot
        for slot in SETTINGS.prediction_model_slots
        if (slot.horizon_minutes == 0) == (dataset == "closing_line")
        and slot.schema_version == BASE_SCHEMA_VERSION
    ]
    builds = read_slot_builds(s3_client=s3, bucket=SETTINGS.s3_bucket, slots=slots)
    now = pd.Timestamp.now(tz="UTC").to_pydatetime()
    rows = []
    for slot in slots:
        outputs = []
        for frame in (pred, swapped):
            with contextlib.redirect_stdout(io.StringIO()):
                outputs.append(
                    load_registry_model_and_predict(
                        s3_client=s3,
                        bucket=SETTINGS.s3_bucket,
                        slot=slot,
                        df=None,
                        prediction_datetime=now,
                        frames={dataset: frame},
                        frame_versions={dataset: BASE_SCHEMA_VERSION},
                        resolved=builds[slot],
                    )
                )
        column = (
            "PRED_SPREAD_ERROR"
            if str(slot.target) == "spread_error"
            else "PRED_LINE_ERROR"
        )
        a, b = (o.set_index("GAME_ID")[column] for o in outputs)
        b = b.reindex(a.index)
        rows.append(
            dict(
                slot=slot.describe(),
                games=len(a),
                mean_abs_move=float((a - b).abs().mean()),
                max_abs_move=float((a - b).abs().max()),
                side_flips=int((np.sign(a) != np.sign(b)).sum()),
            )
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--date", required=True, help="NBA game date, YYYY-MM-DD")
    parser.add_argument(
        "--dataset",
        choices=["intermediate_line", "closing_line", "both"],
        default="both",
    )
    parser.add_argument("--impact", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    datasets = (
        ["intermediate_line", "closing_line"]
        if args.dataset == "both"
        else [args.dataset]
    )
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    summary = {}
    for dataset in datasets:
        print(f"\n=== {dataset} {args.date}")
        keys = ["GAME_ID"] + (
            ["TIME_TO_MATCH_MIN"] if dataset != "closing_line" else []
        )
        pred = prediction_rows(s3, args.date, dataset)
        train = training_rows(s3, args.date, dataset)
        merged, report = compare(pred, train, keys)
        print(
            f"prediction rows {len(pred)}, training rows {len(train)}, matched "
            f"{len(merged)}; games only predicted: "
            f"{sorted(set(pred.GAME_ID.astype(str)) - set(train.GAME_ID.astype(str)))}"
        )
        print(f"columns only in prediction: {sorted(set(pred) - set(train))}")
        print(f"columns only in training: {len(set(train) - set(pred))}")
        features = report[~report["outcome"]].sort_values("rows", ascending=False)
        print(f"{len(features)} non-outcome column(s) differ")
        if not features.empty:
            print(features.drop(columns="outcome").to_string(index=False))
        summary[dataset] = {"matched": len(merged), "differing": len(features)}
        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            report.to_csv(args.out / f"{dataset}_{args.date}_columns.csv", index=False)
        if args.impact and not features.empty:
            matched = pred[
                pred[keys]
                .astype(str)
                .agg("|".join, axis=1)
                .isin(train[keys].astype(str).agg("|".join, axis=1))
            ].reset_index(drop=True)
            swapped = with_training_values(
                matched, train, keys, features["column"].tolist()
            )
            moved = impact(s3, dataset, matched, swapped)
            print("\nprediction moved by the differing columns, per slot:")
            print(moved.round(3).to_string(index=False))
            if args.out:
                moved.to_csv(
                    args.out / f"{dataset}_{args.date}_impact.csv", index=False
                )
    print("\n" + json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
