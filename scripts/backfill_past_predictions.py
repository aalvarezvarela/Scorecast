"""Refit and predict missing past snapshots of the current NBA season.

    python scripts/backfill_past_predictions.py --start-date 2026-10-21 --dry-run
    python scripts/backfill_past_predictions.py --start-date 2026-10-21 --end-date 2026-10-27 --persist

Dry run is the default: real historical fits and predictions, local reports,
no Aiven or S3 writes. --persist archives the historical models and inserts
the missing predictions. No production or staging pointer is changed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nba_ou.config.settings import SETTINGS
from nba_ou.modeling.registry_paths import ModelSlot, parse_horizon
from nba_ou.prediction.historical_backfill import (
    DEFAULT_SNAPSHOT_GRID,
    DEFAULT_TABPFN_BACKFILL_HORIZON,
    backfill_dates,
    run_backfill,
    tabpfn_feature_slots,
)


def parse_slot(value: str) -> ModelSlot:
    parts = value.strip("/").split("/")
    if len(parts) == 3:
        parts.append("main")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Use schema/target/horizon[/variant], e.g. 2_5/line_error/t0060/main"
        )
    try:
        slot = ModelSlot(
            schema_version=parts[0],
            target=parts[1],
            horizon_minutes=parse_horizon(parts[2]),
            variant=parts[3],
            root=SETTINGS.s3_models_prefix or "models/",
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if slot.horizon_minutes is None:
        raise argparse.ArgumentTypeError(
            "Pooled slots are unsupported; use t0000 for closing"
        )
    return slot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--start-date",
        required=True,
        help="First game date YYYY-MM-DD (Eastern), inclusive",
    )
    parser.add_argument("--end-date", help="Last date, inclusive; default: start-date")
    parser.add_argument(
        "--slot",
        action="append",
        type=parse_slot,
        help="Repeatable; default: every enabled intermediate slot",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and predict; write local reports only (default)",
    )
    mode.add_argument(
        "--persist",
        action="store_true",
        help="Also archive historical models/runs in S3 and insert missing Aiven predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/backfill_predictions"),
        help="Local CSV/JSON reports",
    )
    parser.add_argument(
        "--tabpfn",
        action="store_true",
        help="Also fit standard TabPFN for all three targets at one horizon",
    )
    parser.add_argument(
        "--tabpfn-only",
        action="store_true",
        help="Only standard TabPFN for all three targets; no registry-model refits",
    )
    parser.add_argument(
        "--include-closing",
        action="store_true",
        help="Also evaluate closing models at the shared snapshot timepoint",
    )
    parser.add_argument(
        "--tabpfn-horizon-minutes",
        "--snapshot-minutes",
        type=int,
        default=DEFAULT_TABPFN_BACKFILL_HORIZON,
        help="Shared closing/TabPFN snapshot; default: 240 minutes (4 hours)",
    )
    parser.add_argument(
        "--job-matrix",
        action="store_true",
        help="Print GitHub Actions jobs grouped by date/horizon, without external reads or fitting",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        include_tabpfn = args.tabpfn or args.tabpfn_only
        if (include_tabpfn or args.include_closing) and (
            args.tabpfn_horizon_minutes <= 0
            or args.tabpfn_horizon_minutes not in DEFAULT_SNAPSHOT_GRID
        ):
            parser.error("Shared snapshot must be a positive stored grid horizon")
        if args.tabpfn_only and args.slot:
            parser.error("--tabpfn-only cannot be combined with --slot")
        if args.job_matrix:
            if args.persist:
                parser.error("--job-matrix cannot be combined with --persist")
            dates = backfill_dates(args.start_date, args.end_date)
            slots = (
                []
                if args.tabpfn_only
                else (
                    args.slot
                    if args.slot is not None
                    else SETTINGS.prediction_model_slots
                )
            )
            tabpfn_slots = (
                tabpfn_feature_slots(
                    args.tabpfn_horizon_minutes,
                    slots=SETTINGS.prediction_model_slots,
                    include_closing=args.include_closing,
                )
                if include_tabpfn
                else []
            )
            slots = [
                slot
                for slot in slots
                if slot.horizon_minutes is not None
                and (slot.horizon_minutes > 0 or args.include_closing)
            ]

            def input_horizon(slot):
                return (
                    args.tabpfn_horizon_minutes
                    if slot.horizon_minutes == 0
                    else slot.horizon_minutes
                )

            horizons = sorted({input_horizon(slot) for slot in [*slots, *tabpfn_slots]})
            jobs = [
                {
                    "game_date": day,
                    "horizon": horizon,
                    "slots": " ".join(
                        slot.describe()
                        for slot in slots
                        if input_horizon(slot) == horizon
                    ),
                    **(
                        {
                            "tabpfn": horizon == args.tabpfn_horizon_minutes,
                            "tabpfn_only": not any(
                                input_horizon(slot) == horizon for slot in slots
                            ),
                        }
                        if include_tabpfn
                        else {}
                    ),
                }
                for day in dates
                for horizon in horizons
            ]
            if not jobs or len(jobs) > 256:
                raise ValueError(
                    "Select 1 to 256 date/horizon jobs; shorten the range or select fewer slots"
                )
            print(json.dumps({"include": jobs}))
            return 0
        return run_backfill(
            args.start_date,
            args.end_date,
            slots=[] if args.tabpfn_only else args.slot,
            persist=args.persist,
            output_dir=args.output_dir,
            include_tabpfn=include_tabpfn,
            include_closing=args.include_closing,
            tabpfn_horizon_minutes=args.tabpfn_horizon_minutes,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
