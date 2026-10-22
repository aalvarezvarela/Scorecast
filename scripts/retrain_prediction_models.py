"""Refit every enabled slot from its spec and stage the result.

    python scripts/retrain_prediction_models.py
    python scripts/retrain_prediction_models.py --slot 2_5/line_error/t0000/main

Each slot's ``channels/config.json`` names the spec to use, and that spec --
not yesterday's model -- supplies the features, the cleaning thresholds and the
hyperparameters. Nothing here can change a configuration; adopting a new one is
``python -m training_pipeline.promote <run_dir> --to-s3``.

The staged build does not serve until ``scripts/promote_build.py --execute``
flips the production pointer, so a bad refit is visible before it reaches
anything.

The training rows come from the season partitions
(``nba_ou.create_training_data.season_partitions``): each dataset kept in S3
one season per file. The job first brings them up to today -- it builds only
the games since the last cutoff (and rebuilds the last few days, reporting any
stored row that came back different) -- and every slot then reads them: the
closing slots every season, each intermediate slot its own horizon. TabPFN
reads the same partitions.
"""

import argparse
import getpass
import os
import sys
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from nba_ou.config.dataset_versions import BASE_SCHEMA_VERSION
from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.schema_layers import LayerContext
from nba_ou.create_training_data.season_partitions import (
    SeasonPartitions,
    update_from_database,
)
from nba_ou.modeling.refit import (
    TrainingFrameUnavailable,
    model_bytes_of,
    refit_device,
    refit_from_spec,
    refit_random_state,
    require_device,
    resolve_training_frame,
)
from nba_ou.modeling.registry_paths import Channel, ModelSlot, parse_horizon
from nba_ou.modeling.registry_store import (
    SlotNotPromotedError,
    resolve_config_spec,
    set_build_channel,
    write_build,
)
from nba_ou.utils.ci import annotate
from nba_ou.utils.s3_models import make_s3_client

TODAY_TIMEZONE = ZoneInfo("Europe/Madrid")


def configure_tqdm_for_environment() -> None:
    """Keep local tqdm behaviour intact while avoiding noisy CI logs.

    tqdm reads ``TQDM_*`` at bar-creation time, so setting it here disables
    downstream bars without touching their call sites. Set
    ``NBA_OU_GITHUB_ACTIONS_TQDM=keep`` to preserve them in a specific run.
    """
    mode = os.getenv("NBA_OU_GITHUB_ACTIONS_TQDM", "disable").strip().lower()
    if os.getenv("GITHUB_ACTIONS", "").lower() == "true" and mode != "keep":
        os.environ.setdefault("TQDM_DISABLE", "1")


def _today_limit_date() -> str:
    return datetime.now(TODAY_TIMEZONE).strftime("%Y-%m-%d")


def _current_user() -> str | None:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001
        return None


def _parse_slot(value: str, *, root: str) -> ModelSlot:
    parts = [part for part in value.split("/") if part]
    if len(parts) == 3:
        parts.append("main")
    if len(parts) != 4:
        raise SystemExit(
            "--slot must be 'schema_version/target/horizon/variant' "
            f"(variant optional). Got {value!r}."
        )
    schema_version, target, horizon, variant = parts
    return ModelSlot(
        schema_version=schema_version,
        target=target,
        horizon_minutes=parse_horizon(horizon),
        variant=variant,
        root=root,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--slot",
        action="append",
        default=None,
        help="Refit this slot only; repeatable. Default: every enabled slot.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be refitted without fitting or writing.",
    )
    parser.add_argument(
        "--no-stage",
        action="store_true",
        help="Fit and report, but write no build and move no channel.",
    )
    parser.add_argument(
        "--skip-partition-update",
        action="store_true",
        help="Fit on the season partitions as stored, without bringing them "
        "up to today first.",
    )
    return parser


def main() -> int:
    configure_tqdm_for_environment()
    args = _build_arg_parser().parse_args()

    root = SETTINGS.s3_models_prefix or "models/"
    slots = (
        [_parse_slot(value, root=root) for value in args.slot]
        if args.slot
        else SETTINGS.prediction_model_slots
    )

    if not slots:
        # A valid state, not an error: the registry has to be runnable before
        # the first configuration is promoted.
        print(
            "No model slots enabled in [PredictionModels] ENABLED_MODELS. "
            "Nothing to retrain."
        )
        return 0

    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    bucket = SETTINGS.s3_bucket
    actor = _current_user()
    limit_date = _today_limit_date()

    print(f"Refitting {len(slots)} slot(s) up to {limit_date} in {bucket}\n")

    specs = {}
    for slot in slots:
        try:
            specs[slot.describe()] = resolve_config_spec(
                s3_client=s3, bucket=bucket, slot=slot
            )
        except SlotNotPromotedError as exc:
            print(f"{slot.describe()}\n  skipped: {exc}\n")

    # Before the dataset updates, not after them: without a usable GPU
    # XGBoost would fit on CPU with only a warning.
    if not args.dry_run:
        for device in sorted({refit_device(spec) for spec in specs.values()}):
            require_device(device)

    # Every schema version is the base plus layers, so the partitions are the
    # base version's; one per dataset type, brought up to today once.
    partitions: dict[str, SeasonPartitions] = {}
    unusable: dict[str, str] = {}
    for dataset_type in sorted({str(s.identity.dataset_type) for s in specs.values()}):
        try:
            store = SeasonPartitions(
                s3_client=s3,
                bucket=bucket,
                dataset_type=dataset_type,
                schema_version=BASE_SCHEMA_VERSION,
            )
        except ValueError as exc:
            unusable[dataset_type] = str(exc)
            continue
        partitions[dataset_type] = store
        if args.dry_run or args.skip_partition_update:
            print(f"[{dataset_type}] season partitions as stored")
            continue
        print(f"[{dataset_type}] bringing the season partitions up to {limit_date}...")
        try:
            report = update_from_database(store, limit_date=limit_date)
        except Exception as exc:  # noqa: BLE001 -- the other dataset still refits
            unusable[dataset_type] = f"season partitions not updated: {exc}"
            annotate("error", f"[{dataset_type}] {unusable[dataset_type]}")
            continue
        print(report.describe() + "\n")
        if report.not_built:
            annotate(
                "warning",
                f"[{dataset_type}] {len(report.not_built)} finished game(s) could "
                "not be built yet (line history not ingested?); retried next run.",
            )
        if report.drifted:
            annotate(
                "warning",
                f"[{dataset_type}] rebuilt rows differ from the stored ones; see "
                "the log, and rebuild the affected seasons if the change is real.",
            )

    # One frame per (dataset, schema version, horizon), shared by every slot
    # that wants it, and dropped once they are fitted: 16 intermediate
    # horizons would not fit in memory together.
    by_frame: dict[tuple[str, str, int | None], list[ModelSlot]] = defaultdict(list)
    for slot in slots:
        spec = specs.get(slot.describe())
        if spec is not None:
            key = (
                str(spec.identity.dataset_type),
                slot.schema_version,
                spec.identity.horizon_minutes,
            )
            by_frame[key].append(slot)

    # Every schema version starts from the same base frame; newer versions add
    # their layers to it (nba_ou.create_training_data.schema_layers), so the
    # base is read once per run however many versions are enabled.
    base_frames: dict = {}
    layer_context = LayerContext()
    failures = 0
    staged = 0
    unavailable = 0
    for (dataset_type, schema_version, horizon), group in sorted(
        by_frame.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or 0)
    ):
        label = f"{dataset_type} / schema {schema_version} / T-{horizon}"
        if dataset_type in unusable:
            failures += len(group)
            print(f"[{label}] FAILED: {unusable[dataset_type]}\n", file=sys.stderr)
            continue
        frame = None
        if args.dry_run:
            print(f"[{label}] would read the training frame")
        else:
            try:
                frame = resolve_training_frame(
                    specs[group[0].describe()],
                    partitions=partitions,
                    base_frames=base_frames,
                    layer_context=layer_context,
                )
            except TrainingFrameUnavailable as exc:
                # Not a failure: a slot the daily refit cannot build (a pooled
                # or layered intermediate spec) keeps serving the build it has
                # -- nothing is staged, so promote_build leaves it alone.
                # Counting it as a failure made the job exit 1, and the
                # workflow's smoke-test and promote steps (if: success()) then
                # skipped EVERY slot.
                unavailable += len(group)
                print(
                    f"[{label}] no training frame: {exc}\n"
                    f"  keeping the current build for {len(group)} slot(s): "
                    f"{', '.join(slot.describe() for slot in group)}\n",
                    file=sys.stderr,
                )
                continue
            print(
                f"[{label}] {len(frame.df)} rows from {frame.build_id} "
                f"({frame.checksum})"
            )

        for slot in group:
            spec = specs[slot.describe()]
            print(slot.describe())
            print(f"  spec       : {spec.spec_id}")
            for name, value, recorded in (
                ("seed", refit_random_state(spec), spec.training.random_state),
                ("device", refit_device(spec), spec.training.device),
            ):
                note = "" if recorded is not None else " (not in the spec; default)"
                print(f"  {name:<11}: {value}{note}")
            if args.dry_run:
                print("  dry run    : not fitted\n")
                continue
            try:
                result = refit_from_spec(spec, slot=slot, frame=frame, fitted_by=actor)
                window = spec.training.train_games
                if window is not None and result.fit.n_train_games < window:
                    annotate(
                        "warning",
                        f"{slot.describe()}: fitted on {result.fit.n_train_games} "
                        f"games, short of the spec's {window}.",
                    )
                if not args.no_stage:
                    write_build(
                        s3_client=s3,
                        bucket=bucket,
                        slot=slot,
                        fit=result.fit,
                        model_bytes=model_bytes_of(result.model),
                        spec=spec,
                    )
                    set_build_channel(
                        s3_client=s3,
                        bucket=bucket,
                        slot=slot,
                        channel=Channel.STAGING,
                        fit_id=result.fit.fit_id,
                        set_by=actor,
                        reason="daily refit",
                    )
            except Exception as exc:  # noqa: BLE001 - one slot must not stop the rest
                failures += 1
                print(f"  FAILED     : {exc}\n", file=sys.stderr)
                continue

            staged += 1
            print(
                f"  {'fitted' if args.no_stage else 'staged':<11}: {result.fit.fit_id}"
            )
            print(f"  model_name : {result.fit.model_name}")
            print(
                f"  trained on : {result.fit.n_train_games} games "
                f"(spec window {spec.training.train_games}; "
                f"{result.fit.train_date_min.date()} .. "
                f"{result.fit.train_date_max.date()})\n"
            )
        del frame

    print(
        f"{staged} build(s) {'fitted (not staged)' if args.no_stage else 'staged'}, "
        f"{failures} failure(s), {unavailable} slot(s) kept on their current "
        "build (no training frame)."
    )
    if staged and not args.dry_run and not args.no_stage:
        print("Promote them with: python scripts/promote_build.py --execute")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
