"""Predict tonight's games with TabPFN, on its own.

    python scripts/predict_tabpfn.py [--target line_error] [--no-persist]
    python scripts/predict_tabpfn.py --line intermediate [--horizon-minutes 360]

One target and one line per run (``line_error`` at the closing line by
default; ``--line intermediate`` follows the slot of a fixed time before
tip-off, T-360 by default). TabPFN keeps no model, so each run fits it on the
rows and columns of that slot's production build
(``nba_ou.prediction.prediction_tabpfn_client``):

1. The production spec of the target's slot at that line (its features, NaN
   budget and window) and its training rows: the season partitions the daily
   refit keeps current (``nba_ou.create_training_data.season_partitions``),
   read as the refit reads them -- every season for the closing line, the
   horizon's rows for an intermediate slot. Both are loaded before any other
   work, and partitions older than ``--max-frame-age-days`` are refused.
2. The database update, the SBR feed check and tonight's schedule, referees,
   injuries and odds, fetched as the daily predictor fetches them. Tonight's
   rows are built from them, so they carry the lines of this moment: the
   closing rows, or the intermediate row of the horizon -- built from the
   ticks so far when the snapshot is still ahead (``SNAPSHOT_PASSED`` False),
   as the daily predictor does.
3. TabPFN is fitted on the spec's rows and features and predicts tonight's
   rows. The run goes to the prediction store (S3 run archive + Aiven) with
   ``runs.workflow`` 3 and ``model_builds.kind`` 3.

The closing rows keep no raw quote, so the stored bet line is null; the model
line (``inputs.total_model_line``) is the line of this moment.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from nba_ou.config.dataset_versions import BASE_SCHEMA_VERSION
from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict
from nba_ou.create_training_data.get_all_info_for_scheduled_games import (
    get_all_info_for_scheduled_games,
)
from nba_ou.create_training_data.schema_layers import (
    CLOSING_LINE,
    compute_new_columns,
    is_layered,
    parse_version,
)
from nba_ou.create_training_data.season_partitions import SeasonPartitions
from nba_ou.modeling.refit import resolve_training_frame
from nba_ou.modeling.registry_paths import (
    Channel,
    ModelSlot,
    ModelTarget,
    format_horizon,
)
from nba_ou.modeling.registry_store import resolve_channel
from nba_ou.postgre_db.line_history_aiven.feed_freshness import check_live_feed
from nba_ou.postgre_db.predictions_aiven.schema import WORKFLOW_TABPFN
from nba_ou.postgre_db.update_all.update_all_databases import update_all_databases
from nba_ou.prediction.intermediate_frame import (
    PASSTHROUGH_COLUMNS,
    tonight_intermediate_frame,
)
from nba_ou.prediction.prediction_store import (
    PredictionRunRecorder,
    persist_prediction_run,
)
from nba_ou.prediction.prediction_tabpfn_client import (
    DEFAULT_INTERMEDIATE_HORIZON,
    DEFAULT_MODEL_VERSION,
    INTERMEDIATE_LINE,
    feature_slot_for,
    fit_tabpfn,
    model_variant,
    predict_with_tabpfn,
    rows_of_horizon,
    tabpfn_training_set,
)
from nba_ou.utils.ci import annotate
from nba_ou.utils.general_utils import get_season_year_from_date
from nba_ou.utils.s3_models import make_s3_client

#: Partitions older than this many days mean the daily refit has stopped.
DEFAULT_MAX_FRAME_AGE_DAYS = 2
#: ``--line``: the dataset type of the slot TabPFN follows.
LINES = {"closing": CLOSING_LINE, "intermediate": INTERMEDIATE_LINE}


def print_step_header(step_number: int, title: str) -> None:
    clock = datetime.now(ZoneInfo("UTC")).strftime("%H:%M:%S")
    print("-" * 70)
    print(f" STEP {step_number} — {title} [{clock} UTC] ".center(70))
    print("-" * 70)


def print_status(message: str, ok: bool = True) -> None:
    print(f"  {'✓' if ok else '✖'} {message}")


def configure_tqdm_for_environment() -> None:
    """Disable tqdm in GitHub Actions, as the daily predictor does."""
    if os.getenv("GITHUB_ACTIONS", "").lower() == "true":
        os.environ.setdefault("TQDM_DISABLE", "1")


def predict_tabpfn(
    *,
    target: str = "line_error",
    line: str = "closing",
    horizon_minutes: int | None = None,
    feature_slot: ModelSlot | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
    train_games: int | None = None,
    max_frame_age_days: int = DEFAULT_MAX_FRAME_AGE_DAYS,
    update_db: bool = True,
    allow_stale_lines: bool = False,
    persist: bool = True,
) -> None:
    """Fit TabPFN for ``target`` and predict tonight's rows at ``line``.

    ``line`` is ``closing`` or ``intermediate``; ``horizon_minutes`` is the
    intermediate slot's time before tip-off (default
    ``DEFAULT_INTERMEDIATE_HORIZON``). ``feature_slot`` defaults to the enabled
    slot of ``target`` at that line. ``persist=False`` stores nothing (the
    database update still writes).
    """
    configure_tqdm_for_environment()
    dataset_type = LINES[line]
    if dataset_type == CLOSING_LINE:
        if horizon_minutes is not None:
            raise SystemExit("--horizon-minutes applies to --line intermediate only.")
        horizon = 0
    else:
        horizon = horizon_minutes or DEFAULT_INTERMEDIATE_HORIZON
    date_to_predict = datetime.now(ZoneInfo("US/Pacific")).strftime("%Y-%m-%d")
    print(
        f"TabPFN {model_version}, {target} at {format_horizon(horizon)}, "
        f"for {date_to_predict}"
    )

    # Step 1: everything the fit needs from S3, before any other work.
    print_step_header(1, "Loading the Feature Spec and Training Frame")
    slot = feature_slot or feature_slot_for(
        target, SETTINGS.prediction_model_slots, horizon_minutes=horizon
    )
    if slot.target != ModelTarget(target) or slot.horizon_minutes != horizon:
        raise SystemExit(
            f"The feature slot must be a {target} {format_horizon(horizon)} slot; "
            f"got {slot.describe()}."
        )
    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    bucket = SETTINGS.s3_bucket
    resolved = resolve_channel(
        s3_client=s3, bucket=bucket, slot=slot, channel=Channel.PRODUCTION
    )
    spec = resolved.spec
    if str(spec.identity.dataset_type) != dataset_type:
        raise SystemExit(
            f"{slot.describe()} serves a {spec.identity.dataset_type} model, "
            f"not a {dataset_type} one."
        )
    schema_version = spec.identity.schema_version
    print_status(
        f"{slot.describe()}: spec {spec.spec_id}, {spec.features.n_features} "
        f"features, {spec.training.train_games} games, max "
        f"{spec.cleaning.max_na_per_row} NaN per row"
    )

    partitions = SeasonPartitions(
        s3_client=s3,
        bucket=bucket,
        dataset_type=dataset_type,
        schema_version=BASE_SCHEMA_VERSION,
    )
    current_to = partitions.limit_date()
    if current_to is None:
        raise SystemExit(
            f"No {dataset_type} season partitions in S3; create them with "
            "scripts/create_train_data/build_season_partitions.py split."
        )
    age_days = (pd.Timestamp(date_to_predict).date() - current_to).days
    if age_days > max_frame_age_days:
        raise SystemExit(
            f"The {dataset_type} season partitions are current to {current_to}, "
            f"{age_days} days ago (max {max_frame_age_days}); the daily refit has "
            "not run."
        )
    # The refit's own frame: same partitions, horizon and layers.
    frame = resolve_training_frame(spec, partitions={dataset_type: partitions})
    frame_key = partitions.prefix
    training_set = tabpfn_training_set(
        frame.df, spec, frame_key=frame_key, train_games=train_games
    )
    print_status(
        f"Training rows from {frame.build_id} ({frame.checksum}): "
        f"{training_set.n_games} games, "
        f"{training_set.train_date_min.date()} .. {training_set.train_date_max.date()}"
    )

    # Step 2: tonight's rows, with the lines of this moment.
    if update_db:
        print_step_header(2, "Updating All Databases")
        season_to_update = int(get_season_year_from_date(date_to_predict))
        update_all_databases(
            start_season_year=season_to_update,
            end_season_year=season_to_update,
            only_new_games=True,
            headless=SETTINGS.headless,
            exclude_game_date=date_to_predict,
        )
        print_status("Databases updated")

    print_step_header(2, "Checking SBR Line Feed")
    try:
        freshness = check_live_feed(
            datetime.strptime(date_to_predict, "%Y-%m-%d").date()
        )
    except Exception as e:  # noqa: BLE001 -- the check must not cost the run
        annotate("warning", f"SBR feed check failed, continuing unchecked: {e}")
    else:
        print_status(freshness.describe(), ok=not freshness.stale)
        if freshness.stale:
            message = f"SBR line feed looks frozen: {freshness.describe()}"
            if not allow_stale_lines:
                annotate(
                    "error",
                    f"{message}; nothing predicted (--allow-stale-lines overrides).",
                )
                raise SystemExit(1)
            annotate("warning", f"{message}; predicting anyway (--allow-stale-lines).")

    print_step_header(2, "Fetching Scheduled Games & Reports")
    scheduled_data = get_all_info_for_scheduled_games(
        date_to_predict=date_to_predict,
        nba_injury_reports_url=SETTINGS.nba_injury_reports_url,
        headless=SETTINGS.headless,
    )
    if scheduled_data["scheduled_games"].empty:
        print_status("No scheduled games for this date; nothing to predict.")
        return

    if dataset_type == CLOSING_LINE:
        print_step_header(2, "Preparing Tonight's Closing Rows")
        df_to_predict = tonight_closing_rows(
            scheduled_data, schema_version=schema_version, game_date=date_to_predict
        )
        frame_version = schema_version
        print_status(f"{len(df_to_predict)} game(s) to predict")
    else:
        print_step_header(2, f"Preparing Tonight's T-{horizon} Rows")
        # The intermediate builder has no schema-layer step yet.
        if parse_version(schema_version) > parse_version(BASE_SCHEMA_VERSION):
            raise SystemExit(
                f"{slot.describe()} is a {schema_version} model, and tonight's "
                f"intermediate rows are built at {BASE_SCHEMA_VERSION}."
            )
        df_to_predict = rows_of_horizon(
            tonight_intermediate_frame(scheduled_data, game_date=date_to_predict),
            spec,
        )
        frame_version = BASE_SCHEMA_VERSION
        if df_to_predict.empty:
            raise ValueError(f"No T-{horizon} rows were built for tonight's games.")
        passed = int(df_to_predict["SNAPSHOT_PASSED"].sum())
        print_status(
            f"{len(df_to_predict)} game(s) to predict; T-{horizon} passed for "
            f"{passed}, the rest from the lines so far"
        )

    # Step 3: fit and predict.
    label = (
        f"{schema_version}/{target}/{format_horizon(horizon)}/"
        f"{model_variant(model_version)}"
    )
    print_step_header(3, f"Fitting and Predicting ({label})")
    prediction_time = datetime.now(ZoneInfo("Europe/Madrid"))
    model = fit_tabpfn(training_set, model_version=model_version)
    predictions = predict_with_tabpfn(
        df=df_to_predict,
        spec=spec,
        model=model,
        training_set=training_set,
        prediction_datetime=prediction_time,
        passthrough_columns=(
            () if dataset_type == CLOSING_LINE else PASSTHROUGH_COLUMNS
        ),
    )
    if predictions.empty:
        print_status("No game survived the spec's NaN budget; nothing predicted.")
    else:
        value_column = {
            "line_error": "PRED_LINE_ERROR",
            "total_points": "PRED_TOTAL_POINTS",
            "spread_error": "PRED_SPREAD_ERROR",
        }[target]
        print(
            predictions[
                [
                    "GAME_ID",
                    "TEAM_NAME_TEAM_HOME",
                    "TEAM_NAME_TEAM_AWAY",
                    value_column,
                    "PRED_PICK",
                ]
            ].to_string(index=False)
        )

    failures: list[str] = []
    recorder = PredictionRunRecorder(
        workflow=WORKFLOW_TABPFN, game_date=date_to_predict
    )
    recorder.set_schedule(scheduled_data["scheduled_games"])
    recorder.add_frame(dataset_type, df_to_predict, version=frame_version)
    rejected = recorder.add_tabpfn(predictions, label=label)
    if rejected:
        print_status(f"Prediction store rejected {label}: {rejected}", ok=False)
        failures.append(f"{label} (prediction store): {rejected}")

    if not persist:
        print_status("Dry run (--no-persist): nothing stored")
    else:
        print_step_header(4, "Storing Predictions")
        failures += persist_prediction_run(
            recorder, s3_client=s3, bucket=bucket, partial=bool(failures)
        )
    for failure in failures:
        print_status(failure, ok=False)
    if failures:
        raise SystemExit(1)


def tonight_closing_rows(
    scheduled_data: dict, *, schema_version: str, game_date: str
) -> pd.DataFrame:
    """Tonight's closing rows at ``schema_version``, as the daily predictor
    builds them."""
    tonight_ids = scheduled_data["scheduled_games"]["GAME_ID"].astype(str).tolist()
    df_tonight, layer_inputs = create_df_to_predict(
        todays_prediction=True,
        scheduled_data=scheduled_data,
        strict_mode=30,
        normalize_total_lines=True,
        normalize_spread_lines=True,
        null_extreme_spread_prices=True,
        target_game_ids=tonight_ids,
        return_layer_inputs=True,
    )
    if is_layered(schema_version):
        added = compute_new_columns(
            layer_inputs,
            from_version=BASE_SCHEMA_VERSION,
            to_version=schema_version,
            dataset_type=CLOSING_LINE,
            existing_columns=df_tonight.columns,
        )
        df_tonight = pd.concat([df_tonight, added.loc[df_tonight.index]], axis=1)
    rows = df_tonight[df_tonight["GAME_DATE"] == game_date].copy()
    if rows.empty:
        raise ValueError("No closing rows were built for tonight's games.")
    return rows


def _parse_slot(value: str) -> ModelSlot:
    return ModelSlot.parse(value, root=SETTINGS.s3_models_prefix or "models/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--target",
        choices=[target.value for target in ModelTarget],
        default="line_error",
        help="The target to fit and predict (default: line_error).",
    )
    parser.add_argument(
        "--line",
        choices=sorted(LINES),
        default="closing",
        help=(
            "Follow the target's closing slot, or its slot at a fixed time "
            "before tip-off (default: closing)."
        ),
    )
    parser.add_argument(
        "--horizon-minutes",
        type=int,
        help=(
            "With --line intermediate: the slot's minutes before tip-off "
            f"(default: {DEFAULT_INTERMEDIATE_HORIZON})."
        ),
    )
    parser.add_argument(
        "--feature-slot",
        type=_parse_slot,
        metavar="SCHEMA/TARGET/HORIZON/VARIANT",
        help=(
            "The slot whose production spec gives the features and rows. "
            "Default: the enabled slot of --target at --line."
        ),
    )
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        help=f"TabPFN model version (default: {DEFAULT_MODEL_VERSION}).",
    )
    parser.add_argument(
        "--train-games",
        type=int,
        help="Fit on the last N games instead of the spec's window.",
    )
    parser.add_argument(
        "--max-frame-age-days",
        type=int,
        default=DEFAULT_MAX_FRAME_AGE_DAYS,
        help="Refuse season partitions older than this many days.",
    )
    parser.add_argument(
        "--skip-db-update",
        action="store_true",
        help="Do not update the databases first (another job already did).",
    )
    parser.add_argument(
        "--allow-stale-lines",
        action="store_true",
        help="Predict even when SBR's line feed looks frozen (lines hours old).",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Predict without storing: no prediction store rows, no S3 archive",
    )
    args = parser.parse_args()

    predict_tabpfn(
        target=args.target,
        line=args.line,
        horizon_minutes=args.horizon_minutes,
        feature_slot=args.feature_slot,
        model_version=args.model_version,
        train_games=args.train_games,
        max_frame_age_days=args.max_frame_age_days,
        update_db=not args.skip_db_update,
        allow_stale_lines=args.allow_stale_lines,
        persist=not args.no_persist,
    )
