"""Predict tonight's games with TabPFN, on its own.

    python scripts/predict_tabpfn.py [--target line_error] [--no-persist]

One target per run (``line_error`` by default). TabPFN keeps no model, so each
run fits it, from what the daily refit left behind
(``nba_ou.prediction.prediction_tabpfn_client``):

1. The production spec of the target's closing slot (its features, NaN budget
   and window) and the newest closing training frame the refit stored for
   that schema. Both are checked before any other work.
2. The database update, the SBR feed check and tonight's schedule, referees,
   injuries and odds, fetched as the daily predictor fetches them. Tonight's
   closing rows are built from them, so they carry the lines of this moment.
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
from nba_ou.create_training_data.daily_frame_store import (
    latest_daily_frame,
    load_daily_frame,
)
from nba_ou.create_training_data.get_all_info_for_scheduled_games import (
    get_all_info_for_scheduled_games,
)
from nba_ou.create_training_data.schema_layers import (
    CLOSING_LINE,
    compute_new_columns,
    is_layered,
)
from nba_ou.modeling.registry_paths import Channel, ModelSlot, ModelTarget
from nba_ou.modeling.registry_store import resolve_channel
from nba_ou.postgre_db.line_history_aiven.feed_freshness import check_live_feed
from nba_ou.postgre_db.predictions_aiven.schema import WORKFLOW_TABPFN
from nba_ou.postgre_db.update_all.update_all_databases import update_all_databases
from nba_ou.prediction.prediction_store import (
    PredictionRunRecorder,
    persist_prediction_run,
)
from nba_ou.prediction.prediction_tabpfn_client import (
    DEFAULT_MODEL_VERSION,
    feature_slot_for,
    fit_tabpfn,
    model_variant,
    predict_with_tabpfn,
    tabpfn_training_set,
)
from nba_ou.utils.ci import annotate
from nba_ou.utils.general_utils import get_season_year_from_date
from nba_ou.utils.s3_models import make_s3_client

#: A frame older than this many days means the daily refit has stopped.
DEFAULT_MAX_FRAME_AGE_DAYS = 2


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
    feature_slot: ModelSlot | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
    train_games: int | None = None,
    max_frame_age_days: int = DEFAULT_MAX_FRAME_AGE_DAYS,
    update_db: bool = True,
    allow_stale_lines: bool = False,
    persist: bool = True,
) -> None:
    """Fit TabPFN for ``target`` and predict tonight's closing rows.

    ``feature_slot`` defaults to the enabled closing slot of ``target``.
    ``persist=False`` stores nothing (the database update still writes).
    """
    configure_tqdm_for_environment()
    date_to_predict = datetime.now(ZoneInfo("US/Pacific")).strftime("%Y-%m-%d")
    print(f"TabPFN {model_version}, {target}, for {date_to_predict}")

    # Step 1: everything the fit needs from S3, before any other work.
    print_step_header(1, "Loading the Feature Spec and Training Frame")
    slot = feature_slot or feature_slot_for(target, SETTINGS.prediction_model_slots)
    if slot.target != ModelTarget(target) or slot.horizon_minutes != 0:
        raise SystemExit(
            f"The feature slot must be a closing {target} slot; got {slot.describe()}."
        )
    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    bucket = SETTINGS.s3_bucket
    spec = resolve_channel(
        s3_client=s3, bucket=bucket, slot=slot, channel=Channel.PRODUCTION
    ).spec
    schema_version = spec.identity.schema_version
    print_status(
        f"{slot.describe()}: spec {spec.spec_id}, {spec.features.n_features} "
        f"features, {spec.training.train_games} games, max "
        f"{spec.cleaning.max_na_per_row} NaN per row"
    )

    latest = latest_daily_frame(
        s3_client=s3,
        bucket=bucket,
        dataset_type=CLOSING_LINE,
        schema_version=schema_version,
    )
    if latest is None:
        raise SystemExit(
            f"No daily {CLOSING_LINE} {schema_version} training frame in S3; "
            "run the Retrain Prediction Models workflow first."
        )
    frame_key, frame_date = latest
    age_days = (pd.Timestamp(date_to_predict).date() - frame_date).days
    if age_days > max_frame_age_days:
        raise SystemExit(
            f"The newest training frame ({frame_key}) is {age_days} days old "
            f"(max {max_frame_age_days}); the daily refit has not run."
        )
    training_set = tabpfn_training_set(
        load_daily_frame(s3_client=s3, bucket=bucket, key=frame_key),
        spec,
        frame_key=frame_key,
        train_games=train_games,
    )
    print_status(
        f"Training rows from {frame_key}: {training_set.n_games} games, "
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

    print_step_header(2, "Preparing Tonight's Closing Rows")
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
    df_to_predict = df_tonight[df_tonight["GAME_DATE"] == date_to_predict].copy()
    if df_to_predict.empty:
        raise ValueError("No closing rows were built for tonight's games.")
    print_status(f"{len(df_to_predict)} game(s) to predict")

    # Step 3: fit and predict.
    label = f"{schema_version}/{target}/t0000/{model_variant(model_version)}"
    print_step_header(3, f"Fitting and Predicting ({label})")
    prediction_time = datetime.now(ZoneInfo("Europe/Madrid"))
    model = fit_tabpfn(training_set, model_version=model_version)
    predictions = predict_with_tabpfn(
        df=df_to_predict,
        spec=spec,
        model=model,
        training_set=training_set,
        prediction_datetime=prediction_time,
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
    recorder.add_frame(CLOSING_LINE, df_to_predict, version=schema_version)
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
        "--feature-slot",
        type=_parse_slot,
        metavar="SCHEMA/TARGET/t0000/VARIANT",
        help=(
            "The closing slot whose production spec gives the features. "
            "Default: the enabled closing slot of --target."
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
        help="Refuse a training frame older than this many days.",
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
        feature_slot=args.feature_slot,
        model_version=args.model_version,
        train_games=args.train_games,
        max_frame_age_days=args.max_frame_age_days,
        update_db=not args.skip_db_update,
        allow_stale_lines=args.allow_stale_lines,
        persist=not args.no_persist,
    )
