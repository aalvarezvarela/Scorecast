import argparse
import os
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from nba_ou.config.dataset_versions import BASE_SCHEMA_VERSION
from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.create_df_to_predict import (
    create_df_to_predict,
)
from nba_ou.create_training_data.get_all_info_for_scheduled_games import (
    get_all_info_for_scheduled_games,
)
from nba_ou.create_training_data.schema_layers import (
    CLOSING_LINE,
    compute_new_columns,
    is_layered,
    newest_version,
)
from nba_ou.postgre_db.line_history_aiven.feed_freshness import check_live_feed
from nba_ou.postgre_db.update_all.update_all_databases import update_all_databases
from nba_ou.prediction.baseline_predictions import (
    baseline_history_window,
    load_baseline_predictions_for_nba_games,
)
from nba_ou.prediction.intermediate_frame import (
    SNAPSHOT_COLUMNS,
    tonight_intermediate_frame,
)
from nba_ou.prediction.prediction import (
    load_registry_model_and_predict,
)
from nba_ou.prediction.prediction_tabpfn_client import (
    load_and_predict_tabpfn_client_for_nba_games,
)
from nba_ou.utils.ci import annotate
from nba_ou.utils.general_utils import get_season_year_from_date
from nba_ou.utils.s3_models import (
    make_s3_client,
)
from nba_ou.utils.s3_prediction_snapshots import (
    upload_input_features_snapshot,
    upload_model_predictions_snapshot,
)


def print_banner(date_to_predict: str) -> None:
    line = "=" * 70
    title = "NBA OVER/UNDER PREDICTION SYSTEM"
    print(f"\n{line}")
    print(title.center(len(line)))
    print(line)
    print(f"  Prediction Date: {date_to_predict}")
    print(line + "\n")


def print_step_header(step_number: int, title: str) -> None:
    """Print a compact, eye-catching step header."""
    sep = "-" * 70
    clock = datetime.now(ZoneInfo("UTC")).strftime("%H:%M:%S")
    header = f" STEP {step_number} — {title} [{clock} UTC] "
    print(sep)
    print(header.center(len(sep)))
    print(sep)


def print_status(message: str, ok: bool = True) -> None:
    """Print a single-line status message with a check or cross."""
    symbol = "✓" if ok else "✖"
    print(f"  {symbol} {message}")


def configure_tqdm_for_environment() -> None:
    """
    Keep local tqdm behavior intact while avoiding noisy GitHub Actions logs.

    tqdm reads `TQDM_*` environment variables when progress bars are created, so
    setting `TQDM_DISABLE=1` here disables downstream bars in CI without
    touching their call sites.

    Set `NBA_OU_GITHUB_ACTIONS_TQDM=keep` to preserve normal tqdm output inside
    GitHub Actions for a specific workflow run.
    """
    github_actions_tqdm_mode = (
        os.getenv("NBA_OU_GITHUB_ACTIONS_TQDM", "disable").strip().lower()
    )

    if (
        os.getenv("GITHUB_ACTIONS", "").lower() == "true"
        and github_actions_tqdm_mode != "keep"
    ):
        os.environ.setdefault("TQDM_DISABLE", "1")


def predict_nba_games(
    run_tabpfn_client: bool = False,
    normalize_total_lines: bool = True,
    normalize_spread_lines: bool = True,
    null_extreme_spread_prices: bool = True,
    persist: bool = True,
    dump_frames: str | Path | None = None,
    allow_stale_lines: bool = False,
) -> None:
    """
    Main execution function for the NBA prediction pipeline.

    1. Updates the databases.
    2. Fetches tonight's games, referees, injuries and odds.
    3. Builds tonight's closing rows and, when any enabled slot has a horizon,
       tonight's intermediate rows for every horizon.
    4. Predicts with every enabled slot, whatever the time to tip: a closing
       slot predicts every game, an intermediate slot the row of its own
       horizon (passed or not; ``SNAPSHOT_PASSED`` tells them apart).
    5. Runs the baselines.

    A slot that fails does not stop the others; the run exits non-zero at the
    end if anything failed. ``persist=False`` stores no predictions: no
    predictions-table rows and no S3 snapshots. The database update and the
    injury-report sync in steps 1-3 still write. ``dump_frames`` writes the
    feature frames to that local directory, for inspection.

    When SBR's feed has frozen (``feed_freshness``), every line the models would
    read is hours old, so the run stops after the database update with exit 1
    and predicts nothing; ``allow_stale_lines`` predicts anyway.
    """
    configure_tqdm_for_environment()
    if run_tabpfn_client:
        # The feature frame now holds tonight's games only; the TabPFN client
        # trains on history features, which it would have to read from a
        # stored training dataset instead. Fail before any work is done.
        raise SystemExit(
            "TabPFN needs history features, and the prediction frame carries "
            "tonight's games only; run with --no-tabpfn."
        )

    date_to_predict = (datetime.now(ZoneInfo("US/Pacific"))).strftime("%Y-%m-%d")

    # Print welcome banner
    print_banner(date_to_predict)
    configured_slots = SETTINGS.prediction_model_slots
    if not configured_slots:
        # A valid state, not an error: nothing has been promoted yet. The rest
        # of the pipeline (database update, feature build, baselines) is still
        # worth running.
        print("  No model slots enabled in [PredictionModels] ENABLED_MODELS.")
    else:
        print("  Enabled model slots:")
        for slot in configured_slots:
            print(f"    - {slot.describe()}")

    # Step 1: Update the database
    print_step_header(1, "Updating All Databases")
    season_to_update = get_season_year_from_date(date_to_predict)
    try:
        update_all_databases(
            start_season_year=int(season_to_update),
            end_season_year=int(season_to_update),
            only_new_games=True,
            headless=SETTINGS.headless,
            exclude_game_date=date_to_predict,
        )
        print_status("Databases updated")

    except Exception as e:
        print_status(f"Failed to update databases: {e}", ok=False)
        raise

    # Step 1b: SBR's lines must still be moving, or every prediction below is
    # made on lines frozen hours ago. The database update above stands either way.
    print_step_header(1, "Checking SBR Line Feed")
    try:
        freshness = check_live_feed(
            datetime.strptime(date_to_predict, "%Y-%m-%d").date()
        )
    except Exception as e:  # noqa: BLE001 -- the check must not cost the run
        annotate("warning", f"SBR feed check failed, continuing unchecked: {e}")
    else:
        print_status(freshness.describe(), ok=not freshness.stale)
        if freshness.stale:
            message = (
                f"SBR line feed looks frozen: {freshness.describe()}. "
                "Predictions would use stale lines"
            )
            if not allow_stale_lines:
                annotate(
                    "error",
                    f"{message}; nothing predicted (--allow-stale-lines overrides).",
                )
                raise SystemExit(1)
            annotate("warning", f"{message}; predicting anyway (--allow-stale-lines).")

    # Step 2: Fetch scheduled games, referees, injuries and odds
    print_step_header(2, "Fetching Scheduled Games & Reports")
    try:
        scheduled_data = get_all_info_for_scheduled_games(
            date_to_predict=date_to_predict,
            nba_injury_reports_url=SETTINGS.nba_injury_reports_url,
            headless=SETTINGS.headless,
        )
        print_status("Fetched scheduled games, refs, injuries and odds")
        if scheduled_data["scheduled_games"].empty:
            print_status(
                "⚠️ WARNING: No scheduled games found for the specified date.", ok=False
            )
            return 0

    except Exception as e:
        print_status(f"Failed to fetch scheduled data: {e}", ok=False)
        raise

    # Step 3: Build feature DataFrame for prediction
    print_step_header(3, "Preparing Feature DataFrame")
    try:
        # Tonight's games only, on the training build's full history: every
        # row is exactly what a full build would give it, at a fraction of
        # the work (create_df_to_predict's target_game_ids).
        tonight_ids = scheduled_data["scheduled_games"]["GAME_ID"].astype(str).tolist()
        df_to_predict_total, layer_inputs = create_df_to_predict(
            todays_prediction=True,
            scheduled_data=scheduled_data,
            strict_mode=30,
            normalize_total_lines=normalize_total_lines,
            normalize_spread_lines=normalize_spread_lines,
            null_extreme_spread_prices=null_extreme_spread_prices,
            target_game_ids=tonight_ids,
            return_layer_inputs=True,
        )
        # The builder produces the base schema. Each slot reads its own
        # feature_names, and newer versions only add columns, so one frame at
        # the newest version any enabled slot needs serves every slot.
        frame_version = newest_version(slot.schema_version for slot in configured_slots)
        if is_layered(frame_version):
            print(f"Adding schema layers up to {frame_version} ...")
            # Layers calibrate on earlier games, so they run over every built
            # row's inputs; tonight's rows then take their columns.
            added = compute_new_columns(
                layer_inputs,
                from_version=BASE_SCHEMA_VERSION,
                to_version=frame_version,
                dataset_type=CLOSING_LINE,
                existing_columns=df_to_predict_total.columns,
            )
            added = added.loc[df_to_predict_total.index]
            df_to_predict_total = pd.concat([df_to_predict_total, added], axis=1)
        df_to_predict = df_to_predict_total[
            df_to_predict_total["GAME_DATE"] == date_to_predict
        ].copy()
        print_status("Feature DataFrame prepared")

    except Exception as e:
        print_status(f"Failed to prepare features: {e}", ok=False)
        raise

    if df_to_predict.empty:
        print("⚠️  Warning: No games found for the specified date.")
        raise ValueError("df to predict is empty")

    print_status(f"Found {len(df_to_predict)} game(s) to predict")

    # Everything that failed without stopping the run; non-empty means the run
    # exits non-zero once every slot and the baselines have had their turn.
    failures: list[str] = []

    # Intermediate slots predict a fixed time before tip-off: tonight's rows
    # for every horizon, built tonight-only from the live line history. A
    # failure here costs only those slots, never the closing ones.
    frames = {"closing_line": df_to_predict}
    # The intermediate builder has no schema-layer step yet, so its frame is
    # the base; a slot needing a newer version fails by name (see
    # load_registry_model_and_predict) instead of missing columns.
    frame_versions = {
        "closing_line": frame_version,
        "intermediate_line": BASE_SCHEMA_VERSION,
    }
    if any(slot.horizon_minutes for slot in configured_slots):
        print_step_header(3, "Preparing Intermediate Snapshot Rows")
        try:
            intermediate_df = tonight_intermediate_frame(
                scheduled_data, game_date=date_to_predict
            )
            frames["intermediate_line"] = intermediate_df
            print_status(
                f"{len(intermediate_df)} snapshot row(s) over "
                f"{intermediate_df['GAME_ID'].nunique()} game(s), "
                f"{int(intermediate_df['SNAPSHOT_PASSED'].sum())} already passed"
            )
        except Exception as e:  # noqa: BLE001 -- the closing slots still run
            traceback.print_exc()
            print_status(f"Intermediate rows unavailable: {e}", ok=False)
            failures.append(f"intermediate frame: {e}")

    if dump_frames is not None:
        dump_dir = Path(dump_frames)
        dump_dir.mkdir(parents=True, exist_ok=True)
        for dataset_type, frame in frames.items():
            path = dump_dir / f"{dataset_type}_{date_to_predict}.parquet"
            frame.to_parquet(path)
            print_status(f"Frame written to {path}")

    # Initialize S3 client once for all model operations
    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    pipeline_start_time = datetime.now(ZoneInfo("Europe/Madrid"))
    prediction_time = pipeline_start_time

    # Upload input features snapshot before running models
    print_step_header(4, "Uploading Input Features Snapshot")
    if not persist:
        print_status("Dry run (--no-persist): no predictions are stored")
    else:
        try:
            snapshot_key = upload_input_features_snapshot(
                s3_client=s3,
                bucket=SETTINGS.s3_bucket,
                pipeline_start_time=pipeline_start_time,
                date_to_predict=date_to_predict,
                df_to_predict=df_to_predict,
            )
            print_status(
                f"Input features uploaded to s3://{SETTINGS.s3_bucket}/{snapshot_key}"
            )
        except Exception as e:
            print_status(f"Failed to upload input features snapshot: {e}", ok=False)
            # Non-fatal: continue with predictions even if snapshot upload fails

    predicted: list[str] = []
    skipped: list[str] = []
    step_number = 5
    for slot in configured_slots:
        label = slot.describe()
        print_step_header(step_number, f"Generating Predictions ({label})")
        step_number += 1
        try:
            predictions_df = load_registry_model_and_predict(
                s3_client=s3,
                bucket=SETTINGS.s3_bucket,
                slot=slot,
                df=df_to_predict,
                prediction_datetime=prediction_time,
                frames=frames,
                frame_versions=frame_versions,
                passthrough_columns=SNAPSHOT_COLUMNS,
                persist=persist,
            )
        except Exception as e:  # noqa: BLE001 -- the other slots still run
            traceback.print_exc()
            print_status(f"Failed to generate predictions for {label}: {e}", ok=False)
            failures.append(f"{label}: {e}")
            continue
        if predictions_df is None or predictions_df.empty:
            # No frame of this model's dataset type, or no row of its
            # horizon. The reason is printed above; a missing intermediate
            # frame is already in the failures.
            print_status(f"No rows for {label}; skipped")
            skipped.append(label)
            continue

        print_status(
            f"Predictions generated for {label} ({len(predictions_df)} row(s))"
        )
        predicted.append(label)
        if not persist:
            continue

        # Upload model predictions snapshot
        try:
            pred_key = upload_model_predictions_snapshot(
                s3_client=s3,
                bucket=SETTINGS.s3_bucket,
                pipeline_start_time=pipeline_start_time,
                date_to_predict=date_to_predict,
                model_prefix=slot.prefix,
                predictions_df=predictions_df,
            )
            print_status(
                f"Predictions snapshot uploaded to s3://.../{pred_key.split('/', 3)[-1]}"
            )
        except Exception as snap_exc:
            print_status(
                f"Failed to upload predictions snapshot for {label}: {snap_exc}",
                ok=False,
            )

    print_step_header(step_number, "Generating Predictions (Baselines)")
    step_number += 1
    try:
        # The frame holds tonight's games only; the history is every built
        # game's layer inputs (GAME_DATE, TOTAL_POINTS among them).
        baseline_predictions = load_baseline_predictions_for_nba_games(
            df=df_to_predict,
            df_history=baseline_history_window(layer_inputs, date_to_predict),
            prediction_datetime=prediction_time,
            upload_to_postgres=persist,
        )
        print_status("Baseline predictions generated")

        for baseline_name, baseline_predictions_df in baseline_predictions.items():
            if not persist:
                break
            try:
                baseline_key = upload_model_predictions_snapshot(
                    s3_client=s3,
                    bucket=SETTINGS.s3_bucket,
                    pipeline_start_time=pipeline_start_time,
                    date_to_predict=date_to_predict,
                    model_prefix=baseline_name,
                    predictions_df=baseline_predictions_df,
                )
                print_status(
                    "Baseline snapshot uploaded to "
                    f"s3://.../{baseline_key.split('/', 3)[-1]}"
                )
            except Exception as snap_exc:
                print_status(
                    f"Failed to upload baseline snapshot for {baseline_name}: {snap_exc}",
                    ok=False,
                )

    except Exception as e:  # noqa: BLE001 -- reported with the slot failures
        traceback.print_exc()
        print_status(f"Failed to generate baseline predictions: {e}", ok=False)
        failures.append(f"baselines: {e}")

    print_step_header(step_number, "Summary")
    print_status(f"{len(predicted)} slot(s) predicted")
    print_status(f"{len(skipped)} slot(s) skipped", ok=not skipped)
    for label in skipped:
        print(f"      - {label}")
    print_status(f"{len(failures)} failure(s)", ok=not failures)
    for failure in failures:
        print(f"      - {failure}")
    if failures:
        raise SystemExit(1)

    if not run_tabpfn_client:
        print("\nPrediction pipeline completed successfully.")
        return

    # Generate predictions with TabPFN client
    print_step_header(step_number, "Generating Predictions (TabPFN Client)")
    try:
        tabpfn_predictions = load_and_predict_tabpfn_client_for_nba_games(
            df=df_to_predict_total,
            prediction_date=date_to_predict,
            prediction_datetime=prediction_time,
        )
        print_status("TabPFN predictions generated")

        # Upload TabPFN predictions snapshot
        try:
            tabpfn_key = upload_model_predictions_snapshot(
                s3_client=s3,
                bucket=SETTINGS.s3_bucket,
                pipeline_start_time=pipeline_start_time,
                date_to_predict=date_to_predict,
                model_prefix="tabpfn_client",
                predictions_df=tabpfn_predictions,
            )
            print_status(
                f"TabPFN snapshot uploaded to s3://.../{tabpfn_key.split('/', 3)[-1]}"
            )
        except Exception as snap_exc:
            print_status(
                f"Failed to upload TabPFN predictions snapshot: {snap_exc}", ok=False
            )

    except Exception as e:
        print_status(f"Failed to generate TabPFN predictions: {e}", ok=False)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NBA over/under predictions")
    parser.add_argument(
        "--no-tabpfn",
        action="store_true",
        help="Disable TabPFN predictions",
    )
    parser.add_argument(
        "--no-normalize-total-lines",
        action="store_true",
        help="Keep the original asymmetrically priced total lines",
    )
    parser.add_argument(
        "--no-normalize-spread-lines",
        action="store_true",
        help="Keep the original asymmetrically priced spread lines",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Predict without storing: no predictions table rows, no S3 snapshots",
    )
    parser.add_argument(
        "--dump-frames",
        metavar="DIR",
        help="Also write tonight's feature frames to DIR as Parquet",
    )
    parser.add_argument(
        "--keep-extreme-spread-prices",
        action="store_true",
        help="Keep extreme spread price cells instead of setting them to NaN.",
    )
    parser.add_argument(
        "--allow-stale-lines",
        action="store_true",
        help="Predict even when SBR's line feed looks frozen (lines hours old).",
    )
    args = parser.parse_args()

    predict_nba_games(
        run_tabpfn_client=not args.no_tabpfn,
        normalize_total_lines=not args.no_normalize_total_lines,
        normalize_spread_lines=not args.no_normalize_spread_lines,
        null_extreme_spread_prices=not args.keep_extreme_spread_prices,
        persist=not args.no_persist,
        dump_frames=args.dump_frames,
        allow_stale_lines=args.allow_stale_lines,
    )
