import argparse
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.create_df_to_predict import (
    create_df_to_predict,
)
from nba_ou.create_training_data.get_all_info_for_scheduled_games import (
    get_all_info_for_scheduled_games,
)
from nba_ou.postgre_db.update_all.update_all_databases import update_all_databases
from nba_ou.prediction.baseline_predictions import (
    load_baseline_predictions_for_nba_games,
)
from nba_ou.prediction.prediction import (
    load_registry_model_and_predict,
)
from nba_ou.prediction.prediction_tabpfn_client import (
    load_and_predict_tabpfn_client_for_nba_games,
)
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
    header = f" STEP {step_number} — {title} "
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
) -> None:
    """
    Main execution function for the NBA prediction pipeline.

    This function:
    1. Parses command-line arguments for the prediction date
    2. Loads configuration settings
    3. Updates the game database
    4. Processes data with injuries
    5. Generates predictions
    6. Saves results to Excel
    """
    configure_tqdm_for_environment()

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
        df_to_predict_total = create_df_to_predict(
            todays_prediction=True,
            scheduled_data=scheduled_data,
            strict_mode=30,
            normalize_total_lines=normalize_total_lines,
            normalize_spread_lines=normalize_spread_lines,
            null_extreme_spread_prices=null_extreme_spread_prices,
        )
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

    # Initialize S3 client once for all model operations
    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    pipeline_start_time = datetime.now(ZoneInfo("Europe/Madrid"))
    prediction_time = pipeline_start_time

    # Upload input features snapshot before running models
    print_step_header(4, "Uploading Input Features Snapshot")
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
            )
            if predictions_df is None or predictions_df.empty:
                # The horizon guard found no eligible rows: this model is not
                # for the time of day the pipeline is running at. Not a
                # failure, and the other slots still have work to do.
                print_status(f"No eligible rows for {label}; skipped")
                continue

            print_status(f"Predictions generated for {label}")

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

        except Exception as e:
            print_status(f"Failed to generate predictions for {label}: {e}", ok=False)
            raise

    print_step_header(step_number, "Generating Predictions (Baselines)")
    step_number += 1
    try:
        baseline_predictions = load_baseline_predictions_for_nba_games(
            df=df_to_predict,
            df_history=df_to_predict_total,
            prediction_datetime=prediction_time,
        )
        print_status("Baseline predictions generated")

        for baseline_name, baseline_predictions_df in baseline_predictions.items():
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

    except Exception as e:
        print_status(f"Failed to generate baseline predictions: {e}", ok=False)
        raise

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
        "--keep-extreme-spread-prices",
        action="store_true",
        help="Keep extreme spread price cells instead of setting them to NaN.",
    )
    args = parser.parse_args()

    predict_nba_games(
        run_tabpfn_client=not args.no_tabpfn,
        normalize_total_lines=not args.no_normalize_total_lines,
        normalize_spread_lines=not args.no_normalize_spread_lines,
        null_extreme_spread_prices=not args.keep_extreme_spread_prices,
    )
