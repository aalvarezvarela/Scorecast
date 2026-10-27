import argparse
import os
import time
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
from nba_ou.fetch_data.scheduled_game.get_schedule_games import get_schedule_games
from nba_ou.postgre_db.predictions.create.create_ou_predictions_db import (
    upload_predictions_to_postgre,
)
from nba_ou.postgre_db.predictions_aiven.schema import WORKFLOW_DAILY
from nba_ou.postgre_db.update_all.update_all_databases import update_all_databases
from nba_ou.prediction.baseline_predictions import (
    baseline_history_window,
    load_baseline_predictions_for_nba_games,
)
from nba_ou.prediction.input_limits import LATEST_INPUT_COLUMN
from nba_ou.prediction.intermediate_frame import (
    PASSTHROUGH_COLUMNS,
    tonight_intermediate_frame,
)
from nba_ou.prediction.line_source import LINE_SOURCE_COLUMN, check_line_feed
from nba_ou.prediction.prediction import (
    load_registry_model_and_predict,
)
from nba_ou.prediction.prediction_store import (
    PredictionRunRecorder,
    persist_prediction_run,
)
from nba_ou.utils.general_utils import get_season_year_from_date
from nba_ou.utils.phase_timer import PhaseTimer
from nba_ou.utils.s3_models import (
    make_s3_client,
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
    normalize_total_lines: bool = True,
    normalize_spread_lines: bool = True,
    null_extreme_spread_prices: bool = True,
    persist: bool = True,
    dump_frames: str | Path | None = None,
    allow_stale_lines: bool = False,
    intermediate_input: str | None = None,
    line_source: str | None = None,
) -> None:
    """
    Main execution function for the NBA prediction pipeline.

    1. Updates the databases.
    2. Fetches tonight's games, referees, injuries and odds.
    3. Builds tonight's closing rows and, when any enabled slot has a horizon,
       tonight's intermediate rows for every horizon.
    4. Predicts with every enabled slot, whatever the time to tip: a closing
       slot predicts every game, an intermediate slot the row of its own
       horizon (passed or not; ``SNAPSHOT_PASSED`` tells them apart) -- or,
       in ``latest`` mode, each game's latest snapshot (``LATEST_INPUT_SLOTS``).
    5. Runs the baselines.

    ``intermediate_input`` (``latest`` or ``own``) overrides
    ``[PredictionModels] INTERMEDIATE_INPUT_MODE`` for this run.

    A slot that fails does not stop the others; the run exits non-zero at the
    end if anything failed. ``persist=False`` stores no predictions: no
    predictions-table rows and no S3 snapshots. The database update and the
    injury-report sync in steps 1-3 still write. ``dump_frames`` writes the
    feature frames to that local directory, for inspection.

    When SBR's feed has frozen (``feed_freshness``) or serves no odds, tonight's
    lines come from Covers instead (``prediction.line_source``), and each stored
    input row says which (``line_source``). Only when Covers is frozen too
    does the run stop after the database update with exit 1 and predict
    nothing; ``allow_stale_lines`` predicts anyway. ``line_source="covers"``
    forces the fallback.
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
    latest_slots = latest_input_slots(configured_slots, intermediate_input)
    print(
        f"  Intermediate input: {len(latest_slots)} slot(s) read the latest "
        "snapshot, the others their own"
    )

    # Each phase prints a [time] line as it ends, the summary their list.
    timer = PhaseTimer()

    # Step 1: Update the database
    print_step_header(1, "Updating All Databases")
    with timer.phase("database update"):
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
    # made on lines frozen hours ago; Covers stands in when it is not. The
    # database update above stands either way.
    print_step_header(1, "Checking SBR Line Feed")
    with timer.phase("SBR line feed"):
        lines = check_line_feed(
            pd.Timestamp(date_to_predict).date(),
            allow_stale_lines=allow_stale_lines,
            load_scheduled_games=lambda: get_schedule_games(date_to_predict),
            report=print_status,
            force=line_source,
        )

    # Step 2: Fetch scheduled games, referees, injuries and odds
    print_step_header(2, "Fetching Scheduled Games & Reports")
    with timer.phase("scheduled games and reports"):
        try:
            scheduled_data = get_all_info_for_scheduled_games(
                date_to_predict=date_to_predict,
                nba_injury_reports_url=SETTINGS.nba_injury_reports_url,
                headless=SETTINGS.headless,
                line_source=lines,
            )
            print_status("Fetched scheduled games, refs, injuries and odds")
            if scheduled_data["scheduled_games"].empty:
                print_status(
                    "⚠️ WARNING: No scheduled games found for the specified date.",
                    ok=False,
                )
                return 0

        except Exception as e:
            print_status(f"Failed to fetch scheduled data: {e}", ok=False)
            raise

    # Step 3: Build feature DataFrame for prediction
    print_step_header(3, "Preparing Feature DataFrame")
    # The builder produces the base schema. Each slot reads its own
    # feature_names, and newer versions only add columns, so one frame at
    # the newest version any enabled slot needs serves every slot.
    frame_version = newest_version(slot.schema_version for slot in configured_slots)
    tonight_ids = scheduled_data["scheduled_games"]["GAME_ID"].astype(str).tolist()
    # A closing row needs both teams' injury report (games_not_updated). With
    # none filed yet there is no closing row to build: the closing slots and
    # the baselines skip, and the intermediate slots still predict.
    not_filed = {str(game) for game in scheduled_data.get("games_not_updated", [])}
    closing_ids = [game for game in tonight_ids if game not in not_filed]
    with timer.phase("closing rows"):
        layer_inputs = None
        try:
            if not closing_ids:
                df_to_predict = pd.DataFrame()
                print_status(
                    "No game has both injury reports filed yet: no closing rows, "
                    "the closing slots and the baselines skip"
                )
            else:
                df_to_predict, layer_inputs = closing_rows(
                    scheduled_data,
                    tonight_ids=tonight_ids,
                    frame_version=frame_version,
                    date_to_predict=date_to_predict,
                    normalize_total_lines=normalize_total_lines,
                    normalize_spread_lines=normalize_spread_lines,
                    null_extreme_spread_prices=null_extreme_spread_prices,
                )
                print_status("Feature DataFrame prepared")

        except Exception as e:
            print_status(f"Failed to prepare features: {e}", ok=False)
            raise

    if closing_ids and df_to_predict.empty:
        print("⚠️  Warning: No games found for the specified date.")
        raise ValueError("df to predict is empty")

    if not df_to_predict.empty:
        print_status(f"Found {len(df_to_predict)} game(s) to predict")

    # Everything this run predicts, stored at the end in one piece: the S3 run
    # archive and the Aiven prediction store (docs/serving/prediction_storage.md).
    recorder = PredictionRunRecorder(workflow=WORKFLOW_DAILY, game_date=date_to_predict)
    recorder.set_schedule(scheduled_data["scheduled_games"])
    if not df_to_predict.empty:
        recorder.add_frame("closing_line", df_to_predict, version=frame_version)

    # Everything that failed without stopping the run; non-empty means the run
    # exits non-zero once every slot and the baselines have had their turn.
    failures: list[str] = []

    # Intermediate slots predict a fixed time before tip-off: tonight's rows
    # for every horizon, built tonight-only from the live line history. A
    # failure here costs only those slots, never the closing ones.
    # No closing frame when no closing row was built: the closing slots then
    # skip ("not this model's turn"), as a slot of an unbuilt frame does.
    frames = {} if df_to_predict.empty else {"closing_line": df_to_predict}
    # The intermediate builder has no schema-layer step yet, so its frame is
    # the base; a slot needing a newer version fails by name (see
    # load_registry_model_and_predict) instead of missing columns.
    frame_versions = {
        "closing_line": frame_version,
        "intermediate_line": BASE_SCHEMA_VERSION,
    }
    if any(slot.horizon_minutes for slot in configured_slots):
        print_step_header(3, "Preparing Intermediate Snapshot Rows")
        with timer.phase("intermediate rows"):
            try:
                intermediate_df = tonight_intermediate_frame(
                    scheduled_data,
                    game_date=date_to_predict,
                    latest_input=bool(latest_slots),
                    # Every intermediate slot reads the latest row: build it alone.
                    latest_only=all(
                        slot in latest_slots
                        for slot in configured_slots
                        if slot.horizon_minutes
                    ),
                )
                frames["intermediate_line"] = intermediate_df
                recorder.add_frame(
                    "intermediate_line", intermediate_df, version=BASE_SCHEMA_VERSION
                )
                print_status(
                    f"{len(intermediate_df)} snapshot row(s) over "
                    f"{intermediate_df['GAME_ID'].nunique()} game(s), "
                    f"{int(intermediate_df['SNAPSHOT_PASSED'].sum())} already passed"
                )
                if latest_slots:
                    latest_rows = intermediate_df[intermediate_df[LATEST_INPUT_COLUMN]]
                    print_status(
                        "Latest snapshots: "
                        + ", ".join(
                            f"{game} T-{minutes}"
                            for game, minutes in latest_rows[
                                ["GAME_ID", "TIME_TO_MATCH_MIN"]
                            ].itertuples(index=False)
                        )
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
    prediction_time = datetime.now(ZoneInfo("Europe/Madrid"))

    if not persist:
        print_status("Dry run (--no-persist): no predictions are stored")

    predicted: list[str] = []
    skipped: list[str] = []
    # The old Supabase table's failures: reported, but they do not make the
    # prediction store's run partial.
    legacy_failures: list[str] = []
    step_number = 4
    slots_started = time.perf_counter()
    for slot in configured_slots:
        label = slot.describe()
        print_step_header(step_number, f"Generating Predictions ({label})")
        step_number += 1
        slot_started = time.perf_counter()
        try:
            predictions_df = load_registry_model_and_predict(
                s3_client=s3,
                bucket=SETTINGS.s3_bucket,
                slot=slot,
                df=df_to_predict,
                prediction_datetime=prediction_time,
                frames=frames,
                frame_versions=frame_versions,
                passthrough_columns=PASSTHROUGH_COLUMNS,
                # Stored below, each destination on its own: a failing
                # Supabase write must not cost the prediction store its rows.
                persist=False,
                latest_input=slot in latest_slots,
                report_required_max_horizon=SETTINGS.report_required_max_horizon,
            )
        except Exception as e:  # noqa: BLE001 -- the other slots still run
            traceback.print_exc()
            print_status(f"Failed to generate predictions for {label}: {e}", ok=False)
            failures.append(f"{label}: {e}")
            recorder.record_slot(label, "failed", reason=str(e))
            continue
        rejected = recorder.add_predictions(predictions_df, label=label)
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
        if rejected:
            print_status(f"Prediction store rejected {label}: {rejected}", ok=False)
            failures.append(f"{label} (prediction store): {rejected}")
        upload_seconds = 0.0
        if persist:
            upload_started = time.perf_counter()
            legacy_failures += legacy_supabase_upload(predictions_df, label)
            upload_seconds = time.perf_counter() - upload_started
        print(
            f"[time] {label}: {time.perf_counter() - slot_started:.1f} s"
            + (f" (Supabase upload {upload_seconds:.1f} s)" if persist else "")
        )
    timer.record(f"{len(configured_slots)} slots", time.perf_counter() - slots_started)

    print_step_header(step_number, "Generating Predictions (Baselines)")
    step_number += 1
    if df_to_predict.empty:
        # The baselines predict the closing rows; none was built.
        print_status("No closing rows; baselines skipped")
    else:
        with timer.phase("baselines"):
            try:
                # The frame holds tonight's games only; the history is every built
                # game's layer inputs (GAME_DATE, TOTAL_POINTS among them).
                baseline_predictions = load_baseline_predictions_for_nba_games(
                    df=df_to_predict,
                    df_history=baseline_history_window(layer_inputs, date_to_predict),
                    prediction_datetime=prediction_time,
                    upload_to_postgres=False,
                )
                print_status("Baseline predictions generated")
                for (
                    baseline_name,
                    baseline_predictions_df,
                ) in baseline_predictions.items():
                    rejected = recorder.add_baseline(
                        baseline_name,
                        baseline_predictions_df,
                        schema_version=frame_version,
                    )
                    if rejected:
                        print_status(
                            f"Prediction store rejected {baseline_name}: {rejected}",
                            ok=False,
                        )
                        failures.append(
                            f"{baseline_name} (prediction store): {rejected}"
                        )
                    if persist:
                        legacy_failures += legacy_supabase_upload(
                            baseline_predictions_df, baseline_name
                        )

            except Exception as e:  # noqa: BLE001 -- reported with the slot failures
                traceback.print_exc()
                print_status(f"Failed to generate baseline predictions: {e}", ok=False)
                failures.append(f"baselines: {e}")

    if persist:
        print_step_header(step_number, "Storing Predictions")
        step_number += 1
        with timer.phase("storing"):
            failures += persist_prediction_run(
                recorder,
                s3_client=s3,
                bucket=SETTINGS.s3_bucket,
                partial=bool(failures),
            )
    failures += legacy_failures

    print_step_header(step_number, "Summary")
    print(timer.summary())
    print_status(f"{len(predicted)} slot(s) predicted")
    print_status(f"{len(skipped)} slot(s) skipped", ok=not skipped)
    for label in skipped:
        print(f"      - {label}")
    print_status(f"{len(failures)} failure(s)", ok=not failures)
    for failure in failures:
        print(f"      - {failure}")
    if failures:
        raise SystemExit(1)

    print("\nPrediction pipeline completed successfully.")


def closing_rows(
    scheduled_data: dict,
    *,
    tonight_ids: list[str],
    frame_version: str,
    date_to_predict: str,
    normalize_total_lines: bool,
    normalize_spread_lines: bool,
    null_extreme_spread_prices: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tonight's closing rows at ``frame_version``, and every built row's
    layer inputs (the baselines' history)."""
    # Tonight's games only, on the training build's full history: every
    # row is exactly what a full build would give it, at a fraction of
    # the work (create_df_to_predict's target_game_ids).
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
    # Set here, not in the builder: no feature step may ever see it.
    df_to_predict[LINE_SOURCE_COLUMN] = (
        df_to_predict["GAME_ID"].astype(str).map(scheduled_data.get("line_sources", {}))
    )
    return df_to_predict, layer_inputs


def legacy_supabase_upload(predictions: pd.DataFrame, label: str) -> list[str]:
    """Write ``predictions`` to the old Supabase table; the failure, if any.

    Runs in parallel with the prediction store during the transition
    (docs/serving/prediction_storage.md, rollout 3), and independently of it.
    """
    try:
        upload_predictions_to_postgre(predictions)
    except Exception as e:  # noqa: BLE001 -- reported with the run's failures
        print_status(f"Supabase write failed for {label}: {e}", ok=False)
        return [f"{label} (Supabase): {e}"]
    return []


def latest_input_slots(configured_slots: list, mode: str | None = None) -> list:
    """The enabled slots that read the latest snapshot on this run.

    ``mode`` overrides ``[PredictionModels] INTERMEDIATE_INPUT_MODE``. Only an
    intermediate slot reads one; a closing slot in the list keeps its frame.
    """
    mode = mode or SETTINGS.intermediate_input_mode
    if mode == "own":
        return []
    listed = SETTINGS.latest_input_slots
    if listed is None:
        return list(configured_slots)
    unknown = [slot.describe() for slot in listed if slot not in configured_slots]
    if unknown:
        print(f"  ⚠️ LATEST_INPUT_SLOTS rows not in ENABLED_MODELS: {unknown}")
    return [slot for slot in configured_slots if slot in listed]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NBA over/under predictions")
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
    parser.add_argument(
        "--intermediate-input",
        choices=("latest", "own"),
        help=(
            "Which intermediate rows the slots read: latest (each game's latest "
            "snapshot, for LATEST_INPUT_SLOTS) or own (every slot its own "
            "snapshot). Default: [PredictionModels] INTERMEDIATE_INPUT_MODE."
        ),
    )
    parser.add_argument(
        "--line-source",
        choices=("auto", "covers"),
        default="auto",
        help=(
            "Where tonight's lines come from: auto (SBR, Covers when SBR is "
            "frozen or empty) or covers (force the fallback, for tests)."
        ),
    )
    args = parser.parse_args()

    predict_nba_games(
        normalize_total_lines=not args.no_normalize_total_lines,
        normalize_spread_lines=not args.no_normalize_spread_lines,
        null_extreme_spread_prices=not args.keep_extreme_spread_prices,
        persist=not args.no_persist,
        dump_frames=args.dump_frames,
        allow_stale_lines=args.allow_stale_lines,
        intermediate_input=args.intermediate_input,
        line_source=args.line_source,
    )
