import argparse
import gc
import os
import time
import traceback
from collections.abc import Sequence
from dataclasses import replace
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
from nba_ou.postgre_db.config.connection_keep import kept_connections
from nba_ou.postgre_db.predictions_aiven.schema import WORKFLOW_DAILY
from nba_ou.postgre_db.update_all.update_all_databases import update_all_databases
from nba_ou.prediction.input_limits import LATEST_INPUT_COLUMN
from nba_ou.prediction.intermediate_frame import (
    PASSTHROUGH_COLUMNS,
    tonight_intermediate_frame,
)
from nba_ou.prediction.line_source import LINE_SOURCE_COLUMN, check_line_feed
from nba_ou.prediction.prediction import (
    clean_prediction_frames,
    load_registry_model_and_predict,
    read_slot_builds,
)
from nba_ou.prediction.prediction_store import (
    PredictionRunRecorder,
    persist_prediction_run,
)
from nba_ou.prediction.side_build import (
    SideBuild,
    SideBuildFailed,
    side_by_side_reason,
)
from nba_ou.prediction.tabpfn_run import (
    DEFAULT_TARGETS as TABPFN_TARGETS,
)
from nba_ou.prediction.tabpfn_run import (
    TabPFNJob,
    fit_and_store,
    load_fit_inputs,
    parse_jobs,
    tonight_rows,
)
from nba_ou.utils import history_cache
from nba_ou.utils.general_utils import get_season_year_from_date
from nba_ou.utils.phase_timer import PhaseTimer, duration
from nba_ou.utils.run_log import RunLog
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


def slot_overview(slots: list) -> list[str]:
    """The slots one line per schema, target and variant, with their horizons:
    ``2_5/line_error/main: t0000 t0030 t0060 (3)``."""
    horizons: dict[str, list[str]] = {}
    for slot in slots:
        key = f"{slot.schema_version}/{slot.target.value}/{slot.variant}"
        horizons.setdefault(key, []).append(slot.horizon_label)
    return [
        f"{key}: {' '.join(labels)} ({len(labels)})" for key, labels in horizons.items()
    ]


def predict_nba_games(**kwargs) -> None:
    """``run_prediction_pipeline`` with its log in numbered steps
    (``nba_ou.utils.run_log``), its builds sharing their loads of stored
    history (``nba_ou.utils.history_cache``) and its database connections
    (``connection_keep``)."""
    log = RunLog()
    timer = PhaseTimer()
    with timer.report("Prediction phase timings"):
        try:
            with history_cache.history_cache(), kept_connections():
                return run_prediction_pipeline(log, timer=timer, **kwargs)
        finally:
            # Timings and tracebacks stay outside the step's group.
            log.end_group()


def run_prediction_pipeline(
    log: RunLog,
    normalize_total_lines: bool = True,
    normalize_spread_lines: bool = True,
    null_extreme_spread_prices: bool = True,
    persist: bool = True,
    dump_frames: str | Path | None = None,
    allow_stale_lines: bool = False,
    intermediate_input: str | None = None,
    line_source: str | None = None,
    parallel_builds: bool | None = None,
    tabpfn_jobs: Sequence[TabPFNJob] = (),
    tabpfn_refresh: bool = False,
    timer: PhaseTimer | None = None,
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

    The daily baselines run in predict_past_snapshots.py after the slate.

    ``intermediate_input`` (``latest`` or ``own``) overrides
    ``[PredictionModels] INTERMEDIATE_INPUT_MODE`` for this run.

    A slot that fails does not stop the others; the run exits non-zero at the
    end if anything failed. ``persist=False`` stores no predictions: no
    prediction-store rows and no S3 run archive. The database update and the
    injury-report sync in steps 1-3 still write. ``dump_frames`` writes the
    feature frames to that local directory, for inspection.

    When SBR's feed has frozen (``feed_freshness``) or serves no odds, tonight's
    lines come from Covers instead (``prediction.line_source``), and each stored
    input row says which (``line_source``). Only when Covers is frozen too
    does the run stop after the database update with exit 1 and predict
    nothing; ``allow_stale_lines`` predicts anyway. ``line_source="covers"``
    forces the fallback.

    ``parallel_builds`` overrides ``[PredictionModels] PARALLEL_BUILDS``: the
    closing rows build in a side process beside the intermediate rows while
    both stay warm (``nba_ou.prediction.side_build``).

    ``tabpfn_jobs`` (``nba_ou.prediction.tabpfn_run``) run once this run's
    predictions are stored, each a TabPFN run of its own, on the schedule,
    referees, reports and line source fetched above; ``tabpfn_refresh``
    fetches those again and builds the rows at that moment instead of at each
    job's snapshot. A failed job fails the run at the end, as a slot does.
    """
    configure_tqdm_for_environment()

    date_to_predict = (datetime.now(ZoneInfo("US/Pacific"))).strftime("%Y-%m-%d")

    # Print welcome banner
    print_banner(date_to_predict)
    configured_slots = SETTINGS.prediction_model_slots
    if not configured_slots:
        # A valid state, not an error: nothing has been promoted yet. The rest
        # of the pipeline (database update, feature build, TabPFN) is still
        # worth running.
        print("  No model slots enabled in [PredictionModels] ENABLED_MODELS.")
    else:
        print(f"  Enabled model slots ({len(configured_slots)}):")
        for line in slot_overview(configured_slots):
            print(f"    - {line}")
    latest_slots = latest_input_slots(configured_slots, intermediate_input)
    print(
        f"  Intermediate input: {len(latest_slots)} slot(s) read the latest "
        "snapshot, the others their own"
    )
    if not persist:
        print_status("Dry run (--no-persist): no predictions are stored")
    for job in tabpfn_jobs:
        print(
            f"  {describe_tabpfn_job(job)}" + (", refreshed" if tabpfn_refresh else "")
        )

    # Each phase prints a [time] line as it ends, the summary their list.
    timer = timer if timer is not None else PhaseTimer()

    # Step 1: Update the database
    log.step("Updating All Databases")
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
    log.step("Checking SBR Line Feed")
    with timer.phase("SBR line feed"):
        lines = check_line_feed(
            pd.Timestamp(date_to_predict).date(),
            allow_stale_lines=allow_stale_lines,
            load_scheduled_games=lambda: get_schedule_games(date_to_predict),
            report=print_status,
            force=line_source,
        )

    # Step 2: Fetch scheduled games, referees, injuries and odds
    log.step("Fetching Scheduled Games & Reports")
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
                log.warning("No scheduled games found for the specified date.")
                return 0

        except Exception as e:
            print_status(f"Failed to fetch scheduled data: {e}", ok=False)
            raise
    # Nothing this run reads is written from here on, so the builds share
    # their loads of stored history; one read before the update would not do.
    history_cache.clear()

    # Step 3: Build feature DataFrame for prediction
    # The builder produces the base schema. Each slot reads its own
    # feature_names, and newer versions only add columns, so one frame at
    # the newest version any enabled slot needs serves every slot.
    frame_version = newest_version(slot.schema_version for slot in configured_slots)
    tonight_ids = scheduled_data["scheduled_games"]["GAME_ID"].astype(str).tolist()
    # A closing row needs both teams' injury report (games_not_updated). With
    # none filed yet there is no closing row to build: the closing slots skip
    # and intermediate slots still predict.
    not_filed = {str(game) for game in scheduled_data.get("games_not_updated", [])}
    closing_ids = [game for game in tonight_ids if game not in not_filed]
    closing_kwargs = dict(
        tonight_ids=tonight_ids,
        frame_version=frame_version,
        date_to_predict=date_to_predict,
        normalize_total_lines=normalize_total_lines,
        normalize_spread_lines=normalize_spread_lines,
        null_extreme_spread_prices=null_extreme_spread_prices,
    )
    wants_intermediate = any(slot.horizon_minutes for slot in configured_slots)

    # The two builds are independent until the predictions. On a warm run the
    # closing rows build in a side process while this one builds the
    # intermediate rows; a build that turns cold waits for the other and
    # finishes alone (nba_ou.prediction.side_build).
    side = None
    reason = None
    if closing_ids and wants_intermediate:
        if parallel_builds is None:
            parallel_builds = SETTINGS.parallel_builds
        reason = side_by_side_reason() if parallel_builds else "turned off"
        if reason is None:
            side = SideBuild.start(
                closing_rows,
                scheduled_data,
                current_season=int(season_to_update),
                label="closing rows",
                own_label="intermediate rows",
                **closing_kwargs,
            )
            side_started = time.perf_counter()
    if side is None:
        log.step("Preparing Closing Rows")
        if reason is not None:
            print_status(f"Closing and intermediate rows one after the other: {reason}")
        df_to_predict, layer_inputs = timed_closing_rows(
            timer, scheduled_data, closing_ids, **closing_kwargs
        )
        check_closing_rows(df_to_predict, closing_ids)

    # Everything that failed without stopping the run; non-empty means the run
    # exits non-zero once every slot has had its turn.
    failures: list[str] = []

    # Intermediate slots predict a fixed time before tip-off: tonight's rows
    # for every horizon, built tonight-only from the live line history. A
    # failure here costs only those slots, never the closing ones.
    intermediate_df = None
    if wants_intermediate:
        log.step("Preparing Intermediate Snapshot Rows")
        if side is not None:
            print_status(
                "The closing rows build meanwhile in a side process; its log "
                "follows in the next step"
            )
        try:
            with timer.phase("intermediate rows"):
                try:
                    intermediate_df = tonight_intermediate_frame(
                        scheduled_data,
                        game_date=date_to_predict,
                        latest_input=bool(latest_slots),
                        # Every intermediate slot reads the latest row: build
                        # it alone.
                        latest_only=all(
                            slot in latest_slots
                            for slot in configured_slots
                            if slot.horizon_minutes
                        ),
                    )
                    print_status(
                        f"{len(intermediate_df)} snapshot row(s) over "
                        f"{intermediate_df['GAME_ID'].nunique()} game(s), "
                        f"{int(intermediate_df['SNAPSHOT_PASSED'].sum())} "
                        "already passed"
                    )
                    if latest_slots:
                        latest_rows = intermediate_df[
                            intermediate_df[LATEST_INPUT_COLUMN]
                        ]
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
        finally:
            # A side build waiting on the gate must not outlive this process.
            if side is not None:
                side.release_gate()

    if side is not None:
        log.step("Closing Rows (side process)")
        df_to_predict, layer_inputs = side_closing_rows(
            side, timer, log, scheduled_data, closing_ids, **closing_kwargs
        )
        timer.record(
            "closing and intermediate rows, side by side",
            time.perf_counter() - side_started,
        )
        check_closing_rows(df_to_predict, closing_ids)

    # Everything this run predicts, stored at the end in one piece: the S3 run
    # archive and the Aiven prediction store (docs/serving/prediction_storage.md).
    recorder = PredictionRunRecorder(workflow=WORKFLOW_DAILY, game_date=date_to_predict)
    recorder.set_schedule(scheduled_data["scheduled_games"])
    # No closing frame when no closing row was built: the closing slots then
    # skip ("not this model's turn"), as a slot of an unbuilt frame does.
    frames = {}
    if not df_to_predict.empty:
        frames["closing_line"] = df_to_predict
        recorder.add_frame("closing_line", df_to_predict, version=frame_version)
    if intermediate_df is not None:
        frames["intermediate_line"] = intermediate_df
        recorder.add_frame(
            "intermediate_line", intermediate_df, version=BASE_SCHEMA_VERSION
        )
    # The intermediate builder has no schema-layer step yet, so its frame is
    # the base; a slot needing a newer version fails by name (see
    # load_registry_model_and_predict) instead of missing columns.
    frame_versions = {
        "closing_line": frame_version,
        "intermediate_line": BASE_SCHEMA_VERSION,
    }

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

    log.step("Cleaning Rows and Reading Models")
    # Cleaned once here, not once per slot: each slot reads its rows from it.
    with timer.phase("cleaning"):
        clean_frames = clean_prediction_frames(
            frames, passthrough_columns=PASSTHROUGH_COLUMNS
        )

    with timer.phase("model reads"):
        builds = read_slot_builds(
            s3_client=s3, bucket=SETTINGS.s3_bucket, slots=configured_slots
        )

    predicted: list[str] = []
    skipped: list[str] = []
    log.step(f"Generating Predictions ({len(configured_slots)} slots)")
    slots_started = time.perf_counter()
    for index, slot in enumerate(configured_slots, start=1):
        label = slot.describe()
        print(f"[slot {index}/{len(configured_slots)}] {label}", flush=True)
        slot_started = time.perf_counter()
        try:
            build = builds.get(slot)
            if isinstance(build, Exception):
                raise build
            predictions_df = load_registry_model_and_predict(
                s3_client=s3,
                bucket=SETTINGS.s3_bucket,
                slot=slot,
                df=df_to_predict,
                prediction_datetime=prediction_time,
                frames=frames,
                frame_versions=frame_versions,
                passthrough_columns=PASSTHROUGH_COLUMNS,
                latest_input=slot in latest_slots,
                report_required_max_horizon=SETTINGS.report_required_max_horizon,
                clean_frames=clean_frames,
                resolved=build,
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
            print(f"  - {label}: no rows; skipped")
            skipped.append(label)
            continue

        print_status(
            f"{label}: {len(predictions_df)} row(s), "
            f"{time.perf_counter() - slot_started:.1f} s"
        )
        predicted.append(label)
        if rejected:
            print_status(f"Prediction store rejected {label}: {rejected}", ok=False)
            failures.append(f"{label} (prediction store): {rejected}")
    timer.record(f"{len(configured_slots)} slots", time.perf_counter() - slots_started)

    if persist:
        log.step("Storing Predictions")
        with timer.phase("storing"):
            failures += persist_prediction_run(
                recorder,
                s3_client=s3,
                bucket=SETTINGS.s3_bucket,
                partial=bool(failures),
            )

    tabpfn_predicted: list[str] = []
    if tabpfn_jobs:
        # After the stored predictions: a TabPFN failure or slow fit never
        # costs them. Their frames are not read again; free them first.
        del frames, clean_frames, builds, recorder
        del df_to_predict, intermediate_df, layer_inputs
        gc.collect()
        tabpfn_predicted, tabpfn_failures = tabpfn_after_predictions(
            tabpfn_jobs,
            scheduled_data,
            lines,
            s3_client=s3,
            date_to_predict=date_to_predict,
            timer=timer,
            log=log,
            persist=persist,
            refresh=tabpfn_refresh,
        )
        failures += tabpfn_failures

    log.step("Summary", group=False)
    print_status(f"{len(predicted)} slot(s) predicted")
    print_status(f"{len(skipped)} slot(s) skipped", ok=not skipped)
    for label in skipped:
        print(f"      - {label}")
    if tabpfn_jobs:
        print_status(f"{len(tabpfn_predicted)} TabPFN model(s) predicted")
    print_status(f"{len(failures)} failure(s)", ok=not failures)
    for failure in failures:
        log.error(failure)
    log.summary(
        "\n".join(
            [
                f"### Predictions for {date_to_predict}",
                "",
                f"{len(predicted)} slot(s) predicted, {len(skipped)} skipped, "
                + (
                    f"{len(tabpfn_predicted)} TabPFN model(s) predicted, "
                    if tabpfn_jobs
                    else ""
                )
                + f"{len(failures)} failure(s)"
                + ("" if persist else " (dry run: nothing stored)"),
                "",
                *(f"- ✖ {failure}" for failure in failures),
                *(f"- skipped: {label}" for label in skipped),
            ]
        )
    )
    if failures:
        raise SystemExit(1)

    print("\nPrediction pipeline completed successfully.")


def describe_tabpfn_job(job: TabPFNJob) -> str:
    """``TabPFN t0720 (standard: line_error, total_points) at the snapshot of
    2026-10-28 23:00 UTC``."""
    at = (
        "now"
        if job.as_of is None
        else f"the snapshot of {job.as_of.tz_convert('UTC'):%Y-%m-%d %H:%M} UTC"
    )
    return f"{job.name} ({job.plan}) at {at}"


def tabpfn_after_predictions(
    jobs: Sequence[TabPFNJob],
    scheduled_data: dict,
    lines,
    *,
    s3_client,
    date_to_predict: str,
    timer: PhaseTimer,
    log: RunLog,
    persist: bool,
    refresh: bool,
) -> tuple[list[str], list[str]]:
    """Each TabPFN job, a TabPFN run of its own (``nba_ou.prediction.tabpfn_run``):
    the labels it predicted and its failures. A job that fails costs only
    itself.

    The jobs read this run's schedule, referees, reports and line source
    (``lines``), and a scheduled job's rows are cut at its snapshot, as a
    TabPFN run on its own cuts them. With ``refresh`` those are fetched again
    first, and the rows are built at this moment instead.
    """
    predicted: list[str] = []
    failures: list[str] = []
    for job in jobs:
        prefix = f"{job.name}: "
        log.step(f"{prefix}Loading the Feature Spec and Training Frame")
        print_status(describe_tabpfn_job(job))
        try:
            with timer.phase(f"{prefix}spec and training rows"):
                inputs = load_fit_inputs(
                    job,
                    s3_client=s3_client,
                    bucket=SETTINGS.s3_bucket,
                    slots=SETTINGS.prediction_model_slots,
                    date_to_predict=date_to_predict,
                )
            data = scheduled_data
            if refresh:
                log.step(f"{prefix}Fetching Scheduled Games & Reports Again")
                with timer.phase(f"{prefix}scheduled games and reports"):
                    data = get_all_info_for_scheduled_games(
                        date_to_predict=date_to_predict,
                        nba_injury_reports_url=SETTINGS.nba_injury_reports_url,
                        headless=SETTINGS.headless,
                        line_source=lines,
                        dataset_type=job.dataset_type,
                    )
                history_cache.clear()
                # The rows of this moment, not of the booked snapshot.
                job = replace(job, as_of=None, snapshot_games=None)
            started = time.perf_counter()
            rows = tonight_rows(
                job,
                data,
                inputs.specs,
                date_to_predict=date_to_predict,
                log=log,
                step_prefix=prefix,
            )
            if rows is None:
                continue
            timer.record(f"{prefix}tonight's rows", time.perf_counter() - started)
        except Exception as e:  # noqa: BLE001 -- the other jobs still run
            traceback.print_exc()
            failures.append(f"{job.name}: {type(e).__name__}: {e}")
            print_status(failures[-1], ok=False)
            continue
        outcome = fit_and_store(
            job,
            inputs,
            rows,
            data,
            s3_client=s3_client,
            bucket=SETTINGS.s3_bucket,
            date_to_predict=date_to_predict,
            timer=timer,
            log=log,
            persist=persist,
            step_prefix=prefix,
        )
        predicted += outcome.predicted
        failures += outcome.failures
    return predicted, failures


def timed_closing_rows(
    timer: PhaseTimer, scheduled_data: dict, closing_ids: list[str], **closing_kwargs
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The closing rows built in this process, as the "closing rows" phase.

    A run with no game whose both injury reports are filed has none to build.
    """
    with timer.phase("closing rows"):
        try:
            if not closing_ids:
                print_status(
                    "No game has both injury reports filed yet: no closing rows, "
                    "the closing slots skip"
                )
                return pd.DataFrame(), None
            df_to_predict, layer_inputs = closing_rows(scheduled_data, **closing_kwargs)
            print_status("Feature DataFrame prepared")
            return df_to_predict, layer_inputs
        except Exception as e:
            print_status(f"Failed to prepare features: {e}", ok=False)
            raise


def side_closing_rows(
    side: SideBuild,
    timer: PhaseTimer,
    log: RunLog,
    scheduled_data: dict,
    closing_ids,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """The closing rows of the side build; built here if its process died."""
    try:
        outcome = side.result()
    except SideBuildFailed as e:
        print_status(f"Failed to prepare features: {e}", ok=False)
        raise
    if outcome is None:
        log.warning("Closing rows lost with the side process: built here")
        return timed_closing_rows(timer, scheduled_data, closing_ids, **kwargs)
    timer.record("closing rows (side process)", outcome.seconds, peak=outcome.peak)
    # A cold build computed past seasons the history memo should have held:
    # the reason a run is slow, worth seeing on the run's page.
    for build, cause, waited, other in (
        ("Closing rows", outcome.cold_cause, outcome.waited, "intermediate rows"),
        ("Intermediate rows", side.gate.cold_cause, side.gate.waited, "closing rows"),
    ):
        if cause:
            log.warning(
                f"{build} were cold ({cause}): they computed past seasons"
                + (
                    f", after waiting {duration(waited)} for the {other}"
                    if waited >= 1
                    else ""
                )
            )
    print_status("Feature DataFrame prepared")
    return outcome.value


def check_closing_rows(df_to_predict: pd.DataFrame, closing_ids: list[str]) -> None:
    if closing_ids and df_to_predict.empty:
        print("⚠️  Warning: No games found for the specified date.")
        raise ValueError("df to predict is empty")
    if not df_to_predict.empty:
        print_status(f"Found {len(df_to_predict)} game(s) to predict")


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
    layer inputs (used by newer schema layers)."""
    # Tonight's games only, on the training build's full history: every
    # row is exactly what a full build would give it, at a fraction of
    # the work (create_df_to_predict's target_game_ids).
    df_to_predict_total, layer_inputs = create_df_to_predict(
        todays_prediction=True,
        scheduled_data=scheduled_data,
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
        help="Predict without storing: no prediction-store rows, no S3 run archive",
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
    parser.add_argument(
        "--sequential-builds",
        action="store_true",
        help=(
            "Build the closing and intermediate rows one after the other, "
            "never side by side ([PredictionModels] PARALLEL_BUILDS)."
        ),
    )
    parser.add_argument(
        "--tabpfn",
        default="",
        metavar="JOBS",
        help=(
            "TabPFN jobs to run once the predictions are stored, separated by ';': "
            "HORIZON[@AS_OF[@GAME_IDS[@FITS]]], e.g. "
            "'720@2026-10-28T23:00:00+00:00@0022600131' (0: the closing line). "
            "FITS ('standard=line_error,total_points thinking_high=line_error') "
            "replaces --tabpfn-targets and --tabpfn-modes for that job. "
            "Each is a TabPFN run of its own (scripts/predict_tabpfn.py)."
        ),
    )
    parser.add_argument(
        "--tabpfn-targets",
        nargs="+",
        default=list(TABPFN_TARGETS),
        help=f"Targets of the TabPFN jobs (default: {' '.join(TABPFN_TARGETS)}).",
    )
    parser.add_argument(
        "--tabpfn-modes",
        nargs="+",
        default=["standard"],
        help="TabPFN modes of the jobs: standard, thinking_medium, thinking_high.",
    )
    parser.add_argument(
        "--tabpfn-refresh",
        action="store_true",
        help=(
            "Fetch the schedule, reports and odds again before the TabPFN jobs, "
            "and build their rows at that moment instead of at their snapshot."
        ),
    )
    args = parser.parse_args()
    try:
        tabpfn_jobs = parse_jobs(
            args.tabpfn, targets=args.tabpfn_targets, modes=args.tabpfn_modes
        )
    except ValueError as exc:
        parser.error(str(exc))

    predict_nba_games(
        normalize_total_lines=not args.no_normalize_total_lines,
        normalize_spread_lines=not args.no_normalize_spread_lines,
        null_extreme_spread_prices=not args.keep_extreme_spread_prices,
        persist=not args.no_persist,
        dump_frames=args.dump_frames,
        allow_stale_lines=args.allow_stale_lines,
        intermediate_input=args.intermediate_input,
        line_source=args.line_source,
        parallel_builds=False if args.sequential_builds else None,
        tabpfn_jobs=tabpfn_jobs,
        tabpfn_refresh=args.tabpfn_refresh,
    )
