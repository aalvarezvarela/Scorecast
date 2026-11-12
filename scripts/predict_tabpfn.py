"""Predict tonight's games with TabPFN, on its own.

    python scripts/predict_tabpfn.py [--targets line_error total_points spread_error] [--no-persist]
    python scripts/predict_tabpfn.py --modes standard thinking_high
    python scripts/predict_tabpfn.py --fits "standard=line_error,total_points,spread_error thinking_high=line_error,total_points"
    python scripts/predict_tabpfn.py --line intermediate [--horizon-minutes 360]
    python scripts/predict_tabpfn.py --line intermediate --horizon-minutes 720 \
        --game-date 2026-10-26 --as-of 2026-10-26T11:00:00Z

One or more targets and one line per run (``line_error``, ``total_points`` and ``spread_error``
at the closing line by default; ``--line intermediate`` follows the slot of a fixed time before
tip-off, T-360 by default). TabPFN keeps no model, so each run fits it on the
valid historical rows since 2019-01-01, using that slot's production features
plus numeric SEASON_YEAR (the season's starting year) for TabPFN only, and
TabPFN's own NaN budget (800 per row) instead of the spec's
(``nba_ou.prediction.prediction_tabpfn_client``):

1. The production spec of the target's slot at that line (its features) and
   its training rows: the season partitions the daily
   refit keeps current (``nba_ou.create_training_data.season_partitions``),
   read as the refit reads them -- every season for the closing line, the
   horizon's rows for an intermediate slot. TabPFN uses all valid games since
   2019-01-01, without the spec's rolling window, unless --train-games caps
   that history. Both are loaded before any other
   work, and partitions older than ``--max-frame-age-days`` are refused.
2. The database update, the SBR feed check and tonight's schedule, referees,
   injuries and odds, fetched as the daily predictor fetches them. Tonight's
   rows are built from them, so they carry the lines of this moment: the
   closing rows, or the intermediate row of the horizon -- built from the
   ticks so far when the snapshot is still ahead (``SNAPSHOT_PASSED`` False),
   as the daily predictor does. Live/manual runs predict games not yet
   started. Scheduled intermediate jobs (as_of and snapshot_game_ids) rebuild
   the whole slate, including started/final games, at their own pregame
   snapshots; only that horizon's rows are built.
3. TabPFN is fitted separately for each target and mode (``--modes``: the
   standard model by default, or thinking at medium or high effort, each
   stored as a variant of its own) on its spec's rows and features; ``--fits``
   limits each mode to some targets. Tonight's
   rows are prepared once and all outputs share one prediction run.
   The run goes to the prediction store (S3 run archive + Aiven) with
   ``runs.workflow`` 3 and ``model_builds.kind`` 3.

``--with-closing`` (with ``--line intermediate --input latest``; the
dashboard's joint request) predicts the closing line in the same run: the
schedule, the reports and the latest snapshot frame are read once for both
lines and the closing rows reuse the history the intermediate build loaded.
Each line is its own stored run and answers its own request
(``--request-id``, ``--closing-request-id``).

The closing rows keep no raw quote. As in the daily predictor, a closing run
also builds each game's latest snapshot row, only for the store: the bet line,
prices and cross-book lines of this moment come from it
(``PredictionRunRecorder``). If that build fails the run goes on and the bet
line stays null; the model line (``inputs.total_model_line``) is the line of
this moment either way.
"""

from __future__ import annotations

import argparse
import gc
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from nba_ou.config.dataset_versions import BASE_SCHEMA_VERSION
from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.get_all_info_for_scheduled_games import (
    get_all_info_for_scheduled_games,
)
from nba_ou.create_training_data.schema_layers import CLOSING_LINE
from nba_ou.fetch_data.scheduled_game.get_schedule_games import get_schedule_games
from nba_ou.modeling.registry_paths import ModelSlot, format_horizon
from nba_ou.postgre_db.app_aiven.requests import final_status, report_request
from nba_ou.postgre_db.app_aiven.schema import (
    REQUEST_DONE,
    REQUEST_FAILED,
    REQUEST_RUNNING,
)
from nba_ou.postgre_db.config.connection_keep import kept_connections
from nba_ou.postgre_db.update_all.update_all_databases import update_all_databases
from nba_ou.prediction.horizon_choice import assign_horizons, group_by_horizon
from nba_ou.prediction.line_source import check_line_feed
from nba_ou.prediction.prediction_tabpfn_client import (
    DEFAULT_INTERMEDIATE_HORIZON,
    DEFAULT_MODEL_VERSION,
    DEFAULT_MODES,
    INTERMEDIATE_LINE,
    MODES,
    parse_modes,
)
from nba_ou.prediction.tabpfn_run import (
    DEFAULT_MAX_FRAME_AGE_DAYS,
    DEFAULT_TARGETS,
    FitInputs,
    JobOutcome,
    TabPFNJob,
    TabPFNUnavailable,
    TonightRows,
    build_latest_frame,
    fit_and_store,
    latest_rows,
    load_fit_inputs,
    parse_fits,
    parse_game_ids,
    parse_targets,
    restrict_to_games,
    tonight_rows,
)
from nba_ou.utils.general_utils import get_season_year_from_date
from nba_ou.utils.history_cache import history_cache
from nba_ou.utils.phase_timer import PhaseTimer
from nba_ou.utils.run_log import RunLog
from nba_ou.utils.s3_models import make_s3_client

#: ``--line``: the dataset type of the slot TabPFN follows.
LINES = {"closing": CLOSING_LINE, "intermediate": INTERMEDIATE_LINE}


def print_status(message: str, ok: bool = True) -> None:
    print(f"  {'✓' if ok else '✖'} {message}")


def configure_tqdm_for_environment() -> None:
    """Disable tqdm in GitHub Actions, as the daily predictor does."""
    if os.getenv("GITHUB_ACTIONS", "").lower() == "true":
        os.environ.setdefault("TQDM_DISABLE", "1")


def predict_tabpfn(**kwargs) -> None:
    """``run_tabpfn`` with its log in numbered steps (``nba_ou.utils.run_log``).

    With a ``request_id`` (a dashboard request) the request is marked
    running now and finished at the end; a run that predicts nothing or dies
    on the way marks it failed.
    """
    log = RunLog()
    timer = PhaseTimer()
    # A joint run answers two requests: the intermediate one and the closing one.
    request_ids = (kwargs.get("request_id"), kwargs.get("closing_request_id"))
    for request_id in request_ids:
        report_request(request_id, REQUEST_RUNNING)
    with timer.report("TabPFN phase timings"):
        try:
            result = run_tabpfn(log, timer=timer, **kwargs)
        except BaseException as exc:
            # A no-op when the run already wrote its final status.
            for request_id in request_ids:
                report_request(
                    request_id, REQUEST_FAILED, error=f"{type(exc).__name__}: {exc}"
                )
            raise
        finally:
            # Timings and tracebacks stay outside the step's group.
            log.end_group()
    # No-ops for a request the run already finished.
    for request_id in request_ids:
        if result is None:
            report_request(request_id, REQUEST_FAILED, error="Nothing to predict")
        else:
            report_request(request_id, REQUEST_DONE)
    return result


def run_tabpfn(
    log: RunLog,
    *,
    timer: PhaseTimer | None = None,
    targets: Sequence[str] = DEFAULT_TARGETS,
    line: str = "closing",
    horizon_minutes: int | None = None,
    game_date: str | None = None,
    game_ids: Sequence[str] | None = None,
    as_of: str | None = None,
    snapshot_game_ids: Sequence[str] | None = None,
    feature_slot: ModelSlot | None = None,
    model_version: str = DEFAULT_MODEL_VERSION,
    modes: Sequence[str] = DEFAULT_MODES,
    mode_targets: Sequence[tuple[str, Sequence[str]]] = (),
    train_games: int | None = None,
    max_frame_age_days: int = DEFAULT_MAX_FRAME_AGE_DAYS,
    update_db: bool = True,
    allow_stale_lines: bool = False,
    persist: bool = True,
    input_mode: str = "own",
    nearest: bool = False,
    request_id: int | None = None,
    with_closing: bool = False,
    closing_request_id: int | None = None,
) -> bool | None:
    """Fit TabPFN for each target and predict shared tonight's rows at ``line``.

    ``input_mode="latest"`` (intermediate, live; the dashboard's manual runs)
    predicts each game's latest snapshot with the T-``horizon_minutes`` spec
    and training rows. ``nearest`` (latest only) fits instead, per group of
    games, the horizon nearest their minutes to tip-off
    (``nba_ou.prediction.horizon_choice``): games at 4, 8 and 12 hours are
    three fits on the T-240, T-480 and T-720 rows, each one run, all with
    ``request_id``. True once the run has predicted; None when there was
    nothing to predict.

    ``line`` is ``closing`` or ``intermediate``; ``horizon_minutes`` is the
    intermediate slot's time before tip-off (default
    ``DEFAULT_INTERMEDIATE_HORIZON``). ``feature_slot`` defaults to the enabled
    slot of each target at that line. An explicit ``feature_slot`` requires
    one target. ``modes`` are the fits of each target (``MODES``; the standard
    model by default), each stored as its own variant; ``mode_targets``
    limits a mode to some targets (``parse_fits``). ``persist=False``
    stores nothing (the database update still writes).
    ``game_date`` pins a scheduled run to its slate and ``as_of`` the
    intermediate input cutoff, even when the runner is busy.
    ``snapshot_game_ids`` names the last-tip games whose snapshot ``as_of``
    anchors (metadata, not a filter): these scheduled jobs reconstruct the
    whole slate, including started/final games. Without it, the live schedule
    keeps only games not yet started.
    ``game_ids`` (manual runs) keeps only those games.

    ``with_closing`` (latest input only; the dashboard's joint request) also
    predicts the closing line in this run: the schedule, the reports and the
    latest snapshot frame are read once for both lines, the closing rows are
    built in the same history scope, and the closing fits follow the
    intermediate ones. The closing line is its own stored run and answers
    ``closing_request_id``; a closing failure leaves the intermediate
    predictions as they are.
    """
    dataset_type = LINES[line]
    latest = input_mode == "latest"
    if with_closing and not (dataset_type == INTERMEDIATE_LINE and latest):
        raise SystemExit("--with-closing requires --line intermediate --input latest.")
    if with_closing and feature_slot is not None:
        raise SystemExit("--with-closing picks its own feature slots.")
    if closing_request_id is not None and not with_closing:
        raise SystemExit("--closing-request-id requires --with-closing.")
    if dataset_type == CLOSING_LINE:
        if horizon_minutes is not None or nearest:
            raise SystemExit("--horizon-minutes applies to --line intermediate only.")
        if latest:
            raise SystemExit("--input latest applies to --line intermediate only.")
        horizon = 0
    else:
        if nearest and not latest:
            raise SystemExit("--horizon-minutes nearest requires --input latest.")
        if nearest and feature_slot is not None:
            raise SystemExit("--horizon-minutes nearest picks its own feature slots.")
        horizon = horizon_minutes or DEFAULT_INTERMEDIATE_HORIZON
    job = TabPFNJob(
        targets=tuple(parse_targets(targets)),
        dataset_type=dataset_type,
        horizon=horizon,
        modes=tuple(parse_modes(modes)),
        mode_targets=tuple(
            (mode, tuple(targets_of_mode)) for mode, targets_of_mode in mode_targets
        ),
        as_of=None if as_of is None else pd.Timestamp(as_of),
        snapshot_games=(
            None
            if snapshot_game_ids is None
            else tuple(parse_game_ids(snapshot_game_ids))
        ),
        wanted_games=(
            None if game_ids is None else frozenset(parse_game_ids(game_ids))
        ),
        feature_slot=feature_slot,
        model_version=model_version,
        train_games=train_games,
        latest_input=latest,
    )
    configure_tqdm_for_environment()
    date_to_predict = (
        date.fromisoformat(game_date).isoformat()
        if game_date is not None
        else datetime.now(ZoneInfo("US/Pacific")).strftime("%Y-%m-%d")
    )
    at = "each game's nearest horizon" if nearest else format_horizon(horizon)
    if with_closing:
        at += " + closing line"
    print(
        f"TabPFN {model_version} ({job.plan}) at {at}"
        + (", latest snapshots" if latest else "")
        + f", for {date_to_predict}"
        + ("" if request_id is None else f" (request {request_id})")
    )

    # Each phase prints a [time] line as it ends, the summary their list.
    timer = timer if timer is not None else PhaseTimer()

    # Step 1: everything the fit needs from S3, before any other work.
    # (nearest: once the groups are known, after tonight's rows.)
    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    bucket = SETTINGS.s3_bucket
    if not nearest:
        log.step("Loading the Feature Spec and Training Frame")
        with timer.phase("spec and training rows"):
            try:
                inputs = load_fit_inputs(
                    job,
                    s3_client=s3,
                    bucket=bucket,
                    slots=SETTINGS.prediction_model_slots,
                    date_to_predict=date_to_predict,
                    max_frame_age_days=max_frame_age_days,
                )
            except TabPFNUnavailable as exc:
                raise SystemExit(str(exc)) from None

    # Step 2: tonight's rows, with the lines of this moment.
    if update_db:
        log.step("Updating All Databases")
        season_to_update = int(get_season_year_from_date(date_to_predict))
        with timer.phase("database update"):
            update_all_databases(
                start_season_year=season_to_update,
                end_season_year=season_to_update,
                only_new_games=True,
                headless=SETTINGS.headless,
                exclude_game_date=date_to_predict,
            )
        print_status("Databases updated")

    log.step("Checking SBR Line Feed")
    # Covers stands in when SBR is frozen or empty (prediction.line_source).
    with timer.phase("SBR line feed"):
        lines = check_line_feed(
            pd.Timestamp(date_to_predict).date(),
            allow_stale_lines=allow_stale_lines,
            load_scheduled_games=lambda: get_schedule_games(date_to_predict),
            report=print_status,
        )

    log.step("Fetching Scheduled Games & Reports")
    with timer.phase("scheduled games and reports"):
        scheduled_data = get_all_info_for_scheduled_games(
            date_to_predict=date_to_predict,
            nba_injury_reports_url=SETTINGS.nba_injury_reports_url,
            headless=SETTINGS.headless,
            line_source=lines,
            dataset_type=dataset_type,
        )
    if job.wanted_games is not None:
        scheduled_data = restrict_to_games(scheduled_data, job.wanted_games)

    # A joint run builds both lines' rows first, in one history scope: the
    # latest snapshot frame once, then the closing rows on the loads it left.
    latest_frame = None
    closing = None
    with history_cache(), kept_connections():
        if with_closing and not scheduled_data["scheduled_games"].empty:
            log.step("Preparing Tonight's Latest Snapshot Rows")
            with timer.phase("tonight's latest snapshot rows"):
                latest_frame = build_latest_frame(
                    scheduled_data, date_to_predict=date_to_predict
                )
            closing = prepare_closing(
                replace(
                    job,
                    dataset_type=CLOSING_LINE,
                    horizon=0,
                    latest_input=False,
                ),
                scheduled_data,
                latest_frame,
                s3_client=s3,
                bucket=bucket,
                date_to_predict=date_to_predict,
                max_frame_age_days=max_frame_age_days,
                timer=timer,
                log=log,
            )
        rows = None
        if not nearest:
            # The build's two reads of the closing odds (base features,
            # referees) load once, and its reads share their connections.
            with timer.phase("tonight's rows"):
                rows = tonight_rows(
                    job,
                    scheduled_data,
                    inputs.specs,
                    date_to_predict=date_to_predict,
                    log=log,
                    latest_frame=latest_frame,
                )

    if nearest:
        outcome = fit_nearest(
            job,
            scheduled_data,
            s3_client=s3,
            bucket=bucket,
            date_to_predict=date_to_predict,
            max_frame_age_days=max_frame_age_days,
            timer=timer,
            log=log,
            persist=persist,
            request_id=request_id,
            latest_frame=latest_frame,
        )
    elif rows is None:
        outcome = None
    else:
        outcome = fit_and_store(
            job,
            inputs,
            rows,
            scheduled_data,
            s3_client=s3,
            bucket=bucket,
            date_to_predict=date_to_predict,
            timer=timer,
            log=log,
            persist=persist,
            request_id=request_id,
        )
    if not with_closing:
        if outcome is None:
            return None
    else:
        # Each line answers its own request, the intermediate one as soon as
        # its fits end.
        report_outcome(request_id, outcome)
        closing_outcome = closing
        if isinstance(closing, ClosingPart):
            closing_outcome = fit_and_store(
                closing.job,
                closing.inputs,
                closing.rows,
                closing.scheduled_data,
                s3_client=s3,
                bucket=bucket,
                date_to_predict=date_to_predict,
                timer=timer,
                log=log,
                persist=persist,
                step_prefix="closing: ",
                request_id=closing_request_id,
            )
        report_outcome(closing_request_id, closing_outcome)
        parts = [part for part in (outcome, closing_outcome) if part is not None]
        if not parts:
            return None
        outcome = JobOutcome(
            predicted=[label for part in parts for label in part.predicted],
            failures=[failure for part in parts for failure in part.failures],
            n_fits=sum(part.n_fits for part in parts),
        )
    failures = outcome.failures
    log.step("Summary", group=False)
    print_status(f"{len(outcome.predicted)} of {outcome.n_fits} model(s) predicted")
    print_status(f"{len(failures)} failure(s)", ok=not failures)
    for failure in failures:
        log.error(failure)
    log.summary(
        "\n".join(
            [
                f"### TabPFN {model_version} ({job.plan}) for {date_to_predict}, {at}",
                "",
                f"{len(outcome.predicted)} of {outcome.n_fits} model(s) predicted, "
                f"{len(failures)} failure(s)"
                + ("" if persist else " (dry run: nothing stored)"),
                "",
                *(f"- ✖ {failure}" for failure in failures),
            ]
        )
    )
    if not with_closing:
        report_outcome(request_id, outcome)
    if failures:
        raise SystemExit(1)
    return True


def report_outcome(request_id: int | None, outcome: JobOutcome | None) -> None:
    """The final status of the request a line of the run answered (a no-op
    without one); None is a line with nothing to predict."""
    if request_id is None:
        return
    if outcome is None:
        report_request(request_id, REQUEST_FAILED, error="Nothing to predict")
        return
    report_request(
        request_id,
        final_status(outcome.failures, len(outcome.predicted)),
        error="; ".join(outcome.failures) or None,
    )


@dataclass
class ClosingPart:
    """The closing line of a joint run, ready to fit."""

    job: TabPFNJob
    inputs: FitInputs
    rows: TonightRows
    scheduled_data: dict


def prepare_closing(
    job: TabPFNJob,
    scheduled_data: dict,
    latest_frame: pd.DataFrame,
    *,
    s3_client,
    bucket: str,
    date_to_predict: str,
    max_frame_age_days: int,
    timer: PhaseTimer,
    log: RunLog,
) -> ClosingPart | JobOutcome | None:
    """The closing job's spec, training rows and tonight's closing rows.

    ``scheduled_data`` was read for the intermediate line, which keeps every
    game; the closing rows leave out the games missing an injury report, as a
    closing run does. None when there is no closing row to predict; a failure
    comes back as the line's outcome, so the intermediate line still runs.
    """
    closing_data = dict(
        scheduled_data,
        games_not_updated=list(scheduled_data.get("games_missing_injury_reports", [])),
    )
    try:
        log.step("closing: Loading the Feature Spec and Training Frame")
        with timer.phase("closing: spec and training rows"):
            inputs = load_fit_inputs(
                job,
                s3_client=s3_client,
                bucket=bucket,
                slots=SETTINGS.prediction_model_slots,
                date_to_predict=date_to_predict,
                max_frame_age_days=max_frame_age_days,
            )
        with timer.phase("closing: tonight's rows"):
            rows = tonight_rows(
                job,
                closing_data,
                inputs.specs,
                date_to_predict=date_to_predict,
                log=log,
                step_prefix="closing: ",
                latest_frame=latest_frame,
            )
    except Exception as exc:  # noqa: BLE001 -- the intermediate line still runs
        failure = f"closing: {type(exc).__name__}: {exc}"
        print_status(failure, ok=False)
        return JobOutcome(predicted=[], failures=[failure], n_fits=len(job.fits))
    if rows is None:
        return None
    return ClosingPart(job=job, inputs=inputs, rows=rows, scheduled_data=closing_data)


def fit_nearest(
    job: TabPFNJob,
    scheduled_data: dict,
    *,
    s3_client,
    bucket: str,
    date_to_predict: str,
    max_frame_age_days: int,
    timer: PhaseTimer,
    log: RunLog,
    persist: bool,
    request_id: int | None,
    latest_frame: pd.DataFrame | None = None,
) -> JobOutcome | None:
    """Each game's latest snapshot, predicted by TabPFN fitted on the rows of
    its nearest horizon: one fit (and one stored run) per group of games.

    The latest rows are built once for the slate (or filtered from
    ``latest_frame``, when a joint run already built it); a group whose spec
    or training rows cannot be read fails alone. None when no game is left
    before tip-off.
    """
    if latest_frame is None:
        log.step("Preparing Tonight's Latest Snapshot Rows")
    with timer.phase("tonight's rows"), history_cache(), kept_connections():
        rows = latest_rows(
            scheduled_data,
            date_to_predict=date_to_predict,
            games=job.wanted_games,
            frame=latest_frame,
        )
    if rows.empty:
        print_status("No game left before tip-off; nothing to predict.", ok=False)
        return None
    horizons = tabpfn_horizons(job.targets, SETTINGS.prediction_model_slots)
    minutes_left = dict(
        zip(rows["GAME_ID"].astype(str), rows["TIME_TO_MATCH_MIN"], strict=True)
    )
    groups = group_by_horizon(assign_horizons(minutes_left, horizons))
    for horizon, games in groups.items():
        print_status(f"{format_horizon(horizon)}: {', '.join(games)}")

    predicted: list[str] = []
    failures: list[str] = []
    n_fits = 0
    for horizon, games in groups.items():
        group_job = replace(job, horizon=horizon, wanted_games=frozenset(games))
        prefix = f"{format_horizon(horizon)}: "
        log.step(f"{prefix}Loading the Feature Spec and Training Frame")
        try:
            with timer.phase(f"{prefix}spec and training rows"):
                inputs = load_fit_inputs(
                    group_job,
                    s3_client=s3_client,
                    bucket=bucket,
                    slots=SETTINGS.prediction_model_slots,
                    date_to_predict=date_to_predict,
                    max_frame_age_days=max_frame_age_days,
                )
        except Exception as exc:  # noqa: BLE001 -- the other groups still run
            failures.append(f"{prefix}{type(exc).__name__}: {exc}")
            print_status(failures[-1], ok=False)
            n_fits += len(group_job.fits)
            continue
        group_rows = TonightRows(
            rows.loc[rows["GAME_ID"].astype(str).isin(set(games))].copy(),
            BASE_SCHEMA_VERSION,
            scheduled_games=scheduled_data["scheduled_games"],
        )
        outcome = fit_and_store(
            group_job,
            inputs,
            group_rows,
            scheduled_data,
            s3_client=s3_client,
            bucket=bucket,
            date_to_predict=date_to_predict,
            timer=timer,
            log=log,
            persist=persist,
            step_prefix=prefix,
            request_id=request_id,
        )
        del inputs
        gc.collect()
        predicted += outcome.predicted
        failures += outcome.failures
        n_fits += outcome.n_fits
    return JobOutcome(predicted=predicted, failures=failures, n_fits=n_fits)


def tabpfn_horizons(targets: Sequence[str], slots: Sequence[ModelSlot]) -> list[int]:
    """The intermediate horizons every target has an enabled slot at."""
    per_target = [
        {
            int(slot.horizon_minutes)
            for slot in slots
            if slot.horizon_minutes and str(slot.target) == target
        }
        for target in targets
    ]
    common = set.intersection(*per_target) if per_target else set()
    if not common:
        raise SystemExit(
            f"No intermediate horizon is enabled for every target ({', '.join(targets)})."
        )
    return sorted(common)


def _parse_slot(value: str) -> ModelSlot:
    return ModelSlot.parse(value, root=SETTINGS.s3_models_prefix or "models/")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--targets",
        "--target",
        nargs="+",
        help=(
            "Targets separated by spaces or commas "
            f"(default: {' '.join(DEFAULT_TARGETS)})."
        ),
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
        help=(
            "With --line intermediate: the slot's minutes before tip-off "
            f"(default: {DEFAULT_INTERMEDIATE_HORIZON}), or nearest (with "
            "--input latest): each game the horizon nearest its time to tip-off."
        ),
    )
    parser.add_argument(
        "--input",
        choices=("own", "latest"),
        default="own",
        dest="input_mode",
        help=(
            "With --line intermediate: predict each game's row of the horizon "
            "(own, default) or its latest snapshot (latest, live manual runs)."
        ),
    )
    parser.add_argument(
        "--request-id",
        type=int,
        help="The dashboard request this run answers (stored as runs.request_id).",
    )
    parser.add_argument(
        "--with-closing",
        action="store_true",
        help=(
            "With --line intermediate --input latest: also predict the closing "
            "line in this run, on the same schedule, reports and latest "
            "snapshot rows (its own stored run)."
        ),
    )
    parser.add_argument(
        "--closing-request-id",
        type=int,
        help="With --with-closing: the dashboard request the closing line answers.",
    )
    parser.add_argument(
        "--game-date",
        type=date.fromisoformat,
        help="Game date YYYY-MM-DD (default: today's Pacific date).",
    )
    parser.add_argument(
        "--game-ids",
        nargs="+",
        help=(
            "Only these game IDs, separated by spaces or commas (default: every "
            "game not started)."
        ),
    )
    parser.add_argument(
        "--as-of",
        help="Intermediate snapshot cutoff in ISO 8601 with timezone (default: live).",
    )
    parser.add_argument(
        "--snapshot-game-ids",
        nargs="+",
        help=(
            "With --as-of: the games whose snapshot it is, separated by spaces or "
            "commas. Not a filter: every game is still predicted."
        ),
    )
    parser.add_argument(
        "--feature-slot",
        type=_parse_slot,
        metavar="SCHEMA/TARGET/HORIZON/VARIANT",
        help=(
            "The slot whose production spec gives the features and rows. "
            "Requires one target. Default: each target's enabled slot at --line."
        ),
    )
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        help=f"TabPFN model version (default: {DEFAULT_MODEL_VERSION}).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        help=(
            "TabPFN modes separated by spaces or commas, each fitted and stored "
            f"as its own variant: {', '.join(MODES)} "
            f"(default: {' '.join(DEFAULT_MODES)})."
        ),
    )
    parser.add_argument(
        "--fits",
        nargs="+",
        metavar="MODE=TARGET[,TARGET]",
        help=(
            "Each mode with the targets it fits, instead of --targets and --modes, "
            "e.g. 'standard=line_error,total_points thinking_high=line_error'."
        ),
    )
    parser.add_argument(
        "--train-games",
        type=int,
        help="Cap TabPFN's history since 2019-01-01 at the last N valid games.",
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
    args = parser.parse_args(argv)
    nearest = False
    horizon_minutes = None
    if args.horizon_minutes is not None:
        if args.horizon_minutes.strip().lower() == "nearest":
            nearest = True
        else:
            try:
                horizon_minutes = int(args.horizon_minutes)
            except ValueError:
                parser.error("--horizon-minutes takes a number or nearest.")
    if args.input_mode == "latest" and (
        args.line != "intermediate" or args.as_of is not None
    ):
        parser.error("--input latest requires --line intermediate and no --as-of.")
    if args.with_closing and args.input_mode != "latest":
        parser.error("--with-closing requires --line intermediate --input latest.")
    if args.closing_request_id is not None and not args.with_closing:
        parser.error("--closing-request-id requires --with-closing.")
    try:
        if args.fits is not None:
            if args.targets is not None or args.modes is not None:
                raise ValueError("--fits replaces --targets and --modes.")
            targets, modes, mode_targets = parse_fits(" ".join(args.fits))
            targets, modes = list(targets), list(modes)
        else:
            targets = parse_targets(args.targets or DEFAULT_TARGETS)
            modes = parse_modes(args.modes or DEFAULT_MODES)
            mode_targets = ()
        game_ids = None if args.game_ids is None else parse_game_ids(args.game_ids)
        snapshot_game_ids = (
            None
            if args.snapshot_game_ids is None
            else parse_game_ids(args.snapshot_game_ids)
        )
        if snapshot_game_ids is not None and args.as_of is None:
            raise ValueError("--snapshot-game-ids requires --as-of.")
        if args.as_of is not None:
            cutoff = pd.Timestamp(args.as_of)
            if pd.isna(cutoff) or cutoff.tzinfo is None or args.line != "intermediate":
                raise ValueError(
                    "--as-of requires --line intermediate and a timestamp with timezone."
                )
    except ValueError as exc:
        parser.error(str(exc))
    if args.feature_slot is not None and len(targets) != 1:
        parser.error("--feature-slot requires exactly one target.")

    predict_tabpfn(
        targets=targets,
        line=args.line,
        horizon_minutes=horizon_minutes,
        game_date=None if args.game_date is None else args.game_date.isoformat(),
        game_ids=game_ids,
        as_of=args.as_of,
        snapshot_game_ids=snapshot_game_ids,
        feature_slot=args.feature_slot,
        model_version=args.model_version,
        modes=modes,
        mode_targets=mode_targets,
        train_games=args.train_games,
        max_frame_age_days=args.max_frame_age_days,
        update_db=not args.skip_db_update,
        allow_stale_lines=args.allow_stale_lines,
        persist=not args.no_persist,
        input_mode=args.input_mode,
        nearest=nearest,
        request_id=args.request_id,
        with_closing=args.with_closing,
        closing_request_id=args.closing_request_id,
    )


if __name__ == "__main__":
    main()
