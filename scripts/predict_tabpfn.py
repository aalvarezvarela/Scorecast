"""Predict tonight's games with TabPFN, on its own.

    python scripts/predict_tabpfn.py [--targets line_error total_points spread_error] [--no-persist]
    python scripts/predict_tabpfn.py --modes standard thinking_high
    python scripts/predict_tabpfn.py --line intermediate [--horizon-minutes 360]
    python scripts/predict_tabpfn.py --line intermediate --horizon-minutes 720 \
        --game-date 2026-10-26 --as-of 2026-10-26T11:00:00Z

One or more targets and one line per run (``line_error``, ``total_points`` and ``spread_error``
at the closing line by default; ``--line intermediate`` follows the slot of a fixed time before
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
   as the daily predictor does. Live/manual runs predict games not yet
   started. Scheduled intermediate jobs (as_of and snapshot_game_ids) rebuild
   the whole slate, including started/final games, at their own pregame
   snapshots; only that horizon's rows are built.
3. TabPFN is fitted separately for each target and mode (``--modes``: the
   standard model by default, or thinking at medium or high effort, each
   stored as a variant of its own) on its spec's rows and features. Tonight's
   rows are prepared once and all outputs share one prediction run.
   The run goes to the prediction store (S3 run archive + Aiven) with
   ``runs.workflow`` 3 and ``model_builds.kind`` 3.

The closing rows keep no raw quote. As in the daily predictor, a closing run
also builds each game's latest snapshot row, only for the store: the bet line,
prices and cross-book lines of this moment come from it
(``PredictionRunRecorder``). If that build fails the run goes on and the bet
line stays null; the model line (``inputs.total_model_line``) is the line of
this moment either way.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sequence
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.get_all_info_for_scheduled_games import (
    get_all_info_for_scheduled_games,
)
from nba_ou.create_training_data.schema_layers import CLOSING_LINE
from nba_ou.fetch_data.scheduled_game.get_schedule_games import get_schedule_games
from nba_ou.modeling.registry_paths import ModelSlot, format_horizon
from nba_ou.postgre_db.config.connection_keep import kept_connections
from nba_ou.postgre_db.update_all.update_all_databases import update_all_databases
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
    TabPFNJob,
    TabPFNUnavailable,
    fit_and_store,
    load_fit_inputs,
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
    """``run_tabpfn`` with its log in numbered steps (``nba_ou.utils.run_log``)."""
    log = RunLog()
    try:
        return run_tabpfn(log, **kwargs)
    finally:
        # An exception's traceback then prints outside the step's group.
        log.end_group()


def run_tabpfn(
    log: RunLog,
    *,
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
    train_games: int | None = None,
    max_frame_age_days: int = DEFAULT_MAX_FRAME_AGE_DAYS,
    update_db: bool = True,
    allow_stale_lines: bool = False,
    persist: bool = True,
) -> None:
    """Fit TabPFN for each target and predict shared tonight's rows at ``line``.

    ``line`` is ``closing`` or ``intermediate``; ``horizon_minutes`` is the
    intermediate slot's time before tip-off (default
    ``DEFAULT_INTERMEDIATE_HORIZON``). ``feature_slot`` defaults to the enabled
    slot of each target at that line. An explicit ``feature_slot`` requires
    one target. ``modes`` are the fits of each target (``MODES``; the standard
    model by default), each stored as its own variant. ``persist=False``
    stores nothing (the database update still writes).
    ``game_date`` pins a scheduled run to its slate and ``as_of`` the
    intermediate input cutoff, even when the runner is busy.
    ``snapshot_game_ids`` names the last-tip games whose snapshot ``as_of``
    anchors (metadata, not a filter): these scheduled jobs reconstruct the
    whole slate, including started/final games. Without it, the live schedule
    keeps only games not yet started.
    ``game_ids`` (manual runs) keeps only those games.
    """
    dataset_type = LINES[line]
    if dataset_type == CLOSING_LINE:
        if horizon_minutes is not None:
            raise SystemExit("--horizon-minutes applies to --line intermediate only.")
        horizon = 0
    else:
        horizon = horizon_minutes or DEFAULT_INTERMEDIATE_HORIZON
    job = TabPFNJob(
        targets=tuple(parse_targets(targets)),
        dataset_type=dataset_type,
        horizon=horizon,
        modes=tuple(parse_modes(modes)),
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
    )
    configure_tqdm_for_environment()
    date_to_predict = (
        date.fromisoformat(game_date).isoformat()
        if game_date is not None
        else datetime.now(ZoneInfo("US/Pacific")).strftime("%Y-%m-%d")
    )
    print(
        f"TabPFN {model_version} ({', '.join(job.modes)}), {', '.join(job.targets)} "
        f"at {format_horizon(horizon)}, for {date_to_predict}"
    )

    # Each phase prints a [time] line as it ends, the summary their list.
    timer = PhaseTimer()

    # Step 1: everything the fit needs from S3, before any other work.
    log.step("Loading the Feature Spec and Training Frame")
    started = time.perf_counter()
    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    bucket = SETTINGS.s3_bucket
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
    timer.record("spec and training rows", time.perf_counter() - started)

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

    started = time.perf_counter()
    # The build's two reads of the closing odds (base features, referees) load
    # once, and its reads share their connections.
    with history_cache(), kept_connections():
        rows = tonight_rows(
            job, scheduled_data, inputs.specs, date_to_predict=date_to_predict, log=log
        )
    if rows is None:
        return
    timer.record("tonight's rows", time.perf_counter() - started)

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
    )
    failures = outcome.failures
    log.step("Summary", group=False)
    print(timer.summary())
    print_status(f"{len(outcome.predicted)} of {outcome.n_fits} model(s) predicted")
    print_status(f"{len(failures)} failure(s)", ok=not failures)
    for failure in failures:
        log.error(failure)
    log.summary(
        "\n".join(
            [
                f"### TabPFN {model_version} ({', '.join(job.modes)}) for "
                f"{date_to_predict}, {format_horizon(horizon)}",
                "",
                f"{len(outcome.predicted)} of {outcome.n_fits} model(s) predicted, "
                f"{len(failures)} failure(s)"
                + ("" if persist else " (dry run: nothing stored)"),
                "",
                *(f"- ✖ {failure}" for failure in failures),
                "",
                timer.markdown(),
            ]
        )
    )
    if failures:
        raise SystemExit(1)


def _parse_slot(value: str) -> ModelSlot:
    return ModelSlot.parse(value, root=SETTINGS.s3_models_prefix or "models/")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--targets",
        "--target",
        nargs="+",
        default=list(DEFAULT_TARGETS),
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
        type=int,
        help=(
            "With --line intermediate: the slot's minutes before tip-off "
            f"(default: {DEFAULT_INTERMEDIATE_HORIZON})."
        ),
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
        default=list(DEFAULT_MODES),
        help=(
            "TabPFN modes separated by spaces or commas, each fitted and stored "
            f"as its own variant: {', '.join(MODES)} "
            f"(default: {' '.join(DEFAULT_MODES)})."
        ),
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
    args = parser.parse_args(argv)
    try:
        targets = parse_targets(args.targets)
        modes = parse_modes(args.modes)
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
        horizon_minutes=args.horizon_minutes,
        game_date=None if args.game_date is None else args.game_date.isoformat(),
        game_ids=game_ids,
        as_of=args.as_of,
        snapshot_game_ids=snapshot_game_ids,
        feature_slot=args.feature_slot,
        model_version=args.model_version,
        modes=modes,
        train_games=args.train_games,
        max_frame_age_days=args.max_frame_age_days,
        update_db=not args.skip_db_update,
        allow_stale_lines=args.allow_stale_lines,
        persist=not args.no_persist,
    )


if __name__ == "__main__":
    main()
