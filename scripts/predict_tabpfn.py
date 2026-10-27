"""Predict tonight's games with TabPFN, on its own.

    python scripts/predict_tabpfn.py [--targets line_error total_points] [--no-persist]
    python scripts/predict_tabpfn.py --line intermediate [--horizon-minutes 360]
    python scripts/predict_tabpfn.py --line intermediate --horizon-minutes 720 \
        --game-date 2026-10-26 --as-of 2026-10-26T11:00:00Z

One or more targets and one line per run (``line_error`` and ``total_points``
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
   as the daily predictor does. Every game of the slate that has not started
   is predicted, at either line; only that horizon's rows are built.
3. TabPFN is fitted separately for each target on its spec's rows and features.
   Tonight's rows are prepared once and all outputs share one prediction run.
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
from collections.abc import Callable, Collection, Sequence
from datetime import date, datetime
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
    LayerContext,
    compute_new_columns,
    is_layered,
    newest_version,
    parse_version,
)
from nba_ou.create_training_data.season_partitions import SeasonPartitions
from nba_ou.fetch_data.scheduled_game.get_schedule_games import get_schedule_games
from nba_ou.modeling.refit import resolve_training_frame
from nba_ou.modeling.registry_paths import (
    Channel,
    ModelSlot,
    ModelTarget,
    format_horizon,
)
from nba_ou.modeling.registry_store import resolve_channel
from nba_ou.postgre_db.line_history_aiven.ingest import TIPOFF_TOLERANCE_MINUTES
from nba_ou.postgre_db.predictions_aiven.schema import WORKFLOW_TABPFN
from nba_ou.postgre_db.update_all.update_all_databases import update_all_databases
from nba_ou.prediction.intermediate_frame import (
    PASSTHROUGH_COLUMNS,
    tonight_intermediate_frame,
)
from nba_ou.prediction.line_source import LINE_SOURCE_COLUMN, check_line_feed
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
from nba_ou.utils.general_utils import get_season_year_from_date
from nba_ou.utils.phase_timer import PhaseTimer
from nba_ou.utils.s3_models import make_s3_client

#: Partitions older than this many days mean the daily refit has stopped.
DEFAULT_MAX_FRAME_AGE_DAYS = 2
#: The targets of a run when none are given.
DEFAULT_TARGETS = ("line_error", "total_points")
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


def parse_targets(values: Sequence[str]) -> list[str]:
    """Validate a space/comma-separated list, keeping each target once in order."""
    targets = list(
        dict.fromkeys(
            value for item in values for value in item.replace(",", " ").split()
        )
    )
    if not targets:
        raise ValueError("Provide at least one target.")
    for target in targets:
        if target not in {item.value for item in ModelTarget}:
            raise ValueError(
                f"Unknown target {target!r}; choose from "
                + ", ".join(item.value for item in ModelTarget)
            )
    return targets


def parse_game_ids(values: Sequence[str]) -> list[str]:
    """NBA game IDs, preserving leading zeroes and removing duplicates."""
    ids = list(
        dict.fromkeys(
            item for value in values for item in value.replace(",", " ").split()
        )
    )
    if not ids or any(
        len(item) != 10 or not item.isascii() or not item.isdigit() for item in ids
    ):
        raise ValueError(
            "Game IDs must be 10-digit NBA IDs, separated by spaces or commas."
        )
    return ids


def _wait_until(moment: pd.Timestamp) -> None:
    """Sleep until ``moment`` (UTC) has passed."""
    seconds = (moment - pd.Timestamp.now(tz="UTC")).total_seconds()
    if seconds > 0:
        print_status(f"Waiting {seconds:.0f} s for the snapshot at {moment}")
        time.sleep(seconds)


def _rebuild_at_row_snapshot(
    rows: pd.DataFrame,
    cutoff: pd.Timestamp,
    build_rows: Callable[[pd.Timestamp], pd.DataFrame],
    snapshot_games: Collection[str] | None = None,
) -> pd.DataFrame:
    """``rows`` with the games of ``cutoff``'s snapshot cut at their own snapshot.

    The scheduler times ``--as-of`` from the NBA scoreboard's tip-off, the rows
    from the line-history page's. A row of the snapshot that triggered the run
    (``snapshot_games``) up to ``TIPOFF_TOLERANCE_MINUTES`` from the cutoff is
    a few minutes off by rounding between the two: the rows are rebuilt at that
    snapshot, after waiting for it to pass, so none of them is cut before the
    snapshot it is labelled with. Without ``snapshot_games`` (a manual run) the
    snapshot nearest the cutoff is the run's: another game's a few minutes away
    is its own, and has a run of its own. The other rows are the rest of the
    slate: a snapshot already passed reads its own ticks, one still ahead the
    lines so far (``SNAPSHOT_PASSED`` False), as the daily predictor predicts
    them.
    """
    snapshots = pd.to_datetime(rows["SNAPSHOT_TS_UTC"], utc=True)
    distance = (snapshots - cutoff).abs()
    near = distance.le(pd.Timedelta(minutes=TIPOFF_TOLERANCE_MINUTES))
    if snapshot_games is not None:
        near &= rows["GAME_ID"].astype(str).isin(set(snapshot_games))
    elif near.any():
        near &= distance.eq(distance[near].min())
    if snapshots[near].eq(cutoff).all():
        return rows
    # Rows of different snapshots share one cutoff: the latest, so every near
    # row has passed; the earlier ones read their ticks up to their own snapshot.
    snapshot = snapshots[near].max()
    print_status(
        f"Line-history snapshot {snapshot} differs from --as-of {cutoff} "
        "(scoreboard vs page tip-off); rebuilding at the snapshot"
    )
    _wait_until(snapshot)
    return build_rows(snapshot)


def predict_tabpfn(
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
    one target. ``persist=False`` stores nothing (the database update still writes).
    ``game_date`` pins a scheduled run to its slate and ``as_of`` the
    intermediate input cutoff, even when the runner is busy; every game of the
    slate not started when the run reads the schedule is predicted.
    ``snapshot_game_ids`` names the games whose snapshot ``as_of`` is
    (metadata, not a filter: ``_rebuild_at_row_snapshot``). ``game_ids``
    (manual runs) keeps only those games.
    """
    targets = parse_targets(targets)
    wanted_games = None if game_ids is None else set(parse_game_ids(game_ids))
    snapshot_games = (
        None if snapshot_game_ids is None else parse_game_ids(snapshot_game_ids)
    )
    cutoff = None if as_of is None else pd.Timestamp(as_of)
    if cutoff is not None and (
        pd.isna(cutoff) or cutoff.tzinfo is None or line != "intermediate"
    ):
        raise ValueError(
            "--as-of requires --line intermediate and a timestamp with timezone."
        )
    if feature_slot is not None and len(targets) != 1:
        raise ValueError("--feature-slot requires exactly one target.")
    configure_tqdm_for_environment()
    dataset_type = LINES[line]
    if dataset_type == CLOSING_LINE:
        if horizon_minutes is not None:
            raise SystemExit("--horizon-minutes applies to --line intermediate only.")
        horizon = 0
    else:
        horizon = horizon_minutes or DEFAULT_INTERMEDIATE_HORIZON
    date_to_predict = (
        date.fromisoformat(game_date).isoformat()
        if game_date is not None
        else datetime.now(ZoneInfo("US/Pacific")).strftime("%Y-%m-%d")
    )
    print(
        f"TabPFN {model_version}, {', '.join(targets)} at {format_horizon(horizon)}, "
        f"for {date_to_predict}"
    )

    # Each phase prints a [time] line as it ends, the summary their list.
    timer = PhaseTimer()

    # Step 1: everything the fit needs from S3, before any other work.
    print_step_header(1, "Loading the Feature Spec and Training Frame")
    started = time.perf_counter()
    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    bucket = SETTINGS.s3_bucket
    specs = []
    for target in targets:
        slot = feature_slot or feature_slot_for(
            target, SETTINGS.prediction_model_slots, horizon_minutes=horizon
        )
        if slot.target != ModelTarget(target) or slot.horizon_minutes != horizon:
            raise SystemExit(
                f"The feature slot must be a {target} {format_horizon(horizon)} slot; "
                f"got {slot.describe()}."
            )
        spec = resolve_channel(
            s3_client=s3, bucket=bucket, slot=slot, channel=Channel.PRODUCTION
        ).spec
        if str(spec.identity.dataset_type) != dataset_type:
            raise SystemExit(
                f"{slot.describe()} serves a {spec.identity.dataset_type} model, "
                f"not a {dataset_type} one."
            )
        if dataset_type == INTERMEDIATE_LINE and parse_version(
            spec.identity.schema_version
        ) > parse_version(BASE_SCHEMA_VERSION):
            raise SystemExit(
                f"{slot.describe()} is a {spec.identity.schema_version} model, and "
                f"tonight's intermediate rows are built at {BASE_SCHEMA_VERSION}."
            )
        specs.append(spec)
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
    # Base seasons and layer inputs are read once, however many versions.
    training_frames = {}
    base_frames = {}
    layer_context = LayerContext()
    training_sets = []
    for target, spec in zip(targets, specs, strict=True):
        schema_version = spec.identity.schema_version
        if schema_version not in training_frames:
            training_frames[schema_version] = resolve_training_frame(
                spec,
                partitions={dataset_type: partitions},
                base_frames=base_frames,
                layer_context=layer_context,
            )
        frame = training_frames[schema_version]
        training_set = tabpfn_training_set(
            frame.df, spec, frame_key=partitions.prefix, train_games=train_games
        )
        training_sets.append(training_set)
        print_status(
            f"{target}: training rows from {frame.build_id} ({frame.checksum}): "
            f"{training_set.n_games} games, "
            f"{training_set.train_date_min.date()} .. {training_set.train_date_max.date()}"
        )
    # The training sets hold what the fits need; free the full frames.
    del training_frames, base_frames, layer_context, frame
    timer.record("spec and training rows", time.perf_counter() - started)

    # Step 2: tonight's rows, with the lines of this moment.
    if update_db:
        print_step_header(2, "Updating All Databases")
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

    print_step_header(2, "Checking SBR Line Feed")
    # Covers stands in when SBR is frozen or empty (prediction.line_source).
    with timer.phase("SBR line feed"):
        lines = check_line_feed(
            pd.Timestamp(date_to_predict).date(),
            allow_stale_lines=allow_stale_lines,
            load_scheduled_games=lambda: get_schedule_games(date_to_predict),
            report=print_status,
        )

    print_step_header(2, "Fetching Scheduled Games & Reports")
    with timer.phase("scheduled games and reports"):
        scheduled_data = get_all_info_for_scheduled_games(
            date_to_predict=date_to_predict,
            nba_injury_reports_url=SETTINGS.nba_injury_reports_url,
            headless=SETTINGS.headless,
            line_source=lines,
            dataset_type=dataset_type,
        )
    if wanted_games is not None:
        scheduled_data = dict(scheduled_data)
        scheduled = scheduled_data["scheduled_games"]
        missing = wanted_games - set(scheduled["GAME_ID"].astype(str))
        if missing:
            raise ValueError(
                f"Requested game IDs are missing from the schedule: {sorted(missing)}"
            )
        scheduled_data["scheduled_games"] = scheduled.loc[
            scheduled["GAME_ID"].astype(str).isin(wanted_games)
        ].copy()
    if scheduled_data["scheduled_games"].empty:
        print_status("No scheduled games for this date; nothing to predict.")
        return

    latest_quotes = None
    started = time.perf_counter()
    if dataset_type == CLOSING_LINE:
        not_filed = {str(game) for game in scheduled_data["games_not_updated"]}
        if not_filed >= set(scheduled_data["scheduled_games"]["GAME_ID"].astype(str)):
            # Closing rows need both teams' injury report; none is filed yet.
            print_status(
                "No game has both injury reports filed yet; nothing to predict."
            )
            return
        print_step_header(2, "Preparing Tonight's Closing Rows")
        # As in the daily predictor, build the newest schema once; earlier
        # targets select their own features from those rows.
        frame_version = newest_version(spec.identity.schema_version for spec in specs)
        df_to_predict = tonight_closing_rows(
            scheduled_data, schema_version=frame_version, game_date=date_to_predict
        )
        print_status(f"{len(df_to_predict)} game(s) to predict")
        # The closing rows keep no raw quote: the store takes the bet line,
        # prices and cross-book lines from each game's latest snapshot, as
        # in the daily predictor. No model reads these rows.
        try:
            latest_quotes = tonight_intermediate_frame(
                scheduled_data,
                game_date=date_to_predict,
                latest_input=True,
                latest_only=True,
            )
        except Exception as exc:  # noqa: BLE001 -- only the bet line is lost
            print_status(
                f"Latest snapshot rows failed ({type(exc).__name__}: {exc}); "
                "the stored bet line stays null",
                ok=False,
            )
    else:
        print_step_header(2, f"Preparing Tonight's T-{horizon} Rows")
        if cutoff is not None:
            # Current reports were synchronized above. For a queued snapshot,
            # read archived roster observations at its cutoff rather than
            # passing the roster just observed when the runner became free.
            scheduled_data = dict(scheduled_data, roster_snapshots=None)

        def build_rows(as_of: pd.Timestamp | None) -> pd.DataFrame:
            rows = rows_of_horizon(
                tonight_intermediate_frame(
                    scheduled_data,
                    game_date=date_to_predict,
                    as_of=as_of,
                    horizon=horizon,
                ),
                specs[0],
            )
            if wanted_games is not None:
                rows = rows.loc[rows["GAME_ID"].astype(str).isin(wanted_games)].copy()
            if rows.empty:
                raise ValueError(f"No T-{horizon} rows were built for tonight's games.")
            return rows

        df_to_predict = build_rows(cutoff)
        frame_version = BASE_SCHEMA_VERSION
        if cutoff is not None:
            df_to_predict = _rebuild_at_row_snapshot(
                df_to_predict, cutoff, build_rows, snapshot_games
            )
        passed = int(df_to_predict["SNAPSHOT_PASSED"].sum())
        print_status(
            f"{len(df_to_predict)} game(s) to predict; T-{horizon} passed for "
            f"{passed}, the rest from the lines so far"
        )

    timer.record("tonight's rows", time.perf_counter() - started)

    failures: list[str] = []
    recorder = PredictionRunRecorder(
        workflow=WORKFLOW_TABPFN, game_date=date_to_predict
    )
    recorder.set_schedule(scheduled_data["scheduled_games"])
    recorder.add_frame(dataset_type, df_to_predict, version=frame_version)
    if latest_quotes is not None:
        recorder.add_frame(
            INTERMEDIATE_LINE, latest_quotes, version=BASE_SCHEMA_VERSION
        )
    for target, spec, training_set in zip(targets, specs, training_sets, strict=True):
        label = (
            f"{spec.identity.schema_version}/{target}/{format_horizon(horizon)}/"
            f"{model_variant(model_version)}"
        )
        print_step_header(3, f"Fitting and Predicting ({label})")
        try:
            with timer.phase(f"{target} fit"):
                model = fit_tabpfn(training_set, model_version=model_version)
            with timer.phase(f"{target} predict"):
                predictions = predict_with_tabpfn(
                    df=df_to_predict,
                    spec=spec,
                    model=model,
                    training_set=training_set,
                    prediction_datetime=datetime.now(ZoneInfo("Europe/Madrid")),
                    passthrough_columns=(
                        () if dataset_type == CLOSING_LINE else PASSTHROUGH_COLUMNS
                    ),
                )
            if predictions.empty:
                print_status(f"{target}: no game survived the spec's NaN budget.")
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
            rejected = recorder.add_tabpfn(predictions, label=label)
            if rejected:
                failures.append(f"{label} (prediction store): {rejected}")
        except Exception as exc:  # noqa: BLE001 -- persist the other targets
            reason = f"{type(exc).__name__}: {exc}"
            recorder.record_slot(label, "failed", reason=reason)
            failures.append(f"{label}: {reason}")
            print_status(failures[-1], ok=False)

    if not persist:
        print_status("Dry run (--no-persist): nothing stored")
    else:
        print_step_header(4, "Storing Predictions")
        with timer.phase("storing"):
            failures += persist_prediction_run(
                recorder, s3_client=s3, bucket=bucket, partial=bool(failures)
            )
    print(timer.summary())
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
    # Set here, not in the builder: no feature step may ever see it.
    rows[LINE_SOURCE_COLUMN] = (
        rows["GAME_ID"].astype(str).map(scheduled_data.get("line_sources", {}))
    )
    return rows


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
        train_games=args.train_games,
        max_frame_age_days=args.max_frame_age_days,
        update_db=not args.skip_db_update,
        allow_stale_lines=args.allow_stale_lines,
        persist=not args.no_persist,
    )


if __name__ == "__main__":
    main()
