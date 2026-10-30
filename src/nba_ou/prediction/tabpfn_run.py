"""One TabPFN prediction job, from its specs to its stored run.

TabPFN keeps no model (``nba_ou.prediction.prediction_tabpfn_client``): a job
reads its targets' production specs and training rows, builds tonight's rows
of its line, fits each target in each mode and stores the outputs as a run of
their own (``runs.workflow`` 3). Two scripts run jobs:

* ``scripts/predict_tabpfn.py``: TabPFN on its own, after its own database
  update, line-feed check and fetch of the schedule, referees and reports.
* ``scripts/predict_nba_games.py --tabpfn``: after the daily predictor has
  stored its own predictions, on the schedule, referees, reports and line
  source that run fetched (``--tabpfn-refresh``: fetched again, and the rows
  built at the run's own moment).

A scheduled intermediate job carries its snapshot (``as_of``): its rows are
cut there, so a queued run predicts the snapshot it was booked for.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from nba_ou.config.dataset_versions import BASE_SCHEMA_VERSION
from nba_ou.create_training_data.create_df_to_predict import create_df_to_predict
from nba_ou.create_training_data.live_replay import replay_scheduled_games
from nba_ou.create_training_data.schema_layers import (
    CLOSING_LINE,
    LayerContext,
    compute_new_columns,
    is_layered,
    newest_version,
    parse_version,
)
from nba_ou.create_training_data.season_partitions import SeasonPartitions
from nba_ou.data_processing.referees.process_refs_scheduled_game import (
    process_scheduled_referee_assignments,
)
from nba_ou.modeling.refit import resolve_training_frame
from nba_ou.modeling.registry_models import ModelSpec
from nba_ou.modeling.registry_paths import (
    Channel,
    ModelSlot,
    ModelTarget,
    format_horizon,
)
from nba_ou.modeling.registry_store import resolve_channel
from nba_ou.postgre_db.line_history_aiven.ingest import TIPOFF_TOLERANCE_MINUTES
from nba_ou.postgre_db.predictions_aiven.schema import WORKFLOW_TABPFN
from nba_ou.prediction.intermediate_frame import (
    PASSTHROUGH_COLUMNS,
    tonight_intermediate_frame,
)
from nba_ou.prediction.line_source import LINE_SOURCE_COLUMN
from nba_ou.prediction.prediction import clean_prediction_frames
from nba_ou.prediction.prediction_store import (
    PredictionRunRecorder,
    persist_prediction_run,
)
from nba_ou.prediction.prediction_tabpfn_client import (
    DEFAULT_MODEL_VERSION,
    DEFAULT_MODES,
    INTERMEDIATE_LINE,
    STANDARD_MODE,
    TabPFNTrainingSet,
    feature_slot_for,
    fit_tabpfn,
    model_variant,
    parse_modes,
    predict_with_tabpfn,
    rows_of_horizon,
    tabpfn_training_set,
)
from nba_ou.scheduling.tabpfn_jobs import (
    DEFAULT_TARGETS,
    FIELD_SEPARATOR,
    JOB_SEPARATOR,
    MODE_SEPARATOR,
    Fits,
)
from nba_ou.utils.history_cache import cached
from nba_ou.utils.phase_timer import PhaseTimer
from nba_ou.utils.run_log import RunLog

#: Partitions older than this many days mean the daily refit has stopped.
DEFAULT_MAX_FRAME_AGE_DAYS = 2
#: The prediction column of each target, shown as the job predicts it.
VALUE_COLUMNS = {
    "line_error": "PRED_LINE_ERROR",
    "total_points": "PRED_TOTAL_POINTS",
    "spread_error": "PRED_SPREAD_ERROR",
}


class TabPFNUnavailable(RuntimeError):
    """A job's spec or training rows cannot be used: nothing is fitted."""


def _status(message: str, ok: bool = True) -> None:
    print(f"  {'✓' if ok else '✖'} {message}", flush=True)


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


def parse_fits(text: str) -> tuple[tuple[str, ...], tuple[str, ...], Fits]:
    """``(targets, modes, mode_targets)`` of ``MODE=TARGET[,TARGET]`` items
    separated by spaces (``nba_ou.scheduling.tabpfn_jobs``): the targets in
    order of first mention, the modes in order, and each mode's targets."""
    plan: dict[str, tuple[str, ...]] = {}
    for item in text.split():
        mode, separator, targets = item.partition(MODE_SEPARATOR)
        if not separator:
            raise ValueError(
                f"TabPFN fits {item!r}: expected MODE{MODE_SEPARATOR}TARGET[,TARGET]."
            )
        (mode,) = parse_modes([mode])
        if mode in plan:
            raise ValueError(f"TabPFN fits: mode {mode} named twice.")
        plan[mode] = tuple(parse_targets([targets]))
    if not plan:
        raise ValueError(
            f"Provide at least one TabPFN fit, MODE{MODE_SEPARATOR}TARGET[,TARGET]."
        )
    targets = tuple(dict.fromkeys(t for items in plan.values() for t in items))
    return targets, tuple(plan), tuple(plan.items())


@dataclass(frozen=True)
class TabPFNJob:
    """One TabPFN prediction: its targets and modes at one line.

    ``horizon`` is 0 at the closing line. ``as_of`` (intermediate only) is the
    snapshot a scheduled job was booked for, ``snapshot_games`` the games whose
    snapshot it is (metadata, not a filter: ``_rebuild_at_row_snapshot``).
    ``wanted_games`` keeps only those games (manual runs). ``feature_slot``
    (one target only) replaces each target's enabled slot at the line.
    ``mode_targets`` limits a mode to some of the targets (``parse_fits``);
    a mode it does not name fits every target.
    """

    targets: tuple[str, ...] = DEFAULT_TARGETS
    dataset_type: str = CLOSING_LINE
    horizon: int = 0
    modes: tuple[str, ...] = DEFAULT_MODES
    mode_targets: Fits = ()
    as_of: pd.Timestamp | None = None
    snapshot_games: tuple[str, ...] | None = None
    wanted_games: frozenset[str] | None = None
    feature_slot: ModelSlot | None = None
    model_version: str = DEFAULT_MODEL_VERSION
    train_games: int | None = None

    def __post_init__(self) -> None:
        if self.dataset_type not in (CLOSING_LINE, INTERMEDIATE_LINE):
            raise ValueError(f"Unknown TabPFN line {self.dataset_type!r}.")
        if (self.dataset_type == CLOSING_LINE) != (self.horizon == 0):
            raise ValueError("Horizon 0 is the closing line, and only it.")
        if self.as_of is not None and (
            pd.isna(self.as_of)
            or self.as_of.tzinfo is None
            or self.dataset_type != INTERMEDIATE_LINE
        ):
            raise ValueError(
                "--as-of requires --line intermediate and a timestamp with timezone."
            )
        if self.feature_slot is not None and len(self.targets) != 1:
            raise ValueError("--feature-slot requires exactly one target.")
        unknown = {t for _, items in self.mode_targets for t in items} - set(
            self.targets
        )
        if unknown:
            raise ValueError(
                f"TabPFN fits name target(s) {', '.join(sorted(unknown))} "
                "not among the job's targets."
            )
        fitted = {target for target, _ in self.fits}
        unfitted = [target for target in self.targets if target not in fitted]
        if unfitted:
            raise ValueError(f"No TabPFN mode fits {', '.join(unfitted)}.")

    @property
    def fits(self) -> list[tuple[str, str]]:
        """``(target, mode)`` per fit: each target in each mode that fits it."""
        only = dict(self.mode_targets)
        return [
            (target, mode)
            for target in self.targets
            for mode in self.modes
            if target in only.get(mode, (target,))
        ]

    @property
    def plan(self) -> str:
        """``standard: line_error, total_points; thinking_high: line_error``."""
        return "; ".join(
            f"{mode}: "
            + ", ".join(target for target, fit_mode in self.fits if fit_mode == mode)
            for mode in self.modes
            if any(fit_mode == mode for _, fit_mode in self.fits)
        )

    @property
    def name(self) -> str:
        """``TabPFN t0720`` (``TabPFN t0000`` at the closing line)."""
        return f"TabPFN {format_horizon(self.horizon)}"


def parse_jobs(
    text: str,
    *,
    targets: Sequence[str] = DEFAULT_TARGETS,
    modes: Sequence[str] = DEFAULT_MODES,
) -> list[TabPFNJob]:
    """The jobs of ``HORIZON[@AS_OF[@GAME_IDS[@FITS]]]`` items separated by
    ``;`` (``nba_ou.scheduling.tabpfn_jobs``).

    ``720@2026-10-28T23:00:00+00:00@0022600131 0022600132`` is the T-720 job
    of that snapshot, booked for those games; ``0`` the closing-line job (no
    ``AS_OF``). A job with ``FITS`` (``parse_fits``) fits those targets in
    those modes; the others fit ``targets`` in ``modes``.
    """
    targets = tuple(parse_targets(targets))
    modes = tuple(parse_modes(modes))
    jobs = []
    for item in filter(None, (part.strip() for part in text.split(JOB_SEPARATOR))):
        horizon, as_of, games, fits = (item.split(FIELD_SEPARATOR, 3) + [""] * 3)[:4]
        job_targets, job_modes, mode_targets = (
            parse_fits(fits) if fits.strip() else (targets, modes, ())
        )
        try:
            horizon = int(horizon)
        except ValueError:
            raise ValueError(
                f"TabPFN job {item!r}: the horizon is not a number."
            ) from None
        if horizon < 0:
            raise ValueError(f"TabPFN job {item!r}: a negative horizon.")
        jobs.append(
            TabPFNJob(
                targets=job_targets,
                dataset_type=INTERMEDIATE_LINE if horizon else CLOSING_LINE,
                horizon=horizon,
                modes=job_modes,
                mode_targets=mode_targets,
                as_of=pd.Timestamp(as_of) if as_of.strip() else None,
                snapshot_games=(
                    tuple(parse_game_ids([games])) if games.strip() else None
                ),
            )
        )
    return jobs


@dataclass
class FitInputs:
    """Each target's production spec and the rows it is fitted on, in order."""

    specs: list[ModelSpec]
    training_sets: list[TabPFNTrainingSet]


def load_fit_inputs(
    job: TabPFNJob,
    *,
    s3_client,
    bucket: str,
    slots: Sequence[ModelSlot],
    date_to_predict: str,
    max_frame_age_days: int = DEFAULT_MAX_FRAME_AGE_DAYS,
) -> FitInputs:
    """The job's specs and training rows, as the daily refit reads them.

    Raises ``TabPFNUnavailable`` when a spec does not fit the job or the season
    partitions are older than ``max_frame_age_days``.
    """
    specs = []
    for target in job.targets:
        slot = job.feature_slot or feature_slot_for(
            target, slots, horizon_minutes=job.horizon
        )
        if slot.target != ModelTarget(target) or slot.horizon_minutes != job.horizon:
            raise TabPFNUnavailable(
                f"The feature slot must be a {target} "
                f"{format_horizon(job.horizon)} slot; got {slot.describe()}."
            )
        spec = resolve_channel(
            s3_client=s3_client, bucket=bucket, slot=slot, channel=Channel.PRODUCTION
        ).spec
        if str(spec.identity.dataset_type) != job.dataset_type:
            raise TabPFNUnavailable(
                f"{slot.describe()} serves a {spec.identity.dataset_type} model, "
                f"not a {job.dataset_type} one."
            )
        if job.dataset_type == INTERMEDIATE_LINE and parse_version(
            spec.identity.schema_version
        ) > parse_version(BASE_SCHEMA_VERSION):
            raise TabPFNUnavailable(
                f"{slot.describe()} is a {spec.identity.schema_version} model, and "
                f"tonight's intermediate rows are built at {BASE_SCHEMA_VERSION}."
            )
        specs.append(spec)
        _status(
            f"{slot.describe()}: spec {spec.spec_id}, {spec.features.n_features} "
            f"features, {spec.training.train_games} games, max "
            f"{spec.cleaning.max_na_per_row} NaN per row"
        )

    partitions = SeasonPartitions(
        s3_client=s3_client,
        bucket=bucket,
        dataset_type=job.dataset_type,
        schema_version=BASE_SCHEMA_VERSION,
    )
    current_to = partitions.limit_date()
    if current_to is None:
        raise TabPFNUnavailable(
            f"No {job.dataset_type} season partitions in S3; create them with "
            "scripts/create_train_data/build_season_partitions.py split."
        )
    age_days = (pd.Timestamp(date_to_predict).date() - current_to).days
    if age_days > max_frame_age_days:
        raise TabPFNUnavailable(
            f"The {job.dataset_type} season partitions are current to {current_to}, "
            f"{age_days} days ago (max {max_frame_age_days}); the daily refit has "
            "not run."
        )
    # The refit's own frame: same partitions, horizon and layers.
    # Base seasons and layer inputs are read once, however many versions.
    training_frames = {}
    base_frames = {}
    layer_context = LayerContext()
    training_sets = []
    for target, spec in zip(job.targets, specs, strict=True):
        schema_version = spec.identity.schema_version
        if schema_version not in training_frames:
            training_frames[schema_version] = resolve_training_frame(
                spec,
                partitions={job.dataset_type: partitions},
                base_frames=base_frames,
                layer_context=layer_context,
            )
        frame = training_frames[schema_version]
        training_set = tabpfn_training_set(
            frame.df, spec, frame_key=partitions.prefix, train_games=job.train_games
        )
        training_sets.append(training_set)
        _status(
            f"{target}: training rows from {frame.build_id} ({frame.checksum}): "
            f"{training_set.n_games} games, "
            f"{training_set.train_date_min.date()} .. {training_set.train_date_max.date()}"
        )
    # The training sets hold what the fits need; the full frames go with this
    # function's locals.
    return FitInputs(specs=specs, training_sets=training_sets)


def restrict_to_games(scheduled_data: dict, wanted_games: Collection[str]) -> dict:
    """``scheduled_data`` with the schedule cut to ``wanted_games``; raises
    when one of them is not on it."""
    scheduled = scheduled_data["scheduled_games"]
    missing = set(wanted_games) - set(scheduled["GAME_ID"].astype(str))
    if missing:
        raise ValueError(
            f"Requested game IDs are missing from the schedule: {sorted(missing)}"
        )
    return dict(
        scheduled_data,
        scheduled_games=scheduled.loc[
            scheduled["GAME_ID"].astype(str).isin(set(wanted_games))
        ].copy(),
    )


def _wait_until(moment: pd.Timestamp) -> None:
    """Sleep until ``moment`` (UTC) has passed."""
    seconds = (moment - pd.Timestamp.now(tz="UTC")).total_seconds()
    if seconds > 0:
        _status(f"Waiting {seconds:.0f} s for the snapshot at {moment}")
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
    _status(
        f"Line-history snapshot {snapshot} differs from --as-of {cutoff} "
        "(scoreboard vs page tip-off); rebuilding at the snapshot"
    )
    _wait_until(snapshot)
    return build_rows(snapshot)


@dataclass
class TonightRows:
    """Tonight's rows of a job, and (closing) the latest snapshot rows the
    store takes the bet line from."""

    df: pd.DataFrame
    frame_version: str
    latest_quotes: pd.DataFrame | None = None
    scheduled_games: pd.DataFrame | None = None


@cached()
def _slate_schedule(game_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The whole slate, including started/final games, and its referee crews.

    A scheduled TabPFN run is anchored on the last game: earlier games must
    remain available for reconstruction at their own pregame snapshots.
    """
    games = replay_scheduled_games(game_date)
    refs = (
        process_scheduled_referee_assignments(games)
        if not games.empty
        else pd.DataFrame()
    )
    return games, refs


def tonight_rows(
    job: TabPFNJob,
    scheduled_data: dict,
    specs: Sequence[ModelSpec],
    *,
    date_to_predict: str,
    log: RunLog,
    step_prefix: str = "",
) -> TonightRows | None:
    """Tonight's rows of the job's line; None (with the reason printed) when
    there is nothing to predict."""
    if job.as_of is not None and job.snapshot_games is not None:
        games, refs = _slate_schedule(date_to_predict)
        scheduled_data = dict(
            scheduled_data,
            scheduled_games=games,
            df_referees_scheduled=refs,
            roster_snapshots=None,
        )
    if scheduled_data["scheduled_games"].empty:
        _status("No scheduled games for this date; nothing to predict.")
        return None
    if job.dataset_type == CLOSING_LINE:
        not_filed = {str(game) for game in scheduled_data["games_not_updated"]}
        if not_filed >= set(scheduled_data["scheduled_games"]["GAME_ID"].astype(str)):
            # Closing rows need both teams' injury report; none is filed yet.
            _status("No game has both injury reports filed yet; nothing to predict.")
            return None
        log.step(f"{step_prefix}Preparing Tonight's Closing Rows")
        # As in the daily predictor, build the newest schema once; earlier
        # targets select their own features from those rows.
        frame_version = newest_version(spec.identity.schema_version for spec in specs)
        df = tonight_closing_rows(
            scheduled_data, schema_version=frame_version, game_date=date_to_predict
        )
        _status(f"{len(df)} game(s) to predict")
        # The closing rows keep no raw quote: the store takes the bet line,
        # prices and cross-book lines from each game's latest snapshot, as
        # in the daily predictor. No model reads these rows.
        latest_quotes = None
        try:
            latest_quotes = tonight_intermediate_frame(
                scheduled_data,
                game_date=date_to_predict,
                latest_input=True,
                latest_only=True,
            )
        except Exception as exc:  # noqa: BLE001 -- only the bet line is lost
            _status(
                f"Latest snapshot rows failed ({type(exc).__name__}: {exc}); "
                "the stored bet line stays null",
                ok=False,
            )
        return TonightRows(df, frame_version, latest_quotes)

    horizon = job.horizon
    log.step(f"{step_prefix}Preparing Tonight's T-{horizon} Rows")
    if job.as_of is not None:
        # Current reports were synchronized by the caller. For a queued
        # snapshot, read archived roster observations at its cutoff rather
        # than passing the roster just observed when the runner became free.
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
        if job.wanted_games is not None:
            rows = rows.loc[rows["GAME_ID"].astype(str).isin(job.wanted_games)].copy()
        if rows.empty:
            raise ValueError(f"No T-{horizon} rows were built for tonight's games.")
        return rows

    df = build_rows(job.as_of)
    if job.as_of is not None:
        df = _rebuild_at_row_snapshot(df, job.as_of, build_rows, job.snapshot_games)
    passed = int(df["SNAPSHOT_PASSED"].sum())
    _status(
        f"{len(df)} game(s) to predict; T-{horizon} passed for "
        f"{passed}, the rest from the lines so far"
    )
    return TonightRows(
        df, BASE_SCHEMA_VERSION, scheduled_games=scheduled_data["scheduled_games"]
    )


@dataclass
class JobOutcome:
    """What a job predicted (labels) and what failed, of ``n_fits`` fits."""

    predicted: list[str]
    failures: list[str]
    n_fits: int


def fit_and_store(
    job: TabPFNJob,
    inputs: FitInputs,
    rows: TonightRows,
    scheduled_data: dict,
    *,
    s3_client,
    bucket: str,
    date_to_predict: str,
    timer: PhaseTimer,
    log: RunLog,
    persist: bool,
    step_prefix: str = "",
) -> JobOutcome:
    """Each target in each mode fitted and predicted on tonight's rows, then
    stored as one run (``persist=False``: nothing stored).

    A fit that fails costs only itself: the others are still stored, as a
    partial run.
    """
    failures: list[str] = []
    recorder = PredictionRunRecorder(
        workflow=WORKFLOW_TABPFN, game_date=date_to_predict
    )
    recorder.set_schedule(
        rows.scheduled_games
        if rows.scheduled_games is not None
        else scheduled_data["scheduled_games"]
    )
    recorder.add_frame(job.dataset_type, rows.df, version=rows.frame_version)
    if rows.latest_quotes is not None:
        recorder.add_frame(
            INTERMEDIATE_LINE, rows.latest_quotes, version=BASE_SCHEMA_VERSION
        )
    predicted: list[str] = []
    passthrough = () if job.dataset_type == CLOSING_LINE else PASSTHROUGH_COLUMNS
    # Cleaned once for every target, not once per target.
    clean_rows = clean_prediction_frames(
        {job.dataset_type: rows.df}, passthrough_columns=passthrough
    ).get(job.dataset_type)
    # Each target in each mode that fits it: a fit and a variant of its own.
    target_inputs = dict(
        zip(
            job.targets,
            zip(inputs.specs, inputs.training_sets, strict=True),
            strict=True,
        )
    )
    fits = [(target, *target_inputs[target], mode) for target, mode in job.fits]
    for target, spec, training_set, mode in fits:
        label = (
            f"{spec.identity.schema_version}/{target}/{format_horizon(job.horizon)}/"
            f"{model_variant(job.model_version, mode)}"
        )
        # The standard fit keeps its phase names; another mode is named in them.
        phase = target if mode == STANDARD_MODE else f"{target} {mode}"
        log.step(f"{step_prefix}Fitting and Predicting ({label})")
        try:
            with timer.phase(f"{step_prefix}{phase} fit"):
                model = fit_tabpfn(
                    training_set, model_version=job.model_version, mode=mode
                )
            with timer.phase(f"{step_prefix}{phase} predict"):
                predictions = predict_with_tabpfn(
                    df=rows.df if clean_rows is None else clean_rows,
                    spec=spec,
                    model=model,
                    training_set=training_set,
                    prediction_datetime=datetime.now(ZoneInfo("Europe/Madrid")),
                    passthrough_columns=passthrough,
                    cleaned=clean_rows is not None,
                )
            if predictions.empty:
                _status(f"{target}: no game survived the spec's NaN budget.")
            else:
                predicted.append(label)
                print(
                    predictions[
                        [
                            "GAME_ID",
                            "TEAM_NAME_TEAM_HOME",
                            "TEAM_NAME_TEAM_AWAY",
                            VALUE_COLUMNS[target],
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
            _status(failures[-1], ok=False)

    if not persist:
        _status("Dry run (--no-persist): nothing stored")
    else:
        log.step(f"{step_prefix}Storing Predictions")
        with timer.phase(f"{step_prefix}storing"):
            failures += persist_prediction_run(
                recorder, s3_client=s3_client, bucket=bucket, partial=bool(failures)
            )
    return JobOutcome(predicted=predicted, failures=failures, n_fits=len(fits))


def tonight_closing_rows(
    scheduled_data: dict, *, schema_version: str, game_date: str
) -> pd.DataFrame:
    """Tonight's closing rows at ``schema_version``, as the daily predictor
    builds them."""
    tonight_ids = scheduled_data["scheduled_games"]["GAME_ID"].astype(str).tolist()
    df_tonight, layer_inputs = create_df_to_predict(
        todays_prediction=True,
        scheduled_data=scheduled_data,
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
