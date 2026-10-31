"""Refit on earlier saved rows and predict saved snapshots; never rebuild datasets.

No production booster is read. Training partitions and source databases are
read-only; builds are kept under a separate root and never promoted. Dates
are NBA calendar dates (America/New_York). Local reports are written in both
modes; Aiven/S3 writes require persist=True. TabPFN optionally uses its hosted
client for a real standard fit, including during dry runs.
"""

from __future__ import annotations

import hashlib
import json
import traceback
import uuid
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd
from nba_ou.config.dataset_versions import BASE_SCHEMA_VERSION
from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.live_replay import replay_scheduled_games
from nba_ou.create_training_data.season_partitions import SeasonPartitions
from nba_ou.data_processing.line_history.snapshots import DEFAULT_SNAPSHOT_GRID
from nba_ou.modeling.refit import (
    TrainingFrame,
    model_bytes_of,
    refit_device,
    refit_from_spec,
    require_device,
    resolve_training_frame,
)
from nba_ou.modeling.registry_paths import ModelTarget
from nba_ou.modeling.registry_store import (
    ResolvedBuild,
    resolve_config_spec,
    write_build,
    write_spec,
)
from nba_ou.postgre_db.config.connection_keep import kept_connections
from nba_ou.postgre_db.line_history_aiven.fetch import fetch_games
from nba_ou.postgre_db.predictions_aiven.coverage import historical_coverage
from nba_ou.postgre_db.predictions_aiven.schema import (
    KIND_REGISTRY,
    KIND_TABPFN,
    RUN_STATUS_OK,
    RUN_STATUS_PARTIAL,
    TARGET_CODES,
    WORKFLOW_HISTORICAL_BACKFILL,
)
from nba_ou.prediction.historical_inputs import (
    StoredPredictionInputs,
)
from nba_ou.prediction.intermediate_frame import PASSTHROUGH_COLUMNS
from nba_ou.prediction.prediction import load_registry_model_and_predict
from nba_ou.prediction.prediction_store import (
    PredictionRunRecorder,
    no_report_games,
    persist_prediction_run,
)
from nba_ou.prediction.prediction_tabpfn_client import (
    DEFAULT_MODEL_VERSION,
    STANDARD_MODE,
    feature_slot_for,
    fit_tabpfn,
    model_variant,
    predict_with_tabpfn,
    tabpfn_training_set,
)
from nba_ou.prediction.report_requirement import reports_filed, requires_report
from nba_ou.utils.general_utils import get_season_year_from_date
from nba_ou.utils.s3_models import make_s3_client

EASTERN = "America/New_York"
DEFAULT_TABPFN_BACKFILL_HORIZON = 240


def tabpfn_feature_slots(horizon: int, *, slots=None, include_closing=False):
    """One enabled feature specification per target, at exactly one horizon."""
    if horizon <= 0 or horizon not in DEFAULT_SNAPSHOT_GRID:
        raise ValueError(
            f"TabPFN horizon must be one of {sorted(h for h in DEFAULT_SNAPSHOT_GRID if h > 0)}"
        )
    available = SETTINGS.prediction_model_slots if slots is None else slots
    return [
        feature_slot_for(target.value, available, horizon_minutes=point)
        for point in ([0, horizon] if include_closing else [horizon])
        for target in ModelTarget
    ]


def tabpfn_label(slot) -> str:
    return f"{slot.schema_version}/{slot.target}/{slot.horizon_label}/{model_variant(DEFAULT_MODEL_VERSION, STANDARD_MODE)}"


def backfill_dates(start: str, end: str | None = None, *, now=None) -> list[str]:
    """Validate the entire range before any work or external write."""
    first, last = date.fromisoformat(start), date.fromisoformat(end or start)
    today = (
        pd.Timestamp.now(tz=EASTERN)
        if now is None
        else pd.Timestamp(now).tz_convert(EASTERN)
    ).date()
    if first > last:
        raise ValueError("start-date must be on or before end-date")
    if last >= today:
        raise ValueError("Only dates before today's Eastern date can be backfilled")
    season = get_season_year_from_date(today.isoformat())
    if any(
        get_season_year_from_date(day.isoformat()) != season for day in (first, last)
    ):
        raise ValueError("The date range must be in the current NBA season")
    return [day.date().isoformat() for day in pd.date_range(first, last)]


def history_limit(game_date: str) -> pd.Timestamp:
    """Use history through D-1 for every snapshot, regardless of tipoff time."""
    return pd.Timestamp(game_date) - pd.Timedelta(days=1)


def past_training_frame(frame: TrainingFrame, limit: pd.Timestamp) -> TrainingFrame:
    """Cut BEFORE cleaning, weighting and selecting the rolling window."""
    dates = pd.to_datetime(frame.df["GAME_DATE"], errors="raise")
    if dates.isna().any():
        raise ValueError("Training data contains missing game dates")
    rows = frame.df.loc[dates.lt(pd.Timestamp(limit) + pd.Timedelta(days=1))].copy()
    if rows.empty:
        raise ValueError(f"No training games through {limit.date()}")
    digest = hashlib.sha256()
    digest.update(json.dumps(list(rows.columns)).encode())
    digest.update(pd.util.hash_pandas_object(rows, index=False).values.tobytes())
    return TrainingFrame(
        df=rows,
        build_id=f"{frame.build_id}:through={limit.date()}",
        checksum=f"sha256:{digest.hexdigest()[:16]}",
    )


def missing_games(
    scheduled,
    coverage,
    slot,
    spec_id,
    *,
    kind=KIND_REGISTRY,
    variant=None,
    input_horizon_minutes=None,
) -> set[str]:
    ids = set(scheduled.GAME_ID.astype(str))
    input_horizon = (
        slot.horizon_minutes if input_horizon_minutes is None else input_horizon_minutes
    )
    if coverage.empty:
        return ids
    covered = coverage.loc[
        coverage.spec_id.eq(spec_id)
        & coverage.schema_version.eq(slot.schema_version)
        & coverage.target.eq(TARGET_CODES[str(slot.target)])
        & coverage.horizon_minutes.eq(slot.horizon_minutes)
        & coverage.variant.eq(slot.variant if variant is None else variant)
        & coverage.kind.eq(kind),
    ].copy()
    if "dataset_type" in covered:
        covered = covered.loc[
            covered.dataset_type.eq(1 if slot.horizon_minutes == 0 else 2)
        ]
    if "input_horizon_minutes" in covered:
        covered = covered.loc[covered.input_horizon_minutes.eq(input_horizon)]
    elif slot.horizon_minutes == 0:
        return ids
    if covered.empty:
        return ids
    limits = {
        int(game.GAME_ID): history_limit(
            pd.Timestamp(game.GAME_DATE_EST).date().isoformat(),
        )
        for game in scheduled.itertuples()
    }
    eligible = pd.to_datetime(covered.train_date_max).le(covered.game_id.map(limits))
    return ids - {str(int(game)).zfill(10) for game in covered.loc[eligible, "game_id"]}


def run_backfill(
    start_date: str,
    end_date: str | None = None,
    *,
    slots=None,
    persist: bool = False,
    overwrite: bool = False,
    output_dir: Path = Path("data/backfill_predictions"),
    include_tabpfn: bool = False,
    include_closing: bool = False,
    tabpfn_horizon_minutes: int = DEFAULT_TABPFN_BACKFILL_HORIZON,
) -> int:
    dates = backfill_dates(start_date, end_date)
    if (
        include_closing or include_tabpfn
    ) and tabpfn_horizon_minutes not in DEFAULT_SNAPSHOT_GRID:
        raise ValueError("Shared snapshot must be a positive stored grid horizon")
    if (include_closing or include_tabpfn) and tabpfn_horizon_minutes <= 0:
        raise ValueError("Shared snapshot must be a positive stored grid horizon")
    slots = list(SETTINGS.prediction_model_slots if slots is None else slots)
    slots = [
        slot
        for slot in slots
        if slot.horizon_minutes is not None
        and (slot.horizon_minutes > 0 or include_closing)
    ]
    tabpfn_slots = (
        tabpfn_feature_slots(tabpfn_horizon_minutes, include_closing=include_closing)
        if include_tabpfn
        else []
    )
    if not slots and not tabpfn_slots:
        raise ValueError(
            "No model slots selected; enable --include-closing for closing slots (pooled unsupported)"
        )
    if len(set(slots)) != len(slots):
        raise ValueError("Duplicate model slots")
    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    bucket = SETTINGS.s3_bucket
    # Pin configurations once for the whole date range; never resolve a booster.
    specs = {
        slot: resolve_config_spec(s3_client=s3, bucket=bucket, slot=slot)
        for slot in dict.fromkeys([*slots, *tabpfn_slots])
    }
    for slot, spec in specs.items():
        spec.check_matches_slot(slot)
        if (
            str(spec.identity.dataset_type)
            != ("closing_line" if slot.horizon_minutes == 0 else "intermediate_line")
            or slot.schema_version != BASE_SCHEMA_VERSION
        ):
            raise ValueError(f"Unsupported historical dataset: {slot.describe()}")
    for device in sorted({refit_device(specs[slot]) for slot in slots}):
        require_device(device)
    coverage = pd.DataFrame() if overwrite else historical_coverage(dates[0], dates[-1])
    stores = {
        dataset: SeasonPartitions(
            s3_client=s3,
            bucket=bucket,
            dataset_type=dataset,
            schema_version=BASE_SCHEMA_VERSION,
        )
        for dataset in {
            "intermediate_line",
            *(str(spec.identity.dataset_type) for spec in specs.values()),
        }
    }
    stored_inputs = StoredPredictionInputs(stores, tipoff_loader=fetch_games)
    run_id = uuid.uuid4().hex
    destination = output_dir / run_id
    destination.mkdir(parents=True, exist_ok=True)
    print(
        f"Historical backfill: {dates[0]} to {dates[-1]}; {'PERSIST' if persist else 'DRY RUN'}; reports: {destination}"
    )
    failures = []
    for game_date in dates:
        try:
            with kept_connections():
                failures.extend(
                    _backfill_date(
                        game_date,
                        slots,
                        specs,
                        coverage,
                        stores,
                        s3,
                        bucket,
                        persist,
                        destination,
                        run_id,
                        tabpfn_slots=tabpfn_slots,
                        snapshot_minutes=tabpfn_horizon_minutes,
                        stored_inputs=stored_inputs,
                        overwrite=overwrite,
                    )
                )
        except (
            Exception
        ) as exc:  # noqa: BLE001 -- retain other dates and the local report
            traceback.print_exc()
            failures.append(f"{game_date}: {exc}")
            (destination / f"{game_date}.error.json").write_text(
                json.dumps({"error": str(exc)})
            )
    (destination / "summary.json").write_text(
        json.dumps(
            {
                "persist": persist,
                "overwrite": overwrite,
                "input_source": "stored",
                "dates": dates,
                "include_closing": include_closing,
                "snapshot_minutes": tabpfn_horizon_minutes,
                "tabpfn": {
                    "enabled": include_tabpfn,
                    "horizon_minutes": tabpfn_horizon_minutes,
                    "mode": STANDARD_MODE,
                    "model_version": DEFAULT_MODEL_VERSION,
                },
                "specs": {
                    slot.describe(): spec.spec_id for slot, spec in specs.items()
                },
                "failures": failures,
            },
            indent=2,
        )
    )
    print(
        f"Historical backfill finished: {len(failures)} failure(s). Reports: {destination}"
    )
    return 1 if failures else 0


def _backfill_date(
    game_date,
    slots,
    specs,
    coverage,
    stores,
    s3,
    bucket,
    persist,
    destination,
    run_id,
    *,
    tabpfn_slots=(),
    snapshot_minutes=DEFAULT_TABPFN_BACKFILL_HORIZON,
    stored_inputs=None,
    overwrite=False,
):
    scheduled = replay_scheduled_games(game_date)
    if scheduled.empty:
        print(f"{game_date}: no games")
        return []

    def input_horizon(slot):
        return snapshot_minutes if slot.horizon_minutes == 0 else slot.horizon_minutes

    missing = {
        slot: missing_games(
            scheduled,
            coverage,
            slot,
            specs[slot].spec_id,
            input_horizon_minutes=input_horizon(slot),
        )
        for slot in slots
    }
    tabpfn_missing = {
        slot: missing_games(
            scheduled,
            coverage,
            slot,
            specs[slot].spec_id,
            kind=KIND_TABPFN,
            variant=model_variant(DEFAULT_MODEL_VERSION, STANDARD_MODE),
            input_horizon_minutes=input_horizon(slot),
        )
        for slot in tabpfn_slots
    }
    recorder = PredictionRunRecorder(
        workflow=WORKFLOW_HISTORICAL_BACKFILL, game_date=game_date
    )
    recorder.set_schedule(scheduled)
    wanted = set().union(*missing.values(), *tabpfn_missing.values())
    if not wanted:
        print(f"{game_date}: all requested predictions already exist")
        return []
    stored_inputs = stored_inputs or StoredPredictionInputs(
        stores, tipoff_loader=fetch_games
    )
    frames, limits, failures, input_sources = {}, {}, [], {}

    def prediction_frame(horizon, ids, *, required_features=()):
        actual_horizon = snapshot_minutes if horizon == 0 else horizon
        frame = stored_inputs.read(
            game_date,
            actual_horizon,
            ids,
            closing=horizon == 0,
            required_features=required_features,
        )
        input_sources.setdefault(str(horizon), []).append(frame.attrs["input_source"])
        return frame, history_limit(game_date)

    for horizon in sorted({slot.horizon_minutes for slot in [*slots, *tabpfn_slots]}):
        ids = set().union(
            *(missing[slot] for slot in slots if slot.horizon_minutes == horizon),
            *(
                tabpfn_missing[slot]
                for slot in tabpfn_slots
                if slot.horizon_minutes == horizon
            ),
        )
        if not ids:
            continue
        try:
            features = set().union(
                *(
                    set(specs[slot].features.feature_names)
                    for slot in [*slots, *tabpfn_slots]
                    if slot.horizon_minutes == horizon
                )
            )
            frame, limits[horizon] = prediction_frame(
                horizon, ids, required_features=features
            )
            absent = ids - set(frame.GAME_ID.astype(str))
            if absent:
                failures.append(
                    f"{game_date} T-{horizon}: missing or unsafe stored snapshot rows for {sorted(absent)}; dataset reconstruction is disabled"
                )
            frames[horizon] = frame
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            failures.append(f"{game_date} T-{horizon}: {exc}")
            for slot in slots:
                if slot.horizon_minutes == horizon and missing[slot]:
                    recorder.record_slot(slot.describe(), "failed", reason=str(exc))
            for slot in tabpfn_slots:
                if slot.horizon_minutes == horizon and tabpfn_missing[slot]:
                    recorder.record_slot(tabpfn_label(slot), "failed", reason=str(exc))
    # Closing uses the same timestamped quote record as intermediate at the
    # shared timepoint. These rows provide stored bet prices and quote age.
    quote_frames = {h: frame.copy() for h, frame in frames.items() if h > 0}
    if 0 in frames:
        closing_ids = set(frames[0].GAME_ID.astype(str))
        existing = quote_frames.get(snapshot_minutes, pd.DataFrame())
        extra = closing_ids - (
            set(existing.GAME_ID.astype(str)) if not existing.empty else set()
        )
        if extra:
            try:
                quotes, _ = prediction_frame(snapshot_minutes, extra)
                quote_frames[snapshot_minutes] = pd.concat(
                    [existing, quotes], ignore_index=True
                )
            except Exception as exc:  # noqa: BLE001
                # No untimestamped/final-quote fallback for closing inputs.
                failures.append(f"{game_date} stored closing quote snapshot: {exc}")
                frames.pop(0)
        if 0 in frames:
            recorder.add_frame("closing_line", frames[0], version=BASE_SCHEMA_VERSION)
    if quote_frames:
        for h, quotes in quote_frames.items():
            quotes["LATEST_INPUT"] = h == snapshot_minutes
        recorder.add_frame(
            "intermediate_line",
            pd.concat(quote_frames.values(), ignore_index=True),
            version=BASE_SCHEMA_VERSION,
        )
    for horizon, frame in frames.items():
        dataset = "closing_line" if horizon == 0 else "intermediate_line"
        store = stores[dataset]
        training = None
        for slot in [slot for slot in slots if slot.horizon_minutes == horizon]:
            label = slot.describe()
            rows = frame.loc[frame.GAME_ID.astype(str).isin(missing[slot])].copy()
            if rows.empty:
                recorder.record_slot(
                    label, "skipped", reason="already covered or no historical snapshot"
                )
                continue
            try:
                if training is None:
                    training = past_training_frame(
                        resolve_training_frame(specs[slot], partitions=stores),
                        limits[horizon],
                    )
                historical_slot = replace(
                    slot,
                    root=f"{slot.root.rstrip('/')}/historical_backfills/{run_id}/{game_date}/",
                )
                result = refit_from_spec(
                    specs[slot],
                    slot=historical_slot,
                    frame=training,
                    fitted_by="historical_backfill",
                )
                if (
                    pd.Timestamp(result.fit.train_date_max).date()
                    > limits[horizon].date()
                ):
                    raise ValueError(
                        "Refitted model violates the historical training cutoff"
                    )
                # The registry's default fit id has second precision and a
                # spec suffix. Fast fits of several historical dates can
                # otherwise collide in Aiven's build key, even in different
                # S3 directories. Include this execution/date/data identity.
                identity = (
                    f"{run_id}:{game_date}:{result.spec.spec_id}:{training.checksum}"
                )
                suffix = hashlib.sha256(identity.encode()).hexdigest()[:8]
                result = replace(
                    result,
                    fit=result.fit.model_copy(
                        update={"fit_id": f"{result.fit.fit_id.split('-')[0]}-{suffix}"}
                    ),
                )
                resolved = ResolvedBuild(
                    slot=historical_slot,
                    fit=result.fit,
                    spec=result.spec,
                    model_bytes=model_bytes_of(result.model),
                )
                predictions = load_registry_model_and_predict(
                    s3_client=s3,
                    bucket=bucket,
                    slot=historical_slot,
                    df=None,
                    prediction_datetime=recorder.started_at,
                    frames={dataset: rows},
                    frame_versions={dataset: BASE_SCHEMA_VERSION},
                    passthrough_columns=PASSTHROUGH_COLUMNS,
                    latest_input=False,
                    report_required_max_horizon=SETTINGS.report_required_max_horizon,
                    resolved=resolved,
                )
                # A game without both injury reports at a short horizon is
                # skipped as the live slot skips it: recorded, not a failure.
                no_report = set(no_report_games(predictions))
                if predictions is None or predictions.empty:
                    recorder.add_predictions(predictions, label=label)
                    if set(rows.GAME_ID.astype(str)) - no_report:
                        failures.append(
                            f"{game_date} {label}: no eligible prediction rows"
                        )
                    continue
                omitted = (
                    set(rows.GAME_ID.astype(str))
                    - set(predictions.GAME_ID.astype(str))
                    - no_report
                )
                if omitted:
                    failures.append(
                        f"{game_date} {label}: no eligible prediction for {sorted(omitted)}"
                    )
                print(
                    f"{game_date} {label}: {len(predictions)} prediction(s); trained through {result.fit.train_date_max.date()}"
                )
                if persist:
                    write_spec(
                        s3_client=s3,
                        bucket=bucket,
                        slot=historical_slot,
                        spec=result.spec,
                    )
                    write_build(
                        s3_client=s3,
                        bucket=bucket,
                        slot=historical_slot,
                        fit=result.fit,
                        spec=result.spec,
                        model_bytes=resolved.model_bytes,
                    )
                rejected = recorder.add_predictions(predictions, label=label)
                if rejected:
                    raise ValueError(rejected)
            except Exception as exc:  # noqa: BLE001 -- no fallback to any served model
                traceback.print_exc()
                recorder.record_slot(label, "failed", reason=str(exc))
                failures.append(f"{game_date} {label}: {exc}")
        for slot in [slot for slot in tabpfn_slots if slot.horizon_minutes == horizon]:
            label = tabpfn_label(slot)
            rows = frame.loc[
                frame.GAME_ID.astype(str).isin(tabpfn_missing[slot])
            ].copy()
            if rows.empty:
                recorder.record_slot(
                    label, "skipped", reason="already covered or no historical snapshot"
                )
                continue
            try:
                if training is None:
                    training = past_training_frame(
                        resolve_training_frame(specs[slot], partitions=stores),
                        limits[horizon],
                    )
                _predict_historical_tabpfn(
                    recorder,
                    slot,
                    specs[slot],
                    training,
                    rows,
                    limit=limits[horizon],
                    frame_key=store.prefix,
                )
            except Exception as exc:  # noqa: BLE001 -- keep the other targets
                traceback.print_exc()
                recorder.record_slot(label, "failed", reason=str(exc))
                failures.append(f"{game_date} {label}: {exc}")
    tables = recorder.tables(status=RUN_STATUS_PARTIAL if failures else RUN_STATUS_OK)
    tables.predictions.to_csv(destination / f"{game_date}.predictions.csv", index=False)
    tables.model_builds.to_csv(destination / f"{game_date}.models.csv", index=False)
    tables.inputs.to_csv(destination / f"{game_date}.inputs.csv", index=False)
    (destination / f"{game_date}.json").write_text(
        json.dumps(
            {
                "game_date": game_date,
                "persist": persist,
                "overwrite": overwrite,
                "slots": recorder.slots,
                "input_source": "stored",
                "input_sources": input_sources,
                "training_limits": {
                    str(h): str(limit.date()) for h, limit in limits.items()
                },
                "failures": failures,
            },
            indent=2,
            default=str,
        )
    )
    if persist:
        failures.extend(
            persist_prediction_run(
                recorder,
                s3_client=s3,
                bucket=bucket,
                partial=bool(failures),
                replace_historical=overwrite,
            )
        )
    return failures


def _predict_historical_tabpfn(
    recorder, slot, spec, training, rows, *, limit, frame_key
):
    """Standard TabPFN, with the training cutoff enforced before any API call."""
    label = tabpfn_label(slot)
    requested = set(rows.GAME_ID.astype(str))
    input_horizon = (
        int(rows.TIME_TO_MATCH_MIN.min())
        if "TIME_TO_MATCH_MIN" in rows
        else slot.horizon_minutes
    )
    if requires_report(input_horizon, SETTINGS.report_required_max_horizon):
        rows = rows.loc[reports_filed(rows)].copy()
    if rows.empty:
        raise ValueError("No historical game has both injury reports filed")
    training_set = tabpfn_training_set(training.df, spec, frame_key=frame_key)
    if pd.Timestamp(training_set.train_date_max).date() > limit.date():
        raise ValueError("TabPFN training set violates the historical training cutoff")
    model = fit_tabpfn(
        training_set, model_version=DEFAULT_MODEL_VERSION, mode=STANDARD_MODE
    )
    predictions = predict_with_tabpfn(
        df=rows,
        spec=spec,
        model=model,
        training_set=training_set,
        prediction_datetime=recorder.started_at,
        passthrough_columns=PASSTHROUGH_COLUMNS,
    )
    if predictions is None or predictions.empty:
        recorder.add_tabpfn(predictions, label=label)
        raise ValueError("No eligible TabPFN prediction rows")
    # Keep the filtered dataset identity in both the build and the run's
    # manifest. Hosted TabPFN has no booster to upload.
    predictions = predictions.copy()
    predictions["FIT_ID"] = (
        predictions["FIT_ID"].astype(str)
        + "-"
        + training.checksum.removeprefix("sha256:")
    )
    rejected = recorder.add_tabpfn(predictions, label=label)
    if rejected:
        raise ValueError(rejected)
    recorder.slots[-1].update(
        training_build_id=training.build_id,
        training_checksum=training.checksum,
        training_games=training_set.n_games,
        mode=STANDARD_MODE,
        model_version=DEFAULT_MODEL_VERSION,
    )
    print(
        f"{recorder.game_date} {label}: {len(predictions)} prediction(s); trained through {training_set.train_date_max.date()}"
    )
    omitted = requested - set(predictions.GAME_ID.astype(str))
    if omitted:
        raise ValueError(f"No eligible TabPFN prediction for {sorted(omitted)}")
