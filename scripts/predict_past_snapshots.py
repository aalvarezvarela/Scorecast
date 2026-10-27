"""Predict every intermediate slot from its own snapshot of a played date.

The daytime runs read each game's latest snapshot (``INTERMEDIATE_INPUT_MODE =
latest``). This one runs once every game of the date has tipped -- after they
finish is fine -- when every snapshot of every game is in the past: each slot
then predicts exactly the row of its own horizon, the row it was trained on
(``--intermediate-input own``), with no prediction across horizons.

Only the intermediate slots run. A closing slot reads the scheduled odds,
which after tip-off are no longer the closing line, and the daytime runs
already serve it.

The schedule keeps the games already started or finished
(``live_replay.replay_scheduled_games``: the scoreboard, with tip-off from the
season schedule), which the live schedule drops. Referees, rosters, injury
reports and line history are read as the daytime runs read them; each
snapshot only sees what was published before it.

    python scripts/predict_past_snapshots.py [--game-date YYYY-MM-DD] [--no-persist]
"""

from __future__ import annotations

import argparse
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from nba_ou.config.dataset_versions import BASE_SCHEMA_VERSION
from nba_ou.config.settings import SETTINGS
from nba_ou.create_training_data.live_replay import replay_scheduled_games
from nba_ou.data_processing.referees.process_refs_scheduled_game import (
    process_scheduled_referee_assignments,
)
from nba_ou.fetch_data.live_rosters import load_live_rosters
from nba_ou.postgre_db.predictions_aiven.schema import WORKFLOW_PAST_SNAPSHOTS
from nba_ou.postgre_db.update_all.update_all_databases import update_all_databases
from nba_ou.prediction.intermediate_frame import (
    PASSTHROUGH_COLUMNS,
    tonight_intermediate_frame,
)
from nba_ou.prediction.line_source import LineSource
from nba_ou.prediction.prediction import (
    clean_prediction_frames,
    load_registry_model_and_predict,
    read_slot_builds,
)
from nba_ou.prediction.prediction_store import (
    PredictionRunRecorder,
    persist_prediction_run,
)
from nba_ou.utils.general_utils import get_season_year_from_date
from nba_ou.utils.s3_models import make_s3_client

EASTERN = ZoneInfo("America/New_York")


def print_step_header(step_number: int, title: str) -> None:
    clock = datetime.now(ZoneInfo("UTC")).strftime("%H:%M:%S")
    print("-" * 70)
    print(f" STEP {step_number} — {title} [{clock} UTC] ".center(70))
    print("-" * 70)


def print_status(message: str, ok: bool = True) -> None:
    print(f"  {'✓' if ok else '✖'} {message}")


def default_game_date(now: pd.Timestamp | None = None) -> str:
    """The game date a run at ``now`` closes: the Eastern date 12 hours earlier.

    From the late games' final buzzer (~01:00 ET) to noon ET the next day this
    is the night before; on the evening itself it is that evening.
    """
    now = pd.Timestamp.now(tz=EASTERN) if now is None else pd.Timestamp(now)
    return (now.tz_convert(EASTERN) - pd.Timedelta(hours=12)).strftime("%Y-%m-%d")


def intermediate_slots(slots: list) -> list:
    """The slots with a horizon before tip-off; the closing ones are left out."""
    return [slot for slot in slots if slot.horizon_minutes]


def past_scheduled_data(game_date: str) -> dict:
    """What ``tonight_intermediate_frame`` reads of the schedule, for a date
    whose games may have started or finished.

    The lines are SBR's; a game SBR has no ticks for, or whose SBR history
    stops well before tip, is read from Covers (``prediction.line_source``).
    """
    scheduled = replay_scheduled_games(game_date)
    if scheduled.empty:
        return {"scheduled_games": scheduled}
    return {
        "scheduled_games": scheduled,
        "df_referees_scheduled": process_scheduled_referee_assignments(scheduled),
        "roster_snapshots": load_live_rosters(scheduled),
        "line_source": LineSource(pd.Timestamp(game_date).date()),
    }


def keep_passed_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame``'s rows whose snapshot is in the past, reporting the others."""
    passed = frame["SNAPSHOT_PASSED"].astype(bool)
    if not passed.all():
        ahead = frame.loc[~passed]
        print_status(
            f"{len(ahead)} snapshot row(s) of {ahead['GAME_ID'].nunique()} game(s) "
            "still ahead; left out (run again once every game has tipped)",
            ok=False,
        )
    return frame.loc[passed].reset_index(drop=True)


def predict_past_snapshots(
    game_date: str | None = None,
    *,
    persist: bool = True,
    update_databases: bool = True,
) -> None:
    game_date = game_date or default_game_date()
    print(f"\nPast-snapshot predictions for {game_date}")
    slots = intermediate_slots(SETTINGS.prediction_model_slots)
    print(f"  {len(slots)} intermediate slot(s), each on its own snapshot")
    if not slots:
        return

    step = 1
    if update_databases:
        print_step_header(step, "Updating All Databases")
        season = int(get_season_year_from_date(game_date))
        update_all_databases(
            start_season_year=season,
            end_season_year=season,
            only_new_games=True,
            headless=SETTINGS.headless,
            exclude_game_date=game_date,
        )
        print_status("Databases updated")
        step += 1

    print_step_header(step, "Fetching the Date's Games")
    step += 1
    scheduled_data = past_scheduled_data(game_date)
    scheduled = scheduled_data["scheduled_games"]
    if scheduled.empty:
        print_status(f"No games on {game_date}", ok=False)
        return
    print_status(f"{len(scheduled)} game(s) on {game_date}")

    print_step_header(step, "Preparing Intermediate Snapshot Rows")
    step += 1
    frame = tonight_intermediate_frame(scheduled_data, game_date=game_date)
    frame = keep_passed_snapshots(frame)
    if frame.empty:
        print_status("No snapshot has passed yet", ok=False)
        raise SystemExit(1)
    print_status(
        f"{len(frame)} past snapshot row(s) over {frame['GAME_ID'].nunique()} game(s)"
    )

    recorder = PredictionRunRecorder(
        workflow=WORKFLOW_PAST_SNAPSHOTS, game_date=game_date
    )
    recorder.set_schedule(scheduled)
    recorder.add_frame("intermediate_line", frame, version=BASE_SCHEMA_VERSION)

    s3 = make_s3_client(profile=SETTINGS.s3_aws_profile, region=SETTINGS.s3_aws_region)
    started = datetime.now(ZoneInfo("Europe/Madrid"))
    failures: list[str] = []
    # Cleaned once here, not once per slot: each slot reads its rows from it.
    clean_frames = clean_prediction_frames(
        {"intermediate_line": frame}, passthrough_columns=PASSTHROUGH_COLUMNS
    )
    builds = read_slot_builds(s3_client=s3, bucket=SETTINGS.s3_bucket, slots=slots)
    predicted = 0
    for slot in slots:
        label = slot.describe()
        print_step_header(step, f"Generating Predictions ({label})")
        step += 1
        try:
            build = builds.get(slot)
            if isinstance(build, Exception):
                raise build
            predictions = load_registry_model_and_predict(
                s3_client=s3,
                bucket=SETTINGS.s3_bucket,
                slot=slot,
                df=None,
                prediction_datetime=started,
                frames={"intermediate_line": frame},
                frame_versions={"intermediate_line": BASE_SCHEMA_VERSION},
                passthrough_columns=PASSTHROUGH_COLUMNS,
                latest_input=False,
                report_required_max_horizon=SETTINGS.report_required_max_horizon,
                clean_frames=clean_frames,
                resolved=build,
            )
        except Exception as exc:  # noqa: BLE001 -- the other slots still run
            traceback.print_exc()
            print_status(f"Failed for {label}: {exc}", ok=False)
            failures.append(f"{label}: {exc}")
            recorder.record_slot(label, "failed", reason=str(exc))
            continue
        rejected = recorder.add_predictions(predictions, label=label)
        if predictions is None or predictions.empty:
            print_status(f"No rows for {label}; skipped", ok=False)
            continue
        predicted += 1
        print_status(f"{len(predictions)} prediction(s) for {label}")
        if rejected:
            print_status(f"Prediction store rejected {label}: {rejected}", ok=False)
            failures.append(f"{label} (prediction store): {rejected}")

    if persist:
        print_step_header(step, "Storing Predictions")
        step += 1
        failures += persist_prediction_run(
            recorder,
            s3_client=s3,
            bucket=SETTINGS.s3_bucket,
            partial=bool(failures),
        )

    print_step_header(step, "Summary")
    print_status(f"{predicted}/{len(slots)} slot(s) predicted")
    print_status(f"{len(failures)} failure(s)", ok=not failures)
    for failure in failures:
        print(f"      - {failure}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--game-date",
        help="YYYY-MM-DD; default: the Eastern date 12 hours ago (last night's games)",
    )
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Predict without storing: no prediction-store rows, no S3 run archive",
    )
    parser.add_argument(
        "--skip-db-update",
        action="store_true",
        help="Do not update the databases first",
    )
    args = parser.parse_args()
    predict_past_snapshots(
        args.game_date,
        persist=not args.no_persist,
        update_databases=not args.skip_db_update,
    )
