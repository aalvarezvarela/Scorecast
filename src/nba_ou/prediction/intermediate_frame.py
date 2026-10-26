"""Tonight's intermediate-line rows for the prediction job.

The intermediate slots predict a game at a fixed time before tip-off (T-30 ...
T-1080). Tonight's rows come from ``create_intermediate_line_df`` in its
tonight-only mode: tonight's line history read live (``LiveGameDay``), every
earlier game from stored history, so each row is exactly what the full build
gives that (game, snapshot) once the game is stored.

Every horizon is kept, whether its snapshot has passed or not: the daily job
predicts every slot on every run (the user's policy, 2026-10-20). A row whose
snapshot is still ahead is built from the ticks so far, so it is NOT the row
the model was trained on; ``SNAPSHOT_PASSED`` says which kind each row is, and
``SNAPSHOT_TS_UTC`` when its snapshot falls, so the two can be told apart in
the stored predictions.

With ``latest_input`` each game also gets the row of the run's own moment, its
latest snapshot (``latest_snapshot_minutes``), flagged ``LATEST_INPUT``: the
slots in ``LATEST_INPUT_SLOTS`` predict from it whatever their horizon. A game
already tipped has its T-0 row as latest.

With ``latest_only`` as well, only those latest rows are built: when every
intermediate slot reads them, the other horizons' rows would be built for no
prediction. Each row kept is the row the full frame holds.

With ``horizon`` only each game's row of that horizon is built, for a job that
predicts one horizon (TabPFN): every other horizon would cost the injury stage
a full pass over the history for no prediction.
"""

from __future__ import annotations

import math

import pandas as pd
from nba_ou.create_training_data.create_intermediate_line_df import (
    create_intermediate_line_df,
)
from nba_ou.create_training_data.live_game_day import fetch_live_game_day
from nba_ou.data_processing.line_history.snapshots import DEFAULT_SNAPSHOT_GRID
from nba_ou.prediction.input_limits import LATEST_INPUT_COLUMN, LIMITED_COUNT_COLUMN
from nba_ou.prediction.line_source import LINE_SOURCE_COLUMN

#: The display columns the prediction summary reads, from tonight's schedule.
SCHEDULE_COLUMNS = ("GAME_TIME",)


#: Per-row snapshot columns the predictions carry through to their output.
SNAPSHOT_COLUMNS = ("SNAPSHOT_TS_UTC", "SNAPSHOT_PASSED")
#: What an intermediate prediction carries through: when its snapshot falls,
#: its horizon (the model's own or the latest), the cells the training
#: support limits changed (latest input only) and where its lines came from.
PASSTHROUGH_COLUMNS = (
    *SNAPSHOT_COLUMNS,
    "TIME_TO_MATCH_MIN",
    LIMITED_COUNT_COLUMN,
    LINE_SOURCE_COLUMN,
)


def latest_snapshot_minutes(tipoffs: pd.Series, as_of: pd.Timestamp) -> dict[str, int]:
    """Per game id (``tipoffs``' index), whole minutes before tip at ``as_of``.

    Rounded up, so the snapshot never falls after ``as_of``; a game already
    tipped reads 0, its closing snapshot.
    """
    as_of = pd.Timestamp(as_of)
    latest = {}
    for game_id, tipoff in tipoffs.items():
        if pd.isna(tipoff):
            continue
        seconds = (pd.Timestamp(tipoff) - as_of).total_seconds()
        latest[str(game_id)] = max(0, math.ceil(seconds / 60))
    return latest


def mark_snapshots(
    frame: pd.DataFrame, scoring: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    """``frame`` with each row's snapshot time and whether it is at or before ``as_of``.

    ``scoring`` is the builder's sidecar, matched on (game, snapshot): it is
    sorted by game id, the frame by date first.
    """
    keys = ["GAME_ID", "TIME_TO_MATCH_MIN"]
    snapshot = scoring[keys + ["SNAPSHOT_TS_UTC"]].assign(
        GAME_ID=lambda rows: rows["GAME_ID"].astype(str)
    )
    if snapshot.duplicated(keys).any():
        raise ValueError("scoring has more than one row per game and snapshot")
    times = (
        frame[keys]
        .assign(GAME_ID=lambda rows: rows["GAME_ID"].astype(str))
        .merge(snapshot, on=keys, how="left", validate="one_to_one")
    )
    if times["SNAPSHOT_TS_UTC"].isna().any():
        raise ValueError("scoring lacks rows of the frame")
    snapshot_ts = pd.to_datetime(times["SNAPSHOT_TS_UTC"], utc=True).to_numpy()
    marked = frame.drop(columns=list(SNAPSHOT_COLUMNS), errors="ignore").reset_index(
        drop=True
    )
    marked["SNAPSHOT_TS_UTC"] = pd.DatetimeIndex(snapshot_ts, tz="UTC")
    marked["SNAPSHOT_PASSED"] = marked["SNAPSHOT_TS_UTC"] <= pd.Timestamp(as_of)
    return marked


def tonight_intermediate_frame(
    scheduled_data: dict,
    *,
    game_date: str | pd.Timestamp,
    as_of: pd.Timestamp | None = None,
    latest_input: bool = False,
    latest_only: bool = False,
    horizon: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Tonight's (game, snapshot) rows for every horizon, with the schedule's
    display columns and the snapshot marks (``SNAPSHOT_COLUMNS``); with
    ``latest_input``, also each game's latest snapshot (``LATEST_INPUT``), and
    with ``latest_only`` that snapshot alone; with ``horizon``, only each
    game's row of that horizon.

    The run's ``line_source`` (``scheduled_data["line_source"]``) decides
    where the line history comes from; each row's is in ``LINE_SOURCE``."""
    if latest_only and not latest_input:
        raise ValueError("latest_only needs latest_input")
    if horizon is not None and latest_input:
        raise ValueError("horizon and latest_input are exclusive")
    scheduled = scheduled_data["scheduled_games"]
    line_source = scheduled_data.get("line_source")
    live = fetch_live_game_day(
        game_date,
        scheduled_games=scheduled,
        df_referees_scheduled=scheduled_data.get("df_referees_scheduled"),
        roster_snapshots=scheduled_data.get("roster_snapshots"),
        as_of=as_of,
        line_source=line_source,
    )
    latest: dict[str, int] = {}
    if latest_input:
        games = live.line_history.games
        tipoffs = games.set_index(games["game_id"].astype(str))["tipoff_utc"]
        tipoffs = tipoffs[tipoffs.index.isin(live.game_ids)]
        latest = latest_snapshot_minutes(tipoffs, live.as_of)
    if horizon is not None:
        snapshots = {"target_snapshots": dict.fromkeys(live.game_ids, int(horizon))}
    elif latest_only and latest:
        snapshots = {"target_snapshots": latest}
    else:
        snapshots = {
            "extra_target_snapshots": {
                game: minutes
                for game, minutes in latest.items()
                if minutes not in DEFAULT_SNAPSHOT_GRID
            }
        }
    frame, scoring = create_intermediate_line_df(
        live=live,
        target_game_ids=live.game_ids,
        return_scoring=True,
        verbose=verbose,
        **snapshots,
    )
    frame = mark_snapshots(frame, scoring, live.as_of)
    if latest_input:
        frame[LATEST_INPUT_COLUMN] = (
            frame["TIME_TO_MATCH_MIN"]
            .astype(int)
            .eq(frame["GAME_ID"].astype(str).map(latest))
        )
    display = scheduled[["GAME_ID", *SCHEDULE_COLUMNS]].assign(
        GAME_ID=lambda games: games["GAME_ID"].astype(str)
    )
    frame = frame.drop(columns=list(SCHEDULE_COLUMNS), errors="ignore").merge(
        display, on="GAME_ID", how="left", validate="many_to_one"
    )
    if line_source is not None:
        frame[LINE_SOURCE_COLUMN] = (
            frame["GAME_ID"].astype(str).map(line_source.history_sources)
        )
    return frame
