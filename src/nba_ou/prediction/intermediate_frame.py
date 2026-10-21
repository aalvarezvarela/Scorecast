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
"""

from __future__ import annotations

import pandas as pd
from nba_ou.create_training_data.create_intermediate_line_df import (
    create_intermediate_line_df,
)
from nba_ou.create_training_data.live_game_day import fetch_live_game_day

#: The display columns the prediction summary reads, from tonight's schedule.
SCHEDULE_COLUMNS = ("GAME_TIME",)


#: Per-row snapshot columns the predictions carry through to their output.
SNAPSHOT_COLUMNS = ("SNAPSHOT_TS_UTC", "SNAPSHOT_PASSED")


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
    verbose: bool = True,
) -> pd.DataFrame:
    """Tonight's (game, snapshot) rows for every horizon, with the schedule's
    display columns and the snapshot marks (``SNAPSHOT_COLUMNS``)."""
    scheduled = scheduled_data["scheduled_games"]
    live = fetch_live_game_day(
        game_date,
        scheduled_games=scheduled,
        df_referees_scheduled=scheduled_data.get("df_referees_scheduled"),
        roster_snapshots=scheduled_data.get("roster_snapshots"),
        as_of=as_of,
    )
    frame, scoring = create_intermediate_line_df(
        live=live,
        target_game_ids=live.game_ids,
        return_scoring=True,
        verbose=verbose,
    )
    frame = mark_snapshots(frame, scoring, live.as_of)
    display = scheduled[["GAME_ID", *SCHEDULE_COLUMNS]].assign(
        GAME_ID=lambda games: games["GAME_ID"].astype(str)
    )
    frame = frame.drop(columns=list(SCHEDULE_COLUMNS), errors="ignore").merge(
        display, on="GAME_ID", how="left", validate="many_to_one"
    )
    return frame
