"""Tonight's intermediate-line rows for the prediction job.

The intermediate slots predict a game at a fixed time before tip-off (T-30 ...
T-1080). Tonight's rows come from ``create_intermediate_line_df`` in its
tonight-only mode: tonight's line history read live (``LiveGameDay``), every
earlier game from stored history, so each row is exactly what the full build
gives that (game, snapshot) once the game is stored.

Only snapshots that have already passed are kept. The builder also makes rows
for horizons still ahead -- from the latest ticks so far, not from the real
snapshot -- and those must never be predicted from.
"""

from __future__ import annotations

import pandas as pd
from nba_ou.create_training_data.create_intermediate_line_df import (
    create_intermediate_line_df,
)
from nba_ou.create_training_data.live_game_day import fetch_live_game_day

#: The display columns the prediction summary reads, from tonight's schedule.
SCHEDULE_COLUMNS = ("GAME_TIME",)


def passed_snapshots(
    frame: pd.DataFrame, scoring: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    """``frame``'s rows whose snapshot time is at or before ``as_of``.

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
    passed = (
        pd.to_datetime(times["SNAPSHOT_TS_UTC"], utc=True) <= pd.Timestamp(as_of)
    ).to_numpy()
    return frame.loc[passed].reset_index(drop=True)


def tonight_intermediate_frame(
    scheduled_data: dict,
    *,
    game_date: str | pd.Timestamp,
    as_of: pd.Timestamp | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Tonight's (game, passed snapshot) rows, with the schedule's display columns.

    Empty when no snapshot of tonight has passed yet.
    """
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
    frame = passed_snapshots(frame, scoring, live.as_of)
    display = scheduled[["GAME_ID", *SCHEDULE_COLUMNS]].assign(
        GAME_ID=lambda games: games["GAME_ID"].astype(str)
    )
    frame = frame.drop(columns=list(SCHEDULE_COLUMNS), errors="ignore").merge(
        display, on="GAME_ID", how="left", validate="many_to_one"
    )
    return frame
