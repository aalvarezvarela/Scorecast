"""Default loaders for :class:`~.base.LayerContext`.

Used when a layer runs over a dataset file and nothing was seeded. Each loader
repeats the steps ``create_df_to_predict`` takes for the same object, so a
layer computed from a file sees what it would have seen inside the full build.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_ou.create_training_data.predict_data_utils import (
    filter_by_seasons_with_extra_game_ids,
)
from nba_ou.utils.general_utils import get_season_year_from_date
from nba_ou.utils.seasons import get_seasons_between_dates

from .base import GAME_ID_COLUMN, SNAPSHOT_COLUMN, game_id_keys


def player_context_seasons(game_dates: pd.Series) -> list[str]:
    """The output seasons of ``game_dates`` plus one season before the first.

    ``create_df_to_predict`` keeps exactly this as player context: the extra
    season gives the first output season's earliest games a roster history.
    """
    dates = pd.to_datetime(game_dates, errors="coerce").dropna()
    if dates.empty:
        raise ValueError("Cannot load player history for a frame with no game dates.")
    first_season_year = get_season_year_from_date(dates.min())
    start = pd.Timestamp(year=first_season_year - 1, month=10, day=1)
    return get_seasons_between_dates(start, dates.max())


def load_player_history(game_dates: pd.Series) -> pd.DataFrame:
    """Cleaned player box scores up to the last of ``game_dates``.

    The same three steps as ``create_df_to_predict``: load games and players for
    the context seasons, attach dates and numeric minutes with
    ``clear_player_statistics``, then cap by season and date.
    """
    from nba_ou.data_processing.players.attach_player_features import (
        clear_player_statistics,
    )
    from nba_ou.postgre_db import load_all_nba_data_from_db

    seasons = player_context_seasons(game_dates)
    recent_limit = pd.to_datetime(game_dates, errors="coerce").max()
    print(f"Loading player history for seasons {seasons} ...")
    df_games, df_players = load_all_nba_data_from_db(seasons=seasons)
    df_games["GAME_DATE"] = pd.to_datetime(df_games["GAME_DATE"])
    df_games = filter_by_seasons_with_extra_game_ids(
        df_games, seasons=seasons, recent_limit_to_include=recent_limit
    )
    df_players = clear_player_statistics(df_players, df_games)
    return filter_by_seasons_with_extra_game_ids(
        df_players, seasons=seasons, recent_limit_to_include=recent_limit
    )


def _snapshot_keys(frame: pd.DataFrame) -> pd.DataFrame:
    if frame[GAME_ID_COLUMN].isna().any():
        raise ValueError("Snapshot rows must have a GAME_ID")
    horizons = pd.to_numeric(frame[SNAPSHOT_COLUMN], errors="coerce")
    if (
        not np.isfinite(horizons.to_numpy(float)).all()
        or horizons.lt(0).any()
        or horizons.mod(1).ne(0).any()
    ):
        raise ValueError("Snapshot horizons must be nonnegative integer minutes")
    keys = pd.DataFrame(
        {
            GAME_ID_COLUMN: game_id_keys(frame[GAME_ID_COLUMN]).to_numpy(),
            SNAPSHOT_COLUMN: horizons.to_numpy(dtype=np.int64),
        }
    )
    if keys.duplicated().any():
        raise ValueError("Snapshot rows must be unique per GAME_ID and horizon")
    return keys


def _load_tipoffs(inputs: pd.DataFrame) -> pd.DataFrame:
    """Read the same line-history schedule that defined the base snapshots."""
    from nba_ou.postgre_db.line_history_aiven.fetch import fetch_games

    dates = pd.to_datetime(inputs["GAME_DATE"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Cannot resolve snapshot tipoffs without valid GAME_DATEs")
    seasons = sorted({get_season_year_from_date(date) for date in dates})
    schedule = fetch_games(seasons).rename(
        columns={"game_id": GAME_ID_COLUMN, "tipoff_utc": "TIPOFF_UTC"}
    )
    schedule[GAME_ID_COLUMN] = game_id_keys(schedule[GAME_ID_COLUMN])
    if schedule[GAME_ID_COLUMN].duplicated().any():
        raise ValueError("Line-history schedule has duplicate GAME_IDs")
    return schedule[[GAME_ID_COLUMN, "TIPOFF_UTC"]]


def resolve_snapshot_cutoffs(
    inputs: pd.DataFrame, times: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Resolve strict injury-report cutoffs without exposing timestamps as X.

    Embedded timestamps are accepted for in-memory builds. File builds can seed
    the original scoring sidecar in ``LayerContext.snapshot_times``; otherwise
    the authoritative line-history schedule supplies tipoffs. Every row must
    resolve, and supplied tipoff/snapshot timestamps must agree with its horizon.
    """
    keys = _snapshot_keys(inputs)
    timestamp_columns = ["TIPOFF_UTC", "SNAPSHOT_TS_UTC"]
    embedded = [c for c in timestamp_columns if c in inputs.columns]
    if times is not None:
        supplied = [c for c in timestamp_columns if c in times.columns]
        if not supplied:
            raise ValueError("Snapshot time sidecar contains no UTC timestamps")
        source = pd.concat(
            [_snapshot_keys(times), times[supplied].reset_index(drop=True)], axis=1
        )
        resolved = keys.merge(
            source,
            on=[GAME_ID_COLUMN, SNAPSHOT_COLUMN],
            how="left",
            validate="one_to_one",
            sort=False,
        )
        # If an in-memory caller also supplies timestamps, require consistency.
        for column in embedded:
            if column in supplied:
                left = pd.to_datetime(inputs[column], utc=True, errors="coerce")
                right = pd.to_datetime(resolved[column], utc=True, errors="coerce")
                if not left.reset_index(drop=True).eq(right).all():
                    raise ValueError(f"Input and sidecar disagree on {column}")
            else:
                resolved[column] = inputs[column].to_numpy()
    elif embedded:
        resolved = pd.concat([keys, inputs[embedded].reset_index(drop=True)], axis=1)
    else:
        resolved = keys.merge(
            _load_tipoffs(inputs),
            on=GAME_ID_COLUMN,
            how="left",
            validate="many_to_one",
            sort=False,
        )

    delta = pd.to_timedelta(resolved[SNAPSHOT_COLUMN], unit="m")
    for column in timestamp_columns:
        if column in resolved:
            resolved[column] = pd.to_datetime(
                resolved[column], utc=True, errors="coerce"
            )
            if resolved[column].isna().any():
                raise ValueError(f"Missing or invalid {column} for snapshot rows")
    if "TIPOFF_UTC" in resolved:
        expected = resolved["TIPOFF_UTC"] - delta
        if (
            "SNAPSHOT_TS_UTC" in resolved
            and not expected.eq(resolved["SNAPSHOT_TS_UTC"]).all()
        ):
            raise ValueError("Snapshot timestamps do not match tipoff minus horizon")
        resolved["SNAPSHOT_TS_UTC"] = expected
    tipoffs = resolved["SNAPSHOT_TS_UTC"] + delta
    if tipoffs.groupby(resolved[GAME_ID_COLUMN]).nunique().gt(1).any():
        raise ValueError("A game's snapshots imply conflicting tipoff timestamps")
    return resolved[[GAME_ID_COLUMN, SNAPSHOT_COLUMN, "SNAPSHOT_TS_UTC"]].rename(
        columns={
            GAME_ID_COLUMN: "game_id",
            SNAPSHOT_COLUMN: "snapshot_minutes",
            "SNAPSHOT_TS_UTC": "as_of",
        }
    )
