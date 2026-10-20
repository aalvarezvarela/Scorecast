"""The closing lineup family evaluated at each intermediate injury cutoff.

Player history and ratings are shared across horizons on the same history date;
availability and its minute redistribution vary. Earlier-day snapshots also
use earlier history and ratings. Calibration is separate per horizon and phase.
Style traits and the monthly neighbour model are built once for all snapshots.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pandas as pd

from nba_ou.data_processing.injury_status.report_state import InjuryReportState

from .availability import player_out_probabilities, roster_exclusions
from .features import (
    _IMPACT_COLUMNS,
    LINEUP_FEATURE_COLUMNS,
    RatingBook,
    game_nights,
    game_phase,
    project_lineup_games,
    walk_forward_offset,
)
from .game_projection import PlayerNight
from .style_matchup import STYLE_FEATURE_COLUMNS, build_style_matchup_features

SNAPSHOT_COLUMN = "TIME_TO_MATCH_MIN"
KEYS = ["GAME_ID", SNAPSHOT_COLUMN]

#: An Eastern date's box scores and results count as known only from this hour
#: (ET) the next morning. Tip-offs run to 23:00 ET, so with overtime the last
#: games of a date can still be in progress after midnight: a T-18h cutoff of a
#: 19:30 game falls at 01:30 ET (2019-26: 7% of T-1080 rows followed a tip less
#: than three hours earlier).
DAY_SETTLED_HOUR_ET = 5


def snapshot_history_dates(inputs: pd.DataFrame) -> pd.Series:
    """Conservative history cutoff when only box-score/stint dates are known.

    History is read from dates strictly before the returned date: the earlier
    of the game date and the cutoff's Eastern date, where a cutoff before
    ``DAY_SETTLED_HOUR_ET`` belongs to the previous date.
    """
    dates = pd.to_datetime(inputs["GAME_DATE"]).dt.normalize()
    cutoffs = pd.to_datetime(inputs["SNAPSHOT_TS_UTC"], utc=True, errors="coerce")
    if dates.isna().any() or cutoffs.isna().any():
        raise ValueError("Lineup snapshots need valid game dates and UTC cutoffs")
    eastern = cutoffs.dt.tz_convert("America/New_York").dt.tz_localize(None)
    # Hold back the entire cutoff day because publication timestamps are absent,
    # and the previous day until its late games have surely finished.
    cutoff_dates = (eastern - pd.Timedelta(hours=DAY_SETTLED_HOUR_ET)).dt.normalize()
    return dates.where(dates.le(cutoff_dates), cutoff_dates)


def snapshot_lineup_features(
    inputs: pd.DataFrame,
    df_players: pd.DataFrame,
    ratings: RatingBook,
    states: Mapping[int, InjuryReportState],
    *,
    stints: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return only the 17 new columns, aligned to the parent's original rows.

    ``inputs`` carries normalized game IDs and integer horizons, validated by
    the layer's cutoff resolver. ``states`` must be read at those exact UTC
    cutoffs, not at closing. Target-game/same-day box scores never enter the
    shared reference rosters. Both teams must have filed at a snapshot for any
    of its columns to be populated or its game to enter calibration.
    """
    output = pd.DataFrame(
        np.nan, index=pd.RangeIndex(len(inputs)), columns=list(LINEUP_FEATURE_COLUMNS)
    )
    if inputs.empty:
        output.index = inputs.index
        return output

    frame = (
        inputs.rename(
            columns={
                "TEAM_ID_TEAM_HOME": "HOME_TEAM_ID",
                "TEAM_ID_TEAM_AWAY": "AWAY_TEAM_ID",
            }
        )
        .reset_index(drop=True)
        .copy()
    )
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"]).dt.normalize()
    if frame["GAME_DATE"].isna().any():
        raise ValueError("Lineup snapshots need valid game dates")
    # Box scores/stints have dates, not publication times. Conservatively hold
    # back the entire cutoff day, and the previous one before it has settled.
    frame["_HISTORY_DATE"] = snapshot_history_dates(frame)
    frame["_HISTORY_LAG"] = (frame["GAME_DATE"] - frame["_HISTORY_DATE"]).dt.days
    if frame.duplicated(KEYS).any():
        raise ValueError("Lineup snapshots must be unique per game and horizon")
    game_columns = ["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID"]
    consistent = [*game_columns[1:]]
    if "TOTAL_POINTS" in frame:
        consistent.append("TOTAL_POINTS")
    if frame.groupby("GAME_ID")[consistent].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Snapshots of a game disagree on its teams, date or outcome")
    reference = {}
    for lag, group in frame.groupby("_HISTORY_LAG", sort=True):
        games = (
            group[game_columns]
            .assign(GAME_DATE=group["_HISTORY_DATE"])
            .drop_duplicates("GAME_ID")
        )
        reference.update(
            {
                (game, int(lag), team): players
                for (game, team), players in game_nights(games, df_players).items()
            }
        )
    style_nights: dict[tuple, list[PlayerNight]] = {}
    style_positions = []

    for horizon, part in frame.groupby(SNAPSHOT_COLUMN, sort=True):
        horizon = int(horizon)
        if horizon not in states:
            raise ValueError(f"Missing injury report state for horizon {horizon}")
        state = states[horizon]
        probabilities = player_out_probabilities(state.statuses)
        excluded = roster_exclusions(state.statuses)
        covered = {(str(game).zfill(10), str(team)) for game, team in state.covered}
        part = part.sort_values(["GAME_DATE", "GAME_ID"], kind="mergesort")
        lags = dict(zip(part["GAME_ID"], part["_HISTORY_LAG"], strict=True))
        nights = {}
        covered_ids = set()
        for game in part.itertuples(index=False):
            teams = (game.HOME_TEAM_ID, game.AWAY_TEAM_ID)
            if not all((game.GAME_ID, team) in covered for team in teams):
                continue
            covered_ids.add(game.GAME_ID)
            for team in teams:
                players = reference.get((game.GAME_ID, int(lags[game.GAME_ID]), team))
                if players is None:
                    continue
                rotation = [
                    replace(
                        player,
                        p_out=probabilities.get(
                            (game.GAME_ID, team, player.player_id), 0.0
                        ),
                    )
                    for player in players
                    if (game.GAME_ID, team, player.player_id) not in excluded
                ]
                # A completely excluded roster is unknown, not league average.
                if rotation:
                    nights[(game.GAME_ID, team)] = rotation
                    style_nights[(game.GAME_ID, horizon, team)] = rotation

        projected = project_lineup_games(
            part[game_columns].assign(GAME_DATE=part["_HISTORY_DATE"]),
            df_players,
            ratings,
            nights=nights,
        ).set_index("GAME_ID")
        projected = projected.loc[projected.index.isin(covered_ids)]
        # An empty projection is object-typed; the output columns are float.
        aligned = projected.reindex(part["GAME_ID"].to_numpy()).astype(float)
        aligned.index = part.index
        actual = (
            pd.to_numeric(part["TOTAL_POINTS"], errors="coerce")
            if "TOTAL_POINTS" in part
            else pd.Series(np.nan, index=part.index)
        )
        offset = walk_forward_offset(
            part["GAME_DATE"],
            aligned["raw_total"],
            actual,
            phases=game_phase(part["GAME_ID"]),
            cutoff_dates=part["_HISTORY_DATE"],
        )
        output.loc[part.index, "LU_PROJ_TOTAL_BEFORE"] = aligned["raw_total"] + offset
        for column, key in _IMPACT_COLUMNS.items():
            output.loc[part.index, column] = aligned[key]
        style_positions.extend(part.index[aligned["raw_total"].notna()].tolist())

    # Query all rotations in one chronological pass; fitting is shared across
    # horizons and all outputs retain both keys, even for repeated GAME_IDs.
    if stints is not None and style_positions:
        style = build_style_matchup_features(
            stints,
            frame.loc[style_positions, [*game_columns, SNAPSHOT_COLUMN]].assign(
                GAME_DATE=frame.loc[style_positions, "_HISTORY_DATE"]
            ),
            style_nights,
            snapshot_column=SNAPSHOT_COLUMN,
        ).set_index(KEYS)
        aligned_style = style.reindex(pd.MultiIndex.from_frame(frame[KEYS]))
        output.loc[:, list(STYLE_FEATURE_COLUMNS)] = aligned_style[
            list(STYLE_FEATURE_COLUMNS)
        ].to_numpy(float)

    output.index = inputs.index
    covered_count = output["LU_ABSENCE_IMPACT_PTS_BEFORE"].notna().sum()
    print(f"Lineup projection covers {covered_count:,} of {len(output):,} snapshots")
    return output
