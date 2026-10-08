"""Game graphs at the intermediate snapshots (T minutes before tip).

Availability is read at each snapshot's UTC cutoff; everything else (rosters,
recent minutes, ratings, guarding and overlap history, positions) is read as
of the snapshot's **history date**, exactly as the 2_6 intermediate layer does
(``lineups.intermediate_features``):

* history date = the earlier of the game date and the cutoff's Eastern date,
  where a cutoff before 05:00 ET belongs to the previous date (late games may
  still be running), via ``snapshot_history_dates``;
* reference rosters come from ``game_nights`` with the game dated on its
  history date; each snapshot then sets ``p_out`` from its own report state,
  drops roster exclusions (G League), and covers a game only when both teams
  had filed by the cutoff.

At T-0 to T-420 the history date is the game date for virtually every game;
at T-960 and T-1080 it is mostly the day before.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pandas as pd

from nba_ou.data_processing.lineups.availability import (
    player_out_probabilities,
    roster_exclusions,
)
from nba_ou.data_processing.lineups.features import game_nights
from nba_ou.data_processing.lineups.game_projection import PlayerNight
from nba_ou.data_processing.lineups.intermediate_features import (
    snapshot_history_dates,
)

from .as_of import PointInTimeData
from .game_graph import (
    DEFAULT_PARAMS,
    GameGraphParams,
    GameGraphs,
    build_snapshot_graphs,
)

SNAPSHOT_COLUMN = "TIME_TO_MATCH_MIN"


def snapshot_frame(games: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
    """One row per game and snapshot, with its history date.

    ``games``: ``GAME_ID``, ``GAME_DATE``, ``HOME_TEAM_ID``, ``AWAY_TEAM_ID``.
    ``cutoffs``: ``GAME_ID``, ``TIME_TO_MATCH_MIN``, ``SNAPSHOT_TS_UTC`` (the
    intermediate scoring sidecar).
    """
    games = games.assign(GAME_ID=games["GAME_ID"].astype(str).str.zfill(10))
    cutoffs = cutoffs.assign(GAME_ID=cutoffs["GAME_ID"].astype(str).str.zfill(10))
    frame = cutoffs[["GAME_ID", SNAPSHOT_COLUMN, "SNAPSHOT_TS_UTC"]].merge(
        games, on="GAME_ID", how="inner"
    )
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"]).dt.normalize()
    frame["AS_OF_DATE"] = snapshot_history_dates(frame)
    return frame.reset_index(drop=True)


def reference_nights(
    frame: pd.DataFrame, df_players: pd.DataFrame
) -> dict[tuple[str, pd.Timestamp, str], list[PlayerNight]]:
    """``(game_id, as_of_date, team_id) ->`` roster read before that date."""
    out = {}
    columns = ["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID"]
    for as_of_date, group in frame.groupby("AS_OF_DATE", sort=True):
        games = group[columns].assign(GAME_DATE=as_of_date).drop_duplicates("GAME_ID")
        out.update(
            {
                (game, as_of_date, team): players
                for (game, team), players in game_nights(games, df_players).items()
            }
        )
    return out


def snapshot_nights(
    part: pd.DataFrame,
    reference: Mapping[tuple[str, pd.Timestamp, str], list[PlayerNight]],
    state,
) -> tuple[dict[tuple[str, str], list[PlayerNight]], set[tuple[str, str]]]:
    """One snapshot's ``(nights, report_covered)`` from its report state."""
    probabilities = player_out_probabilities(state.statuses)
    excluded = roster_exclusions(state.statuses)
    covered = {(str(game).zfill(10), str(team)) for game, team in state.covered}
    nights = {}
    for game in part.itertuples(index=False):
        for team in (game.HOME_TEAM_ID, game.AWAY_TEAM_ID):
            players = reference.get((game.GAME_ID, game.AS_OF_DATE, team))
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
            if rotation:
                nights[(game.GAME_ID, team)] = rotation
    return nights, covered


def build_intermediate_graphs(
    data: PointInTimeData,
    frame: pd.DataFrame,
    states: Mapping[int, object],
    params: GameGraphParams = DEFAULT_PARAMS,
) -> dict[int, GameGraphs]:
    """Game graphs per snapshot horizon (minutes before tip).

    ``frame`` comes from :func:`snapshot_frame`; ``states`` maps each horizon
    to its ``InjuryReportState`` read at the snapshot cutoffs
    (``report_state.load_snapshot_report_states``). Snapshots sharing a
    history date share their expected lifts and guarding rates.
    """
    reference = reference_nights(frame, data.box_scores)
    out: dict[int, GameGraphs] = {}
    # Group by history date so each date's rates are computed once for every
    # horizon reading it.
    for _, by_date in frame.groupby("AS_OF_DATE", sort=True):
        snapshots = {}
        for horizon, part in by_date.groupby(SNAPSHOT_COLUMN, sort=True):
            horizon = int(horizon)
            if horizon not in states:
                raise ValueError(f"Missing injury report state for horizon {horizon}")
            nights, covered = snapshot_nights(part, reference, states[horizon])
            snapshots[horizon] = (nights, covered, set(part["GAME_ID"]))
        games = by_date.drop_duplicates("GAME_ID")[
            ["GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID", "AS_OF_DATE"]
        ]
        built = build_snapshot_graphs(data, games, snapshots, params)
        for horizon, graphs in built.items():
            out.setdefault(horizon, []).append(graphs)
    return {horizon: _concat(parts) for horizon, parts in out.items()}


def _concat(parts: list[GameGraphs]) -> GameGraphs:
    skipped = {"uncovered": 0, "no_roster": 0}
    for part in parts:
        for key, value in part.metadata["skipped_games"].items():
            skipped[key] += value
    return GameGraphs(
        scenarios=pd.concat([p.scenarios for p in parts], ignore_index=True),
        nodes=pd.concat([p.nodes for p in parts], ignore_index=True),
        edges=pd.concat([p.edges for p in parts], ignore_index=True),
        metadata={**parts[0].metadata, "skipped_games": skipped},
    )
