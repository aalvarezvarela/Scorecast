"""Schema 2_6: starter history and the rotation-based lineup projection.

Both families are per game and read nothing from the 2_5 feature columns, so the
layer needs only each game's date, teams and (for the lineup calibration) the
final total of *earlier* games:

* ``STARTER_*_BEFORE_TEAM_{HOME,AWAY}`` -- four prior-game starter-history
  features per side (``data_processing.players.starter_history``). Both
  datasets.
* ``LU_*_BEFORE`` -- the lineup projection built from stints, the walk-forward
  rating cache and the last injury report before tip
  (``data_processing.lineups.features``). Closing dataset only: an intermediate
  snapshot needs availability as of the snapshot, not the closing report, and
  that is a later version's job.
"""

from __future__ import annotations

import pandas as pd

from nba_ou.data_processing.lineups.features import LINEUP_FEATURE_COLUMNS
from nba_ou.data_processing.players.starter_history import (
    STARTER_HISTORY_COLUMNS,
    add_starter_history_features,
)

from .base import (
    CLOSING_LINE,
    GAME_ID_COLUMN,
    INTERMEDIATE_LINE,
    LayerContext,
    SchemaLayer,
    game_id_keys,
    team_id_keys,
)

_SIDES = ("_TEAM_HOME", "_TEAM_AWAY")
_HOME, _AWAY = "TEAM_ID_TEAM_HOME", "TEAM_ID_TEAM_AWAY"

STARTER_COLUMNS = tuple(
    f"{column}{side}" for side in _SIDES for column in STARTER_HISTORY_COLUMNS
)


def _games(inputs: pd.DataFrame) -> pd.DataFrame:
    """One row per game in (date, GAME_ID) order, keys normalized."""
    games = pd.DataFrame(
        {
            GAME_ID_COLUMN: game_id_keys(inputs[GAME_ID_COLUMN]),
            "GAME_DATE": pd.to_datetime(inputs["GAME_DATE"]).dt.normalize(),
            _HOME: team_id_keys(inputs[_HOME]),
            _AWAY: team_id_keys(inputs[_AWAY]),
            "TOTAL_POINTS": (
                pd.to_numeric(inputs["TOTAL_POINTS"], errors="coerce")
                if "TOTAL_POINTS" in inputs.columns
                else float("nan")
            ),
        }
    )
    # Sorted so the lineup calibration does not depend on how the parent file
    # happened to order its rows: its trailing window is cut by position.
    return (
        games.drop_duplicates(GAME_ID_COLUMN)
        .sort_values(["GAME_DATE", GAME_ID_COLUMN], kind="mergesort")
        .reset_index(drop=True)
    )


def _starter_history(games: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """``STARTER_COLUMNS`` per GAME_ID."""
    team_games = pd.concat(
        [
            games[[GAME_ID_COLUMN, "GAME_DATE"]].assign(TEAM_ID=games[team], _SIDE=side)
            for team, side in ((_HOME, _SIDES[0]), (_AWAY, _SIDES[1]))
        ],
        ignore_index=True,
    )
    history = add_starter_history_features(team_games, players)
    wide = history.pivot(
        index=GAME_ID_COLUMN, columns="_SIDE", values=list(STARTER_HISTORY_COLUMNS)
    )
    wide.columns = [f"{column}{side}" for column, side in wide.columns]
    return wide.reindex(columns=list(STARTER_COLUMNS)).astype(float)


def _lineup(
    games: pd.DataFrame, players: pd.DataFrame, ctx: LayerContext
) -> pd.DataFrame:
    """``LINEUP_FEATURE_COLUMNS`` per GAME_ID."""
    from nba_ou.data_processing.lineups.features import (
        DATA_ROOT,
        attach_lineup_features,
        load_rating_book,
    )
    from nba_ou.data_processing.lineups.stint_store import read_stints

    state = ctx.injury_report()
    projected = attach_lineup_features(
        games,
        players,
        enabled=True,
        injury_statuses=state.statuses,
        report_covered=state.covered,
        ratings=ctx.get_or_load("lineup_rating_book", load_rating_book),
        stints=ctx.get_or_load(
            "lineup_stints", lambda: read_stints(None, local_root=DATA_ROOT)
        ),
    )
    return projected.set_index(GAME_ID_COLUMN)[list(LINEUP_FEATURE_COLUMNS)]


def build_2_6(
    inputs: pd.DataFrame, ctx: LayerContext, dataset_type: str
) -> pd.DataFrame:
    games = _games(inputs)
    players = ctx.players(games["GAME_DATE"])
    per_game = _starter_history(games, players)
    if dataset_type == CLOSING_LINE:
        per_game = per_game.join(_lineup(games, players, ctx))
    # Back onto the parent's rows: one per game (closing) or per snapshot.
    keys = game_id_keys(inputs[GAME_ID_COLUMN])
    out = per_game.reindex(keys.to_numpy())
    out.index = inputs.index
    return out


LAYER_2_6 = SchemaLayer(
    version="2_6",
    parent="2_5",
    summary=(
        "Prior-game starter history (both datasets) and the rotation-based "
        "lineup projection LU_* (closing only)."
    ),
    columns={
        CLOSING_LINE: STARTER_COLUMNS + tuple(LINEUP_FEATURE_COLUMNS),
        INTERMEDIATE_LINE: STARTER_COLUMNS,
    },
    requires=("GAME_DATE", _HOME, _AWAY),
    optional=("TOTAL_POINTS",),
    build=build_2_6,
)
