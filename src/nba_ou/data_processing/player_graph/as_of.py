"""Point-in-time access to the data the player graph reads (plan phase 0).

Every 2_7 component reads stints, matchups and box scores through
:meth:`PointInTimeData.as_of`, so nothing dated on or after the cutoff can be
read by accident. The 2_6 modules each implement the same cutoff themselves and
are left as they are: their columns are frozen by the layer rule.

Two kinds of cutoff:

* **Game data** (stints, matchups, box scores) is cut by game date: a view as of
  ``D`` holds only games with ``game_date < D``. A game finished earlier on the
  same day is excluded too, as in 2_6.
* **Injury reports** are cut by timestamp, upstream in SQL
  (``postgre_db.injury_report_aiven.fetch``): each ``InjuryReportState`` already
  holds, per game, the last report strictly before its tip (closing) or its
  snapshot cutoff. They are kept here per horizon, not re-cut.

Everything is loaded once and sorted by date; a view slices with
``searchsorted`` and returns copies, so a consumer that edits a frame cannot
change what the next view sees.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Key of the closing-line injury state in :attr:`PointInTimeData.injuries`;
#: intermediate states are keyed by their horizon in minutes.
CLOSING = "closing"

STINT_DATE = "game_date"
MATCHUP_DATE = "game_date"
BOX_SCORE_DATE = "GAME_DATE"


def _game_ids(values: pd.Series) -> pd.Series:
    return values.astype(str).str.zfill(10)


def _with_columns(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """Give an empty, columnless frame (``read_stints`` with nothing validated)
    the key columns, so it slices like any other."""
    if not frame.empty:
        return frame
    frame = frame.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype=object)
    return frame


def _freeze_sequences(frame: pd.DataFrame) -> pd.DataFrame:
    """Arrays inside object columns become tuples.

    ``DataFrame.copy`` does not copy the objects in a cell, so a view holding
    the stints' ``home_lineup`` arrays could edit the store in place.
    """
    for column in frame.columns[frame.dtypes.eq(object)]:
        values = frame[column].dropna()
        if len(values) and isinstance(values.iloc[0], (np.ndarray, list)):
            frame[column] = frame[column].map(
                lambda value: (
                    tuple(value) if isinstance(value, (np.ndarray, list)) else value
                )
            )
    return frame


def _sorted_by_date(frame: pd.DataFrame, date_col: str, id_col: str) -> pd.DataFrame:
    frame = _with_columns(frame, id_col, date_col).copy()
    frame[id_col] = _game_ids(frame[id_col])
    frame[date_col] = pd.to_datetime(frame[date_col]).dt.normalize()
    if frame[date_col].isna().any():
        raise ValueError(f"Rows without {date_col} cannot be placed in time")
    frame = frame.sort_values([date_col, id_col], kind="mergesort", ignore_index=True)
    return _freeze_sequences(frame)


def _attach_matchup_dates(
    matchups: pd.DataFrame, game_dates: pd.Series
) -> pd.DataFrame:
    """Matchup rows carry only ``game_id``; date them from ``game_dates``."""
    matchups = _with_columns(matchups, "game_id").copy()
    matchups["game_id"] = _game_ids(matchups["game_id"])
    matchups[MATCHUP_DATE] = matchups["game_id"].map(game_dates)
    undated = matchups.loc[matchups[MATCHUP_DATE].isna(), "game_id"].unique()
    if len(undated):
        raise ValueError(
            f"{len(undated)} matchup games have no known date, e.g. {sorted(undated)[:3]}"
        )
    return matchups


def game_date_index(*frames: tuple[pd.DataFrame, str, str]) -> pd.Series:
    """``game_id -> game_date`` from ``(frame, id_col, date_col)`` sources.

    Raises if two sources disagree on a game's date.
    """
    parts = [
        pd.DataFrame(
            {
                "game_id": _game_ids(frame[id_col]),
                "game_date": pd.to_datetime(frame[date_col]).dt.normalize(),
            }
        ).drop_duplicates()
        for frame, id_col, date_col in frames
        if not frame.empty
    ]
    if not parts:
        return pd.Series(dtype="datetime64[ns]")
    dates = pd.concat(parts, ignore_index=True).drop_duplicates()
    conflicts = dates.loc[dates["game_id"].duplicated(), "game_id"]
    if len(conflicts):
        raise ValueError(f"Conflicting dates for games {sorted(conflicts)[:3]}")
    return dates.set_index("game_id")["game_date"]


@dataclass(frozen=True)
class AsOfView:
    """Game data strictly before :attr:`cutoff`."""

    data: PointInTimeData
    cutoff: pd.Timestamp

    def _slice(
        self, frame: pd.DataFrame, date_col: str, since: Any | None
    ) -> pd.DataFrame:
        dates = frame[date_col].to_numpy()
        stop = np.searchsorted(dates, self.cutoff.to_datetime64(), side="left")
        start = 0
        if since is not None:
            start = np.searchsorted(
                dates, pd.Timestamp(since).normalize().to_datetime64(), side="left"
            )
        return frame.iloc[start : max(start, stop)].copy()

    def stints(self, since: Any | None = None) -> pd.DataFrame:
        """Validated stints of games dated in ``[since, cutoff)``."""
        return self._slice(self.data.stints, STINT_DATE, since)

    def matchups(self, since: Any | None = None) -> pd.DataFrame:
        """Who-guarded-whom rows of games dated in ``[since, cutoff)``."""
        return self._slice(self.data.matchups, MATCHUP_DATE, since)

    def box_scores(self, since: Any | None = None) -> pd.DataFrame:
        """Player box-score rows of games dated in ``[since, cutoff)``."""
        return self._slice(self.data.box_scores, BOX_SCORE_DATE, since)


@dataclass(frozen=True)
class PointInTimeData:
    """Stints, matchups, box scores and injury states, sorted by game date.

    Build it with :meth:`from_frames` (tests, already-loaded frames) or
    :meth:`load` (local stores plus the database).
    """

    stints: pd.DataFrame
    matchups: pd.DataFrame
    box_scores: pd.DataFrame
    injuries: Mapping[str | int, Any] = field(default_factory=dict)

    @classmethod
    def from_frames(
        cls,
        stints: pd.DataFrame,
        matchups: pd.DataFrame,
        box_scores: pd.DataFrame,
        injuries: Mapping[str | int, Any] | None = None,
        game_dates: pd.DataFrame | None = None,
    ) -> PointInTimeData:
        """Normalize ids and dates and sort.

        Matchups are dated from ``game_dates`` (``GAME_ID``, ``GAME_DATE``: the
        games table), the box scores and the stints. The games table matters:
        player box scores start in 2018-19, matchups in 2017-18.
        """
        stints = _sorted_by_date(stints, STINT_DATE, "game_id")
        box_scores = _sorted_by_date(box_scores, BOX_SCORE_DATE, "GAME_ID")
        if matchups.empty or MATCHUP_DATE not in matchups.columns:
            sources = [
                (box_scores, "GAME_ID", BOX_SCORE_DATE),
                (stints, "game_id", STINT_DATE),
            ]
            if game_dates is not None:
                sources.append((game_dates, "GAME_ID", "GAME_DATE"))
            matchups = _attach_matchup_dates(matchups, game_date_index(*sources))
        matchups = _sorted_by_date(matchups, MATCHUP_DATE, "game_id")
        return cls(stints, matchups, box_scores, dict(injuries or {}))

    @classmethod
    def load(
        cls,
        season_years: Iterable[int],
        *,
        local_root: Path = Path("data"),
        injuries: Mapping[str | int, Any] | None = None,
        closing_injuries: bool = False,
    ) -> PointInTimeData:
        """Read the local stint and matchup stores and the box scores from the DB.

        ``season_years`` are start years (2019 = 2019-20). Box scores also cover
        the season before the first, as player context, like the 2_6 layer.
        ``closing_injuries`` reads the closing ``InjuryReportState`` from the
        database; intermediate states need snapshot cutoffs and are passed in
        through ``injuries`` (``load_snapshot_report_states``).
        """
        from nba_ou.create_training_data.schema_layers.inputs import (
            load_player_history,
        )
        from nba_ou.data_processing.lineups.stint_store import read_stints
        from nba_ou.fetch_data.nba_lineups.matchups import MatchupStore
        from nba_ou.postgre_db import load_games_from_db

        seasons = sorted(set(season_years))
        stints = read_stints(seasons, local_root=local_root)
        store = MatchupStore(local_root)
        held = [store.read(season) for season in seasons]
        held = [frame for frame in held if not frame.empty] or held[:1]
        matchups = pd.concat(held, ignore_index=True)
        games = load_games_from_db(
            seasons=[f"{year}-{(year + 1) % 100:02d}" for year in seasons]
        )
        if games is None or games.empty:
            raise RuntimeError(f"No games in the games table for seasons {seasons}")
        games.columns = games.columns.str.upper()
        # The last real game, not a calendar date: the 2019-20 Finals ran into
        # October 2020.
        span = pd.Series(
            [pd.Timestamp(seasons[0], 10, 1), pd.to_datetime(games["GAME_DATE"]).max()]
        )
        box_scores = load_player_history(span)
        states = dict(injuries or {})
        if closing_injuries:
            from nba_ou.data_processing.injury_status.report_state import (
                load_injury_report_state,
            )

            states[CLOSING] = load_injury_report_state()
        return cls.from_frames(stints, matchups, box_scores, states, games)

    def as_of(self, date: Any) -> AsOfView:
        """Game data dated strictly before ``date`` (time of day is ignored)."""
        return AsOfView(self, pd.Timestamp(date).normalize())

    def injury_report(self, horizon: str | int = CLOSING) -> Any:
        """The ``InjuryReportState`` at ``horizon``: ``CLOSING`` or minutes."""
        try:
            return self.injuries[horizon]
        except KeyError:
            loaded = sorted(map(str, self.injuries))
            raise KeyError(
                f"No injury state for horizon {horizon!r}; loaded: {loaded}"
            ) from None
