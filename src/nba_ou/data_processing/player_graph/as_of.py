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

Box scores come from the database, which keeps them from 2018-19 on. Earlier
2_7 seasons (2016-17, 2017-18) are read from the per-season CSVs the database
was loaded from (identical on 2018-19, the season both hold), and every row
says where it came from in ``BOX_SOURCE``. The game-graph rosters read only
the database rows (:attr:`PointInTimeData.box_scores_2_6`), as 2_6 does, so
v0 keeps reproducing it.
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
PAIR_GAME_DATE = "game_date"
MATCHUP_DATE = "game_date"
BOX_SCORE_DATE = "GAME_DATE"

#: First season (start year) 2_7 reads. Stints exist from 2012-13, but the
#: seasons before 2016-17 have no matchup tracking and are a future extension
#: (``docs/player_graph/frozen_decisions_and_future_checks.md``).
FIRST_SEASON = 2016

#: Where a box-score row came from: ``"db"`` or ``"csv"``.
BOX_SOURCE = "BOX_SOURCE"
DEFAULT_BOX_CSV_DIR = Path("data/season_games_data")


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


def season_of(game_ids: pd.Series) -> pd.Series:
    """Start year encoded in the game id (``0021700001`` -> 2017)."""
    return 2000 + _game_ids(game_ids).str[3:5].astype(int)


def local_box_scores(
    season_years: Iterable[int],
    games: pd.DataFrame,
    data_dir: Path = DEFAULT_BOX_CSV_DIR,
) -> pd.DataFrame:
    """Box scores of ``season_years`` from the per-season CSVs.

    Cleaned like the database rows (``clear_player_statistics``): dated from
    ``games`` (``GAME_ID``, ``GAME_DATE``: the games table), rows of games it
    does not hold dropped, minutes parsed. The CSVs call turnovers ``TO``.
    """
    from nba_ou.data_processing.players.attach_player_features import (
        clear_player_statistics,
    )

    ids = {column: str for column in ("GAME_ID", "TEAM_ID", "PLAYER_ID", "MIN")}
    frames = []
    for year in sorted(set(season_years)):
        path = data_dir / f"nba_players_{year}_{(year + 1) % 100:02d}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No box-score CSV for season {year}: {path}")
        frames.append(pd.read_csv(path, dtype=ids))
    if not frames:
        return pd.DataFrame(columns=["GAME_ID", BOX_SCORE_DATE])
    box = pd.concat(frames, ignore_index=True).rename(columns={"TO": "TOV"})
    box["GAME_ID"] = _game_ids(box["GAME_ID"])
    games = games.assign(
        GAME_ID=_game_ids(games["GAME_ID"]),
        GAME_DATE=pd.to_datetime(games["GAME_DATE"]),
    )
    return clear_player_statistics(box, games)


def with_local_box_scores(
    box_scores: pd.DataFrame,
    season_years: Iterable[int],
    games: pd.DataFrame,
    data_dir: Path = DEFAULT_BOX_CSV_DIR,
) -> pd.DataFrame:
    """Database box scores plus the CSV rows of the seasons it lacks entirely.

    Only whole seasons fall back: a season the database holds is never mixed
    with CSV rows. Every row gets ``BOX_SOURCE``.
    """
    box_scores = box_scores.assign(**{BOX_SOURCE: "db"})
    held = set(season_of(box_scores["GAME_ID"])) if len(box_scores) else set()
    missing = [year for year in sorted(set(season_years)) if year not in held]
    if not missing:
        return box_scores
    local = local_box_scores(missing, games, data_dir).assign(**{BOX_SOURCE: "csv"})
    parts = [frame for frame in (box_scores, local) if not frame.empty]
    return pd.concat(parts or [box_scores], ignore_index=True)


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

    def pair_game(self, since: Any | None = None) -> pd.DataFrame:
        """Observed ``pair_game`` rows of games dated in ``[since, cutoff)``."""
        return self._slice(self.data.pair_game, PAIR_GAME_DATE, since)

    def overlap_game(self, since: Any | None = None) -> pd.DataFrame:
        """Per-game shared floor time of player pairs, ``[since, cutoff)``."""
        return self._slice(self.data.overlap_game, PAIR_GAME_DATE, since)


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
    pair_game: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["game_id", PAIR_GAME_DATE])
    )
    overlap_game: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["game_id", PAIR_GAME_DATE])
    )

    @classmethod
    def from_frames(
        cls,
        stints: pd.DataFrame,
        matchups: pd.DataFrame,
        box_scores: pd.DataFrame,
        injuries: Mapping[str | int, Any] | None = None,
        game_dates: pd.DataFrame | None = None,
        pair_game: pd.DataFrame | None = None,
        overlap_game: pd.DataFrame | None = None,
    ) -> PointInTimeData:
        """Normalize ids and dates and sort.

        Matchups are dated from ``game_dates`` (``GAME_ID``, ``GAME_DATE``: the
        games table), the box scores and the stints. The games table matters:
        the database keeps box scores from 2018-19 on, matchups start in 2017-18.
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
        pairs = _sorted_by_date(
            pair_game if pair_game is not None else pd.DataFrame(),
            PAIR_GAME_DATE,
            "game_id",
        )
        overlaps = _sorted_by_date(
            overlap_game if overlap_game is not None else pd.DataFrame(),
            PAIR_GAME_DATE,
            "game_id",
        )
        return cls(stints, matchups, box_scores, dict(injuries or {}), pairs, overlaps)

    @classmethod
    def load(
        cls,
        season_years: Iterable[int],
        *,
        local_root: Path = Path("data"),
        injuries: Mapping[str | int, Any] | None = None,
        closing_injuries: bool = False,
    ) -> PointInTimeData:
        """Read the local stint, matchup and ``pair_game`` stores and the DB.

        ``season_years`` are start years (2019 = 2019-20), none before
        :data:`FIRST_SEASON`. Box scores also cover the season before the first
        when the database has it, as player context, like the 2_6 layer; a
        requested season the database lacks is read from the CSVs
        (:func:`with_local_box_scores`).
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

        from .overlap import build_overlap_game
        from .pair_game import read_pair_game

        seasons = sorted(set(season_years))
        if seasons and seasons[0] < FIRST_SEASON:
            raise ValueError(
                f"2_7 starts in {FIRST_SEASON}; asked for {seasons[0]} (the earlier "
                "stints are a future extension, see the frozen decisions doc)"
            )
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
        box_scores = with_local_box_scores(
            load_player_history(span),
            seasons,
            games,
            data_dir=local_root / "season_games_data",
        )
        states = dict(injuries or {})
        if closing_injuries:
            from nba_ou.data_processing.injury_status.report_state import (
                load_injury_report_state,
            )

            states[CLOSING] = load_injury_report_state()
        pairs = read_pair_game(seasons, local_root=local_root)
        overlaps = pd.concat(
            [
                build_overlap_game(stints.loc[stints["season_year"].eq(season)])
                for season in (seasons if not stints.empty else [])
            ]
            or [pd.DataFrame()],
            ignore_index=True,
        )
        return cls.from_frames(
            stints, matchups, box_scores, states, games, pairs, overlaps
        )

    @property
    def box_scores_2_6(self) -> pd.DataFrame:
        """The box scores 2_6 reads: database rows only.

        Game-graph rosters and recent minutes use these, so v0 reproduces 2_6;
        2_7's own history (positions, node profiles, usage) reads all rows.
        """
        if BOX_SOURCE not in self.box_scores.columns:
            return self.box_scores
        return self.box_scores.loc[self.box_scores[BOX_SOURCE].ne("csv")]

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
