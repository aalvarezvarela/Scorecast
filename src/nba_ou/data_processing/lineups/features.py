"""Phase G: the lineup projection as ``_BEFORE`` features for the closing dataset.

Everything here composes the phase-F pieces that already exist -- rosters and
availability from ``availability``, the projection and its full-health
counterfactual from ``game_projection``, ratings from the walk-forward cache --
into one row per game. The evaluation script
(``scripts/lineups/evaluate_game_projection.py``) calls the same
:func:`project_lineup_games`, so the numbers the plan reports and the columns
the model sees cannot drift apart.

**The column set is deliberately small** (``docs/lineup_projection_plan.md``
section 8.1: "add more only on evidence"). The only column with a measured,
non-null relation to ``LINE_ERROR`` is the absence counterfactual, positive in
all five seasons 2021-2025 and stronger after controlling for the existing
injury columns. Synergy and ``proj_total - line`` measured null and are left
out; the level, possessions and scenario spread come along because they are
free once the projection has run and the model may use them as context.

**Temporal contract**, inherited from the pieces and checked by
``tests/test_lineup_leakage.py``:

- rosters and minutes come from box scores on strictly earlier dates;
- availability is the last injury report before tip-off;
- ratings are read from the latest cache date **on or before** the game date,
  and every cache row was fitted on games strictly before its own date;
- the level calibration uses outcomes of games on strictly earlier dates.

**Coverage.** Ratings start in 2021-22, so earlier games get NaN in every
column. That step makes the family season-gated for any training window that
starts earlier, and ``find_season_gated_columns`` will drop it there; evaluate
it on a 2021-22+ window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .availability import RECENT_GAMES, build_player_nights
from .game_projection import TeamRatings, project_with_counterfactual

#: Where ``scripts/lineups/build_player_ratings.py`` writes the cache by default.
DEFAULT_RATING_CACHE = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "lineup_ratings"
    / "player_ratings.parquet"
)

LINEUP_FEATURE_COLUMNS = (
    "LU_PROJ_TOTAL_BEFORE",
    "LU_PROJ_POSS_BEFORE",
    "LU_PROJ_TOTAL_SD_BEFORE",
    "LU_ABSENCE_IMPACT_PTS_BEFORE",
    "LU_ABSENCE_IMPACT_PACE_BEFORE",
)

#: Prior games behind the walk-forward level calibration (``total_offset``).
#:
#: Swept 2026-09-23 over {100, 200, 300, 500, 800} games and against day-based
#: windows: MAE is flat (14.66-14.72) and 200 games has the smallest bias
#: (-0.04). Counted in games rather than days so a season opener still has a
#: full window, and bounded rather than expanding so a training build, which
#: loads many seasons, and same-day serving, which loads two, compute the same
#: offset.
OFFSET_WINDOW_GAMES = 200

#: The oldest a rating date may be and still be used for a game. The cache is
#: refitted on every game date, so a larger gap means the cache was not
#: refreshed, and projecting from months-old ratings would be silently wrong.
RATING_MAX_STALENESS_DAYS = 14


class RatingBook:
    """Walk-forward ratings, looked up by game date.

    A game on date D reads the latest cache date on or before D. Each cache row
    was fitted only on games strictly before its own date, so the lookup can
    never see date D itself.
    """

    def __init__(
        self,
        ratings: pd.DataFrame,
        max_staleness_days: int = RATING_MAX_STALENESS_DAYS,
    ) -> None:
        required = {
            "as_of_date",
            "player_id",
            "o_rating",
            "d_rating",
            "pace_rating",
            "league_ortg",
            "league_pace",
            "fit_max_game_date",
        }
        if missing := required - set(ratings.columns):
            raise ValueError(f"Ratings are missing {sorted(missing)}")
        frame = ratings.copy()
        frame["as_of_date"] = pd.to_datetime(frame["as_of_date"]).dt.normalize()
        frame["fit_max_game_date"] = pd.to_datetime(frame["fit_max_game_date"])
        if not frame["fit_max_game_date"].lt(frame["as_of_date"]).all():
            raise ValueError("Every rating fit must end strictly before its as-of date")
        frame["player_id"] = frame["player_id"].astype(str)
        self.max_staleness = pd.Timedelta(days=max_staleness_days)
        self._dates = np.array(
            sorted(frame["as_of_date"].unique()), dtype="datetime64[ns]"
        )
        self._frames = dict(tuple(frame.groupby("as_of_date", sort=True)))
        self._cache: dict[pd.Timestamp, tuple[TeamRatings, float, float]] = {}

    def as_of(self, date: pd.Timestamp) -> pd.Timestamp | None:
        """The cache date a game on ``date`` reads, or None."""
        date = pd.Timestamp(date).normalize()
        position = np.searchsorted(self._dates, np.datetime64(date), side="right") - 1
        if position < 0:
            return None
        chosen = pd.Timestamp(self._dates[position])
        if date - chosen > self.max_staleness:
            return None
        return chosen

    def for_date(self, date: pd.Timestamp) -> tuple[TeamRatings, float, float] | None:
        """``(ratings, league_ortg, league_pace)`` for a game on ``date``."""
        chosen = self.as_of(date)
        if chosen is None:
            return None
        if chosen not in self._cache:
            frame = self._frames[chosen]
            self._cache[chosen] = (
                TeamRatings(
                    dict(zip(frame["player_id"], frame["o_rating"], strict=True)),
                    dict(zip(frame["player_id"], frame["d_rating"], strict=True)),
                    dict(zip(frame["player_id"], frame["pace_rating"], strict=True)),
                ),
                float(frame["league_ortg"].iloc[0]),
                float(frame["league_pace"].iloc[0]),
            )
        return self._cache[chosen]


def project_lineup_games(
    games: pd.DataFrame,
    df_players: pd.DataFrame,
    ratings: RatingBook,
    p_out: dict[tuple[str, str, str], float] | None = None,
    recent_games: int = RECENT_GAMES,
    roster_games: int | None = None,
) -> pd.DataFrame:
    """Uncalibrated projection and absence counterfactual, one row per game.

    ``games`` has ``GAME_ID``, ``GAME_DATE``, ``HOME_TEAM_ID`` and
    ``AWAY_TEAM_ID``. ``df_players`` is box-score history with ``GAME_ID``,
    ``TEAM_ID``, ``PLAYER_ID``, ``GAME_DATE`` and numeric ``MIN``; rows for the
    target games themselves may be present and are never read for them.

    Returns ``GAME_ID``, ``raw_total`` (before the level calibration),
    ``healthy_total``, ``possessions``, ``total_sd``, ``impact_points`` and
    ``impact_possessions``. Games without ratings or without a roster on either
    side are absent from the result.
    """
    required = {"GAME_ID", "GAME_DATE", "HOME_TEAM_ID", "AWAY_TEAM_ID"}
    if missing := required - set(games.columns):
        raise ValueError(f"Games are missing {sorted(missing)}")
    targets = games[list(required)].copy()
    targets["GAME_ID"] = targets["GAME_ID"].astype(str).str.zfill(10)
    for column in ("HOME_TEAM_ID", "AWAY_TEAM_ID"):
        targets[column] = targets[column].astype(str)
    targets["GAME_DATE"] = pd.to_datetime(targets["GAME_DATE"]).dt.normalize()
    targets = targets.drop_duplicates("GAME_ID")

    team_rows = pd.concat(
        [
            targets[["GAME_ID", "GAME_DATE", side]].rename(columns={side: "TEAM_ID"})
            for side in ("HOME_TEAM_ID", "AWAY_TEAM_ID")
        ],
        ignore_index=True,
    )
    nights = build_player_nights(
        team_rows,
        df_players,
        p_out,
        recent_games=recent_games,
        roster_games=roster_games,
    )

    rows = []
    for game in targets.itertuples(index=False):
        home = nights.get((game.GAME_ID, game.HOME_TEAM_ID))
        away = nights.get((game.GAME_ID, game.AWAY_TEAM_ID))
        rated = ratings.for_date(game.GAME_DATE)
        if home is None or away is None or rated is None:
            continue
        team_ratings, league_ortg, league_pace = rated
        result = project_with_counterfactual(
            home, away, team_ratings, team_ratings, league_ortg, league_pace
        )
        if "absence_impact_points" not in result:
            continue
        rows.append(
            {
                "GAME_ID": game.GAME_ID,
                "raw_total": result["total"],
                "healthy_total": result["healthy_total"],
                "possessions": result["possessions"],
                "total_sd": result["total_sd"],
                "impact_points": result["absence_impact_points"],
                "impact_possessions": result["absence_impact_possessions"],
            }
        )
    columns = [
        "GAME_ID",
        "raw_total",
        "healthy_total",
        "possessions",
        "total_sd",
        "impact_points",
        "impact_possessions",
    ]
    return pd.DataFrame(rows, columns=columns)


def walk_forward_offset(
    game_dates: pd.Series,
    raw_totals: pd.Series,
    actual_totals: pd.Series,
    window: int = OFFSET_WINDOW_GAMES,
) -> pd.Series:
    """The level calibration for each row, from strictly earlier dates only.

    Mean of ``actual - raw`` over the last ``window`` games with a known result
    on dates before the row's own; games sharing a date are held back together.
    See ``game_projection.project_totals`` for why the offset exists at all.
    """
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(game_dates).dt.normalize().to_numpy(),
            "residual": (
                pd.to_numeric(actual_totals, errors="coerce")
                - pd.to_numeric(raw_totals, errors="coerce")
            ).to_numpy(),
        },
        index=game_dates.index,
    )
    known = frame.dropna(subset=["residual"]).sort_values("date", kind="mergesort")
    known_dates = known["date"].to_numpy()
    known_residuals = known["residual"].to_numpy()
    offsets = {}
    for date in frame["date"].dropna().unique():
        end = np.searchsorted(known_dates, np.datetime64(date), side="left")
        start = max(0, end - window)
        offsets[pd.Timestamp(date)] = (
            known_residuals[start:end].mean() if end > start else np.nan
        )
    return frame["date"].map(offsets).astype(float)


def add_lineup_features(
    df_merged: pd.DataFrame,
    df_players: pd.DataFrame,
    ratings: RatingBook,
    p_out: dict[tuple[str, str, str], float] | None = None,
    recent_games: int = RECENT_GAMES,
) -> pd.DataFrame:
    """Attach ``LINEUP_FEATURE_COLUMNS`` to the merged one-row-per-game frame.

    ``df_merged`` is the output of ``merge_home_away_data``: it carries
    ``GAME_ID``, ``GAME_DATE``, ``TEAM_ID_TEAM_HOME``, ``TEAM_ID_TEAM_AWAY`` and,
    for finished games, ``TOTAL_POINTS``. The outcome is read only for the
    calibration of *later* dates (:func:`walk_forward_offset`).

    Games the projection cannot cover -- before the rating cache starts, or
    with no roster history -- get NaN, which is the honest value: there is no
    projection, not a neutral one.
    """
    required = {"GAME_ID", "GAME_DATE", "TEAM_ID_TEAM_HOME", "TEAM_ID_TEAM_AWAY"}
    if missing := required - set(df_merged.columns):
        raise ValueError(f"Merged games are missing {sorted(missing)}")
    games = pd.DataFrame(
        {
            "GAME_ID": df_merged["GAME_ID"].astype(str).str.zfill(10),
            "GAME_DATE": pd.to_datetime(df_merged["GAME_DATE"]).dt.normalize(),
            "HOME_TEAM_ID": df_merged["TEAM_ID_TEAM_HOME"].astype(str),
            "AWAY_TEAM_ID": df_merged["TEAM_ID_TEAM_AWAY"].astype(str),
        },
        index=df_merged.index,
    )
    projected = project_lineup_games(
        games, df_players, ratings, p_out, recent_games=recent_games
    ).set_index("GAME_ID")

    aligned = projected.reindex(games["GAME_ID"])
    aligned.index = df_merged.index
    actual = (
        df_merged["TOTAL_POINTS"]
        if "TOTAL_POINTS" in df_merged.columns
        else pd.Series(np.nan, index=df_merged.index)
    )
    offset = walk_forward_offset(games["GAME_DATE"], aligned["raw_total"], actual)

    result = df_merged.drop(
        columns=[c for c in LINEUP_FEATURE_COLUMNS if c in df_merged.columns]
    )
    features = pd.DataFrame(
        {
            "LU_PROJ_TOTAL_BEFORE": aligned["raw_total"] + offset,
            "LU_PROJ_POSS_BEFORE": aligned["possessions"],
            "LU_PROJ_TOTAL_SD_BEFORE": aligned["total_sd"],
            "LU_ABSENCE_IMPACT_PTS_BEFORE": aligned["impact_points"],
            "LU_ABSENCE_IMPACT_PACE_BEFORE": aligned["impact_possessions"],
        },
        index=df_merged.index,
    )
    covered = features["LU_ABSENCE_IMPACT_PTS_BEFORE"].notna().sum()
    print(f"Lineup projection covers {covered:,} of {len(features):,} games")
    return pd.concat([result, features], axis=1)


def attach_lineup_features(
    df_merged: pd.DataFrame,
    df_players: pd.DataFrame,
    *,
    enabled: bool,
    injury_statuses: pd.DataFrame | None,
    ratings: RatingBook | None = None,
    rating_cache: Path | str | None = None,
) -> pd.DataFrame:
    """The ``lineup_features`` switch, as the dataset builder calls it.

    Off, the frame is returned **unchanged** -- the same object -- so the
    control arm of the phase-G campaign is byte-for-byte today's dataset. On,
    exactly ``LINEUP_FEATURE_COLUMNS`` are added and nothing else changes.

    ``injury_statuses`` is ``InjuryReportState.statuses``, the last report
    before each tip: the same source the report-derived injury columns read, so
    availability is scored identically in both families.
    """
    if not enabled:
        return df_merged
    if injury_statuses is None:
        raise ValueError("lineup_features needs the pre-game injury report statuses")
    from .availability import player_out_probabilities

    book = ratings if ratings is not None else load_rating_book(rating_cache)
    return add_lineup_features(
        df_merged, df_players, book, player_out_probabilities(injury_statuses)
    )


def load_rating_book(path: Path | str | None = None) -> RatingBook:
    """Read the walk-forward rating cache written by ``build_player_ratings.py``."""
    source = Path(path) if path is not None else DEFAULT_RATING_CACHE
    if not source.exists():
        raise FileNotFoundError(
            f"No lineup rating cache at {source}. Build it with "
            "scripts/lineups/build_player_ratings.py."
        )
    return RatingBook(pd.read_parquet(source))
