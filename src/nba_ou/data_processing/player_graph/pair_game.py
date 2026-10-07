"""Observed attacker-defender pairs per game: the ``pair_game`` table.

One row per game x attacker x defender, for every pair that shared the floor
in a validated stint, plus the few matchup rows whose pair never did. It is
**what happened** in the game, so it is a label and history table only: the
encoder never reads a game's own row as input (plan principle 7). Expected
shares for a game are built from earlier games' rows (``expected_guard``).

Guarding rate::

    guard_rate = matchup_seconds / cofloor_seconds

``cofloor_seconds`` comes from our stints, not from the NBA's
``pct_total_time_both_on``, whose denominator is about 0.39 x the co-floor
time (offensive tracked time, measured 2023-24) and is rounded to 3 decimals.
The raw parts are kept so the definition can change later.

Data quirks, measured 2017-18 to 2025-26:

* Shared the floor but no matchup row (~6.5% of directed pairs): a real zero,
  kept with zero counts (``has_matchup_row`` False).
* Matchup row but no shared stint (3-151 rows a season, <0.06%): tracking and
  our rotations disagree slightly. Kept with ``cofloor_seconds`` 0 and a NaN
  rate.
* ``matchup_seconds`` above ``cofloor_seconds`` (10-31 rows a season, up to
  34 s over): the rate is clipped at 1; the raw columns are not.

Only games with both validated stints and tracking are built; a game with one
of them has no co-floor time or no matchups to divide.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

PAIR_GAME_DIRNAME = Path("player_graph") / "pair_game"

KEY_COLUMNS = ("game_id", "off_player_id", "def_player_id")

#: Copied from the matchup row; zero when the pair shared the floor without one.
MATCHUP_COUNT_COLUMNS = (
    "matchup_seconds",
    "partial_possessions",
    "player_points",
    "team_points",
    "matchup_assists",
    "matchup_potential_assists",
    "matchup_turnovers",
    "matchup_blocks",
    "matchup_fgm",
    "matchup_fga",
    "matchup_fg3m",
    "matchup_fg3a",
    "matchup_ftm",
    "matchup_fta",
    "shooting_fouls",
)

#: The NBA's own time shares, kept for reference; NaN without a matchup row.
NBA_SHARE_COLUMNS = (
    "pct_defender_total_time",
    "pct_offensive_total_time",
    "pct_total_time_both_on",
)

COLUMNS = (
    "game_id",
    "game_date",
    "season_year",
    "off_team_id",
    "def_team_id",
    "off_player_id",
    "def_player_id",
    "attacker_is_home",
    "cofloor_seconds",
    *MATCHUP_COUNT_COLUMNS,
    "guard_rate",
    "has_matchup_row",
    *NBA_SHARE_COLUMNS,
)


def cofloor_seconds(stints: pd.DataFrame) -> pd.DataFrame:
    """Seconds each home x away pair shared the floor, per game.

    Returns ``game_id``, ``home_player_id``, ``away_player_id``,
    ``cofloor_seconds``.
    """
    if stints.empty:
        return pd.DataFrame(
            columns=["game_id", "home_player_id", "away_player_id", "cofloor_seconds"]
        )
    home = np.stack(stints["home_lineup"].map(np.asarray).to_numpy()).astype(str)
    away = np.stack(stints["away_lineup"].map(np.asarray).to_numpy()).astype(str)
    if home.shape[1] != 5 or away.shape[1] != 5:
        raise ValueError("Every stint must have five players a side")
    pairs = pd.DataFrame(
        {
            "game_id": np.repeat(stints["game_id"].astype(str).to_numpy(), 25),
            # Row-major: home player k meets away players 0..4.
            "home_player_id": np.repeat(home, 5, axis=1).ravel(),
            "away_player_id": np.tile(away, (1, 5)).ravel(),
            "cofloor_seconds": np.repeat(stints["seconds"].to_numpy(float), 25),
        }
    )
    return pairs.groupby(
        ["game_id", "home_player_id", "away_player_id"], as_index=False, sort=False
    )["cofloor_seconds"].sum()


def _directed(cofloor: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Each co-floor pair twice: once per side attacking."""
    games = games.set_index("game_id")
    frames = []
    for attacker_is_home, off, dfn in (
        (True, "home", "away"),
        (False, "away", "home"),
    ):
        frame = pd.DataFrame(
            {
                "game_id": cofloor["game_id"],
                "off_player_id": cofloor[f"{off}_player_id"],
                "def_player_id": cofloor[f"{dfn}_player_id"],
                "attacker_is_home": attacker_is_home,
                "cofloor_seconds": cofloor["cofloor_seconds"],
            }
        )
        frame["off_team_id"] = frame["game_id"].map(games[f"{off}_team_id"])
        frame["def_team_id"] = frame["game_id"].map(games[f"{dfn}_team_id"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_pair_game(stints: pd.DataFrame, matchups: pd.DataFrame) -> pd.DataFrame:
    """The ``pair_game`` rows of every game with both stints and matchups."""
    stints = stints.copy()
    matchups = matchups.copy()
    for frame, columns in (
        (stints, ("game_id", "home_team_id", "away_team_id")),
        (
            matchups,
            (
                "game_id",
                "home_team_id",
                "off_team_id",
                "def_team_id",
                "off_player_id",
                "def_player_id",
            ),
        ),
    ):
        for column in columns:
            if column in frame.columns:
                frame[column] = frame[column].astype(str)
    if "game_id" in stints.columns:
        stints["game_id"] = stints["game_id"].str.zfill(10)
    if "game_id" in matchups.columns:
        matchups["game_id"] = matchups["game_id"].str.zfill(10)
    if stints.empty or matchups.empty:
        return pd.DataFrame(columns=list(COLUMNS))

    built = sorted(set(stints["game_id"]) & set(matchups["game_id"]))
    stints = stints.loc[stints["game_id"].isin(built)]
    matchups = matchups.loc[matchups["game_id"].isin(built)]
    games = stints.drop_duplicates("game_id")[
        ["game_id", "game_date", "season_year", "home_team_id", "away_team_id"]
    ]

    directed = _directed(cofloor_seconds(stints), games)
    observed = matchups[
        [
            *KEY_COLUMNS,
            "off_team_id",
            "def_team_id",
            *MATCHUP_COUNT_COLUMNS,
            *NBA_SHARE_COLUMNS,
        ]
    ]
    if observed.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError("Matchups have more than one row per game and pair")
    pairs = directed.merge(
        observed,
        on=list(KEY_COLUMNS),
        how="outer",
        suffixes=("", "_matchup"),
        indicator=True,
    )
    pairs["has_matchup_row"] = pairs["_merge"].ne("left_only")
    matchup_only = pairs["_merge"].eq("right_only")
    for team in ("off_team_id", "def_team_id"):
        pairs[team] = pairs[team].fillna(pairs[f"{team}_matchup"])
    home = pairs["game_id"].map(games.set_index("game_id")["home_team_id"])
    pairs.loc[matchup_only, "attacker_is_home"] = pairs["off_team_id"].eq(home)
    pairs["attacker_is_home"] = pairs["attacker_is_home"].astype(bool)
    pairs["cofloor_seconds"] = pairs["cofloor_seconds"].fillna(0.0)
    pairs[list(MATCHUP_COUNT_COLUMNS)] = pairs[list(MATCHUP_COUNT_COLUMNS)].fillna(0)

    with np.errstate(divide="ignore", invalid="ignore"):
        rate = pairs["matchup_seconds"] / pairs["cofloor_seconds"]
    pairs["guard_rate"] = rate.where(pairs["cofloor_seconds"] > 0).clip(upper=1.0)

    by_game = games.set_index("game_id")
    pairs["game_date"] = pd.to_datetime(pairs["game_id"].map(by_game["game_date"]))
    pairs["season_year"] = pairs["game_id"].map(by_game["season_year"])
    return (
        pairs[list(COLUMNS)]
        .sort_values(["game_date", *KEY_COLUMNS], kind="mergesort")
        .reset_index(drop=True)
    )


def season_path(season_year: int, *, local_root: Path = Path("data")) -> Path:
    return Path(local_root) / PAIR_GAME_DIRNAME / f"season={season_year}.parquet"


def read_pair_game(
    season_years: Iterable[int], *, local_root: Path = Path("data")
) -> pd.DataFrame:
    """The stored ``pair_game`` rows of ``season_years`` (missing seasons skipped)."""
    frames = [
        pd.read_parquet(path)
        for season in sorted(set(season_years))
        if (path := season_path(season, local_root=local_root)).exists()
    ]
    if not frames:
        return pd.DataFrame(columns=list(COLUMNS))
    return pd.concat(frames, ignore_index=True)
