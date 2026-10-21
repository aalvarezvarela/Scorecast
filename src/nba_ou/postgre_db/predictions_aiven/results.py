"""Final scores for the prediction store's ``games``, from the NBA API.

``LeagueGameFinder`` returns one row per team and game; a game is final when
both of its rows carry a result (``WL``) and points (``PTS``). No total-points
heuristic: a game counts as final on the official result alone. A game that is
not final is left out and looked up again on the next run.
"""

from __future__ import annotations

import pandas as pd


def season_label(season_year: int) -> str:
    """``2026`` -> ``"2026-27"``, the API's season format."""
    return f"{season_year}-{str(season_year + 1)[-2:]}"


def final_scores(team_games: pd.DataFrame) -> pd.DataFrame:
    """One row per final game (``game_id`` int, ``home_points``, ``away_points``).

    ``team_games`` are ``LeagueGameFinder`` rows: ``GAME_ID``, ``MATCHUP``
    ("BOS vs. TOR" at home, "TOR @ BOS" away), ``PTS``, ``WL``.
    """
    rows = team_games.assign(
        game_id=pd.to_numeric(team_games["GAME_ID"].astype(str)).astype("int64"),
        is_home=team_games["MATCHUP"].str.contains("vs.", regex=False, na=False),
    )
    rows["scored"] = pd.to_numeric(rows["PTS"], errors="coerce").notna()
    rows["decided"] = rows["WL"].notna()
    complete = rows.groupby("game_id").agg(
        teams=("is_home", "size"),
        homes=("is_home", "sum"),
        results=("decided", "sum"),
        scores=("scored", "sum"),
    )
    final = complete.index[
        (complete["teams"] == 2)
        & (complete["homes"] == 1)
        & (complete["results"] == 2)
        & (complete["scores"] == 2)
    ]
    rows = rows[rows["game_id"].isin(final)]
    points = pd.to_numeric(rows["PTS"])
    home = points[rows["is_home"]].set_axis(rows.loc[rows["is_home"], "game_id"])
    away = points[~rows["is_home"]].set_axis(rows.loc[~rows["is_home"], "game_id"])
    scores = pd.DataFrame({"home_points": home, "away_points": away}).astype("int64")
    return scores.reset_index().rename(columns={"index": "game_id"})


def fetch_final_scores(game_ids: list[int], season_years: list[int]) -> pd.DataFrame:
    """Final scores of ``game_ids``, one ``LeagueGameFinder`` call per season."""
    from nba_api.stats.endpoints import LeagueGameFinder

    frames = [
        LeagueGameFinder(
            season_nullable=season_label(int(season)), league_id_nullable="00"
        ).get_data_frames()[0]
        for season in sorted(set(season_years))
    ]
    if not frames:
        return pd.DataFrame(columns=["game_id", "home_points", "away_points"])
    scores = final_scores(pd.concat(frames, ignore_index=True))
    return scores[scores["game_id"].isin(set(game_ids))].reset_index(drop=True)
