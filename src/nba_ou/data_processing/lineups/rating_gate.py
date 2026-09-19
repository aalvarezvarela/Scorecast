"""Phase C ratings-only game projection and walk-forward acceptance gate."""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd


def _game_table(games: pd.DataFrame) -> pd.DataFrame:
    required = {"GAME_ID", "GAME_DATE", "TEAM_ID", "HOME", "PTS"}
    if missing := required - set(games):
        raise ValueError(f"Games are missing {sorted(missing)}")
    frame = games[list(required)].copy()
    frame["GAME_ID"] = frame.GAME_ID.astype(str).str.zfill(10)
    frame["GAME_DATE"] = pd.to_datetime(frame.GAME_DATE).dt.normalize()
    frame["TEAM_ID"] = frame.TEAM_ID.astype(str)
    frame["PTS"] = pd.to_numeric(frame.PTS, errors="coerce")
    if not pd.api.types.is_bool_dtype(frame.HOME):
        frame["HOME"] = (
            frame.HOME.astype(str)
            .str.lower()
            .map({"true": True, "1": True, "false": False, "0": False})
        )
    rows = []
    for game_id, game in frame.groupby("GAME_ID", sort=False):
        home = game.loc[game.HOME.eq(True)]
        away = game.loc[game.HOME.eq(False)]
        if len(home) != 1 or len(away) != 1:
            continue
        rows.append(
            {
                "game_id": game_id,
                "game_date": home.GAME_DATE.iloc[0],
                "home_team_id": home.TEAM_ID.iloc[0],
                "away_team_id": away.TEAM_ID.iloc[0],
                "home_points": home.PTS.iloc[0],
                "away_points": away.PTS.iloc[0],
                "actual_total": home.PTS.iloc[0] + away.PTS.iloc[0],
            }
        )
    return pd.DataFrame(rows).sort_values(["game_date", "game_id"])


def _rating_maps(ratings: pd.DataFrame) -> tuple[dict, dict]:
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
    if missing := required - set(ratings):
        raise ValueError(f"Ratings are missing {sorted(missing)}")
    frame = ratings.copy()
    frame["as_of_date"] = pd.to_datetime(frame.as_of_date).dt.normalize()
    frame["fit_max_game_date"] = pd.to_datetime(frame.fit_max_game_date).dt.normalize()
    if not frame.fit_max_game_date.lt(frame.as_of_date).all():
        raise ValueError("Every rating fit must end strictly before its as-of date")
    values = {
        (date, str(player)): (float(off), float(defense), float(pace))
        for date, player, off, defense, pace in frame[
            ["as_of_date", "player_id", "o_rating", "d_rating", "pace_rating"]
        ].itertuples(index=False, name=None)
    }
    daily = frame.groupby("as_of_date", sort=False)[
        ["league_ortg", "league_pace", "fit_max_game_date"]
    ].first()
    return values, daily.to_dict("index")


def ratings_only_game_projections(
    games: pd.DataFrame,
    players: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    validation_from: object,
    validation_to: object,
    recent_games: int = 5,
) -> pd.DataFrame:
    """Project totals from prior player minutes and ratings, holding back each day."""
    if recent_games <= 0:
        raise ValueError("recent_games must be positive")
    game_table = _game_table(games)
    if game_table.empty:
        return pd.DataFrame()
    required_players = {"GAME_ID", "TEAM_ID", "PLAYER_ID", "MIN"}
    if missing := required_players - set(players):
        raise ValueError(f"Players are missing {sorted(missing)}")
    player_rows = players[list(required_players)].copy()
    player_rows["GAME_ID"] = player_rows.GAME_ID.astype(str).str.zfill(10)
    player_rows["TEAM_ID"] = player_rows.TEAM_ID.astype(str)
    player_rows["PLAYER_ID"] = player_rows.PLAYER_ID.astype(str)
    player_rows["MIN"] = pd.to_numeric(player_rows.MIN, errors="coerce").fillna(0.0)
    player_rows = player_rows.merge(
        game_table[["game_id", "game_date"]],
        left_on="GAME_ID",
        right_on="game_id",
        how="inner",
    ).drop_duplicates(["GAME_ID", "TEAM_ID", "PLAYER_ID"])
    games_by_date = {
        date: frame for date, frame in game_table.groupby("game_date", sort=True)
    }
    players_by_date = {
        date: frame for date, frame in player_rows.groupby("game_date", sort=True)
    }
    rating_values, daily_ratings = _rating_maps(ratings)
    start = pd.Timestamp(validation_from).normalize()
    end = pd.Timestamp(validation_to).normalize()
    if start > end:
        raise ValueError("validation_from must be on or before validation_to")

    assignments: dict[str, str] = {}
    rosters: dict[str, set[str]] = defaultdict(set)
    minute_history: dict[tuple[str, str], deque[float]] = defaultdict(
        lambda: deque(maxlen=recent_games)
    )
    points_history: dict[str, deque[float]] = defaultdict(
        lambda: deque(maxlen=recent_games)
    )
    output = []

    def team_projection(
        team_id: str, day: pd.Timestamp
    ) -> tuple[float, float, float] | None:
        recent = {
            player: float(np.mean(minute_history[(team_id, player)]))
            for player in rosters[team_id]
            if minute_history[(team_id, player)]
        }
        total_minutes = sum(recent.values())
        if total_minutes <= 0:
            return None
        scale = 240.0 / total_minutes
        result = np.zeros(3)
        for player, minutes in recent.items():
            result += (
                minutes
                * scale
                / 48.0
                * np.asarray(rating_values.get((day, player), (0.0, 0.0, 0.0)))
            )
        return tuple(float(value) for value in result)

    for day in sorted(games_by_date):
        day_games = games_by_date[day]
        if start <= day <= end and day in daily_ratings:
            league = daily_ratings[day]
            for game in day_games.itertuples(index=False):
                home = team_projection(game.home_team_id, day)
                away = team_projection(game.away_team_id, day)
                if home is None or away is None:
                    continue
                home_off, home_def, home_pace = home
                away_off, away_def, away_pace = away
                possessions = float(league["league_pace"]) + home_pace + away_pace
                projected_total = (
                    possessions
                    / 100.0
                    * (
                        2 * float(league["league_ortg"])
                        + home_off
                        + away_off
                        - home_def
                        - away_def
                    )
                )
                home_points = points_history[game.home_team_id]
                away_points = points_history[game.away_team_id]
                baseline = (
                    float(np.mean(home_points)) + float(np.mean(away_points))
                    if home_points and away_points
                    else np.nan
                )
                output.append(
                    {
                        "game_id": game.game_id,
                        "game_date": day,
                        "actual_total": game.actual_total,
                        "proj_total": projected_total,
                        "baseline_total": baseline,
                        "projected_possessions": possessions,
                        "fit_max_game_date": league["fit_max_game_date"],
                    }
                )

        # A date's outcomes and box scores become history only after every game
        # on that date has been projected.
        for game in day_games.itertuples(index=False):
            if pd.notna(game.home_points):
                points_history[game.home_team_id].append(float(game.home_points))
            if pd.notna(game.away_points):
                points_history[game.away_team_id].append(float(game.away_points))
        for row in players_by_date.get(day, pd.DataFrame()).itertuples(index=False):
            player = row.PLAYER_ID
            team = row.TEAM_ID
            previous = assignments.get(player)
            if previous is not None and previous != team:
                rosters[previous].discard(player)
            assignments[player] = team
            rosters[team].add(player)
            minute_history[(team, player)].append(max(0.0, float(row.MIN)))
    return pd.DataFrame(output)


def score_rating_gate(projections: pd.DataFrame) -> dict[str, float | int | bool]:
    """Compare both methods on the same games and return the Phase C verdict."""
    required = {"actual_total", "proj_total", "baseline_total"}
    if missing := required - set(projections):
        raise ValueError(f"Projections are missing {sorted(missing)}")
    scored = projections.dropna(subset=list(required))
    if scored.empty:
        raise ValueError("No games have both ratings and baseline projections")
    ratings_mae = float((scored.actual_total - scored.proj_total).abs().mean())
    baseline_mae = float((scored.actual_total - scored.baseline_total).abs().mean())
    improvement = baseline_mae - ratings_mae
    return {
        "games": len(scored),
        "ratings_mae": ratings_mae,
        "baseline_mae": baseline_mae,
        "mae_improvement": improvement,
        "passed": improvement > 0,
    }
