"""Walk-forward selection of ridge penalties from next-date stint error."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .player_ratings import _lineup, _poss, walk_forward_player_ratings


def _rating_lookup(ratings: pd.DataFrame, column: str) -> dict:
    return {
        (pd.Timestamp(date).normalize(), str(player)): float(value)
        for date, player, value in ratings[
            ["as_of_date", "player_id", column]
        ].itertuples(index=False, name=None)
    }


def score_stint_predictions(
    stints: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    score_offdef: bool = True,
    score_pace: bool = True,
) -> dict[str, float]:
    """Possession/seconds weighted MAE using each date's pre-game ratings."""
    if ratings.empty:
        return {"offdef_mae": np.nan, "pace_mae": np.nan}
    frame = stints.copy()
    frame["game_date"] = pd.to_datetime(frame.game_date).dt.normalize()
    frame["home_lineup"] = frame.home_lineup.map(_lineup)
    frame["away_lineup"] = frame.away_lineup.map(_lineup)
    covered = set(pd.to_datetime(ratings.as_of_date).dt.normalize())
    frame = frame.loc[frame.game_date.isin(covered)]
    offense = _rating_lookup(ratings, "o_rating")
    defense = _rating_lookup(ratings, "d_rating")
    pace = _rating_lookup(ratings, "pace_rating")
    intercepts = ratings.groupby("as_of_date", sort=False)[
        ["league_ortg", "league_pace"]
    ].first()
    intercepts.index = pd.to_datetime(intercepts.index).normalize()
    off_abs = off_weight = pace_abs = pace_weight = 0.0
    for row in frame.to_dict("records"):
        date = row["game_date"]
        home, away = row["home_lineup"], row["away_lineup"]
        if score_offdef:
            for attacking, defending, side in (
                (home, away, "home"),
                (away, home, "away"),
            ):
                poss = _poss(row, side)
                if poss <= 0:
                    continue
                predicted = float(intercepts.loc[date, "league_ortg"])
                predicted += sum(offense.get((date, pid), 0.0) for pid in attacking)
                predicted -= sum(defense.get((date, pid), 0.0) for pid in defending)
                actual = 100 * float(row[f"{side}_pts"]) / poss
                off_abs += poss * abs(actual - predicted)
                off_weight += poss
        if score_pace:
            seconds = (float(row["end_ds"]) - float(row["start_ds"])) / 10
            game_poss = (_poss(row, "home") + _poss(row, "away")) / 2
            if seconds <= 0 or game_poss <= 0:
                continue
            predicted = float(intercepts.loc[date, "league_pace"])
            predicted += sum(pace.get((date, pid), 0.0) for pid in home + away)
            actual = game_poss * 2880 / seconds
            pace_abs += seconds * abs(actual - predicted)
            pace_weight += seconds
    return {
        "offdef_mae": off_abs / off_weight if off_weight else np.nan,
        "pace_mae": pace_abs / pace_weight if pace_weight else np.nan,
    }


def tune_rating_lambdas(
    stints: pd.DataFrame,
    validation_dates: Iterable,
    *,
    offdef_candidates: Iterable[float],
    pace_candidates: Iterable[float],
    half_life_days: float = 180.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Evaluate explicit candidate grids; no production penalty is assumed."""
    dates = sorted({pd.Timestamp(date).normalize() for date in validation_dates})
    off_values = sorted({float(value) for value in offdef_candidates})
    pace_values = sorted({float(value) for value in pace_candidates})
    if not dates or not off_values or not pace_values:
        raise ValueError("Validation dates and both lambda grids must be nonempty")
    if any(value <= 0 for value in off_values + pace_values):
        raise ValueError("Every lambda candidate must be positive")
    rows = []
    # The unused penalty does not affect the fitted block being scored.
    for alpha in off_values:
        ratings = walk_forward_player_ratings(
            stints,
            dates,
            lambda_offdef=alpha,
            lambda_pace=pace_values[0],
            half_life_days=half_life_days,
        )
        score = score_stint_predictions(
            stints, ratings, score_offdef=True, score_pace=False
        )
        rows.append({"rating_type": "offdef", "lambda": alpha, **score})
    for alpha in pace_values:
        ratings = walk_forward_player_ratings(
            stints,
            dates,
            lambda_offdef=off_values[0],
            lambda_pace=alpha,
            half_life_days=half_life_days,
        )
        score = score_stint_predictions(
            stints, ratings, score_offdef=False, score_pace=True
        )
        rows.append({"rating_type": "pace", "lambda": alpha, **score})
    results = pd.DataFrame(rows)
    if results.offdef_mae.dropna().empty or results.pace_mae.dropna().empty:
        raise ValueError("Validation dates have no scoreable prior-history stints")
    best = {
        "lambda_offdef": float(
            results.loc[results.rating_type.eq("offdef")]
            .sort_values(["offdef_mae", "lambda"])
            .iloc[0]["lambda"]
        ),
        "lambda_pace": float(
            results.loc[results.rating_type.eq("pace")]
            .sort_values(["pace_mae", "lambda"])
            .iloc[0]["lambda"]
        ),
    }
    return results, best
