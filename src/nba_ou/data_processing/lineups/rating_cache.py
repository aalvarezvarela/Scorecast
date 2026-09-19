"""Build a reproducible cache of leakage-safe player lineup ratings."""

from __future__ import annotations

import pandas as pd

from .player_ratings import walk_forward_player_ratings


def build_player_rating_cache(
    stints: pd.DataFrame,
    *,
    as_of_from: object,
    as_of_to: object,
    lambda_offdef: float,
    lambda_pace: float,
    half_life_days: float = 180.0,
) -> pd.DataFrame:
    """Fit ratings for game dates in a range and always include its end date."""
    if stints.empty:
        raise ValueError("Cannot build a rating cache without stints")
    start = pd.Timestamp(as_of_from).normalize()
    end = pd.Timestamp(as_of_to).normalize()
    if start > end:
        raise ValueError("as_of_from must be on or before as_of_to")
    game_dates = pd.to_datetime(stints.game_date).dt.normalize()
    dates = set(game_dates.loc[game_dates.between(start, end)])
    dates.add(end)
    ratings = walk_forward_player_ratings(
        stints,
        sorted(dates),
        lambda_offdef=lambda_offdef,
        lambda_pace=lambda_pace,
        half_life_days=half_life_days,
    )
    if ratings.empty:
        raise ValueError("No prior stints exist for the requested rating dates")
    if not ratings.fit_max_game_date.lt(ratings.as_of_date).all():
        raise AssertionError("Rating cache contains a non-prior fit date")
    return ratings
