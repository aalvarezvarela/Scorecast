"""Default loaders for :class:`~.base.LayerContext`.

Used when a layer runs over a dataset file and nothing was seeded. Each loader
repeats the steps ``create_df_to_predict`` takes for the same object, so a
layer computed from a file sees what it would have seen inside the full build.
"""

from __future__ import annotations

import pandas as pd

from nba_ou.create_training_data.predict_data_utils import (
    filter_by_seasons_with_extra_game_ids,
)
from nba_ou.utils.general_utils import get_season_year_from_date
from nba_ou.utils.seasons import get_seasons_between_dates


def player_context_seasons(game_dates: pd.Series) -> list[str]:
    """The output seasons of ``game_dates`` plus one season before the first.

    ``create_df_to_predict`` keeps exactly this as player context: the extra
    season gives the first output season's earliest games a roster history.
    """
    dates = pd.to_datetime(game_dates, errors="coerce").dropna()
    if dates.empty:
        raise ValueError("Cannot load player history for a frame with no game dates.")
    first_season_year = get_season_year_from_date(dates.min())
    start = pd.Timestamp(year=first_season_year - 1, month=10, day=1)
    return get_seasons_between_dates(start, dates.max())


def load_player_history(game_dates: pd.Series) -> pd.DataFrame:
    """Cleaned player box scores up to the last of ``game_dates``.

    The same three steps as ``create_df_to_predict``: load games and players for
    the context seasons, attach dates and numeric minutes with
    ``clear_player_statistics``, then cap by season and date.
    """
    from nba_ou.data_processing.players.attach_player_features import (
        clear_player_statistics,
    )
    from nba_ou.postgre_db import load_all_nba_data_from_db

    seasons = player_context_seasons(game_dates)
    recent_limit = pd.to_datetime(game_dates, errors="coerce").max()
    print(f"Loading player history for seasons {seasons} ...")
    df_games, df_players = load_all_nba_data_from_db(seasons=seasons)
    df_games["GAME_DATE"] = pd.to_datetime(df_games["GAME_DATE"])
    df_games = filter_by_seasons_with_extra_game_ids(
        df_games, seasons=seasons, recent_limit_to_include=recent_limit
    )
    df_players = clear_player_statistics(df_players, df_games)
    return filter_by_seasons_with_extra_game_ids(
        df_players, seasons=seasons, recent_limit_to_include=recent_limit
    )
