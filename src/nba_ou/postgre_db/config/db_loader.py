from nba_ou.postgre_db.games.fetch_data_from_db.fetch_data_from_games_db import (
    load_games_from_db,
)
from nba_ou.postgre_db.players.fetch_players_data_from_db import (
    load_players_from_db,
)
from nba_ou.utils.history_cache import cached
from nba_ou.utils.seasons import drop_exhibition_games


@cached()
def load_all_nba_data_from_db(seasons=None, extra_game_ids=None):
    """Load games and players from Postgres with uppercase column names.

    Args:
        seasons (list[str], optional): Seasons such as ["2023-24", "2024-25"].
            None or an empty list leaves the season filter unset.
        extra_game_ids (list, optional): Game IDs included in addition to the
            selected seasons. With no seasons, these IDs restrict the query.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Game rows and player rows, without
        preseason and All-Star games (``drop_exhibition_games``): every
        consumer is a feature builder, and no feature reads them.

    Raises:
        ValueError: If either database loader returns None.
    """
    print("Loading NBA data from PostgreSQL...")

    df_games = load_games_from_db(seasons=seasons, extra_game_ids=extra_game_ids)
    df_players = load_players_from_db(seasons=seasons, extra_game_ids=extra_game_ids)

    if df_games is None or df_players is None:
        print("Failed to load data from databases")
        raise ValueError("Failed to load data from databases")

    # Match your previous pipeline expectations (uppercase columns)
    df_games.columns = df_games.columns.str.upper()
    df_players.columns = df_players.columns.str.upper()
    df_games = drop_exhibition_games(df_games, "GAME_ID")
    df_players = drop_exhibition_games(df_players, "GAME_ID")

    print(
        f"Successfully loaded {len(df_games)} games and {len(df_players)} player records"
    )
    return df_games, df_players
