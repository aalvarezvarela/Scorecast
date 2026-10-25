import pandas as pd

from nba_ou.config.constants import EXHIBITION_SEASON_TYPES, SEASON_TYPE_MAP
from nba_ou.utils.general_utils import get_season_year_from_date

#: Game-id prefixes of ``EXHIBITION_SEASON_TYPES`` (``001``, ``003``).
EXHIBITION_PREFIXES = frozenset(
    prefix
    for prefix, season_type in SEASON_TYPE_MAP.items()
    if season_type in EXHIBITION_SEASON_TYPES
)


def classify_season_type(game_id: str) -> str:
    """Return the canonical season type for an NBA game ID."""
    return SEASON_TYPE_MAP.get(game_id[:3], "Unknown")


def is_exhibition_game_id(game_ids: pd.Series) -> pd.Series:
    """True where a game id is a preseason or All-Star game.

    Classified by the id's prefix, never by a ``SEASON_TYPE`` text column
    (which labels play-in games "Playoffs"). Ids stored without their leading
    zeros (``22400001``) are padded first; a missing id is not an exhibition.
    """
    prefixes = game_ids.astype("string").str.zfill(10).str[:3]
    return prefixes.isin(EXHIBITION_PREFIXES).fillna(False).astype(bool)


def drop_exhibition_games(
    df: pd.DataFrame | None, game_id_col: str | None = None
) -> pd.DataFrame | None:
    """``df`` without its preseason and All-Star rows (``EXHIBITION_SEASON_TYPES``).

    ``game_id_col`` defaults to ``GAME_ID``, else ``game_id``. A frame without
    either column, or None, is returned unchanged.
    """
    if df is None:
        return None
    if game_id_col is None:
        game_id_col = next((c for c in ("GAME_ID", "game_id") if c in df.columns), None)
        if game_id_col is None:
            return df
    keep = ~is_exhibition_game_id(df[game_id_col])
    return df if keep.all() else df.loc[keep.to_numpy()].reset_index(drop=True)


def get_all_seasons_from_2006(date_to_train_until):
    """
    Get all NBA seasons from 2006-07 until the season containing date_to_train_until.

    Args:
        date_to_train_until (datetime): The target date

    Returns:
        list: List of season strings in format "YYYY-YY" (e.g., ["2006-07", "2007-08", ...])
    """
    if isinstance(date_to_train_until, str):
        date_to_train_until = pd.to_datetime(date_to_train_until)

    end_season_year = get_season_year_from_date(date_to_train_until)

    # Generate all seasons from 2006 to end_season_year
    seasons = []
    for year in range(2006, end_season_year + 1):
        season_str = f"{year}-{str(year + 1)[-2:]}"
        seasons.append(season_str)

    return seasons


def get_season_start_date_n_seasons_back(
    n_seasons: int, reference_date=None
) -> pd.Timestamp:
    """
    Get the start date of the season N seasons back from reference_date.

    NBA seasons start in October. For example, if n_seasons=2 and we're in
    the 2025-26 season, returns the start of the 2024-25 season (Oct 2024).

    Args:
        n_seasons (int): Number of seasons to include (1 = current season only,
                        2 = current + previous, etc.)
        reference_date (datetime, optional): Reference date to compute from.
                                             Defaults to today.

    Returns:
        pd.Timestamp: Start date (October 1st) of the earliest season to include.
    """
    if reference_date is None:
        reference_date = pd.Timestamp.now()
    if isinstance(reference_date, str):
        reference_date = pd.to_datetime(reference_date)

    current_season_start_year = get_season_year_from_date(reference_date)

    # Go back n_seasons - 1 (since n_seasons=1 means current season only)
    target_season_start_year = current_season_start_year - (n_seasons - 1)

    return pd.Timestamp(year=target_season_start_year, month=10, day=1)


def get_seasons_between_dates(date_from, date_to):
    """
    Get all NBA seasons between two dates (inclusive).

    Args:
        date_from (datetime or str): The start date
        date_to (datetime or str): The end date

    Returns:
        list: List of season strings in format "YYYY-YY" (e.g., ["2006-07", "2007-08", ...])
    """
    if isinstance(date_from, str):
        date_from = pd.to_datetime(date_from)
    if isinstance(date_to, str):
        date_to = pd.to_datetime(date_to)

    start_season_year = get_season_year_from_date(date_from)
    end_season_year = get_season_year_from_date(date_to)

    # Generate all seasons between start and end
    seasons = []
    for year in range(start_season_year, end_season_year + 1):
        season_str = f"{year}-{str(year + 1)[-2:]}"
        seasons.append(season_str)

    return seasons
