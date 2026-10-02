import pandas as pd
from nba_ou.data_processing.team.cleaning_teams import fix_home_away_parsing_errors
from nba_ou.postgre_db.config.db_config import (
    connect_nba_db,
    get_schema_name_games,
)
from psycopg import sql


def _normalize_game_ids(game_ids) -> list[str]:
    """Normalize GAME_ID inputs into a unique, non-empty string list."""
    if game_ids is None:
        return []

    normalized = []
    seen = set()
    for game_id in game_ids:
        if pd.isna(game_id):
            continue
        game_id_str = str(game_id).strip()
        if not game_id_str or game_id_str in seen:
            continue
        seen.add(game_id_str)
        normalized.append(game_id_str)
    return normalized


def load_games_from_db(seasons=None, extra_game_ids=None) -> pd.DataFrame | None:
    """Load NBA game rows from Postgres.

    Args:
        seasons (list[str], optional): Seasons such as ["2023-24", "2024-25"].
            None or an empty list leaves the season filter unset.
        extra_game_ids (list, optional): Game IDs included in addition to the
            selected seasons. With no seasons, these IDs restrict the query.

    Returns:
        pd.DataFrame | None: Matching rows in descending game-date order, or
            None if loading fails.
    """
    schema = get_schema_name_games()
    table = schema  # convention: schema == table

    conn = None
    try:
        conn = connect_nba_db()
        where_parts = []
        query_params = []

        if seasons is not None and len(seasons) > 0:
            season_years = [int(s.split("-")[0]) for s in seasons]
            where_parts.append(sql.SQL("season_year = ANY(%s)"))
            query_params.append(season_years)

        normalized_extra_game_ids = _normalize_game_ids(extra_game_ids)
        if normalized_extra_game_ids:
            where_parts.append(sql.SQL("game_id = ANY(%s)"))
            query_params.append(normalized_extra_game_ids)

        where_clause = sql.SQL("")
        if where_parts:
            where_clause = sql.SQL("WHERE ") + sql.SQL(" OR ").join(where_parts)

        query_obj = sql.SQL("""
            SELECT *
            FROM {}.{}
            {}
            ORDER BY game_date DESC, game_id DESC
        """).format(
            sql.Identifier(schema),
            sql.Identifier(table),
            where_clause,
        )

        query = query_obj.as_string(conn)
        if query_params:
            df = pd.read_sql_query(query, conn, params=tuple(query_params))
        else:
            df = pd.read_sql_query(query, conn)

        print(f"Loaded {len(df)} game records from database")
        return df

    except Exception as e:
        print(f"Error loading games from database: {e}")
        import traceback

        traceback.print_exc()
        return None

    finally:
        if conn is not None:
            conn.close()


def get_historical_game_ids_for_home_away_matchups(
    home_away_pairs: list[tuple[str, str]],
    exclude_game_ids=None,
    max_game_date=None,
    min_season_year: int | None = None,
) -> list[str]:
    """
    Fetch historical GAME_IDs for the exact same home/away matchup orientation.

    Args:
        home_away_pairs: list of (home_team_id, away_team_id) tuples.
        exclude_game_ids: optional GAME_IDs to exclude from results.
        max_game_date: optional max game date (inclusive) to cap returned games.
        min_season_year: optional lower bound on season_year. If None, uses
            the full history available in the games table.

    Returns:
        list[str]: Historical game IDs ordered by most recent game date first.
    """
    if not home_away_pairs:
        return []

    normalized_pairs = []
    seen_pairs = set()
    for home_team_id, away_team_id in home_away_pairs:
        if pd.isna(home_team_id) or pd.isna(away_team_id):
            continue
        home_team_id = str(home_team_id).strip()
        away_team_id = str(away_team_id).strip()
        if not home_team_id or not away_team_id:
            continue
        pair = (home_team_id, away_team_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        normalized_pairs.append(pair)

    if not normalized_pairs:
        return []

    schema = get_schema_name_games()
    table = schema  # convention: schema == table
    excluded_game_ids = _normalize_game_ids(exclude_game_ids)

    conn = None
    try:
        conn = connect_nba_db()

        values_sql = sql.SQL(", ").join([sql.SQL("(%s, %s)") for _ in normalized_pairs])
        where_exclude = sql.SQL("")
        where_date_cap = sql.SQL("")
        where_min_season = sql.SQL("")
        query_params = []

        for home_team_id, away_team_id in normalized_pairs:
            query_params.extend([home_team_id, away_team_id])

        if excluded_game_ids:
            where_exclude = sql.SQL(" AND NOT (g_home.game_id = ANY(%s))")
            query_params.append(excluded_game_ids)

        if max_game_date is not None:
            where_date_cap = sql.SQL(" AND g_home.game_date <= %s")
            query_params.append(pd.to_datetime(max_game_date).date())

        if min_season_year is not None:
            where_min_season = sql.SQL(" AND g_home.season_year >= %s")
            query_params.append(int(min_season_year))

        query_obj = sql.SQL("""
            WITH requested_matchups(home_team_id, away_team_id) AS (
                VALUES {}
            )
            SELECT g_home.game_id, MAX(g_home.game_date) AS game_date
            FROM {}.{} AS g_home
            JOIN {}.{} AS g_away
                ON g_home.game_id = g_away.game_id
            JOIN requested_matchups AS rm
                ON g_home.team_id = rm.home_team_id
                AND g_away.team_id = rm.away_team_id
            WHERE g_home.home = TRUE
                AND g_away.home = FALSE
                AND COALESCE(g_home.season_type, '') NOT IN ('All Star', 'Preseason')
                AND COALESCE(g_away.season_type, '') NOT IN ('All Star', 'Preseason')
                {}
                {}
                {}
            GROUP BY g_home.game_id
            ORDER BY game_date DESC
        """).format(
            values_sql,
            sql.Identifier(schema),
            sql.Identifier(table),
            sql.Identifier(schema),
            sql.Identifier(table),
            where_exclude,
            where_date_cap,
            where_min_season,
        )

        query = query_obj.as_string(conn)
        rows = pd.read_sql_query(query, conn, params=tuple(query_params))
        game_ids = rows["game_id"].astype(str).tolist() if not rows.empty else []
        print(
            f"Loaded {len(game_ids)} historical game IDs for {len(normalized_pairs)} home/away matchup pairs"
        )
        return game_ids

    except Exception as e:
        print(f"Error loading historical game IDs for matchups: {e}")
        import traceback

        traceback.print_exc()
        return []

    finally:
        if conn is not None:
            conn.close()


def load_cleaned_games_for_odds(
    season_year: str | int | None = None,
) -> pd.DataFrame:
    """
    Load games from DB, filter out All Star and Preseason games, and fix home/away issues.

    Args:
        season_year: Optional season start year (e.g., 2025 for 2025-26 season).
                     If None, loads all seasons.

    Returns:
        Cleaned DataFrame with regular season and playoff games only.
    """
    # Load games from database
    if season_year is not None:
        season_str = f"{season_year}-{str(int(season_year) + 1)[-2:]}"
        df = load_games_from_db(seasons=[season_str])
    else:
        df = load_games_from_db(seasons=None)

    if df is None or df.empty:
        return pd.DataFrame()

    # Filter out All Star and Preseason games
    if "season_type" in df.columns:
        df = df[~df["season_type"].isin(["All Star", "Preseason"])]

    # Fix home/away parsing errors
    df = fix_home_away_parsing_errors(df)

    df = df.drop_duplicates(keep="first").reset_index(drop=True)

    return df
