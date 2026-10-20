import random
import re
import time

import pandas as pd
from nba_api.stats.endpoints import (
    BoxScoreAdvancedV3,
    BoxScoreTraditionalV3,
    LeagueGameFinder,
)
from nba_ou.config.constants import SEASON_TYPE_MAP as SEASON_TYPE_MAPPING
from nba_ou.fetch_data.nba_api_session import reset_nba_http_session
from nba_ou.postgre_db.games.update_games.mapping_v3_v2 import (
    V3_TO_V2_ADVANCED_PLAYER_MAP,
    V3_TO_V2_ADVANCED_TEAM_MAP,
    V3_TO_V2_TRADITIONAL_MAP,
)
from tqdm import tqdm


def classify_season_type(game_id: str) -> str:
    """Determines the season type based on the game ID prefix."""
    return SEASON_TYPE_MAPPING.get(game_id[:3], "Unknown")


def fetch_box_score_data(game_id: str, n_tries: int = 3):
    """Fetches traditional and advanced box score data with retries."""
    limit_reached = False
    box_score_traditional = None
    box_score_advanced = None

    for api_call in [BoxScoreTraditionalV3, BoxScoreAdvancedV3]:
        attempts = 0
        time.sleep(random.uniform(0.05, 0.1))  # Avoid rate limiting

        while attempts < n_tries:
            try:
                data = api_call(game_id=game_id, timeout=10)

                if api_call == BoxScoreTraditionalV3:
                    box_score_traditional = data
                elif api_call == BoxScoreAdvancedV3:
                    box_score_advanced = data

                break  # Exit the retry loop for this API call if successful

            except AttributeError as e:
                # If the error is 'NoneType' object has no attribute 'get', skip and continue
                if "'NoneType' object has no attribute 'get'" in str(e):
                    print(f"Skipping {game_id} ({api_call.__name__}): {e}")
                    time.sleep(1)  # Brief pause before continuing
                    break  # Do not retry, just skip this game/api_call
                else:
                    attempts += 1
                    print(
                        f"Attempt {attempts} failed for {game_id} ({api_call.__name__}). Error: {e}"
                    )
                    reset_nba_http_session()
                    if attempts < n_tries:
                        print(f"Retrying in {5 * attempts} seconds...")
                        time.sleep(5 * attempts)
                    else:
                        print(
                            f"Failed to fetch data for {game_id} ({api_call.__name__}). Max attempts reached."
                        )
                        limit_reached = True
            except Exception as e:
                attempts += 1
                print(
                    f"Attempt {attempts} failed for {game_id} ({api_call.__name__}). Error: {e}"
                )
                reset_nba_http_session()
                if attempts < n_tries:
                    print(f"Retrying in {5 * attempts} seconds...")
                    time.sleep(5 * attempts)
                else:
                    print(
                        f"Failed to fetch data for {game_id} ({api_call.__name__}). Max attempts reached."
                    )
                    limit_reached = True

    if not box_score_traditional or not box_score_advanced:
        print(
            f"Warning: Missing data for game_id {game_id}. Traditional: {box_score_traditional is not None}, Advanced: {box_score_advanced is not None}"
        )

    return box_score_traditional, box_score_advanced, limit_reached


def merge_stats(player_trad, player_adv, team_trad, team_adv, game_id):
    """Merges traditional and advanced stats for both players and teams."""
    try:
        player_stats = pd.merge(
            player_trad,
            player_adv,
            on=["PLAYER_ID", "GAME_ID", "TEAM_ID"],
            suffixes=("", "_drop"),
        )
        player_stats = player_stats.loc[:, ~player_stats.columns.str.endswith("_drop")]
    except Exception as e:
        print(f"Failed to merge player stats for game_id {game_id}: {e}")
        player_stats = pd.DataFrame()

    try:
        team_stats = pd.merge(
            team_trad, team_adv, on=["TEAM_ID", "GAME_ID"], suffixes=("", "_drop")
        )
        team_stats = team_stats.loc[:, ~team_stats.columns.str.endswith("_drop")]
    except Exception as e:
        print(f"Failed to merge team stats for game_id {game_id}: {e}")
        team_stats = pd.DataFrame()

    return player_stats, team_stats


def fetch_nba_data(
    season_nullable: str,
    existing_game_ids: set = None,
    n_tries: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Fetches and processes NBA game data for a specified season.

    This function retrieves game statistics from the NBA API for the given season, processes the data,
    and returns updated game and player statistics. It avoids duplicate entries by checking against
    previously fetched data.

    Args:
        season_nullable (str): The NBA season in the format 'YYYY-YY' (e.g., '2023-24').
        input_df (pd.DataFrame, optional): An existing DataFrame containing previously fetched game data.
                                           If provided, new game data will be appended, avoiding duplicates.
                                           Deprecated - use existing_game_ids instead.
        input_player_stats (pd.DataFrame, optional): An existing DataFrame containing previously fetched
                                                     player statistics. If provided, new player stats
                                                     will be appended, avoiding duplicates.
        existing_game_ids (set, optional): A set of game IDs already present in the database.
                                           If provided, only games not in this set will be fetched.
        n_tries (int, optional): The number of attempts to fetch each game's box score data in case of failures.
                                 Default is 3.

    Returns:
        tuple:
            - pd.DataFrame: Updated game data including team statistics.
            - pd.DataFrame: Updated player statistics.
            - bool: Flag indicating if the rate limit was reached during data fetching.

    Raises:
        ValueError: If the `season_nullable` format is incorrect.

    Notes:
        - The function avoids redundant requests by skipping games already present in `existing_game_ids` or `input_df`.
        - The function introduces a delay between API requests to prevent hitting rate limits.
        - If the rate limit is reached (299 games fetched), the function pauses for 5 seconds
          and resets the HTTP session before stopping further requests.
    """

    limit_reached = False
    if not re.match(r"^\d{4}-\d{2}$", season_nullable):
        raise ValueError("Invalid season format. Expected format: 'YYYY-YY'")

    # Retry LeagueGameFinder up to 3 times if it fails
    for attempt in range(1, 4):
        try:
            time.sleep(random.uniform(0.1, 0.3))  # Avoid rate limiting

            print("Attempting to fetch scheduled games for a given season")
            game_finder = LeagueGameFinder(
                season_nullable=season_nullable,
                league_id_nullable="00",
            )
            time.sleep(random.uniform(0.1, 0.3))  # Avoid rate limiting
            games = game_finder.get_data_frames()[0]
            print(
                f"Fetched {len(games)} games for season {season_nullable}, attempt {attempt}"
            )
            break
        except Exception as e:
            print(
                f"LeagueGameFinder attempt {attempt} failed: {e}. Season: {season_nullable}"
            )
            if attempt == 3:
                raise
            time.sleep(5 * attempt)

    games["SEASON_TYPE"] = games["GAME_ID"].apply(classify_season_type)
    games = games[~games["SEASON_TYPE"].isin(["Preseason", "All Star"])].copy()
    print(f"Filtered to {len(games)} games (excluded Preseason and All Star)")
    games["HOME"] = games["MATCHUP"].str.contains("vs.")

    # Determine existing game IDs from provided set or input_df (backward compatibility)
    if existing_game_ids is not None:
        existing_ids = existing_game_ids

    else:
        existing_ids = set()

    game_ids = set(games["GAME_ID"]) - existing_ids

    if not game_ids:
        print(
            f"No new games found for season {season_nullable}. Skipping data fetch..."
        )
        return None, None, limit_reached

    all_player_stats, all_team_stats = [], []
    fetched_counter = 0

    for game_id in tqdm(game_ids, desc="Fetching NBA Game Data"):
        time.sleep(random.uniform(0.1, 0.5))  # Avoid rate limiting

        box_score_traditional, box_score_advanced, limit_reached = fetch_box_score_data(
            game_id, n_tries
        )

        if not box_score_traditional or not box_score_advanced:
            continue

        player_trad = box_score_traditional.get_data_frames()[0].fillna("")
        team_trad = box_score_traditional.get_data_frames()[2].fillna("")
        player_adv = box_score_advanced.get_data_frames()[0].fillna("")
        team_adv = box_score_advanced.get_data_frames()[1].fillna("")

        # Apply V3 to V2 mappings to maintain consistency with historical data
        player_trad.rename(columns=V3_TO_V2_TRADITIONAL_MAP, inplace=True)
        team_trad.rename(columns=V3_TO_V2_TRADITIONAL_MAP, inplace=True)
        player_adv.rename(columns=V3_TO_V2_ADVANCED_PLAYER_MAP, inplace=True)
        team_adv.rename(columns=V3_TO_V2_ADVANCED_TEAM_MAP, inplace=True)

        player_stats, team_stats = merge_stats(
            player_trad, player_adv, team_trad, team_adv, game_id
        )
        all_player_stats.append(player_stats)
        all_team_stats.append(team_stats)

        fetched_counter += 1
        if fetched_counter == 299:
            print("Waiting 5 secs to avoid rate limit...")
            time.sleep(5)
            reset_nba_http_session()
            limit_reached = True

        if limit_reached:
            break

    player_stats_df = (
        pd.concat(all_player_stats, ignore_index=True)
        if all_player_stats
        else pd.DataFrame()
    )
    team_stats_df = (
        pd.concat(all_team_stats, ignore_index=True)
        if all_team_stats
        else pd.DataFrame()
    )
    team_stats_df.rename(columns={"TO": "TOV"}, inplace=True)

    # Extract season year from season_nullable (first 4 digits)
    season_year = season_nullable[:4]
    if team_stats_df.empty:
        print(f"No team stats fetched for season {season_nullable}.")
        return None, None, limit_reached
    merged_games = pd.merge(
        games, team_stats_df, on=["GAME_ID", "TEAM_ID"], suffixes=("", "_drop")
    )
    merged_games = merged_games.loc[:, ~merged_games.columns.str.endswith("_drop")]
    merged_games["SEASON_YEAR"] = season_year

    # Remove duplicates even if no input_df (safety measure)
    merged_games = merged_games.drop_duplicates(
        subset=["GAME_ID", "TEAM_ID", "SEASON_ID"], keep="last"
    )

    # Remove duplicates even if no input_player_stats (safety measure)
    if not player_stats_df.empty:
        player_stats_df = player_stats_df.drop_duplicates(
            subset=["GAME_ID", "PLAYER_ID", "TEAM_ID"], keep="last"
        )

    # Add season_year to player_stats_df if it's not empty
    if not player_stats_df.empty:
        player_stats_df["SEASON_YEAR"] = season_year

    return merged_games, player_stats_df, limit_reached
