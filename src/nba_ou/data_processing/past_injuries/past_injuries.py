from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd

# Constants for top player statistics
N_TOP_PLAYERS_NON_INJURED = 6
N_TOP_PLAYERS_INJURED = 4


def get_injured_players_dict(df_injuries, df_players=None):
    """
    Build a dictionary: injured_dict[game_id][team_id] -> list of injured players for that game/team.

    Injured players are collected from:
    - `df_injuries` (inactive/injury feed)
    - `df_players` comments when injury wording appears (e.g. "DND - Injury/Illness")

    Args:
        df_injuries (pd.DataFrame): Injury data with GAME_ID, TEAM_ID, PLAYER_ID
        df_players (pd.DataFrame, optional): Player boxscore data with GAME_ID, TEAM_ID,
            PLAYER_ID and COMMENT/COMMENTS column

    Returns:
        dict: Nested dictionary mapping game_id -> team_id -> list of injured player_ids
    """
    injured_dict = defaultdict(lambda: defaultdict(set))

    # Source 1: official injuries table
    if (
        df_injuries is not None
        and not df_injuries.empty
        and {"GAME_ID", "TEAM_ID", "PLAYER_ID"}.issubset(df_injuries.columns)
    ):
        valid_injuries = df_injuries.loc[
            df_injuries["GAME_ID"].notna()
            & df_injuries["TEAM_ID"].notna()
            & df_injuries["PLAYER_ID"].notna(),
            ["GAME_ID", "TEAM_ID", "PLAYER_ID"],
        ].drop_duplicates()

        for game_id, team_id, player_id in valid_injuries.itertuples(index=False):
            injured_dict[game_id][team_id].add(player_id)

    # Source 2: player comment field includes injury text
    if (
        df_players is not None
        and not df_players.empty
        and {"GAME_ID", "TEAM_ID", "PLAYER_ID"}.issubset(df_players.columns)
    ):
        comment_col = None
        for candidate in ["COMMENT", "COMMENTS", "comment", "comments"]:
            if candidate in df_players.columns:
                comment_col = candidate
                break

        if comment_col is not None:
            injury_mask = (
                df_players[comment_col]
                .fillna("")
                .astype(str)
                .str.contains(r"injur|injry", case=False, regex=True)
            )
            valid_comment_injuries = df_players.loc[
                injury_mask
                & df_players["GAME_ID"].notna()
                & df_players["TEAM_ID"].notna()
                & df_players["PLAYER_ID"].notna(),
                ["GAME_ID", "TEAM_ID", "PLAYER_ID"],
            ].drop_duplicates()

            for game_id, team_id, player_id in valid_comment_injuries.itertuples(
                index=False
            ):
                injured_dict[game_id][team_id].add(player_id)

    injured_dict = {
        game_id: {team_id: list(player_ids) for team_id, player_ids in team_map.items()}
        for game_id, team_map in injured_dict.items()
    }

    return injured_dict


def create_player_lookup(df_players, injured_dict=None):
    """
    Precompute all necessary indexes for fast player lookups.
    Returns a function that can be called with (season_id, team_id, date_to_filter)
    to get the same result as get_players_for_team_in_season but much faster.

    Args:
        df_players (pd.DataFrame): Player statistics DataFrame (already sorted by PLAYER_ID, GAME_DATE)

    Returns:
        callable: A lookup function with signature
            (season_id, team_id, date_to_filter, game_id=None) -> pd.DataFrame
    """
    # Ensure GAME_DATE is datetime
    df_players = df_players.copy()
    df_players["GAME_DATE"] = pd.to_datetime(df_players["GAME_DATE"], errors="coerce")

    # Rows returned by this lookup are roster/stat-state evidence, not a list of
    # players who appeared in the target game.  Keeping MIN=0 box-score rows is
    # load-bearing: a DNP who is not on the inactive list was available, and
    # dropping them here makes roster membership depend on the rotation the coach
    # actually used after tip-off.  The cumulative columns attached to every row
    # are shifted by precompute_cumulative_avg_stat, so current-game PTS/MIN never
    # enter the value returned for that game.
    #
    # Synthetic scheduled-game placeholders (MIN=None) obey the same contract and
    # remain included naturally.  Rows without a player id cannot contribute to a
    # roster and are the only ones discarded.
    df_valid = df_players[df_players["PLAYER_ID"].notna()].copy()

    # Sort for season_team_groups indexing
    df_valid.sort_values(
        ["SEASON_ID", "TEAM_ID", "PLAYER_ID", "GAME_DATE"], inplace=True
    )

    # Returning a reported injured player may require statistics from their prior
    # team (for example, their first game after a trade). Keep a whole-season
    # slice as well as the team membership indexes built below.
    season_groups = {
        season_id: group_df for season_id, group_df in df_valid.groupby("SEASON_ID")
    }
    season_year_groups = {
        int(season_year): group_df
        for season_year, group_df in df_valid.groupby("SEASON_YEAR")
    }

    # For tracking player movements, use ALL data (including MIN=0 games)
    # because a player's last game might be a DNP (MIN=0) but they're still on the team
    df_all = df_players.copy()
    df_all.sort_values(["SEASON_ID", "PLAYER_ID", "GAME_DATE"], inplace=True)
    df_all_by_year = df_players.copy()
    df_all_by_year.sort_values(["SEASON_YEAR", "PLAYER_ID", "GAME_DATE"], inplace=True)

    # Pre-compute each player's assignment history. Real box-score rows are known
    # only after the game and therefore assign a team starting on the following
    # game. Scheduled placeholders have MIN=None and are generated from already
    # known history, so they remain valid same-day pregame evidence.
    player_timeline = defaultdict(
        list
    )  # (season_id, player_id) -> [(date, team_id), ...]
    for (season_id, player_id), grp in df_all.groupby(["SEASON_ID", "PLAYER_ID"]):
        # Sorted by date (chronologically across all teams)
        dates = grp["GAME_DATE"].values
        teams = grp["TEAM_ID"].values
        placeholders = grp["MIN"].isna().values
        player_timeline[(season_id, str(player_id))] = list(
            zip(dates, teams, placeholders, strict=True)
        )

    player_timeline_by_season_year = defaultdict(list)
    for (season_year, player_id), grp in df_all_by_year.groupby(
        ["SEASON_YEAR", "PLAYER_ID"]
    ):
        dates = grp["GAME_DATE"].values
        teams = grp["TEAM_ID"].values
        placeholders = grp["MIN"].isna().values
        player_timeline_by_season_year[(int(season_year), str(player_id))] = list(
            zip(dates, teams, placeholders, strict=True)
        )

    # Get unique players per (season, team) - use ALL data for membership tracking
    players_by_season_team = {}
    for (season_id, team_id), group_df in df_all.groupby(["SEASON_ID", "TEAM_ID"]):
        players_by_season_team[(season_id, str(team_id))] = set(
            group_df["PLAYER_ID"].astype(str).unique()
        )

    players_by_season_year_team = {}
    for (season_year, team_id), group_df in df_all_by_year.groupby(
        ["SEASON_YEAR", "TEAM_ID"]
    ):
        players_by_season_year_team[(int(season_year), str(team_id))] = set(
            group_df["PLAYER_ID"].astype(str).unique()
        )

    empty_df = pd.DataFrame(columns=df_players.columns)

    injured_team_by_game_player = {}
    if injured_dict:
        for game_id, team_map in injured_dict.items():
            game_key = str(game_id)
            player_to_teams = defaultdict(set)
            for listed_team_id, player_ids in team_map.items():
                team_key = str(listed_team_id)
                for player_id in player_ids:
                    if pd.isna(player_id):
                        continue
                    player_to_teams[str(player_id)].add(team_key)
            injured_team_by_game_player[game_key] = player_to_teams

    def _season_year_from_season_id(season_id):
        try:
            return int(str(season_id)[-4:])
        except (TypeError, ValueError):
            return None

    def _lookup_by_key(
        bucket_key,
        team_id,
        date_to_filter,
        game_id,
        players_by_bucket_team,
        player_timeline_by_bucket,
        bucket_groups,
    ):
        team_key = str(team_id)
        game_injury_map = (
            injured_team_by_game_player.get(str(game_id), {})
            if game_id is not None
            else {}
        )

        # Prior box scores provide the ordinary roster candidates. A same-game
        # injury report is authoritative pregame evidence and can add a player who
        # has not yet logged a box score for this team.
        candidate_players = set(
            players_by_bucket_team.get((bucket_key, team_key), set())
        )
        candidate_players.update(
            player_id
            for player_id, listed_teams in game_injury_map.items()
            if team_key in listed_teams
        )
        if not candidate_players:
            return empty_df

        # Convert date_to_filter to numpy datetime64 for comparison
        date_np = np.datetime64(date_to_filter)
        # A current injury report overrides older assignment evidence. Otherwise,
        # use the last box-score assignment strictly before the target game. This
        # mirrors what is available for a real pregame prediction.
        valid_players = []
        for player_id in candidate_players:
            listed_teams = game_injury_map.get(str(player_id), set())
            if listed_teams:
                if listed_teams == {team_key}:
                    valid_players.append(str(player_id))
                continue

            timeline = player_timeline_by_bucket.get((bucket_key, player_id), [])
            if not timeline:
                continue

            last_team = None
            for game_date, game_team, is_scheduled_placeholder in timeline:
                if game_date < date_np or (
                    game_date == date_np and is_scheduled_placeholder
                ):
                    last_team = str(game_team)
                else:
                    break

            if last_team == team_key:
                valid_players.append(str(player_id))

        if not valid_players:
            return empty_df

        df_bucket = bucket_groups.get(bucket_key)
        if df_bucket is None or df_bucket.empty:
            return empty_df

        valid_players_set = set(valid_players)

        # Keep a player's history across teams so a same-game injury assignment
        # after a trade can still use their prior, already known statistics.
        mask = df_bucket["PLAYER_ID"].astype(str).isin(valid_players_set) & (
            df_bucket["GAME_DATE"] <= date_to_filter
        )
        result = df_bucket[mask]

        return result

    def lookup(season_id, team_id, date_to_filter, game_id=None):
        """
        Fast lookup for players on a team in a season before a given date.
        """
        result = _lookup_by_key(
            season_id,
            team_id,
            date_to_filter,
            game_id,
            players_by_season_team,
            player_timeline,
            season_groups,
        )
        if not result.empty:
            return result

        season_year = _season_year_from_season_id(season_id)
        if season_year is None:
            return empty_df

        return _lookup_by_key(
            season_year,
            team_id,
            date_to_filter,
            game_id,
            players_by_season_year_team,
            player_timeline_by_season_year,
            season_year_groups,
        )

    return lookup


def get_players_for_team_in_season(df_players, season_id, team_id, date_to_filter):
    """
    Returns rows from df_players belonging to (season_id, team_id),
    only for players who had not left by date_to_filter (based on last game).

    NOTE: For batch processing, use create_player_lookup() instead for much better performance.

    Args:
        df_players (pd.DataFrame): Player statistics DataFrame
        season_id (str): Season identifier
        team_id (str): Team identifier
        date_to_filter (datetime): Date to filter by

    Returns:
        pd.DataFrame: Filtered player data for the team in the season
    """
    # Filter same season
    df_season = df_players[df_players["SEASON_ID"] == season_id].copy()
    if df_season.empty:
        return pd.DataFrame(columns=df_players.columns)

    # Only games BEFORE this date
    df_season = df_season[df_season["GAME_DATE"] < date_to_filter]

    # Only consider players who played for this team at least once
    df_with_target_team = df_season[df_season["TEAM_ID"] == team_id]
    if df_with_target_team.empty:
        return pd.DataFrame(columns=df_players.columns)

    # Players who appeared for that team
    possible_ids = set(df_with_target_team["PLAYER_ID"].unique())
    df_season = df_season[df_season["PLAYER_ID"].isin(possible_ids)]

    # Sort by date so we can see each player's last appearance
    df_season.sort_values(["PLAYER_ID", "GAME_DATE"], inplace=True)

    # For each player, take the last row to see final team
    df_last_game = df_season.groupby("PLAYER_ID", as_index=False).tail(1)
    final_player_ids = df_last_game.loc[
        df_last_game["TEAM_ID"] == team_id, "PLAYER_ID"
    ].unique()

    # Return the relevant rows for these players who truly remain on the team
    df_result = df_players[
        (df_players["SEASON_ID"] == season_id)
        & (df_players["TEAM_ID"] == team_id)
        & (df_players["PLAYER_ID"].isin(final_player_ids))
    ].copy()

    # Filter out players who have not played
    df_result = df_result[df_result["MIN"] > 0]
    df_result = df_result[df_result["GAME_DATE"] <= date_to_filter]

    # Drop rows with NaN points
    df_result = df_result.dropna(subset=["PTS"])

    return df_result


def _season_year_from_value(season_value):
    if pd.isna(season_value):
        return None
    # Prefer direct numeric season-year values when available.
    try:
        season_int = int(season_value)
        if 1900 <= season_int <= 2200:
            return season_int
    except (TypeError, ValueError):
        pass

    digits = "".join(ch for ch in str(season_value) if ch.isdigit())
    if len(digits) < 4:
        return None
    try:
        return int(digits[-4:])
    except ValueError:
        return None


def create_injury_streak_lookup(df_team, injured_dict, max_seasons_back=2):
    """
    Build a lookup for consecutive injured-game streaks.

    Returns a callable:
        lookup(game_id, team_id, player_id) -> int

    Streak is counted for consecutive team games up to and including `game_id`.
    Search is limited to the current + previous `max_seasons_back - 1` seasons.
    """
    if df_team is None or df_team.empty:
        return lambda game_id, team_id, player_id: 0

    season_col = "SEASON_YEAR" if "SEASON_YEAR" in df_team.columns else "SEASON_ID"
    df_games = df_team[["GAME_ID", "TEAM_ID", "GAME_DATE", season_col]].copy()
    df_games["GAME_DATE"] = pd.to_datetime(df_games["GAME_DATE"], errors="coerce")
    df_games = df_games.dropna(subset=["GAME_ID", "TEAM_ID", "GAME_DATE"])
    df_games = df_games.drop_duplicates(subset=["GAME_ID", "TEAM_ID"], keep="first")

    team_games = {}
    game_pos_by_team = {}
    season_year_by_team_game = {}

    for team_id, grp in df_games.groupby("TEAM_ID"):
        team_key = str(team_id)
        grp_sorted = grp.sort_values(["GAME_DATE", "GAME_ID"], kind="mergesort")
        games_list = []
        pos_map = {}
        season_map = {}

        for idx, row in enumerate(grp_sorted.itertuples(index=False)):
            game_key = str(row.GAME_ID)
            games_list.append(game_key)
            pos_map[game_key] = idx
            season_map[game_key] = _season_year_from_value(getattr(row, season_col))

        team_games[team_key] = games_list
        game_pos_by_team[team_key] = pos_map
        season_year_by_team_game[team_key] = season_map

    injured_sets = {}
    if injured_dict:
        for game_id, team_map in injured_dict.items():
            game_key = str(game_id)
            per_team = {}
            for team_id, player_ids in team_map.items():
                team_key = str(team_id)
                per_team[team_key] = {
                    str(pid)
                    for pid in player_ids
                    if not pd.isna(pid) and str(pid) not in {"", "0", "None"}
                }
            injured_sets[game_key] = per_team

    @lru_cache(maxsize=300_000)
    def lookup(game_id, team_id, player_id):
        if pd.isna(game_id) or pd.isna(team_id) or pd.isna(player_id):
            return 0

        game_key = str(game_id)
        team_key = str(team_id)
        player_key = str(player_id)

        games_list = team_games.get(team_key)
        if not games_list:
            return 0

        pos_map = game_pos_by_team.get(team_key, {})
        current_pos = pos_map.get(game_key)
        if current_pos is None:
            return 0

        current_season_year = season_year_by_team_game.get(team_key, {}).get(game_key)
        min_allowed_season_year = None
        if current_season_year is not None:
            min_allowed_season_year = current_season_year - max(1, max_seasons_back) + 1

        streak = 0
        for idx in range(current_pos, -1, -1):
            hist_game_key = games_list[idx]
            if min_allowed_season_year is not None:
                hist_season_year = season_year_by_team_game.get(team_key, {}).get(
                    hist_game_key
                )
                if (
                    hist_season_year is not None
                    and hist_season_year < min_allowed_season_year
                ):
                    break

            injured_for_team = injured_sets.get(hist_game_key, {}).get(team_key, set())
            if player_key in injured_for_team:
                streak += 1
            else:
                break

        return streak

    return lookup
