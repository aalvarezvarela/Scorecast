"""``create_player_lookup`` before its per-player row index, verbatim.

Reference for tests/test_player_lookup_speedups.py only.
"""

from collections import defaultdict

import numpy as np
import pandas as pd


def create_player_lookup_before(df_players, injured_dict=None):
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
