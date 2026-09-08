import re

import pandas as pd
from tqdm import tqdm

from nba_ou.data_processing.past_injuries.past_injuries import (
    N_TOP_PLAYERS_INJURED,
    N_TOP_PLAYERS_NON_INJURED,
    create_injury_streak_lookup,
    create_player_lookup,
    get_injured_players_dict,
)
from nba_ou.data_processing.players.players_statistics import (
    get_top_n_averages_with_names,
    precompute_cumulative_avg_stat,
)
from nba_ou.utils.general_utils import _with_before_suffix

#: The top-N player columns come in three flavours per statistic: the value
#: (``TOP1_PLAYER_PTS_BEFORE``), the player's id (``TOP1_PLAYER_ID_PTS_BEFORE``)
#: and their name (``TOP1_PLAYER_NAME_PTS_BEFORE``). Only the first is a feature.
#:
#: The id and name are bookkeeping: they say *who* the top scorer was, which is
#: how ``add_top3_availability_effect_features_for_columns`` locates the player,
#: not a quantity a model can learn from. An id is worse than useless as a
#: numeric feature -- player 1610612747 is not "greater than" player 201939, so
#: every split on it is arbitrary -- and a name is a string that only survives to
#: be discarded later by ``clean_df_for_training``'s string-column pass.
#:
#: They are dropped at the very end of the pipeline, after the availability-effect
#: features have consumed the id columns.
PLAYER_IDENTIFIER_PATTERN = re.compile(r"^TOP\d+_(?:INJURED_)?PLAYER_(?:ID|NAME)_")


def is_player_identifier_column(column: str) -> bool:
    """True for a top-N player id/name bookkeeping column."""
    return bool(PLAYER_IDENTIFIER_PATTERN.match(column))


def drop_player_identifier_columns(df: pd.DataFrame, *, verbose: bool = False):
    """Remove the top-N player id and name columns from a training frame.

    Must run *after* ``add_top3_availability_effect_features_for_columns``,
    which reads the id columns to find each team's leading players.
    """
    identifiers = [c for c in df.columns if is_player_identifier_column(c)]
    if not identifiers:
        return df
    if verbose:
        print(f"Dropping {len(identifiers)} player id/name bookkeeping columns")
    return df.drop(columns=identifiers)


def _injured_player_ids_for_team(injured_dict, game_id, team_id) -> set[str]:
    """Return canonical player ids from either historical or scheduled reports."""
    game_map = next(
        (value for key, value in injured_dict.items() if str(key) == str(game_id)),
        {},
    )
    player_ids = next(
        (value for key, value in game_map.items() if str(key) == str(team_id)),
        [],
    )
    return {str(player_id) for player_id in player_ids if pd.notna(player_id)}


def _parse_minutes_series(min_series: pd.Series) -> pd.Series:
    def parse_value(value) -> float:
        if pd.isna(value) or value == "":
            return 0.0
        value_str = str(value)
        if ":" in value_str:
            parts = value_str.split(":")
            if len(parts) == 2:
                minutes_part = parts[0].strip()
                seconds_part = parts[1].strip()
                try:
                    minutes_val = float(minutes_part)
                except ValueError:
                    return 0.0
                if seconds_part.isdigit():
                    return minutes_val + int(seconds_part) / 60.0
            return 0.0
        try:
            return float(value_str)
        except ValueError:
            return 0.0

    return min_series.apply(parse_value)


def clear_player_statistics(df_players, df_team):
    """
    Process player statistics and prepare for training.

    This function handles:
    - Merging player data with game dates from team data
    - Converting player minutes from MM:SS format to decimal
    - Cleaning and deduplicating player data

    Args:
        df_players (pd.DataFrame): Player statistics DataFrame
        df (pd.DataFrame): Processed team DataFrame with GAME_ID, GAME_DATE, SEASON_ID

    Returns:
        pd.DataFrame: Processed player DataFrame
    """
    desired_cols = ["GAME_ID", "GAME_DATE", "SEASON_ID", "SEASON_YEAR"]
    merge_cols = ["GAME_ID"] + [
        c
        for c in desired_cols[1:]
        if c in df_team.columns and c not in df_players.columns
    ]
    df_players = df_players.merge(
        df_team[merge_cols].drop_duplicates(subset=["GAME_ID"]),
        on="GAME_ID",
        how="left",
    )
    df_players = df_players.dropna(subset=["GAME_DATE"])

    df_players["GAME_DATE"] = pd.to_datetime(df_players["GAME_DATE"], format="%Y-%m-%d")
    df_players["MIN"] = _parse_minutes_series(df_players["MIN"]).round(3).fillna(0)

    df_players = df_players.drop_duplicates(keep="first")

    return df_players


def _build_bench_stats_lookup(df_players):
    """
    Precompute per-player bench stats from valid games (MIN > 2).

    Returns a closure that, for a given player at a given date, provides
    their average minutes, PTS per minute, and PACE_PER40 from their
    last 5 valid games before that date.

    Args:
        df_players (pd.DataFrame): Full player boxscore history with
            PLAYER_ID, GAME_DATE, MIN, PTS, and optionally PACE_PER40.

    Returns:
        function: bench_lookup(player_id, game_date) ->
            (avg_min, avg_pts_per_min, avg_pace_per40) or None
    """
    df = df_players.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce").fillna(0)
    df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce").fillna(0)

    has_pace = "PACE_PER40" in df.columns
    if has_pace:
        df["PACE_PER40"] = pd.to_numeric(df["PACE_PER40"], errors="coerce").fillna(0)
    else:
        df["PACE_PER40"] = 0.0

    # Filter to valid games (MIN > 2) and compute PTS per minute
    valid = df[df["MIN"] > 2].copy()
    valid["PTS_PER_MIN"] = valid["PTS"] / valid["MIN"]
    valid = valid.sort_values(["PLAYER_ID", "GAME_DATE"])

    # Build per-player sorted history: (game_date, min, pts_per_min, pace_per40)
    player_history = {}
    cols = ["GAME_DATE", "MIN", "PTS_PER_MIN", "PACE_PER40"]
    for pid, grp in valid.groupby("PLAYER_ID"):
        player_history[pid] = list(grp[cols].itertuples(index=False, name=None))

    def bench_lookup(player_id, game_date):
        """Returns (avg_min, avg_pts_per_min, avg_pace_per40) or None."""
        hist = player_history.get(player_id)
        if not hist:
            return None
        prior = [h for h in hist if h[0] < game_date]
        if not prior:
            return None
        last5 = prior[-5:]
        avg_min = sum(h[1] for h in last5) / len(last5)
        avg_pts_pm = sum(h[2] for h in last5) / len(last5)
        avg_pace = sum(h[3] for h in last5) / len(last5)
        return (avg_min, avg_pts_pm, avg_pace)

    return bench_lookup


def _build_prior_roster_lookup(df_players):
    """Infer a roster from assignments strictly before the target game.

    This fallback is mainly needed at season openers, where the season-scoped
    lookup has no earlier assignment. Player rows from the target game may carry
    already-shifted statistics, but they never decide who belongs to the team.
    """
    required = {"PLAYER_ID", "TEAM_ID", "GAME_DATE"}
    if not required.issubset(df_players.columns):
        return lambda team_id, game_date: None

    players = df_players.copy()
    players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"], errors="coerce")
    players = players.dropna(subset=list(required)).sort_values(
        ["PLAYER_ID", "GAME_DATE"], kind="mergesort"
    )

    def prior_roster_lookup(team_id, game_date):
        prior = players[players["GAME_DATE"] < game_date]
        if prior.empty:
            return None
        last_assignment = prior.groupby("PLAYER_ID", as_index=False).tail(1)
        roster_ids = set(
            last_assignment.loc[
                last_assignment["TEAM_ID"].astype(str).eq(str(team_id)), "PLAYER_ID"
            ].astype(str)
        )
        if not roster_ids:
            return None
        return players[
            players["PLAYER_ID"].astype(str).isin(roster_ids)
            & players["GAME_DATE"].le(game_date)
        ]

    return prior_roster_lookup


def add_player_history_features(
    df_team, df_players, df_injuries, stat_cols=["PTS"], injury_dict_scheduled=None
):
    """
    Main function to attach top player statistics and injured player stats to team data.

    - Precomputes cumulative avg of `stat_cols` in df_players
    - Builds an injured_dict from df_injuries
    - For each row in df_team, finds top-n average of `stat_cols` among non-injured
      and injured players who belong to that team on that date

    Args:
        df_team (pd.DataFrame): Team-level game data
        df_players (pd.DataFrame): Player-level boxscore data
        df_injuries (pd.DataFrame): Injury data per (GAME_ID, TEAM_ID, PLAYER_ID)
        stat_cols (list or str): Statistics columns to compute averages for
        injury_dict_scheduled (dict, optional): Dictionary of scheduled injury data

    Returns:
        pd.DataFrame: Updated df_team with extra columns for top players and injured players
        dict: Updated injured players dictionary
    """
    # Build injuries lookup
    injured_dict = get_injured_players_dict(df_injuries, df_players=df_players)

    if injury_dict_scheduled:
        injured_dict.update(injury_dict_scheduled)

    if isinstance(stat_cols, str):
        stat_cols = [stat_cols]

    # Collect all column names first to avoid fragmentation
    all_new_cols = []
    for stat_col in stat_cols:
        # 1) Precompute cumulative averages for the chosen stat
        df_players = precompute_cumulative_avg_stat(df_players, stat_col=stat_col)

        # 2) Dynamically name new columns based on `stat_col`
        new_cols = [
            # Top 3 non-injured player columns
            *[
                f"TOP{i}_PLAYER_ID_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_NON_INJURED + 1)
            ],
            *[
                f"TOP{i}_PLAYER_NAME_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_NON_INJURED + 1)
            ],
            *[
                f"TOP{i}_PLAYER_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_NON_INJURED + 1)
            ],
            # Top 3 injured player columns
            *[
                f"TOP{i}_INJURED_PLAYER_ID_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_INJURED + 1)
            ],
            *[
                f"TOP{i}_INJURED_PLAYER_NAME_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_INJURED + 1)
            ],
            *[
                f"TOP{i}_INJURED_PLAYER_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_INJURED + 1)
            ],
            # Streak columns only for PTS to avoid repetition
            *(
                [
                    f"TOP{i}_INJURED_STREAK_{stat_col}"
                    for i in range(1, N_TOP_PLAYERS_INJURED + 1)
                ]
                if stat_col == "PTS"
                else []
            ),
            f"AVG_INJURED_{stat_col}",
            # Aggregation over the INJURED set only. That set is resolved from
            # each player's last game strictly BEFORE this one (the
            # ``injured=True`` branch of get_top_n_averages_with_names), so it
            # carries no information from tonight's box score. Its non-injured
            # twin did, and is gone -- see below.
            f"TOTAL_INJURED_PLAYER_{stat_col}",
            # Player count columns only for PTS to avoid repetition
            *(["N_INJURED_PLAYERS"] if stat_col == "PTS" else []),
        ]
        all_new_cols.extend([_with_before_suffix(c) for c in new_cols])

    # Bench player columns (stat-independent, added once)
    bench_cols = [
        "BENCH_AVG_PTS_PER_MIN",
        "BENCH_MAX_PTS_PER_MIN",
        "BENCH_AVG_PACE_PER40",
        "BENCH_MAX_PACE_PER40",
        "N_BENCH_PLAYERS",
    ]
    all_new_cols.extend([_with_before_suffix(c) for c in bench_cols])

    # Create all columns at once to avoid fragmentation
    new_cols_df = pd.DataFrame(None, index=df_team.index, columns=all_new_cols)
    df_team = pd.concat([df_team, new_cols_df], axis=1)

    # Sort df_players once before the loop for optimal performance
    df_players = df_players.copy()
    df_players["GAME_DATE"] = pd.to_datetime(df_players["GAME_DATE"], errors="coerce")
    df_players.sort_values(["PLAYER_ID", "GAME_DATE"], kind="mergesort", inplace=True)

    # Create optimized lookup function (precomputes indexes once)
    player_lookup = create_player_lookup(df_players, injured_dict=injured_dict)
    injury_streak_lookup = create_injury_streak_lookup(
        df_team, injured_dict, max_seasons_back=2
    )
    bench_lookup = _build_bench_stats_lookup(df_players)
    prior_roster_lookup = _build_prior_roster_lookup(df_players)

    # 3) Iterate over each row in df_team (only needed columns for efficiency)
    cols_needed = ["GAME_ID", "TEAM_ID", "SEASON_ID", "GAME_DATE"]

    # Collect all updates in a list for bulk assignment
    updates_list = []

    for _, (game_id, team_id, season_id, game_date) in enumerate(
        tqdm(
            df_team[cols_needed].itertuples(index=False, name=None),
            total=len(df_team),
            desc="Adding players data",
        )
    ):
        # Resolve the roster without consulting who logged minutes in this game.
        # Availability is defined below solely as roster minus injured/inactive.
        df_roster = player_lookup(season_id, team_id, game_date, game_id=game_id)
        if df_roster.empty:
            # At a season opener, fall back to the latest assignments from the
            # prior season without consulting the opener's box-score membership.
            fallback_roster = prior_roster_lookup(team_id, game_date)
            if fallback_roster is not None:
                df_roster = fallback_roster
        if df_roster.empty:
            updates_list.append({})
            continue

        # Who is injured for this game/team?
        injured_players = _injured_player_ids_for_team(injured_dict, game_id, team_id)
        player_ids = df_roster["PLAYER_ID"].astype(str)

        # No target-game MIN rule: a rostered DNP remains available unless the
        # injury/inactive feed places them in the injured set.
        df_non_inj = df_roster[~player_ids.isin(injured_players)]
        df_inj = df_roster[player_ids.isin(injured_players)]

        row_update = {}

        # Bench player stats (available non-starters with 7-21 avg MIN)
        bench_players = []
        for pid in df_non_inj["PLAYER_ID"].unique():
            stats = bench_lookup(pid, game_date)
            if stats is not None:
                avg_min, pts_per_min, pace = stats
                if 7 <= avg_min <= 21:
                    bench_players.append((pts_per_min, pace))

        if bench_players:
            pts_pm_vals = [b[0] for b in bench_players]
            pace_vals = [b[1] for b in bench_players]
            row_update[_with_before_suffix("BENCH_AVG_PTS_PER_MIN")] = sum(
                pts_pm_vals
            ) / len(pts_pm_vals)
            row_update[_with_before_suffix("BENCH_MAX_PTS_PER_MIN")] = max(pts_pm_vals)
            row_update[_with_before_suffix("BENCH_AVG_PACE_PER40")] = sum(
                pace_vals
            ) / len(pace_vals)
            row_update[_with_before_suffix("BENCH_MAX_PACE_PER40")] = max(pace_vals)
        row_update[_with_before_suffix("N_BENCH_PLAYERS")] = len(bench_players)

        for stat_col in stat_cols:
            n_players_noninj = N_TOP_PLAYERS_NON_INJURED
            n_players_inj = N_TOP_PLAYERS_INJURED

            # Get ALL non-injured players for aggregation
            all_non_inj = get_top_n_averages_with_names(
                df_non_inj,
                date=game_date,
                stat_col=stat_col,
                injured=False,
                n_players=df_non_inj["PLAYER_ID"].nunique(),
            )
            # Get ALL injured players for aggregation
            all_inj = get_top_n_averages_with_names(
                df_inj,
                date=game_date,
                stat_col=stat_col,
                n_players=df_inj["PLAYER_ID"].nunique(),
                injured=True,
            )

            # Top 3 are the first N from the sorted lists
            topn_non_inj = all_non_inj[:n_players_noninj]
            topn_inj = all_inj[:n_players_inj]

            # Pad to required length with None for IDs/names, 0 for stats
            while len(topn_non_inj) < n_players_noninj:
                topn_non_inj.append((None, None, 0))
            while len(topn_inj) < n_players_inj:
                topn_inj.append((None, None, 0))

            # Store top 3 non-injured individual columns
            for i in range(n_players_noninj):
                row_update[_with_before_suffix(f"TOP{i + 1}_PLAYER_ID_{stat_col}")] = (
                    topn_non_inj[i][0]
                )
                row_update[
                    _with_before_suffix(f"TOP{i + 1}_PLAYER_NAME_{stat_col}")
                ] = topn_non_inj[i][1]
                row_update[_with_before_suffix(f"TOP{i + 1}_PLAYER_{stat_col}")] = (
                    topn_non_inj[i][2]
                )

            # Store top 3 injured individual columns + streaks (streaks only for PTS)
            for i in range(n_players_inj):
                row_update[
                    _with_before_suffix(f"TOP{i + 1}_INJURED_PLAYER_ID_{stat_col}")
                ] = topn_inj[i][0]
                row_update[
                    _with_before_suffix(f"TOP{i + 1}_INJURED_PLAYER_NAME_{stat_col}")
                ] = topn_inj[i][1]
                row_update[
                    _with_before_suffix(f"TOP{i + 1}_INJURED_PLAYER_{stat_col}")
                ] = topn_inj[i][2]
                if stat_col == "PTS":
                    injured_pid = topn_inj[i][0]
                    row_update[
                        _with_before_suffix(f"TOP{i + 1}_INJURED_STREAK_{stat_col}")
                    ] = (
                        injury_streak_lookup(game_id, team_id, injured_pid)
                        if injured_pid is not None
                        else 0
                    )

            # Average of top 3 injured players
            inj_values = [val for (_, _, val) in topn_inj if val != 0]
            row_update[_with_before_suffix(f"AVG_INJURED_{stat_col}")] = (
                sum(inj_values) / len(inj_values) if inj_values else 0
            )

            # Aggregation: sum of cum avg for ALL players (not just top 3)
            row_update[_with_before_suffix(f"TOTAL_INJURED_PLAYER_{stat_col}")] = sum(
                val for (_, _, val) in all_inj if val != 0
            )

            # NOT BUILT: TOTAL_NON_INJURED_PLAYER_<stat> and N_ACTIVE_PLAYERS.
            #
            # In archived datasets both aggregated over ``all_non_inj`` after it
            # had been reduced to THIS GAME'S ``MIN > 0`` box-score rows. Thus
            # ``len(all_non_inj)`` meant "how many players the coach actually used
            # tonight", a function of how the game went: it correlates +0.55 with
            # |HOME_MARGIN| (bench-emptying in blowouts) and is lower in overtime
            # games, which stay close and rotate short.
            #
            # The per-player VALUES are properly lagged, which is what made this
            # survive review for so long: only the membership of the set leaks.
            # A spread_error regressor given these 14 columns scored 66.7%
            # against the closing spread; without them, 52.4%.
            #
            # The individual TOP{i}_PLAYER_* and bench columns are retained: their
            # membership is now roster-minus-injured and their values are shifted
            # pre-game estimates.  These two broad aggregate families remain
            # absent so archived CSVs carrying their old, leaky meaning stay
            # distinguishable and can continue to be rejected centrally.
            #
            # Player counts only for PTS to avoid repetition across stat_cols
            if stat_col == "PTS":
                row_update[_with_before_suffix("N_INJURED_PLAYERS")] = len(all_inj)

        updates_list.append(row_update)

    # Apply all updates at once using a DataFrame
    updates_df = pd.DataFrame(updates_list, index=df_team.index)
    for col in updates_df.columns:
        df_team[col] = updates_df[col]

    streak_cols = [c for c in df_team.columns if "_INJURED_STREAK_" in c]
    if streak_cols:
        df_team[streak_cols] = (
            df_team[streak_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(int)
        )

    return df_team, injured_dict
