import re

import pandas as pd
from tqdm import tqdm

from nba_ou.data_processing.injury_status.report_state import (
    apply_report_out_overrides,
    nested_status_dict,
    union_membership_dicts,
)
from nba_ou.data_processing.past_injuries.past_injuries import (
    N_TOP_PLAYERS_INJURED,
    N_TOP_PLAYERS_NON_INJURED,
    N_TOP_PLAYERS_QUESTIONABLE,
    create_injury_streak_lookup,
    create_player_lookup,
    get_injured_players_dict,
)
from nba_ou.data_processing.players.feature_profile import ACTIVE_PROFILE
from nba_ou.data_processing.players.fresh_absence import (
    STREAK_STAT_COLS,
    add_fresh_absence_features,
)
from nba_ou.data_processing.players.players_statistics import (
    get_top_n_averages_with_names,
    precompute_cumulative_avg_stat,
)
from nba_ou.utils.general_utils import _with_before_suffix
from nba_ou.utils.row_cache import RowCache

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
PLAYER_IDENTIFIER_PATTERN = re.compile(
    r"^TOP\d+_(?:INJURED_|QUESTIONABLE_)?PLAYER_(?:ID|NAME)_"
)


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


#: Available roster players who logged minutes within this many days before the
#: game. Membership never reads the target game's box score.
AVAILABLE_ROSTER_RECENT_DAYS = 30
N_AVAILABLE_ROSTER_PLAYERS_COL = "N_AVAILABLE_ROSTER_PLAYERS"


def _index_injured_dict(injured_dict) -> dict[str, dict[str, set[str]]]:
    """String-keyed ``{game: {team: {player}}}`` for constant-time lookups."""
    index: dict[str, dict[str, set[str]]] = {}
    for game_id, team_map in (injured_dict or {}).items():
        game_bucket = index.setdefault(str(game_id), {})
        for team_id, player_ids in team_map.items():
            game_bucket.setdefault(str(team_id), set()).update(
                str(player_id) for player_id in player_ids if pd.notna(player_id)
            )
    return index


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


def _restrict_snapshot_membership(
    membership_dict: dict,
    snapshot_game_ids: set[str],
    report_out_overrides: dict | None,
    questionable_dict: dict,
    report_listed_players: dict | None,
) -> dict:
    """Keep only as-of report evidence for current-game roster additions."""
    reported = union_membership_dicts(
        nested_status_dict(report_out_overrides),
        questionable_dict,
        report_listed_players or {},
    )
    ids = {str(game) for game in snapshot_game_ids}
    out = {
        str(game): teams
        for game, teams in membership_dict.items()
        if str(game) not in ids
    }
    out.update({str(game): teams for game, teams in reported.items() if str(game) in ids})
    return out


def _game_membership_signatures(membership_dict: dict) -> dict[str, frozenset]:
    """``game -> {(team, player)}``, keyed the way ``create_player_lookup`` reads it."""
    signatures = {}
    for game_id, team_map in (membership_dict or {}).items():
        # A later duplicate string key replaces the earlier one, exactly as in
        # the lookup's ``injured_team_by_game_player``.
        signatures[str(game_id)] = frozenset(
            (str(team_id), str(player_id))
            for team_id, player_ids in team_map.items()
            for player_id in player_ids
            if not pd.isna(player_id)
        )
    return signatures


def add_player_history_features(
    df_team,
    df_players,
    df_injuries,
    stat_cols=("PTS",),
    injury_dict_scheduled=None,
    return_availability_dict: bool = False,
    report_out_overrides: dict[tuple[str, str], list[str]] | None = None,
    report_questionable_sets: dict[tuple[str, str], list[str]] | None = None,
    include_available_roster_count: bool = False,
    snapshot_game_ids: set[str] | None = None,
    snapshot_report_listed_players: dict | None = None,
    row_cache: RowCache | None = None,
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
        report_out_overrides (dict, optional): ``(game_id, team_id) -> players``
            from the last injury report before tip, for every team-game the
            report covers (see ``injury_status.report_state``). Those
            team-games take their out set from the report; every other
            team-game keeps the inactive-list/comment set. History -- streaks
            before this game and the returned availability map -- always uses
            realized absences, so a player who was listed but played is never
            recorded as having missed that game.
        report_questionable_sets (dict, optional): ``(game_id, team_id) ->
            players`` listed Questionable on that same report. They form a third
            availability group: not in the injured set, not in the available
            set, and given the same family of top-N columns as both. Omit it
            (the default) and the roster splits in two exactly as before.
        include_available_roster_count (bool): Emit
            ``N_AVAILABLE_ROSTER_PLAYERS_BEFORE``. Off by default for builds
            without injury-report features.
        snapshot_game_ids (set, optional): Current games whose roster
            supplementation may use only the as-of report. Settled injuries
            remain available as history for later games.
        snapshot_report_listed_players (dict, optional): All players named by
            the as-of report, including Probable and Available designations.
        row_cache (RowCache, optional): Shared across repeated calls on the
            same inputs that differ only in the report arguments. A row reads
            the report through its own game alone -- roster membership for that
            game, and the out and questionable sets for that team-game -- while
            everything else it touches (earlier games' realized absences,
            player history) is the same in every call. Rows whose game sees the
            same report sets as before are reused; the result is identical to
            building every row.

    Returns:
        pd.DataFrame: Updated df_team with extra columns for top players and injured players
        dict: Updated injured players dictionary
        dict, optional: Per-game available/injured roster classification. Returned
            only when ``return_availability_dict=True`` and kept separate from the
            injury-report dictionary.
    """
    # Build injuries lookup
    injured_dict = get_injured_players_dict(df_injuries, df_players=df_players)

    if injury_dict_scheduled:
        injured_dict.update(injury_dict_scheduled)

    # The third group, keyed the same way. Its members are removed from BOTH the
    # injured and the available frames below, so the three groups partition the
    # roster.
    questionable_dict = nested_status_dict(report_questionable_sets)

    # Pre-game out set for the game being built vs realized absences for games
    # already played. Without overrides the two are the same dict.
    if report_out_overrides:
        pregame_dict = apply_report_out_overrides(injured_dict, report_out_overrides)
        # Roster membership only: the report naming a player on a team is
        # authoritative evidence he is on it, whichever group he lands in.
        membership_dict = union_membership_dicts(
            pregame_dict, injured_dict, questionable_dict
        )
    else:
        pregame_dict = injured_dict
        membership_dict = injured_dict
    if snapshot_game_ids:
        # The normal union includes the settled inactive list and comments.
        # Those are valid history later, but future information for an earlier
        # snapshot of this game. Current-game roster supplementation can only
        # read players explicitly named by the as-of report.
        membership_dict = _restrict_snapshot_membership(
            membership_dict,
            snapshot_game_ids,
            report_out_overrides,
            questionable_dict,
            snapshot_report_listed_players,
        )
    pregame_index = _index_injured_dict(pregame_dict)
    realized_index = _index_injured_dict(injured_dict)
    questionable_index = {
        game_id: {team_id: set(players) for team_id, players in team_map.items()}
        for game_id, team_map in questionable_dict.items()
    }
    build_questionable_group = report_questionable_sets is not None

    if isinstance(stat_cols, str):
        stat_cols = [stat_cols]

    # Collect all column names first to avoid fragmentation
    all_new_cols = []
    for stat_col in stat_cols:
        # 1) Precompute cumulative averages for the chosen stat
        df_players = precompute_cumulative_avg_stat(df_players, stat_col=stat_col)

        # 2) Dynamically name new columns based on `stat_col`
        profile = ACTIVE_PROFILE
        new_cols = [
            # Top-N non-injured player columns. The id/name pair is bookkeeping
            # for the availability-effect features and is dropped at the end of
            # the pipeline; only the slots those features read are produced.
            *[
                f"TOP{i}_PLAYER_ID_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_NON_INJURED + 1)
                if profile.emits_identifier(stat_col, i)
            ],
            *[
                f"TOP{i}_PLAYER_NAME_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_NON_INJURED + 1)
                if profile.emits_identifier(stat_col, i)
            ],
            *[
                f"TOP{i}_PLAYER_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_NON_INJURED + 1)
                if profile.emits_value(stat_col, i, injured=False)
            ],
            # Top-N injured player columns
            *[
                f"TOP{i}_INJURED_PLAYER_ID_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_INJURED + 1)
                if profile.emits_identifier(stat_col, i)
            ],
            *[
                f"TOP{i}_INJURED_PLAYER_NAME_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_INJURED + 1)
                if profile.emits_identifier(stat_col, i)
            ],
            *[
                f"TOP{i}_INJURED_PLAYER_{stat_col}"
                for i in range(1, N_TOP_PLAYERS_INJURED + 1)
                if profile.emits_value(stat_col, i, injured=True)
            ],
            # Streak columns only for the statistics that rank a player's
            # importance (see STREAK_STAT_COLS); the rest would repeat the same
            # absence over a differently sorted top-N list.
            *(
                [
                    f"TOP{i}_INJURED_STREAK_{stat_col}"
                    for i in range(1, N_TOP_PLAYERS_INJURED + 1)
                ]
                if stat_col in STREAK_STAT_COLS
                else []
            ),
            *(
                [f"AVG_INJURED_{stat_col}"]
                if stat_col in profile.avg_injured_stats
                else []
            ),
            # Questionable players remain a separate group and follow the
            # same reduced profile as injured players.
            *(
                [
                    *[
                        f"TOP{i}_QUESTIONABLE_PLAYER_ID_{stat_col}"
                        for i in range(1, N_TOP_PLAYERS_QUESTIONABLE + 1)
                        if profile.emits_identifier(stat_col, i)
                    ],
                    *[
                        f"TOP{i}_QUESTIONABLE_PLAYER_NAME_{stat_col}"
                        for i in range(1, N_TOP_PLAYERS_QUESTIONABLE + 1)
                        if profile.emits_identifier(stat_col, i)
                    ],
                    *[
                        f"TOP{i}_QUESTIONABLE_PLAYER_{stat_col}"
                        for i in range(1, N_TOP_PLAYERS_QUESTIONABLE + 1)
                        if profile.emits_value(stat_col, i, injured=True)
                    ],
                    *(
                        [
                            f"TOP{i}_QUESTIONABLE_STREAK_{stat_col}"
                            for i in range(1, N_TOP_PLAYERS_QUESTIONABLE + 1)
                        ]
                        if stat_col in STREAK_STAT_COLS
                        else []
                    ),
                    *(
                        [f"AVG_QUESTIONABLE_{stat_col}"]
                        if stat_col in profile.avg_injured_stats
                        else []
                    ),
                    *(
                        [f"TOTAL_QUESTIONABLE_PLAYER_{stat_col}"]
                        if stat_col in profile.total_injured_stats
                        else []
                    ),
                    *(["N_QUESTIONABLE_PLAYERS"] if stat_col == "PTS" else []),
                ]
                if build_questionable_group
                else []
            ),
            # Aggregation over the INJURED set only. That set is resolved from
            # each player's last game strictly BEFORE this one (the
            # ``injured=True`` branch of get_top_n_averages_with_names), so it
            # carries no information from tonight's box score. Its non-injured
            # twin did, and is gone -- see below.
            #
            # Emitted only for counting statistics: summing a RATE over players
            # is not a quantity, and measurably just re-counts the injured
            # players (r = 0.98 with N_INJURED_PLAYERS). See feature_profile.py.
            *(
                [f"TOTAL_INJURED_PLAYER_{stat_col}"]
                if stat_col in profile.total_injured_stats
                else []
            ),
            # Player count columns only for PTS to avoid repetition
            *(["N_INJURED_PLAYERS"] if stat_col == "PTS" else []),
        ]
        all_new_cols.extend([_with_before_suffix(c) for c in new_cols])

    # Bench player columns (stat-independent, added once)
    if include_available_roster_count:
        all_new_cols.append(_with_before_suffix(N_AVAILABLE_ROSTER_PLAYERS_COL))
    all_new_cols.extend([_with_before_suffix(c) for c in ACTIVE_PROFILE.bench_cols])

    # Minutes-weighted rate aggregates, one per side per rate statistic. These
    # replace the per-slot rate columns: ranking rotation players by a rate
    # returns nearly the same value in every slot (r = 0.97 between slots 2 and
    # 3), while the weighted aggregate says what the per-slot columns were
    # reaching for and stays defined when a slot is empty.
    all_new_cols.extend(
        [
            _with_before_suffix(f"{group}_WEIGHTED_{stat_col}")
            for group in (
                ("ACTIVE", "INJURED", "QUESTIONABLE")
                if build_questionable_group
                else ("ACTIVE", "INJURED")
            )
            for stat_col in ACTIVE_PROFILE.weighted_rate_stats
        ]
    )

    # Create all columns at once to avoid fragmentation
    new_cols_df = pd.DataFrame(None, index=df_team.index, columns=all_new_cols)
    df_team = pd.concat([df_team, new_cols_df], axis=1)

    # Sort df_players once before the loop for optimal performance
    df_players = df_players.copy()
    df_players["GAME_DATE"] = pd.to_datetime(df_players["GAME_DATE"], errors="coerce")
    df_players.sort_values(["PLAYER_ID", "GAME_DATE"], kind="mergesort", inplace=True)

    # Create optimized lookup function (precomputes indexes once)
    player_lookup = create_player_lookup(df_players, injured_dict=membership_dict)
    injury_streak_lookup = create_injury_streak_lookup(
        df_team,
        injured_dict,
        max_seasons_back=2,
        current_game_injured_dict=pregame_dict if report_out_overrides else None,
    )
    # A player Questionable tonight is not in the current out set, so the injured
    # streak would break at this game and read 0 for him. Counting the current
    # game from the questionable set instead makes the column mean the same thing
    # as its injured twin: consecutive team games, this one included, on which
    # the player has been unavailable or at risk.
    questionable_streak_lookup = (
        create_injury_streak_lookup(
            df_team,
            injured_dict,
            max_seasons_back=2,
            current_game_injured_dict=questionable_dict,
        )
        if build_questionable_group
        else None
    )
    bench_lookup = _build_bench_stats_lookup(df_players)
    prior_roster_lookup = _build_prior_roster_lookup(df_players)

    # 3) Iterate over each row in df_team (only needed columns for efficiency)
    cols_needed = ["GAME_ID", "TEAM_ID", "SEASON_ID", "GAME_DATE"]

    # Collect all updates in a list for bulk assignment
    updates_list = []
    availability_dict = {}

    if row_cache is not None:
        row_cache.bind(
            (
                tuple(stat_cols),
                include_available_roster_count,
                build_questionable_group,
                len(df_team),
                len(df_players),
            )
        )
        membership_signatures = _game_membership_signatures(membership_dict)

    for _, (game_id, team_id, season_id, game_date) in enumerate(
        tqdm(
            df_team[cols_needed].itertuples(index=False, name=None),
            total=len(df_team),
            desc="Adding players data",
        )
    ):
        if row_cache is not None:
            # Everything the row reads that can differ between calls: its own
            # game's roster membership, out, questionable and realized sets.
            game_key, team_key = str(game_id), str(team_id)
            game_questionable = questionable_index.get(game_key, {})
            cache_key = (
                game_key,
                team_key,
                season_id,
                game_date,
                membership_signatures.get(game_key),
                frozenset(pregame_index.get(game_key, {}).get(team_key, ())),
                team_key in game_questionable,
                frozenset(game_questionable.get(team_key, ())),
                frozenset(realized_index.get(game_key, {}).get(team_key, ())),
            )
            cached = row_cache.rows.get(cache_key)
            if cached is not None:
                row_cache.hits += 1
                cached_update, cached_availability = cached
                updates_list.append(cached_update)
                if cached_availability is not None:
                    available, injured = cached_availability
                    availability_dict.setdefault(game_key, {})[team_key] = {
                        "available": list(available),
                        "injured": list(injured),
                    }
                continue
            row_cache.misses += 1

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
            if row_cache is not None:
                row_cache.rows[cache_key] = ({}, None)
            continue

        # Who is injured for this game/team? The injury sources already encode
        # the reason for the absence (inactive list, plus the DNP reason comment
        # filtered by UNAVAILABLE_COMMENT_PATTERN), and every reason they accept
        # was settled before tip-off. Target-game MIN takes no part in the split:
        # minutes cannot distinguish a late scratch from a coach's decision, and
        # reading them would make roster membership a function of the game being
        # predicted -- the failure documented in ``nba_ou.config.leakage``.
        injured_players = pregame_index.get(str(game_id), {}).get(str(team_id), set())
        # Questionable is its own group: neither subtracted from the team as an
        # absence nor counted among the available. A player listed both Out and
        # Questionable on one report cannot happen, but injured wins if it does.
        # A team-game the report does not cover has no questionable group at
        # all: "nobody Questionable" is unknown there, not zero, so its
        # questionable columns are left NaN (rule 4 of ``report_state``).
        questionable_covered = str(team_id) in questionable_index.get(str(game_id), {})
        questionable_players = questionable_index.get(str(game_id), {}).get(
            str(team_id), set()
        ) - set(injured_players)
        player_ids = df_roster["PLAYER_ID"].astype(str)

        df_non_inj = df_roster[
            ~player_ids.isin(injured_players) & ~player_ids.isin(questionable_players)
        ]
        df_inj = df_roster[player_ids.isin(injured_players)]
        df_questionable = df_roster[player_ids.isin(questionable_players)]

        # The history map records what actually happened, for later games'
        # present/absent comparisons -- never the pre-game report.
        realized_injured = realized_index.get(str(game_id), {}).get(str(team_id), set())
        roster_ids = set(player_ids)
        availability_dict.setdefault(str(game_id), {})[str(team_id)] = {
            "available": sorted(roster_ids - realized_injured),
            "injured": sorted(roster_ids & realized_injured),
        }

        row_update = {}

        if include_available_roster_count:
            recent = df_non_inj[
                (df_non_inj["GAME_DATE"] < game_date)
                & (
                    df_non_inj["GAME_DATE"]
                    >= game_date - pd.Timedelta(days=AVAILABLE_ROSTER_RECENT_DAYS)
                )
                & (pd.to_numeric(df_non_inj["MIN"], errors="coerce").fillna(0) > 0)
            ]
            row_update[_with_before_suffix(N_AVAILABLE_ROSTER_PLAYERS_COL)] = recent[
                "PLAYER_ID"
            ].nunique()

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
            bench_values = {
                "BENCH_AVG_PTS_PER_MIN": sum(pts_pm_vals) / len(pts_pm_vals),
                "BENCH_MAX_PTS_PER_MIN": max(pts_pm_vals),
                "BENCH_AVG_PACE_PER40": sum(pace_vals) / len(pace_vals),
                "BENCH_MAX_PACE_PER40": max(pace_vals),
            }
            for col, value in bench_values.items():
                if col in ACTIVE_PROFILE.bench_cols:
                    row_update[_with_before_suffix(col)] = value
        if "N_BENCH_PLAYERS" in ACTIVE_PROFILE.bench_cols:
            row_update[_with_before_suffix("N_BENCH_PLAYERS")] = len(bench_players)

        # Per-player values for this row, keyed by statistic, so the
        # minutes-weighted rate aggregates can pair a player's rate with their
        # own minutes after the per-statistic loop below.
        per_player_values: dict[str, dict] = {}

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

            # The questionable group, resolved exactly like the injured one:
            # ``injured=True`` reads each player's last game strictly BEFORE
            # this one, so membership and values are both independent of
            # tonight's box score. Using the available branch instead would pick
            # up a same-day row that exists only if the player ended up playing.
            all_questionable = (
                get_top_n_averages_with_names(
                    df_questionable,
                    date=game_date,
                    stat_col=stat_col,
                    n_players=df_questionable["PLAYER_ID"].nunique(),
                    injured=True,
                )
                if build_questionable_group and questionable_covered
                else []
            )

            # Top 3 are the first N from the sorted lists
            topn_non_inj = all_non_inj[:n_players_noninj]
            topn_inj = all_inj[:n_players_inj]
            topn_questionable = all_questionable[:N_TOP_PLAYERS_QUESTIONABLE]

            # Pad to required length with None for IDs/names, 0 for stats
            while len(topn_non_inj) < n_players_noninj:
                topn_non_inj.append((None, None, 0))
            while len(topn_inj) < n_players_inj:
                topn_inj.append((None, None, 0))
            while len(topn_questionable) < N_TOP_PLAYERS_QUESTIONABLE:
                topn_questionable.append((None, None, 0))

            # Keep every player's value for this statistic, so the weighted
            # aggregates below can pair a rate with the same player's minutes.
            per_player_values[stat_col] = {
                "active": {
                    pid: val for (pid, _, val) in all_non_inj if pid is not None
                },
                "injured": {pid: val for (pid, _, val) in all_inj if pid is not None},
                "questionable": {
                    pid: val for (pid, _, val) in all_questionable if pid is not None
                },
            }

            # Store top-N non-injured individual columns
            for i in range(n_players_noninj):
                slot = i + 1
                if ACTIVE_PROFILE.emits_identifier(stat_col, slot):
                    row_update[
                        _with_before_suffix(f"TOP{slot}_PLAYER_ID_{stat_col}")
                    ] = topn_non_inj[i][0]
                    row_update[
                        _with_before_suffix(f"TOP{slot}_PLAYER_NAME_{stat_col}")
                    ] = topn_non_inj[i][1]
                if ACTIVE_PROFILE.emits_value(stat_col, slot, injured=False):
                    row_update[_with_before_suffix(f"TOP{slot}_PLAYER_{stat_col}")] = (
                        topn_non_inj[i][2]
                    )

            # Store top-N injured individual columns + streaks
            for i in range(n_players_inj):
                slot = i + 1
                if ACTIVE_PROFILE.emits_identifier(stat_col, slot):
                    row_update[
                        _with_before_suffix(f"TOP{slot}_INJURED_PLAYER_ID_{stat_col}")
                    ] = topn_inj[i][0]
                    row_update[
                        _with_before_suffix(f"TOP{slot}_INJURED_PLAYER_NAME_{stat_col}")
                    ] = topn_inj[i][1]
                if ACTIVE_PROFILE.emits_value(stat_col, slot, injured=True):
                    row_update[
                        _with_before_suffix(f"TOP{slot}_INJURED_PLAYER_{stat_col}")
                    ] = topn_inj[i][2]
                if stat_col in STREAK_STAT_COLS:
                    injured_pid = topn_inj[i][0]
                    row_update[
                        _with_before_suffix(f"TOP{i + 1}_INJURED_STREAK_{stat_col}")
                    ] = (
                        injury_streak_lookup(game_id, team_id, injured_pid)
                        if injured_pid is not None
                        else 0
                    )

            # The questionable group's own slots, streaks and aggregates.
            if build_questionable_group and questionable_covered:
                for i in range(N_TOP_PLAYERS_QUESTIONABLE):
                    slot = i + 1
                    if ACTIVE_PROFILE.emits_identifier(stat_col, slot):
                        row_update[
                            _with_before_suffix(
                                f"TOP{slot}_QUESTIONABLE_PLAYER_ID_{stat_col}"
                            )
                        ] = topn_questionable[i][0]
                        row_update[
                            _with_before_suffix(
                                f"TOP{slot}_QUESTIONABLE_PLAYER_NAME_{stat_col}"
                            )
                        ] = topn_questionable[i][1]
                    if ACTIVE_PROFILE.emits_value(stat_col, slot, injured=True):
                        row_update[
                            _with_before_suffix(
                                f"TOP{slot}_QUESTIONABLE_PLAYER_{stat_col}"
                            )
                        ] = topn_questionable[i][2]
                    if stat_col in STREAK_STAT_COLS:
                        questionable_pid = topn_questionable[i][0]
                        # Games missed in a row coming in: a Questionable player
                        # on a five-game absence streak is a different
                        # proposition from one who played last night.
                        row_update[
                            _with_before_suffix(
                                f"TOP{slot}_QUESTIONABLE_STREAK_{stat_col}"
                            )
                        ] = (
                            questionable_streak_lookup(
                                game_id, team_id, questionable_pid
                            )
                            if questionable_pid is not None
                            else 0
                        )
                if stat_col in ACTIVE_PROFILE.avg_injured_stats:
                    questionable_values = [
                        val for (_, _, val) in topn_questionable if val != 0
                    ]
                    row_update[_with_before_suffix(f"AVG_QUESTIONABLE_{stat_col}")] = (
                        sum(questionable_values) / len(questionable_values)
                        if questionable_values
                        else 0
                    )
                if stat_col in ACTIVE_PROFILE.total_injured_stats:
                    row_update[
                        _with_before_suffix(f"TOTAL_QUESTIONABLE_PLAYER_{stat_col}")
                    ] = sum(val for (_, _, val) in all_questionable if val != 0)
                if stat_col == "PTS":
                    row_update[_with_before_suffix("N_QUESTIONABLE_PLAYERS")] = len(
                        all_questionable
                    )

            # Average of top 3 injured players
            if stat_col in ACTIVE_PROFILE.avg_injured_stats:
                inj_values = [val for (_, _, val) in topn_inj if val != 0]
                row_update[_with_before_suffix(f"AVG_INJURED_{stat_col}")] = (
                    sum(inj_values) / len(inj_values) if inj_values else 0
                )

            # Aggregation: sum of cum avg for ALL players (not just top 3).
            # Counting statistics only -- a sum of rates is not a quantity.
            if stat_col in ACTIVE_PROFILE.total_injured_stats:
                row_update[_with_before_suffix(f"TOTAL_INJURED_PLAYER_{stat_col}")] = (
                    sum(val for (_, _, val) in all_inj if val != 0)
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

        # Minutes-weighted rate aggregates, replacing the per-slot rate columns.
        # Weighting by each player's own minutes is what makes this a team
        # quantity rather than an average over a ranking: a 34-minute starter
        # and a 6-minute reserve do not shape the game equally.
        minutes_by_player = per_player_values.get("MIN", {})
        for group in (
            ("ACTIVE", "INJURED", "QUESTIONABLE")
            if build_questionable_group
            else ("ACTIVE", "INJURED")
        ):
            if group == "QUESTIONABLE" and not questionable_covered:
                continue
            key = group.lower()
            weights = minutes_by_player.get(key, {})
            for stat_col in ACTIVE_PROFILE.weighted_rate_stats:
                values = per_player_values.get(stat_col, {}).get(key, {})
                weighted = sum(
                    values[pid] * weights.get(pid, 0.0)
                    for pid in values
                    if values[pid] != 0
                )
                total_minutes = sum(
                    weights.get(pid, 0.0) for pid in values if values[pid] != 0
                )
                # 0 is the right no-evidence value: with nobody on this side of
                # the split (no injuries, or no history yet) there is no rate to
                # report, and the column sits alongside N_INJURED_PLAYERS which
                # says which of the two it is.
                row_update[_with_before_suffix(f"{group}_WEIGHTED_{stat_col}")] = (
                    weighted / total_minutes if total_minutes > 0 else 0
                )

        updates_list.append(row_update)
        if row_cache is not None:
            entry = availability_dict[str(game_id)][str(team_id)]
            row_cache.rows[cache_key] = (
                row_update,
                (tuple(entry["available"]), tuple(entry["injured"])),
            )

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
    # Questionable streaks are numeric but stay float: a team-game the report
    # does not cover has no questionable group, and zero-filling it would read
    # as "nobody at risk".
    questionable_streak_cols = [
        c for c in df_team.columns if "_QUESTIONABLE_STREAK_" in c
    ]
    if questionable_streak_cols:
        df_team[questionable_streak_cols] = df_team[questionable_streak_cols].apply(
            pd.to_numeric, errors="coerce"
        )

    df_team = add_fresh_absence_features(df_team)

    if return_availability_dict:
        return df_team, injured_dict, availability_dict
    return df_team, injured_dict
