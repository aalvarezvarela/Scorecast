"""Minutes-weighted roster continuity features."""

from bisect import bisect_right
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from nba_ou.config.constants import SEASON_TYPE_MAP

ROSTER_CONTINUITY_COLUMN = "ROSTER_MINUTES_CONTINUITY_PCT_BEFORE"
IMMEDIATE_ROSTER_CONTINUITY_COLUMN = "ROSTER_MINUTES_CONTINUITY_2M_PCT_BEFORE"
NEW_PLAYER_MINUTES_COLUMN = "ROSTER_NEW_PLAYER_MINUTES_PCT_BEFORE"
IMMEDIATE_NEW_PLAYER_MINUTES_COLUMN = "ROSTER_NEW_PLAYER_MINUTES_2M_PCT_BEFORE"
NET_ROSTER_MINUTES_COLUMN = "ROSTER_NET_MINUTES_PCT_BEFORE"
IMMEDIATE_NET_ROSTER_MINUTES_COLUMN = "ROSTER_NET_MINUTES_2M_PCT_BEFORE"
NBA_REGULATION_TEAM_MINUTES = 240.0

_ROSTER_SEASON_TYPES = frozenset({"Regular Season", "Playoffs", "Play-In Tournament"})
_PLAYER_EVENT_PRIORITY = 1
_INJURY_EVENT_PRIORITY = 2
_ONE_DAY_NS = pd.Timedelta(days=1).value


def _canonical_id(value: Any) -> str | None:
    """Normalize numeric/string IDs without changing meaningful digit strings."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))

    value_str = str(value).strip()
    if not value_str:
        return None
    if value_str.endswith(".0") and value_str[:-2].isdigit():
        return value_str[:-2]
    return value_str


def _canonical_game_id(value: Any) -> str | None:
    game_id = _canonical_id(value)
    if game_id is None:
        return None
    return game_id.zfill(10) if game_id.isdigit() else game_id


def _date_ns(value: Any) -> int | None:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return None
    timestamp = pd.Timestamp(date)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp.normalize().value


def _season_year(value: Any) -> int | None:
    try:
        season_year = int(value)
    except (TypeError, ValueError):
        return None
    return season_year if 1900 <= season_year <= 2200 else None


def _is_regular_season_or_postseason(game_id: Any) -> bool:
    """Use the repository's canonical game-ID-to-season-type mapping."""
    game_key = _canonical_game_id(game_id)
    if game_key is None:
        return False
    season_type = SEASON_TYPE_MAP.get(game_key[:3], "Unknown")
    return season_type in _ROSTER_SEASON_TYPES


def _late_previous_season_start_ns(target_season_year: int) -> int:
    # For 2025-26, the late-2024-25 window begins on March 15, 2025.
    return pd.Timestamp(year=target_season_year, month=3, day=15).value


def _immediate_trade_window_start_ns(game_date_ns: int) -> int:
    """Return two months before the game, extending offseason starts to March 1."""
    game_date = pd.Timestamp(game_date_ns)
    start = (game_date - pd.DateOffset(months=2)).normalize()
    if start.month in {6, 7, 8, 9}:
        start = pd.Timestamp(year=start.year, month=3, day=1)
    return start.value


def _build_game_metadata(df_team: pd.DataFrame) -> dict[str, tuple[int, int]]:
    metadata: dict[str, tuple[int, int]] = {}
    required = ["GAME_ID", "GAME_DATE", "SEASON_YEAR"]
    for game_id, game_date, season_year_value in df_team[required].itertuples(
        index=False, name=None
    ):
        game_key = _canonical_game_id(game_id)
        date_value = _date_ns(game_date)
        season_year = _season_year(season_year_value)
        if game_key is None or date_value is None or season_year is None:
            continue
        metadata[game_key] = (date_value, season_year)
    return metadata


def add_roster_continuity_feature(
    df_team: pd.DataFrame,
    df_players: pd.DataFrame,
    injured_dict: dict[Any, dict[Any, list[Any]]] | None,
    *,
    output_col: str = ROSTER_CONTINUITY_COLUMN,
    immediate_output_col: str = IMMEDIATE_ROSTER_CONTINUITY_COLUMN,
    new_player_output_col: str = NEW_PLAYER_MINUTES_COLUMN,
    immediate_new_player_output_col: str = IMMEDIATE_NEW_PLAYER_MINUTES_COLUMN,
    net_output_col: str = NET_ROSTER_MINUTES_COLUMN,
    immediate_net_output_col: str = IMMEDIATE_NET_ROSTER_MINUTES_COLUMN,
    df_game_context: pd.DataFrame | None = None,
    scheduled_game_ids: list[Any] | set[Any] | tuple[Any, ...] | None = None,
    target_game_ids: set[str] | None = None,
    current_rosters: dict | None = None,
) -> pd.DataFrame:
    """Add continuity, incoming, and net minute shares for two roster horizons.

    A target season's roster window begins on March 15 during the preceding NBA
    season and ends at the target game. A player enters the window when either a
    regular-season/postseason boxscore or injury report assigns them to a team.
    They count as lost only when their latest known assignment in that window is
    to another team.

    Every roster candidate is weighted by minutes played for the target team per
    target-team game in the target season, falling back to the previous season.
    Continuity is the retained candidates' share of all candidate minute value.
    Current-game boxscore rows become available on the following day;
    current-game injury reports are treated as before-game information.

    The immediate version normally uses the two calendar months before the game.
    If that start falls from June through September, it moves to March 1 to span
    the offseason and retain a usable pre-summer roster baseline.

    Incoming players are those whose latest assignment is the target team and
    whose preceding distinct assignment inside the window was another team.
    Their value is their average minutes for that previous team, divided by 240.

    Net roster minutes are incoming share minus lost share. Since lost share is
    ``1 - continuity``, the equivalent calculation is
    ``incoming share + continuity - 1``. Positive values mean more minutes were
    brought in than lost; negative values mean the opposite.

    ``target_game_ids``, when given, computes only those games' rows; a row
    reads the roster events, not other rows, and the rest stay NaN.
    """
    team_required = {"GAME_ID", "GAME_DATE", "SEASON_YEAR", "TEAM_ID"}
    player_required = {
        "GAME_ID",
        "GAME_DATE",
        "SEASON_YEAR",
        "TEAM_ID",
        "PLAYER_ID",
        "MIN",
    }
    missing_team_cols = sorted(team_required.difference(df_team.columns))
    missing_player_cols = sorted(player_required.difference(df_players.columns))
    if missing_team_cols or missing_player_cols:
        raise ValueError(
            "Cannot compute roster continuity; missing columns: "
            f"team={missing_team_cols}, players={missing_player_cols}"
        )

    # Event tuple: (known_date_ns, actual_date_ns, priority, team_id, player_id,
    # season_year). Boxscore membership is known after that game; injury reports
    # are legitimate current-game, before-tip information.
    events: set[tuple[int, int, int, str, str, int]] = set()
    minute_appearances: dict[tuple[str, int, str], tuple[int, float, str]] = {}
    scheduled_game_keys = {
        game_key
        for value in (scheduled_game_ids or [])
        if (game_key := _canonical_game_id(value)) is not None
    }

    player_cols = [
        "GAME_ID",
        "GAME_DATE",
        "SEASON_YEAR",
        "TEAM_ID",
        "PLAYER_ID",
        "MIN",
    ]
    player_rows = df_players[player_cols].itertuples(index=False, name=None)
    for (
        game_id,
        game_date,
        season_year_value,
        team_id_value,
        player_id_value,
        minutes_value,
    ) in tqdm(
        player_rows,
        total=len(df_players),
        desc="Building roster assignments",
        unit="rows",
    ):
        if not _is_regular_season_or_postseason(game_id):
            continue
        actual_date_ns = _date_ns(game_date)
        season_year = _season_year(season_year_value)
        team_id = _canonical_id(team_id_value)
        player_id = _canonical_id(player_id_value)
        game_key = _canonical_game_id(game_id)
        if (
            actual_date_ns is None
            or season_year is None
            or team_id is None
            or player_id is None
            or game_key is None
        ):
            continue

        # Historical boxscore rows become known the next day. Scheduled
        # placeholders are different: they are constructed from information
        # available before tip and are identified by the explicit scheduled ID
        # plus the MIN-is-NaN placeholder contract.
        is_scheduled_placeholder = game_key in scheduled_game_keys and pd.isna(
            minutes_value
        )
        known_date_ns = (
            actual_date_ns if is_scheduled_placeholder else actual_date_ns + _ONE_DAY_NS
        )
        events.add(
            (
                known_date_ns,
                actual_date_ns,
                _PLAYER_EVENT_PRIORITY,
                team_id,
                player_id,
                season_year,
            )
        )

        minutes = pd.to_numeric(minutes_value, errors="coerce")
        if pd.notna(minutes) and float(minutes) > 0:
            minute_appearances[(player_id, season_year, game_key)] = (
                actual_date_ns,
                float(minutes),
                team_id,
            )

    game_metadata_source = df_team
    if df_game_context is not None:
        context_cols = ["GAME_ID", "GAME_DATE", "SEASON_YEAR"]
        missing_context_cols = sorted(set(context_cols).difference(df_game_context))
        if missing_context_cols:
            raise ValueError(
                "Cannot use roster game context; missing columns: "
                f"{missing_context_cols}"
            )
        game_metadata_source = pd.concat(
            [df_game_context[context_cols], df_team[context_cols]],
            ignore_index=True,
        )

    game_metadata = _build_game_metadata(game_metadata_source)
    for game_id, team_map in (injured_dict or {}).items():
        game_key = _canonical_game_id(game_id)
        metadata = game_metadata.get(game_key) if game_key is not None else None
        if metadata is None:
            continue
        actual_date_ns, season_year = metadata
        if not _is_regular_season_or_postseason(game_key):
            continue
        for team_id_value, player_ids in team_map.items():
            team_id = _canonical_id(team_id_value)
            if team_id is None:
                continue
            for player_id_value in player_ids:
                player_id = _canonical_id(player_id_value)
                if player_id is None:
                    continue
                events.add(
                    (
                        actual_date_ns,
                        actual_date_ns,
                        _INJURY_EVENT_PRIORITY,
                        team_id,
                        player_id,
                        season_year,
                    )
                )

    # A bucket combines the target season with the prior season from March 1.
    # Each item is (known_ns, actual_ns, player_id, is_previous_season). Keeping
    # both dates lets boxscores become known the next day while window membership
    # still uses the date on which the player was assigned to the team.
    team_buckets: dict[tuple[int, str], list[tuple[int, int, str, bool]]] = defaultdict(
        list
    )
    player_timelines: dict[tuple[int, str], list[tuple[int, int, int, str]]] = (
        defaultdict(list)
    )

    for known_ns, actual_ns, priority, team_id, player_id, event_season in events:
        team_buckets[(event_season, team_id)].append(
            (known_ns, actual_ns, player_id, False)
        )
        player_timelines[(event_season, player_id)].append(
            (known_ns, actual_ns, priority, team_id)
        )

        next_season = event_season + 1
        previous_march_start = pd.Timestamp(year=next_season, month=3, day=1).value
        if actual_ns >= previous_march_start:
            team_buckets[(next_season, team_id)].append(
                (known_ns, actual_ns, player_id, True)
            )
            player_timelines[(next_season, player_id)].append(
                (known_ns, actual_ns, priority, team_id)
            )

    team_bucket_known_dates: dict[tuple[int, str], list[int]] = {}
    for key, bucket in team_buckets.items():
        bucket.sort()
        team_bucket_known_dates[key] = [item[0] for item in bucket]

    player_timeline_keys: dict[tuple[int, str], list[tuple[int, int, int]]] = {}
    for key, timeline in player_timelines.items():
        timeline.sort()
        player_timeline_keys[key] = [item[:3] for item in timeline]

    team_minute_histories: dict[tuple[str, int, str], list[tuple[int, float]]] = (
        defaultdict(list)
    )
    team_game_date_sets: dict[tuple[int, str], set[int]] = defaultdict(set)
    for (player_id, season_year, _), (
        appearance_date,
        minutes,
        appearance_team,
    ) in minute_appearances.items():
        appearance = (appearance_date, minutes)
        team_minute_histories[(player_id, season_year, appearance_team)].append(
            appearance
        )
        team_game_date_sets[(season_year, appearance_team)].add(appearance_date)

    team_minute_dates: dict[tuple[str, int, str], list[int]] = {}
    team_minute_prefix_sums: dict[tuple[str, int, str], list[float]] = {}
    for key, appearances in team_minute_histories.items():
        appearances.sort()
        team_minute_dates[key] = [appearance[0] for appearance in appearances]
        cumulative = 0.0
        prefix = []
        for _, minutes in appearances:
            cumulative += minutes
            prefix.append(cumulative)
        team_minute_prefix_sums[key] = prefix

    team_game_dates = {
        key: sorted(game_dates) for key, game_dates in team_game_date_sets.items()
    }

    def target_team_minutes_per_game_before(
        player_id: str,
        season_year: int,
        target_team_id: str,
        date_ns: int,
    ) -> float:
        """Measure the player's actual minute contribution to the target team."""
        key = (player_id, season_year, target_team_id)
        dates = team_minute_dates.get(key, [])
        appearance_count = bisect_right(dates, date_ns - 1)
        if appearance_count:
            team_games = bisect_right(
                team_game_dates.get((season_year, target_team_id), []), date_ns - 1
            )
            if team_games:
                return team_minute_prefix_sums[key][appearance_count - 1] / team_games

        previous_season = season_year - 1
        previous_key = (player_id, previous_season, target_team_id)
        previous_dates = team_minute_dates.get(previous_key, [])
        previous_appearance_count = bisect_right(previous_dates, date_ns - 1)
        if previous_appearance_count:
            previous_team_games = bisect_right(
                team_game_dates.get((previous_season, target_team_id), []), date_ns - 1
            )
            if previous_team_games:
                return (
                    team_minute_prefix_sums[previous_key][previous_appearance_count - 1]
                    / previous_team_games
                )
        return 0.0

    def mean_previous_team_minutes_before(
        player_id: str,
        season_year: int,
        previous_team_id: str,
        date_ns: int,
    ) -> float:
        key = (player_id, season_year, previous_team_id)
        dates = team_minute_dates.get(key, [])
        count = bisect_right(dates, date_ns - 1)
        if count:
            return team_minute_prefix_sums[key][count - 1] / count

        previous_key = (player_id, season_year - 1, previous_team_id)
        previous_dates = team_minute_dates.get(previous_key, [])
        previous_count = bisect_right(previous_dates, date_ns - 1)
        if previous_count:
            return (
                team_minute_prefix_sums[previous_key][previous_count - 1]
                / previous_count
            )
        return 0.0

    def roster_metrics_for_window(
        *,
        season_year: int,
        team_id: str,
        date_ns: int,
        window_start_ns: int,
        require_previous_season_baseline: bool,
        members: dict | None = None,
    ) -> tuple[float, float]:
        bucket_key = (season_year, team_id)
        bucket = team_buckets.get(bucket_key, [])
        end = bisect_right(team_bucket_known_dates.get(bucket_key, []), date_ns)
        window_bucket = [item for item in bucket[:end] if item[1] >= window_start_ns]
        if not window_bucket:
            return np.nan, np.nan
        if require_previous_season_baseline and not any(
            is_previous for _, _, _, is_previous in window_bucket
        ):
            return np.nan, np.nan

        candidate_players = {player_id for _, _, player_id, _ in window_bucket}
        if members is not None:
            candidate_players.update(members)
        lost_minutes = 0.0
        candidate_minutes = 0.0
        new_player_minutes = 0.0
        for player_id in candidate_players:
            timeline_key = (season_year, player_id)
            timeline = player_timelines.get(timeline_key, [])
            timeline_keys = player_timeline_keys.get(timeline_key, [])
            assignment_index = bisect_right(
                timeline_keys, (date_ns, np.iinfo(np.int64).max, 99)
            )
            if assignment_index == 0:
                continue
            player_target_team_minutes = target_team_minutes_per_game_before(
                player_id,
                season_year,
                team_id,
                date_ns,
            )
            candidate_minutes += player_target_team_minutes
            latest_team = timeline[assignment_index - 1][3]
            prior_index = assignment_index - 2
            if members is not None:
                if player_id in members:
                    if latest_team != team_id:
                        prior_index = assignment_index - 1
                    latest_team = team_id
                else:
                    latest_team = None
            if latest_team != team_id:
                lost_minutes += player_target_team_minutes
                continue

            previous_team_id = None
            for timeline_index in range(prior_index, -1, -1):
                _, actual_ns, _, assigned_team = timeline[timeline_index]
                # The bisect already limits these indexes to known_ns <= date_ns.
                # In reverse chronological order, crossing the window boundary
                # means every remaining assignment is older too.
                if actual_ns < window_start_ns:
                    break
                if assigned_team != team_id:
                    previous_team_id = assigned_team
                    break

            if previous_team_id is not None:
                new_player_minutes += mean_previous_team_minutes_before(
                    player_id,
                    season_year,
                    previous_team_id,
                    date_ns,
                )

        continuity = (
            float(np.clip(1.0 - lost_minutes / candidate_minutes, 0.0, 1.0))
            if candidate_minutes > 0
            else np.nan
        )
        new_player_share = float(
            np.clip(new_player_minutes / NBA_REGULATION_TEAM_MINUTES, 0.0, 1.0)
        )
        return continuity, new_player_share

    values: list[float] = []
    immediate_values: list[float] = []
    new_player_values: list[float] = []
    immediate_new_player_values: list[float] = []
    row_cols = ["GAME_DATE", "SEASON_YEAR", "TEAM_ID", "GAME_ID"]
    targets = None if target_game_ids is None else {str(g) for g in target_game_ids}
    for game_date, season_year_value, team_id_value, game_id in df_team[
        row_cols
    ].itertuples(index=False, name=None):
        if targets is not None and str(game_id) not in targets:
            values.append(np.nan)
            immediate_values.append(np.nan)
            new_player_values.append(np.nan)
            immediate_new_player_values.append(np.nan)
            continue
        date_ns = _date_ns(game_date)
        season_year = _season_year(season_year_value)
        team_id = _canonical_id(team_id_value)
        if date_ns is None or season_year is None or team_id is None:
            values.append(np.nan)
            immediate_values.append(np.nan)
            new_player_values.append(np.nan)
            immediate_new_player_values.append(np.nan)
            continue

        continuity, new_player_share = roster_metrics_for_window(
            season_year=season_year,
            team_id=team_id,
            date_ns=date_ns,
            window_start_ns=_late_previous_season_start_ns(season_year),
            require_previous_season_baseline=True,
            members=(current_rosters or {}).get((str(game_id), team_id)),
        )
        immediate_continuity, immediate_new_player_share = roster_metrics_for_window(
            season_year=season_year,
            team_id=team_id,
            date_ns=date_ns,
            window_start_ns=_immediate_trade_window_start_ns(date_ns),
            require_previous_season_baseline=False,
            members=(current_rosters or {}).get((str(game_id), team_id)),
        )
        values.append(continuity)
        immediate_values.append(immediate_continuity)
        new_player_values.append(new_player_share)
        immediate_new_player_values.append(immediate_new_player_share)

    result = df_team.copy()
    result[output_col] = values
    result[immediate_output_col] = immediate_values
    result[new_player_output_col] = new_player_values
    result[immediate_new_player_output_col] = immediate_new_player_values
    result[net_output_col] = result[new_player_output_col] + result[output_col] - 1.0
    result[immediate_net_output_col] = (
        result[immediate_new_player_output_col] + result[immediate_output_col] - 1.0
    )
    return result
