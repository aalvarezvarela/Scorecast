"""Starter-rotation history available before a team's next game.

The box score's START_POSITION is an outcome of that game. It may describe a
*later* game only after that game has finished; the target game's starters and
minutes must never contribute to its own features. Games on the same calendar
date are held back together because this pipeline does not have a reliable
publication timestamp for their box scores.
"""

from collections import defaultdict

import numpy as np
import pandas as pd

STARTER_HISTORY_COLUMNS = (
    "STARTER_OVERLAP_LAST_TWO_GAMES_BEFORE",
    "STARTER_UNIQUE_LAST_5_GAMES_BEFORE",
    "STARTER_REPEAT_RATE_LAST_5_GAMES_BEFORE",
    "STARTER_LATEST_FIVE_MINUTES_SHARE_LAST_5_GAMES_BEFORE",
)


def add_starter_history_features(
    df_team: pd.DataFrame, df_players: pd.DataFrame
) -> pd.DataFrame:
    """Attach four prior-game starter features to each team-game row.

    Only completed box scores with exactly five starters enter history. A
    malformed or missing box score is skipped, never interpreted as a lineup
    change. ``df_players`` must have numeric minutes, as returned by
    ``clear_player_statistics``. The recent-minutes share asks what fraction of the team's minutes
    over its last five games were played by its *most recent* starting five.
    """
    team_required = {"GAME_ID", "TEAM_ID", "GAME_DATE"}
    player_required = team_required | {"PLAYER_ID", "START_POSITION", "MIN"}
    if missing := team_required - set(df_team.columns):
        raise ValueError(f"Team games are missing {sorted(missing)}")
    if missing := player_required - set(df_players.columns):
        raise ValueError(f"Player history is missing {sorted(missing)}")

    players = df_players[list(player_required)].copy()
    players["GAME_DATE"] = pd.to_datetime(players["GAME_DATE"], errors="coerce").dt.normalize()
    players["MIN"] = pd.to_numeric(players["MIN"], errors="coerce")
    players = players.dropna(subset=["GAME_DATE", "GAME_ID", "TEAM_ID", "PLAYER_ID"])
    players = players.drop_duplicates(["GAME_ID", "TEAM_ID", "PLAYER_ID"])
    players["_starter"] = players["START_POSITION"].fillna("").astype(str).str.strip().ne("")

    # Build one event per finished team-game. Scheduled placeholder rows have
    # no real playing time and therefore cannot pass this check.
    events: dict[str, list[tuple[pd.Timestamp, str, frozenset[str], dict[str, float]]]] = defaultdict(list)
    for (game_id, team_id), game in players.groupby(["GAME_ID", "TEAM_ID"], sort=False):
        starters = frozenset(game.loc[game["_starter"], "PLAYER_ID"].astype(str))
        minutes = dict(zip(game["PLAYER_ID"].astype(str), game["MIN"], strict=True))
        if len(starters) != 5 or not any(v > 0 for v in minutes.values() if pd.notna(v)):
            continue
        date = game["GAME_DATE"].iloc[0]
        events[str(team_id)].append((date, str(game_id), starters, minutes))

    for team_events in events.values():
        team_events.sort(key=lambda event: (event[0], event[1]))

    result = df_team.copy()
    for column in STARTER_HISTORY_COLUMNS:
        result[column] = np.nan

    for team_id, indices in result.groupby(result["TEAM_ID"].astype(str), sort=False).groups.items():
        team_events = events.get(team_id, [])
        targets = result.loc[indices, ["GAME_DATE"]].copy()
        targets["GAME_DATE"] = pd.to_datetime(targets["GAME_DATE"], errors="coerce").dt.normalize()
        targets = targets.sort_values("GAME_DATE", kind="mergesort")
        history = []
        next_event = 0
        for target_date, date_rows in targets.groupby("GAME_DATE", sort=True):
            if pd.isna(target_date):
                continue
            while next_event < len(team_events) and team_events[next_event][0] < target_date:
                history.append(team_events[next_event])
                next_event += 1
            if not history:
                continue
            recent = history[-5:]
            latest_starters = recent[-1][2]
            values = {
                "STARTER_UNIQUE_LAST_5_GAMES_BEFORE": len(set().union(*(item[2] for item in recent))),
            }
            if len(recent) >= 2:
                values["STARTER_OVERLAP_LAST_TWO_GAMES_BEFORE"] = len(recent[-1][2] & recent[-2][2])
                values["STARTER_REPEAT_RATE_LAST_5_GAMES_BEFORE"] = sum(
                    recent[i][2] == recent[i - 1][2] for i in range(1, len(recent))
                ) / (len(recent) - 1)
            total_minutes = sum(
                float(value)
                for _, _, _, minutes in recent
                for value in minutes.values()
                if pd.notna(value) and value > 0
            )
            if total_minutes > 0:
                latest_minutes = sum(
                    float(minutes.get(player_id, 0))
                    for _, _, _, minutes in recent
                    for player_id in latest_starters
                    if pd.notna(minutes.get(player_id, 0))
                )
                values["STARTER_LATEST_FIVE_MINUTES_SHARE_LAST_5_GAMES_BEFORE"] = (
                    latest_minutes / total_minutes
                )
            for column, value in values.items():
                result.loc[date_rows.index, column] = value

    return result
