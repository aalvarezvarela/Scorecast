"""Keep current roster membership independent of box-score participation."""

import pandas as pd

from nba_ou.data_processing.players.players_statistics import (
    precompute_cumulative_avg_stat,
)


def supplement_rosters(rosters, listed):
    """A report may add players to an official roster, never remove them."""
    result = {key: dict(players) for key, players in (rosters or {}).items()}
    for (game, team), players in result.items():
        for player in (listed or {}).get(game, {}).get(team, []):
            players.setdefault(str(player), str(player))
    return result


def report_membership(state):
    result = {}
    if state is not None:
        for game, team, player in state.statuses[
            ["game_id", "team_id", "player_id"]
        ].itertuples(index=False, name=None):
            if (game, team) in state.covered:
                result.setdefault(str(game), {}).setdefault(str(team), []).append(
                    str(player)
                )
    return result


def live_roster_history(
    players, members, *, game_id, team_id, season_id, game_date, stat_cols
):
    """Prepared history plus an available-player state for each roster member.

    Historical rows and their shifted statistics are retained exactly, including
    DNP rows and missing MIN. Only synthetic available-player states are
    calculated anew. Out/Questionable selection uses strictly earlier history,
    as in training, and never reads these same-day synthetic states.
    """
    dates = pd.to_datetime(players.GAME_DATE)
    history = players.loc[
        players.PLAYER_ID.astype(str).isin(members) & dates.lt(game_date)
    ].copy()
    latest = (
        history.assign(_id=history.PLAYER_ID.astype(str))
        .sort_values("GAME_DATE", kind="stable")
        .drop_duplicates("_id", keep="last")
        .set_index("_id")
    )
    names = latest.PLAYER_NAME.to_dict()
    identifiers = latest.PLAYER_ID.to_dict()
    rows = []
    for player, name in members.items():
        row = dict.fromkeys(players.columns)
        row.update(
            GAME_ID=str(game_id),
            TEAM_ID=str(team_id),
            SEASON_ID=str(season_id),
            SEASON_YEAR=int(str(season_id)[-4:]),
            GAME_DATE=pd.Timestamp(game_date),
            PLAYER_ID=identifiers.get(
                player,
                int(player)
                if pd.api.types.is_numeric_dtype(players.PLAYER_ID)
                else str(player),
            ),
            PLAYER_NAME=names.get(player, name),
        )
        rows.append(row)
    placeholders = pd.DataFrame(rows)
    # Preserve numeric missingness explicitly across pandas concat versions.
    for column in history.select_dtypes(include="number"):
        placeholders[column] = pd.to_numeric(placeholders[column], errors="coerce")
    combined = pd.concat([history, placeholders], ignore_index=True)
    for column in stat_cols:
        combined = precompute_cumulative_avg_stat(combined, stat_col=column)
    placeholders = combined.loc[combined.GAME_DATE.eq(game_date)].copy()
    known = set(
        history.loc[
            pd.to_numeric(history.MIN, errors="coerce").gt(0), "PLAYER_ID"
        ].astype(str)
    )
    unknown = placeholders.PLAYER_ID.astype(str).isin(set(members) - known)
    placeholders.loc[unknown, [f"{s}_CUM_AVG" for s in stat_cols]] = float("nan")
    return pd.concat([history, placeholders], ignore_index=True).sort_values(
        ["PLAYER_ID", "GAME_DATE"], kind="stable"
    )
