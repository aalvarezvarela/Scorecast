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


def game_day_rosters(players, listed, *, exclude_games=()):
    """Every stored team-game's roster: its box score plus the players listed.

    The box score lists every player dressed for the game, minutes or not, and
    ``listed`` (``game -> team -> players``: the inactive list, the injury
    report) adds the ones who were not. Together they are the roster on game
    day, which the official roster gives for tonight before tip and which does
    not depend on who ended up playing. A listed player belongs to the team
    even without a box-score row.

    Returns ``(game_id, team_id) -> {player_id: name}`` for every team-game
    with a box score, except ``exclude_games`` (tonight's: their rows are
    placeholders, not a box score).
    """
    exclude = {str(game) for game in exclude_games}
    box = players[players["PLAYER_ID"].notna() & players["GAME_ID"].notna()]
    games = box["GAME_ID"].astype(str).to_numpy()
    teams = box["TEAM_ID"].astype(str).to_numpy()
    ids = box["PLAYER_ID"].astype(str).to_numpy()
    names = box["PLAYER_NAME"].to_numpy() if "PLAYER_NAME" in box else ids
    result: dict[tuple[str, str], dict[str, str]] = {}
    for game, team, player, name in zip(games, teams, ids, names, strict=True):
        if game not in exclude:
            result.setdefault((game, team), {}).setdefault(player, name)
    for game, team_map in (listed or {}).items():
        for team, team_players in team_map.items():
            members = result.get((str(game), str(team)))
            if members is None:
                continue
            for player in team_players:
                if pd.notna(player):
                    members.setdefault(str(player), str(player))
    return result


def previous_game_rosters(team_games, players, injured, targets, rosters=None):
    """Rosters for target team-games the official roster does not cover.

    Training reads a stored game's roster from :func:`game_day_rosters`.
    Tonight has no box score, so the official roster stands in; where that
    fetch failed, the team's previous game's roster -- its box score plus its
    inactive list -- does. Returns ``(game_id, team_id) -> {player_id: name}``
    for those team-games only; one whose team has no earlier game with a box
    score is left to the inference.

    Args:
        team_games: Team-game rows with GAME_ID, TEAM_ID and GAME_DATE,
            including the history before the targets.
        players: Box-score rows with GAME_ID, TEAM_ID and PLAYER_ID (and
            PLAYER_NAME where known).
        injured: ``game -> team -> players`` realized absences (inactive list).
        targets: Target game ids (tonight's games).
        rosters: ``(game_id, team_id) -> members`` already known; skipped.
    """
    targets = {str(game) for game in (() if targets is None else targets)}
    if not targets:
        return {}
    games = pd.DataFrame(
        {
            "GAME_ID": team_games["GAME_ID"].astype(str).to_numpy(),
            "TEAM_ID": team_games["TEAM_ID"].astype(str).to_numpy(),
            "GAME_DATE": pd.to_datetime(team_games["GAME_DATE"]).to_numpy(),
        }
    ).drop_duplicates()
    stored_rosters = game_day_rosters(players, injured, exclude_games=targets)
    stored = games[
        pd.Series(
            [
                key in stored_rosters
                for key in zip(games.GAME_ID, games.TEAM_ID, strict=True)
            ],
            index=games.index,
        )
    ].sort_values(["GAME_DATE", "GAME_ID"], kind="stable")
    result = {}
    for game, team, date in games.loc[
        games.GAME_ID.isin(targets), ["GAME_ID", "TEAM_ID", "GAME_DATE"]
    ].itertuples(index=False, name=None):
        if (game, team) in (rosters or {}):
            continue
        earlier = stored[stored.TEAM_ID.eq(team) & stored.GAME_DATE.lt(date)]
        if earlier.empty:
            continue
        result[(game, team)] = dict(stored_rosters[(earlier.GAME_ID.iloc[-1], team)])
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
    # A member with no earlier appearance with minutes reads NaN, from the same
    # rule training applies (precompute_cumulative_avg_stat).
    placeholders = combined.loc[combined.GAME_DATE.eq(game_date)].copy()
    return pd.concat([history, placeholders], ignore_index=True).sort_values(
        ["PLAYER_ID", "GAME_DATE"], kind="stable"
    )
