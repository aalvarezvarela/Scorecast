"""Who is expected to play tonight, and for how long, before tip-off.

This is the input side of the phase F projection: it turns box-score history
and the pre-game injury report into the ``PlayerNight`` rosters
``game_projection`` consumes.

The temporal contract matches ``rating_gate`` and ``starter_history``: minutes
come only from games on **strictly earlier** dates, so games sharing a calendar
date are held back together, and the report is the last one filed before tip.

Two rules that look like details and are not:

**Recent minutes must mean recently played minutes.** A player who has missed
the team's whole recent window still has an average in his history, and letting
it stand would have him absorb minutes from whoever actually replaced him --
precisely inverting what an absence is supposed to do. He is dropped from the
roster until he plays again.

**A player belongs to one team.** When he appears for a new one he leaves the
old one's roster, so a trade does not leave him projected for both.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .game_projection import PlayerNight

#: Team games behind a player's recent-minutes average.
RECENT_GAMES = 5

_TEAM_REQUIRED = {"GAME_ID", "TEAM_ID", "GAME_DATE"}
_PLAYER_REQUIRED = _TEAM_REQUIRED | {"PLAYER_ID", "MIN"}


def player_out_probabilities(
    statuses: pd.DataFrame,
) -> dict[tuple[str, str, str], float]:
    """``(game_id, team_id, player_id) -> p_out`` from the pre-game report.

    ``statuses`` is ``InjuryReportState.statuses``: the last status filed before
    each tip-off. The probability itself comes from ``injury_status.news
    .chance_out``, which is what every other injury feature in the repo uses, so
    a listed player is scored the same way here as there.
    """
    from nba_ou.data_processing.injury_status.news import chance_out

    if statuses.empty:
        return {}
    probabilities = chance_out(statuses)
    return {
        (str(game_id).zfill(10), str(team_id), str(player_id)): float(value)
        for (game_id, team_id, player_id), value in zip(
            statuses[["game_id", "team_id", "player_id"]].itertuples(
                index=False, name=None
            ),
            probabilities,
            strict=True,
        )
    }


def build_player_nights(
    df_team: pd.DataFrame,
    df_players: pd.DataFrame,
    p_out: dict[tuple[str, str, str], float] | None = None,
    recent_games: int = RECENT_GAMES,
) -> dict[tuple[str, str], list[PlayerNight]]:
    """``(game_id, team_id) ->`` the roster expected to share tonight's minutes.

    A player absent from ``p_out`` is taken as playing. That is the same
    convention the report itself uses: it lists the players with something to
    say about them, not the squad, so silence means available rather than
    unknown.
    """
    if recent_games <= 0:
        raise ValueError("recent_games must be positive")
    if missing := _TEAM_REQUIRED - set(df_team.columns):
        raise ValueError(f"Team games are missing {sorted(missing)}")
    if missing := _PLAYER_REQUIRED - set(df_players.columns):
        raise ValueError(f"Player history is missing {sorted(missing)}")
    p_out = p_out or {}

    players = df_players[list(_PLAYER_REQUIRED)].copy()
    for column in ("GAME_ID", "TEAM_ID", "PLAYER_ID"):
        players[column] = players[column].astype(str)
    players["GAME_ID"] = players["GAME_ID"].str.zfill(10)
    players["GAME_DATE"] = pd.to_datetime(
        players["GAME_DATE"], errors="coerce"
    ).dt.normalize()
    players["MIN"] = pd.to_numeric(players["MIN"], errors="coerce").fillna(0.0)
    players = players.dropna(subset=["GAME_DATE"]).drop_duplicates(
        ["GAME_ID", "TEAM_ID", "PLAYER_ID"]
    )

    targets = df_team[list(_TEAM_REQUIRED)].copy()
    for column in ("GAME_ID", "TEAM_ID"):
        targets[column] = targets[column].astype(str)
    targets["GAME_ID"] = targets["GAME_ID"].str.zfill(10)
    targets["GAME_DATE"] = pd.to_datetime(
        targets["GAME_DATE"], errors="coerce"
    ).dt.normalize()
    targets = targets.dropna(subset=["GAME_DATE"]).drop_duplicates(
        ["GAME_ID", "TEAM_ID"]
    )

    history_by_date = {
        date: frame for date, frame in players.groupby("GAME_DATE", sort=True)
    }
    minutes: dict[tuple[str, str], deque[float]] = defaultdict(
        lambda: deque(maxlen=recent_games)
    )
    rosters: dict[str, set[str]] = defaultdict(set)
    assignments: dict[str, str] = {}
    games_played: dict[str, int] = defaultdict(int)
    last_seen: dict[tuple[str, str], int] = {}
    result: dict[tuple[str, str], list[PlayerNight]] = {}

    for date in sorted(set(targets["GAME_DATE"]) | set(history_by_date)):
        for row in targets.loc[targets["GAME_DATE"].eq(date)].itertuples(index=False):
            team = row.TEAM_ID
            cutoff = games_played[team] - recent_games
            roster = [
                player
                for player in rosters[team]
                if minutes[(team, player)]
                and last_seen.get((team, player), -1) > cutoff
            ]
            if not roster:
                continue
            result[(row.GAME_ID, team)] = [
                PlayerNight(
                    player_id=player,
                    base_minutes=float(np.mean(minutes[(team, player)])),
                    p_out=p_out.get((row.GAME_ID, team, player), 0.0),
                )
                for player in sorted(roster)
            ]
        # A date's box scores become history only once every game on it has
        # been built, so the second game of a date cannot read the first.
        day = history_by_date.get(date)
        if day is None:
            continue
        for team in day["TEAM_ID"].unique():
            games_played[team] += 1
        for row in day.itertuples(index=False):
            player, team = row.PLAYER_ID, row.TEAM_ID
            previous = assignments.get(player)
            if previous is not None and previous != team:
                rosters[previous].discard(player)
            assignments[player] = team
            rosters[team].add(player)
            minutes[(team, player)].append(max(0.0, float(row.MIN)))
            last_seen[(team, player)] = games_played[team]
    return result
