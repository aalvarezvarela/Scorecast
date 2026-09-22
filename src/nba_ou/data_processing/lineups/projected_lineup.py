"""Projected starting five and replacement identification, before tip-off.

Tonight's starting five is an outcome of tonight's game, so it can never be
read from the target game's box score. This module projects it from the team's
prior starting fives and the pre-game out set, following the same temporal
contract as ``players/starter_history.py``: only finished box scores enter
history, and games on the same calendar date are held back together because
this pipeline has no reliable publication timestamp for their box scores.

The projection is deliberately simple, because it is a prerequisite for the
phase D/E/F work rather than a model in its own right (see
``docs/lineup_projection_plan.md`` step D0):

1. Start from the five who started the team's most recent prior game.
2. Drop the ones on tonight's pre-game out set.
3. For each dropped starter, find who started in his place the last time he
   did not play, and use that player.
4. If his absence has no precedent, fall back to the available non-starter
   with the most minutes over the team's recent games.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

#: Team games behind the fallback replacement's recent-minutes ranking.
FALLBACK_MINUTES_WINDOW = 5

PROJECTED_LINEUP_COLUMNS = (
    "LU_PROJ_STARTERS_REPLACED_BEFORE",
    "LU_REPL_MIN_STARTS_SEASON_BEFORE",
    "LU_REPL_PRIOR_EVIDENCE_BEFORE",
)

_TEAM_REQUIRED = {"GAME_ID", "TEAM_ID", "GAME_DATE"}
_PLAYER_REQUIRED = _TEAM_REQUIRED | {"PLAYER_ID", "START_POSITION", "MIN"}


class _TeamGame:
    """One finished team-game, reduced to what the projection reads."""

    __slots__ = ("date", "game_id", "starters", "minutes", "played", "season")

    def __init__(self, date, game_id, starters, minutes, played, season):
        self.date = date
        self.game_id = game_id
        self.starters = starters
        self.minutes = minutes
        self.played = played
        self.season = season


def _history_events(players: pd.DataFrame) -> dict[str, list[_TeamGame]]:
    """One event per finished team-game, keyed by team, in date order.

    A box score without exactly five starters, or with no playing time at all,
    is a scheduled placeholder or a malformed row. It is skipped rather than
    interpreted as a lineup change, exactly as ``starter_history`` does.
    """
    events: dict[str, list[_TeamGame]] = defaultdict(list)
    for (game_id, team_id), game in players.groupby(["GAME_ID", "TEAM_ID"], sort=False):
        starters = frozenset(game.loc[game["_starter"], "PLAYER_ID"])
        minutes = dict(zip(game["PLAYER_ID"], game["MIN"], strict=True))
        if len(starters) != 5:
            continue
        played = frozenset(
            player
            for player, value in minutes.items()
            if pd.notna(value) and float(value) > 0
        )
        if not played:
            continue
        events[team_id].append(
            _TeamGame(
                date=game["GAME_DATE"].iloc[0],
                game_id=game_id,
                starters=starters,
                minutes={
                    player: float(value)
                    for player, value in minutes.items()
                    if pd.notna(value)
                },
                played=played,
                season=game["_season"].iloc[0],
            )
        )
    for team_events in events.values():
        team_events.sort(key=lambda event: (event.date, event.game_id))
    return events


def _replacement_for(
    absent: str,
    remaining: frozenset[str],
    history: list[_TeamGame],
) -> tuple[str | None, int]:
    """Who started in ``absent``'s place, and on how many prior games we saw it.

    Looks back for games ``absent`` missed and takes the starter who is not one
    of tonight's remaining four. When two of tonight's starters were both out
    in that game there are two such candidates and no way to tell which
    replaced whom, so the one with more minutes wins; the evidence count says
    how often the chosen player actually appeared, which is what a consumer
    should weight by.
    """
    tally: dict[str, int] = defaultdict(int)
    best: str | None = None
    for event in reversed(history):
        if absent in event.played:
            continue
        candidates = event.starters - remaining
        if not candidates:
            continue
        choice = max(candidates, key=lambda player: event.minutes.get(player, 0.0))
        tally[choice] += 1
        if best is None:
            best = choice
    return best, tally.get(best, 0) if best is not None else 0


def _fallback_replacement(
    history: list[_TeamGame],
    excluded: set[str],
) -> str | None:
    """The most-used available player who is not already in the projected five."""
    totals: dict[str, float] = defaultdict(float)
    for event in history[-FALLBACK_MINUTES_WINDOW:]:
        for player, value in event.minutes.items():
            if player not in excluded:
                totals[player] += max(0.0, value)
    if not totals:
        return None
    return max(totals, key=lambda player: (totals[player], player))


def _starts_this_season(history: list[_TeamGame], player: str, season) -> int:
    """Starts the player already has for this team in the target game's season.

    Falls back to the whole prior history when the caller supplied no season,
    which keeps the column meaningful rather than absent.
    """
    return sum(
        1
        for event in history
        if player in event.starters and (season is None or event.season == season)
    )


def project_starting_five(
    history: list[_TeamGame],
    expected_out: frozenset[str],
    season=None,
) -> dict | None:
    """Project one team-game's five from its prior games and tonight's out set."""
    if not history:
        return None
    latest = history[-1].starters
    replaced = frozenset(latest & expected_out)
    remaining = frozenset(latest - expected_out)
    projected = set(remaining)
    replacements: list[str] = []
    evidence: list[int] = []
    for absent in sorted(replaced):
        choice, seen = _replacement_for(absent, remaining, history)
        if choice is None or choice in projected or choice in expected_out:
            choice = _fallback_replacement(history, projected | expected_out)
            seen = 0
        if choice is None:
            continue
        projected.add(choice)
        replacements.append(choice)
        evidence.append(seen)
    return {
        "projected_five": frozenset(projected),
        "latest_five": latest,
        "replaced": replaced,
        "replacements": tuple(replacements),
        "n_replaced": len(replaced),
        "min_starts_season": (
            min(_starts_this_season(history, player, season) for player in replacements)
            if replacements
            else np.nan
        ),
        "min_prior_evidence": min(evidence) if evidence else np.nan,
    }


def add_projected_lineup_features(
    df_team: pd.DataFrame,
    df_players: pd.DataFrame,
    out_sets: dict[tuple[str, str], list[str]] | None = None,
) -> pd.DataFrame:
    """Attach the projected-five diagnostics to each team-game row.

    ``out_sets`` maps ``(game_id, team_id)`` to the pre-game out set, as
    ``injury_status.report_state.report_out_overrides`` returns it. A team-game
    that is absent from it has no report, so nobody is projected out and the
    projected five is simply the latest five; that is the same
    covered/uncovered distinction the injury-status features already draw.

    ``df_players`` must carry numeric minutes, as ``clear_player_statistics``
    returns them.
    """
    if missing := _TEAM_REQUIRED - set(df_team.columns):
        raise ValueError(f"Team games are missing {sorted(missing)}")
    if missing := _PLAYER_REQUIRED - set(df_players.columns):
        raise ValueError(f"Player history is missing {sorted(missing)}")

    players = df_players[
        list(_PLAYER_REQUIRED | ({"SEASON_YEAR"} & set(df_players.columns)))
    ].copy()
    for column in ("GAME_ID", "TEAM_ID", "PLAYER_ID"):
        players[column] = players[column].astype(str)
    players["GAME_DATE"] = pd.to_datetime(
        players["GAME_DATE"], errors="coerce"
    ).dt.normalize()
    players["MIN"] = pd.to_numeric(players["MIN"], errors="coerce")
    players["_season"] = (
        players["SEASON_YEAR"]
        if "SEASON_YEAR" in players
        else pd.Series(None, index=players.index)
    )
    players = players.dropna(subset=["GAME_DATE", "GAME_ID", "TEAM_ID", "PLAYER_ID"])
    players = players.drop_duplicates(["GAME_ID", "TEAM_ID", "PLAYER_ID"])
    players["_starter"] = (
        players["START_POSITION"].fillna("").astype(str).str.strip().ne("")
    )

    events = _history_events(players)

    result = df_team.copy()
    for column in PROJECTED_LINEUP_COLUMNS:
        result[column] = np.nan
    projections: dict[tuple[str, str], dict] = {}

    seasons = (
        df_team["SEASON_YEAR"]
        if "SEASON_YEAR" in df_team
        else pd.Series(None, index=df_team.index)
    )
    targets = pd.DataFrame(
        {
            "GAME_ID": result["GAME_ID"].astype(str),
            "TEAM_ID": result["TEAM_ID"].astype(str),
            "GAME_DATE": pd.to_datetime(
                result["GAME_DATE"], errors="coerce"
            ).dt.normalize(),
            "SEASON_YEAR": seasons.to_numpy(),
        },
        index=result.index,
    )

    for team_id, team_rows in targets.groupby("TEAM_ID", sort=False):
        team_events = events.get(team_id, [])
        history: list[_TeamGame] = []
        next_event = 0
        ordered = team_rows.sort_values("GAME_DATE", kind="mergesort")
        for target_date, date_rows in ordered.groupby("GAME_DATE", sort=True):
            if pd.isna(target_date):
                continue
            # Strictly earlier dates only: same-date games are held back
            # together, so a team's second game of a date cannot read the first.
            while (
                next_event < len(team_events)
                and team_events[next_event].date < target_date
            ):
                history.append(team_events[next_event])
                next_event += 1
            if not history:
                continue
            for index, row in date_rows.iterrows():
                key = (row["GAME_ID"], team_id)
                listed = out_sets.get(key) if out_sets is not None else None
                projection = project_starting_five(
                    history,
                    frozenset(str(player) for player in listed or ()),
                    season=row["SEASON_YEAR"],
                )
                if projection is None:
                    continue
                projections[key] = projection
                result.loc[index, "LU_PROJ_STARTERS_REPLACED_BEFORE"] = projection[
                    "n_replaced"
                ]
                result.loc[index, "LU_REPL_MIN_STARTS_SEASON_BEFORE"] = projection[
                    "min_starts_season"
                ]
                result.loc[index, "LU_REPL_PRIOR_EVIDENCE_BEFORE"] = projection[
                    "min_prior_evidence"
                ]

    result.attrs["projected_lineups"] = projections
    return result
