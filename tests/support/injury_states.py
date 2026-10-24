"""Games and injury states shared by row-cache and player parity tests."""

import numpy as np
import pandas as pd
from nba_ou.data_processing.players.attach_player_features import (
    add_player_history_features,
)

SEASON_ID = "22024"


TEAMS = ("1610612738", "1610612744")


ROSTERS = {
    TEAMS[0]: ["a1", "a2", "a3", "a4", "a5"],
    TEAMS[1]: ["b1", "b2", "b3", "b4", "b5"],
}


def _games(n_games: int = 8):
    team_rows, player_rows = [], []
    rng = np.random.default_rng(5)
    for i in range(n_games):
        game_id = f"00224{i:05d}"
        game_date = pd.Timestamp("2024-10-20") + pd.Timedelta(days=2 * i)
        for home, team in ((True, TEAMS[0]), (False, TEAMS[1])):
            team_rows.append(
                {
                    "GAME_ID": game_id,
                    "TEAM_ID": team,
                    "HOME": home,
                    "SEASON_ID": SEASON_ID,
                    "SEASON_YEAR": 2024,
                    "GAME_DATE": game_date,
                }
            )
            for player in ROSTERS[team]:
                sits = player.endswith("5") and i % 3 == 0
                minutes = 0.0 if sits else float(rng.uniform(10, 36))
                player_rows.append(
                    {
                        "GAME_ID": game_id,
                        "TEAM_ID": team,
                        "SEASON_ID": SEASON_ID,
                        "SEASON_YEAR": 2024,
                        "GAME_DATE": game_date,
                        "PLAYER_ID": player,
                        "PLAYER_NAME": f"Player {player}",
                        "MIN": minutes,
                        "PTS": None if sits else float(rng.uniform(2, 30)),
                        "PACE_PER40": None if sits else float(rng.uniform(90, 105)),
                        "COMMENT": "DNP - Injury/Illness" if sits else "",
                    }
                )
    return pd.DataFrame(team_rows), pd.DataFrame(player_rows)


def _report_states(team_games: pd.DataFrame):
    """Report states for a sequence of horizons over the last three games."""
    games = sorted(team_games["GAME_ID"].unique())
    late = games[-3:]
    first = {
        "report_out_overrides": {},
        "report_questionable_sets": {},
        "snapshot_game_ids": set(late),
        "snapshot_report_listed_players": {},
    }
    second = {
        "report_out_overrides": {(late[0], TEAMS[0]): ["a1"], (late[1], TEAMS[1]): []},
        "report_questionable_sets": {
            (late[0], TEAMS[0]): ["a2"],
            (late[1], TEAMS[1]): [],
        },
        "snapshot_game_ids": set(late),
        "snapshot_report_listed_players": {
            late[0]: {TEAMS[0]: ["a1", "a2", "a3"]},
            late[1]: {TEAMS[1]: ["b4"]},
        },
    }
    third = {
        "report_out_overrides": {
            (late[0], TEAMS[0]): ["a1", "a2"],
            (late[1], TEAMS[1]): ["b1"],
            (late[2], TEAMS[0]): ["a3"],
        },
        "report_questionable_sets": {
            (late[0], TEAMS[0]): [],
            (late[1], TEAMS[1]): ["b2"],
            (late[2], TEAMS[0]): ["a4"],
        },
        "snapshot_game_ids": set(late),
        "snapshot_report_listed_players": {
            late[0]: {TEAMS[0]: ["a1", "a2"]},
            late[1]: {TEAMS[1]: ["b1", "b2"]},
            late[2]: {TEAMS[0]: ["a3", "a4"]},
        },
    }
    # Horizons that differ from ``probable`` in exactly one thing the row
    # reads, so a cache key missing any one of them would serve a stale row.
    game, team = late[0], TEAMS[0]

    def filed(out, questionable, listed):
        return {
            "report_out_overrides": {(game, team): out},
            "report_questionable_sets": {(game, team): questionable},
            "snapshot_game_ids": set(late),
            "snapshot_report_listed_players": {game: {team: listed}},
        }

    probable = filed([], [], ["a1"])
    out_only = filed(["a1"], [], ["a1"])  # same roster, same questionable set
    questionable_only = filed([], ["a1"], ["a1"])  # same roster, same out set
    # Same out and questionable sets; the report now names an opponent player
    # for this team (a trade), which only changes roster membership.
    membership_only = filed([], [], ["a1", "b3"])
    # The opponent files an empty report: nothing listed, so its roster and
    # sets are unchanged and only its coverage differs (NaN -> 0 questionable).
    opponent_files = filed([], [], ["a1"])
    opponent_files["report_out_overrides"][(game, TEAMS[1])] = []
    opponent_files["report_questionable_sets"][(game, TEAMS[1])] = []
    return [
        first,
        second,
        third,
        second,
        probable,
        out_only,
        questionable_only,
        membership_only,
        probable,
        opponent_files,
    ]


def _run(team_games, players, state, row_cache=None):
    return add_player_history_features(
        team_games.copy(),
        players.copy(),
        pd.DataFrame(columns=["GAME_ID", "TEAM_ID", "PLAYER_ID"]),
        ["PTS", "MIN", "PACE_PER40"],
        return_availability_dict=True,
        include_available_roster_count=True,
        row_cache=row_cache,
        **state,
    )
