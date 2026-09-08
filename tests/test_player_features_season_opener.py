"""Player features must survive a team's season opener.

``create_player_lookup`` resolves a team's roster by asking who last played for it
*earlier in the same season*, and ``precompute_cumulative_avg_stat`` groups its
EWMA by ``(SEASON_YEAR, PLAYER_ID)``. Both reset at the season boundary, so at a
team's opener the lookup returned nobody and every top-N player column on that row
stayed missing -- ~190 numeric columns, enough for the downstream NaN-per-row
limit to discard the row entirely.

Filling those with 0 would be wrong for most of them: ``OFF_RATING`` runs 87-165
in real data and ``TS_PCT`` 0.38-1.33, so a 0 is not a neutral value but an
extreme that never occurs, and a tree would read the opener as "worst offense on
record". The players did play last season, so the fix is the same fallback used
for team trends: current season -> previous REGULAR season -> 0.
"""

from __future__ import annotations

import pandas as pd
import pytest
from nba_ou.data_processing.players.attach_player_features import (
    add_player_history_features,
    drop_player_identifier_columns,
    is_player_identifier_column,
)

TEAM = "1610612738"
REGULAR, PLAYOFFS = "002", "004"

TOP1_VALUE = "TOP1_PLAYER_PTS_BEFORE"
TOP1_ID = "TOP1_PLAYER_ID_PTS_BEFORE"


def _team_games(season_year, prefix, n, start, tag=""):
    return pd.DataFrame(
        {
            "GAME_ID": [f"{prefix}{season_year}{tag}{i:03d}" for i in range(n)],
            "TEAM_ID": TEAM,
            "SEASON_ID": f"2{season_year}",
            "SEASON_YEAR": season_year,
            "GAME_DATE": pd.date_range(start, periods=n, freq="3D"),
        }
    )


def _player_rows(df_team, points_by_player):
    rows = []
    for game in df_team.itertuples():
        for player_id, points in points_by_player.items():
            rows.append(
                {
                    "GAME_ID": game.GAME_ID,
                    "TEAM_ID": TEAM,
                    "SEASON_ID": game.SEASON_ID,
                    "SEASON_YEAR": game.SEASON_YEAR,
                    "GAME_DATE": game.GAME_DATE,
                    "PLAYER_ID": player_id,
                    "PLAYER_NAME": f"Player {player_id}",
                    "MIN": 30.0,
                    "PTS": points,
                    "OFF_RATING": 110.0,
                    "DEF_RATING": 105.0,
                    "TS_PCT": 0.55,
                    "PACE_PER40": 99.0,
                }
            )
    return pd.DataFrame(rows)


def _no_injuries():
    return pd.DataFrame(columns=["GAME_ID", "TEAM_ID", "PLAYER_ID"])


def _run(df_team, df_players):
    out, _ = add_player_history_features(
        df_team.copy(), df_players, _no_injuries(), stat_cols=["PTS"]
    )
    return out.sort_values(["SEASON_YEAR", "GAME_DATE"])


def test_opener_inherits_the_previous_regular_season_average():
    previous = _team_games(2023, REGULAR, 12, "2023-11-01")
    current = _team_games(2024, REGULAR, 3, "2024-11-01")
    df_team = pd.concat([previous, current], ignore_index=True)
    # The leading scorer averaged 28 last season and has no games yet this one.
    df_players = _player_rows(df_team, {"1": 28.0, "2": 12.0})

    out = _run(df_team, df_players)
    opener = out[out.SEASON_YEAR == 2024].iloc[0]

    assert opener[TOP1_VALUE] == pytest.approx(28.0)
    assert opener[TOP1_ID] == "1"


def test_the_opener_value_is_neither_missing_nor_zero():
    """The two options the fallback exists to avoid."""
    df_team = pd.concat(
        [
            _team_games(2023, REGULAR, 12, "2023-11-01"),
            _team_games(2024, REGULAR, 3, "2024-11-01"),
        ],
        ignore_index=True,
    )
    df_players = _player_rows(df_team, {"1": 28.0, "2": 12.0})

    opener = _run(df_team, df_players)[lambda d: d.SEASON_YEAR == 2024].iloc[0]

    assert not pd.isna(opener[TOP1_VALUE])
    assert opener[TOP1_VALUE] != 0


def test_the_player_fallback_reads_the_regular_season_not_the_playoffs():
    df_team = pd.concat(
        [
            _team_games(2023, REGULAR, 12, "2023-11-01"),
            # A scoring collapse in the playoffs, chronologically the last thing
            # played before the new season.
            _team_games(2023, PLAYOFFS, 6, "2024-04-20", tag="p"),
            _team_games(2024, REGULAR, 3, "2024-11-01"),
        ],
        ignore_index=True,
    )
    regular_games = df_team[df_team.GAME_ID.str.startswith(REGULAR)]
    playoff_games = df_team[df_team.GAME_ID.str.startswith(PLAYOFFS)]
    df_players = pd.concat(
        [
            _player_rows(regular_games, {"1": 28.0, "2": 12.0}),
            _player_rows(playoff_games, {"1": 4.0, "2": 12.0}),
        ],
        ignore_index=True,
    )

    opener = _run(df_team, df_players)[lambda d: d.SEASON_YEAR == 2024].iloc[0]

    assert opener[TOP1_VALUE] == pytest.approx(28.0)
    assert opener[TOP1_VALUE] != pytest.approx(4.0)


def test_opener_without_any_prior_roster_history_stays_missing():
    df_team = _team_games(2024, REGULAR, 3, "2024-11-01")
    df_players = _player_rows(df_team, {"1": 28.0, "2": 12.0})

    opener = _run(df_team, df_players).iloc[0]

    assert pd.isna(opener[TOP1_VALUE])


def test_the_opener_is_no_emptier_than_a_settled_row():
    """The whole point: the opener row must stop being the one that gets
    discarded for having far more missing values than any other."""
    df_team = pd.concat(
        [
            _team_games(2023, REGULAR, 12, "2023-11-01"),
            _team_games(2024, REGULAR, 4, "2024-11-01"),
        ],
        ignore_index=True,
    )
    df_players = _player_rows(df_team, {"1": 28.0, "2": 12.0})

    out = _run(df_team, df_players)
    current = out[out.SEASON_YEAR == 2024]
    player_cols = [c for c in out.columns if "PLAYER" in c or c.startswith("BENCH_")]

    opener_na = current[player_cols].isna().sum(axis=1).iloc[0]
    settled_na = current[player_cols].isna().sum(axis=1).iloc[-1]

    assert opener_na <= settled_na


# --- identifier columns ----------------------------------------------------


@pytest.mark.parametrize(
    "column",
    [
        "TOP1_PLAYER_ID_PTS_BEFORE_TEAM_HOME",
        "TOP3_PLAYER_NAME_OFF_RATING_BEFORE_TEAM_AWAY",
        "TOP1_INJURED_PLAYER_ID_MIN_BEFORE_TEAM_HOME",
        "TOP2_INJURED_PLAYER_NAME_TS_PCT_BEFORE_TEAM_AWAY",
    ],
)
def test_identifier_columns_are_recognised(column):
    assert is_player_identifier_column(column)


@pytest.mark.parametrize(
    "column",
    [
        # The value columns -- these are the actual features and must survive.
        "TOP1_PLAYER_PTS_BEFORE_TEAM_HOME",
        "TOP1_PLAYER_MIN_BEFORE_TEAM_HOME",
        "TOP1_INJURED_PLAYER_OFF_RATING_BEFORE_TEAM_AWAY",
        # Player-level keys used inside the pipeline, not top-N bookkeeping.
        "PLAYER_ID",
        "PLAYER_NAME",
        "N_ACTIVE_PLAYERS_BEFORE_TEAM_HOME",
    ],
)
def test_non_identifier_columns_are_kept(column):
    assert not is_player_identifier_column(column)


def test_dropping_identifiers_keeps_every_value_column():
    df = pd.DataFrame(
        {
            "GAME_ID": ["1"],
            "TOP1_PLAYER_PTS_BEFORE_TEAM_HOME": [28.0],
            "TOP1_PLAYER_ID_PTS_BEFORE_TEAM_HOME": ["201939"],
            "TOP1_PLAYER_NAME_PTS_BEFORE_TEAM_HOME": ["Stephen Curry"],
            "TOP2_INJURED_PLAYER_ID_MIN_BEFORE_TEAM_AWAY": ["1610612747"],
            "TOP2_INJURED_PLAYER_NAME_MIN_BEFORE_TEAM_AWAY": ["Someone Else"],
            "N_ACTIVE_PLAYERS_BEFORE_TEAM_HOME": [10],
        }
    )

    out = drop_player_identifier_columns(df)

    assert list(out.columns) == [
        "GAME_ID",
        "TOP1_PLAYER_PTS_BEFORE_TEAM_HOME",
        "N_ACTIVE_PLAYERS_BEFORE_TEAM_HOME",
    ]


def test_dropping_identifiers_is_a_noop_when_there_are_none():
    df = pd.DataFrame({"GAME_ID": ["1"], "TOP1_PLAYER_PTS_BEFORE_TEAM_HOME": [28.0]})

    out = drop_player_identifier_columns(df)

    assert list(out.columns) == list(df.columns)
