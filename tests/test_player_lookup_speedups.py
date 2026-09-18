"""``create_player_lookup`` must return exactly what it returned before.

The lookup now selects a roster's rows through a per-player position index
instead of masking the whole season on every call. Every call is compared with
the previous implementation (``tests/legacy_player_lookup.py``): same rows, same
order, same index, dtypes and values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.past_injuries.past_injuries import create_player_lookup

from .frame_identity import assert_identical
from .legacy_player_lookup import create_player_lookup_before

TEAMS = [1610612737 + t for t in range(4)]


def players_frame(seed: int = 0) -> pd.DataFrame:
    """Box scores over two seasons with trades, DNPs, scheduled placeholders,
    playoff season ids that fall back to the season year, and missing ids."""
    rng = np.random.default_rng(seed)
    rows = []
    for season_year in (2022, 2023):
        start = pd.Timestamp(f"{season_year - 1}-10-20")
        roster = {team: list(range(team * 100, team * 100 + 9)) for team in TEAMS}
        for day in range(40):
            date = start + pd.Timedelta(days=day * 2)
            season_id = f"4{season_year}" if day >= 36 else f"2{season_year}"
            if day == 20:
                # A trade: two players swap teams mid-season.
                a, b = TEAMS[0], TEAMS[1]
                roster[a][0], roster[b][0] = roster[b][0], roster[a][0]
            for pair in range(len(TEAMS) // 2):
                home, away = TEAMS[2 * pair], TEAMS[2 * pair + 1]
                if day % 3 == 0:
                    home, away = TEAMS[pair], TEAMS[3 - pair]
                game_id = f"00{season_year - 2000}{day:04d}{pair}"
                for team in (home, away):
                    for player in roster[team]:
                        if rng.random() < 0.1:
                            continue
                        minutes = rng.choice(
                            [0.0, 12.0, 30.0, np.nan], p=[0.15, 0.4, 0.4, 0.05]
                        )
                        rows.append(
                            {
                                "GAME_ID": game_id,
                                "SEASON_ID": season_id,
                                "SEASON_YEAR": season_year,
                                "TEAM_ID": team,
                                "PLAYER_ID": player,
                                "PLAYER_NAME": f"P{player}",
                                "GAME_DATE": date,
                                "MIN": minutes,
                                "PTS": float(rng.integers(0, 30)),
                            }
                        )
    df = pd.DataFrame(rows)
    df.loc[rng.choice(len(df), 5, replace=False), "PLAYER_ID"] = np.nan
    df = df.sample(frac=1.0, random_state=seed)
    df.index = rng.permutation(len(df)) + 1000
    return df.sort_values(["PLAYER_ID", "GAME_DATE"], kind="mergesort")


def injured_dict(df: pd.DataFrame, seed: int) -> dict:
    """Listed players, including one on a team he has no box score for."""
    rng = np.random.default_rng(seed)
    games = df[["GAME_ID", "TEAM_ID"]].drop_duplicates().to_numpy()
    out: dict = {}
    for game_id, team_id in games[rng.random(len(games)) < 0.4]:
        players = df.loc[
            (df["GAME_ID"] == game_id) & (df["TEAM_ID"] == team_id), "PLAYER_ID"
        ].dropna()
        listed = list(players.sample(min(2, len(players)), random_state=1).astype(int))
        if rng.random() < 0.2:
            listed.append(TEAMS[3] * 100 + 1)
        out.setdefault(game_id, {})[team_id] = listed
    return out


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("with_injuries", [False, True])
def test_lookup_returns_the_same_frames(seed, with_injuries):
    df = players_frame(seed)
    injuries = injured_dict(df, seed) if with_injuries else None
    new = create_player_lookup(df, injured_dict=injuries)
    old = create_player_lookup_before(df, injured_dict=injuries)

    games = df[["SEASON_ID", "TEAM_ID", "GAME_DATE", "GAME_ID"]].drop_duplicates()
    non_empty = 0
    for season_id, team_id, date, game_id in games.itertuples(index=False):
        for args in (
            (season_id, team_id, date),
            (season_id, str(team_id), date + pd.Timedelta(hours=12)),
            (season_id, team_id, date.to_datetime64()),
        ):
            for kwargs in ({"game_id": game_id}, {}):
                expected = old(*args, **kwargs)
                actual = new(*args, **kwargs)
                assert_identical(actual, expected)
                non_empty += not expected.empty
    assert non_empty > 100


# ---- add_player_history_features ----------------------------------------------


def _use_previous_player_code(monkeypatch):
    """Route the builder through the per-statistic calls and the lookup it had."""
    from nba_ou.data_processing.players import attach_player_features

    from .legacy_top_n_averages import get_top_n_averages_with_names_before

    monkeypatch.setattr(
        attach_player_features,
        "latest_player_states",
        lambda df, date, *, injured: (df, date, injured),
    )
    monkeypatch.setattr(
        attach_player_features,
        "rank_player_states",
        lambda states, stat_col, *, n_players: get_top_n_averages_with_names_before(
            states[0],
            date=states[1],
            stat_col=stat_col,
            injured=states[2],
            n_players=n_players,
        ),
    )
    monkeypatch.setattr(
        attach_player_features, "create_player_lookup", create_player_lookup_before
    )


def test_player_history_features_match_the_previous_code_path(monkeypatch):
    from nba_ou.utils.row_cache import RowCache

    from .test_injury_stage_caching import _games, _report_states, _run

    team_games, players = _games()
    states = _report_states(team_games)
    actual_cache = RowCache()
    actual = [_run(team_games, players, state, actual_cache) for state in states]

    _use_previous_player_code(monkeypatch)
    expected_cache = RowCache()
    for state, (team, injured, availability) in zip(states, actual, strict=True):
        expected_team, expected_injured, expected_availability = _run(
            team_games, players, state, expected_cache
        )
        assert_identical(team, expected_team)
        assert injured == expected_injured
        assert availability == expected_availability
        # Uncached as well, so a stale cache entry cannot mask a difference.
        uncached = _run(team_games, players, state)
        assert_identical(team, uncached[0])


def test_top_n_split_matches_the_previous_function():
    from nba_ou.data_processing.players.players_statistics import (
        get_top_n_averages_with_names,
        precompute_cumulative_avg_stat,
    )

    from .legacy_top_n_averages import get_top_n_averages_with_names_before

    df = players_frame(4).dropna(subset=["PLAYER_ID"])
    for stat in ("PTS", "MIN"):
        df = precompute_cumulative_avg_stat(df, stat_col=stat)
    dates = sorted(df["GAME_DATE"].unique())[::7]
    for team in TEAMS:
        roster = df[df["TEAM_ID"] == team]
        for date in dates:
            for injured in (False, True):
                for stat, drop_min in (
                    ("PTS", False),
                    ("MIN", False),
                    ("PTS", True),
                    ("DEF_RATING", False),
                ):
                    frame = roster.drop(columns="MIN_CUM_AVG") if drop_min else roster
                    if stat == "DEF_RATING":
                        frame = frame.assign(DEF_RATING_CUM_AVG=frame["PTS_CUM_AVG"])
                    for n in (3, 50):
                        assert get_top_n_averages_with_names(
                            frame, date, stat_col=stat, injured=injured, n_players=n
                        ) == get_top_n_averages_with_names_before(
                            frame, date, stat_col=stat, injured=injured, n_players=n
                        )


def test_top_n_ranking_keeps_pandas_tie_and_nan_order():
    """Ties, NaN and signed zeros in the ranked value, both sort directions."""
    from nba_ou.data_processing.players.players_statistics import (
        get_top_n_averages_with_names,
        precompute_cumulative_avg_stat,
    )

    from .legacy_top_n_averages import get_top_n_averages_with_names_before

    rng = np.random.default_rng(9)
    df = players_frame(5).dropna(subset=["PLAYER_ID"])
    for stat in ("PTS", "MIN"):
        df = precompute_cumulative_avg_stat(df, stat_col=stat)
    coarse = np.round(df["PTS_CUM_AVG"].to_numpy() / 5) * 5
    coarse[rng.random(len(coarse)) < 0.1] = np.nan
    coarse[rng.random(len(coarse)) < 0.1] = -0.0
    df = df.assign(PTS_CUM_AVG=coarse, DEF_RATING_CUM_AVG=coarse)
    dates = sorted(df["GAME_DATE"].unique())[::5]
    compared = 0
    for team in TEAMS:
        roster = df[df["TEAM_ID"] == team]
        for date in dates:
            for injured in (False, True):
                for stat in ("PTS", "DEF_RATING", "MIN"):
                    for n in (2, 50):
                        actual = get_top_n_averages_with_names(
                            roster, date, stat_col=stat, injured=injured, n_players=n
                        )
                        expected = get_top_n_averages_with_names_before(
                            roster, date, stat_col=stat, injured=injured, n_players=n
                        )
                        # NaN != NaN, so compare the text form of each tuple.
                        assert list(map(repr, actual)) == list(map(repr, expected))
                        compared += len(expected)
    assert compared > 1000


def test_precomputed_stat_players_change_nothing():
    """Passing the once-per-stage precompute equals computing it per call."""
    from nba_ou.data_processing.players.attach_player_features import (
        add_player_history_features,
        precompute_stat_players,
    )

    from .test_injury_stage_caching import _games, _report_states

    team_games, players = _games()
    stats = ["PTS", "MIN", "PACE_PER40"]
    stat_players = precompute_stat_players(players.copy(), stats)
    frozen = stat_players.copy()
    for state in _report_states(team_games)[:4]:
        args = (
            team_games.copy(),
            players.copy(),
            pd.DataFrame(columns=["GAME_ID", "TEAM_ID", "PLAYER_ID"]),
            stats,
        )
        kwargs = {
            "return_availability_dict": True,
            "include_available_roster_count": True,
            **state,
        }
        expected = add_player_history_features(*args, **kwargs)
        actual = add_player_history_features(*args, stat_players=stat_players, **kwargs)
        assert_identical(actual[0], expected[0])
        assert actual[1:] == expected[1:]
    # Shared across calls, so it must never be modified by one.
    assert_identical(stat_players, frozen)
