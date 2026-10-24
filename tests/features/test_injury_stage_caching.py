"""The snapshot injury stage's speed-ups must not change a single value.

``RowCache`` reuses rows across the per-horizon calls of
``add_player_history_features`` and ``add_all_star_voting_features``; ``_TeamHistoryIndex`` replaces a pandas filter
over the whole team-game history. Both are checked against the computation they
replace, bit for bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.past_injuries.injury_effects import (
    EFFECT_METRICS,
    _continuous_effect_and_se,
    _empty_effect_result,
    _TeamHistoryIndex,
)
from nba_ou.utils.row_cache import RowCache

# ---- _TeamHistoryIndex ------------------------------------------------------
from tests.support.injury_states import (
    _games,
    _report_states,
    _run,
)


def _frame_effect(df_hist, team_id, season_years, before_date, injured, available):
    """The frame-filter implementation ``_TeamHistoryIndex`` replaced."""
    y1, y2 = season_years
    team_hist = df_hist.loc[
        (df_hist["TEAM_ID"].astype("int64") == int(team_id))
        & (df_hist["SEASON_YEAR"].astype("int64").isin([y1, y2]))
        & (df_hist["GAME_DATE"] < before_date)
    ]
    if team_hist.empty:
        return _empty_effect_result()
    game_ids = pd.to_numeric(team_hist["GAME_ID"], errors="coerce").astype("Int64")
    team_hist = team_hist.assign(_GAME_ID_INT=game_ids)
    inj_mask = team_hist["_GAME_ID_INT"].isin(list(injured))
    if available is None:
        present_mask = ~inj_mask
    else:
        present_mask = team_hist["_GAME_ID_INT"].isin(list(available)) & ~inj_mask
    df_inj, df_present = team_hist.loc[inj_mask], team_hist.loc[present_mask]
    n_inj, n_present = len(df_inj), len(df_present)
    if n_inj == 0 or n_present == 0:
        return _empty_effect_result(n_inj, n_present, n_inj + n_present)
    results = [
        _continuous_effect_and_se(df_present[m], df_inj[m]) for m in EFFECT_METRICS
    ]
    return (
        tuple(r[0] for r in results),
        tuple(r[1] for r in results),
        tuple(r[2] for r in results),
        tuple(r[3] for r in results),
        n_inj,
        n_present,
        n_inj + n_present,
    )


def _history(n_games: int = 900, seed: int = 3) -> pd.DataFrame:
    """Two rows per game, as the effect builder lays it out, with gaps."""
    rng = np.random.default_rng(seed)
    teams = np.array([1610612737 + t for t in range(6)])
    home = rng.choice(teams, n_games)
    away = np.array([rng.choice(teams[teams != h]) for h in home])
    dates = pd.Timestamp("2021-10-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 3 * 365, n_games)), unit="D"
    )
    season = np.where(dates.month >= 10, dates.year + 1, dates.year)
    game_ids = np.array([f"00{22100000 + i}" for i in range(n_games)], dtype=object)
    game_ids[rng.choice(n_games, 5, replace=False)] = "not-a-game"
    total = rng.normal(225, 15, n_games)
    total[rng.choice(n_games, 40, replace=False)] = np.nan
    diff = rng.normal(0, 14, n_games)
    spread = rng.normal(0, 12, n_games)
    frames = [
        pd.DataFrame(
            {
                "TEAM_ID": team,
                "GAME_DATE": dates,
                "SEASON_YEAR": pd.array(season, dtype="Int64"),
                "TOTAL_POINTS": total,
                "DIFF_FROM_LINE": diff,
                "SPREAD_ERROR": sign * spread,
                "GAME_ID": game_ids,
            }
        )
        for team, sign in ((home, 1.0), (away, -1.0))
    ]
    return pd.concat(frames, ignore_index=True)


def test_team_history_index_matches_the_frame_filter_bit_for_bit():
    df_hist = _history()
    index = _TeamHistoryIndex(df_hist)
    rng = np.random.default_rng(11)
    valid_ids = pd.to_numeric(df_hist["GAME_ID"], errors="coerce").dropna()
    all_ids = valid_ids.astype("int64").unique()

    checked_nonempty = 0
    for _ in range(400):
        team = int(rng.choice(df_hist["TEAM_ID"].unique()))
        before = pd.Timestamp("2021-10-01") + pd.Timedelta(
            days=int(rng.integers(0, 3 * 365 + 30))
        )
        season_year = int(rng.integers(2021, 2026))
        injured = set(rng.choice(all_ids, int(rng.integers(0, 80)), replace=False))
        available = (
            None
            if rng.random() < 0.3
            else set(rng.choice(all_ids, int(rng.integers(0, 600)), replace=False))
        )
        years = (season_year - 1, season_year)
        expected = _frame_effect(df_hist, team, years, before, injured, available)
        actual = index.player_effect(team, years, before, injured, available)
        # repr compares floats exactly and treats NaN as equal to NaN.
        assert repr(actual) == repr(expected)
        checked_nonempty += expected[4] > 0 and expected[5] > 0
    assert checked_nonempty > 50, "the sample never exercised the effect branch"


def test_team_history_index_memos_match_the_frame_filter_bit_for_bit():
    """Windows shared across players and memberships reused across dates."""
    df_hist = _history()
    index = _TeamHistoryIndex(df_hist)
    rng = np.random.default_rng(12)
    all_ids = (
        pd.to_numeric(df_hist["GAME_ID"], errors="coerce")
        .dropna()
        .astype("int64")
        .unique()
    )
    teams = [int(t) for t in df_hist["TEAM_ID"].unique()]
    players = {}
    for team in teams:
        for player in range(8):
            injured = set(rng.choice(all_ids, int(rng.integers(0, 80)), replace=False))
            available = (
                None
                if player == 0
                else set(rng.choice(all_ids, int(rng.integers(0, 600)), replace=False))
            )
            players[(team, player)] = (injured, available)

    checked_nonempty = 0
    # Few distinct dates, so windows and memberships are both reused heavily,
    # and more queries than the window memo holds, so it is also cleared.
    dates = [
        pd.Timestamp("2021-10-01") + pd.Timedelta(days=int(d))
        for d in rng.integers(0, 3 * 365 + 30, 60)
    ]
    for _ in range(3 * _TeamHistoryIndex._MAX_WINDOWS):
        team = int(rng.choice(teams))
        player = int(rng.integers(0, 8))
        before = dates[int(rng.integers(0, len(dates)))]
        season_year = int(rng.integers(2021, 2026))
        years = (season_year - 1, season_year)
        injured, available = players[(team, player)]
        expected = _frame_effect(df_hist, team, years, before, injured, available)
        actual = index.player_effect(
            team, years, before, injured, available, player_key=player
        )
        assert repr(actual) == repr(expected)
        checked_nonempty += expected[4] > 0 and expected[5] > 0
    assert checked_nonempty > 50, "the sample never exercised the effect branch"


def test_effect_cache_shared_across_instances_changes_nothing():
    """The snapshot builder shares one cache across families and horizons."""
    df_hist = _history()
    cache = RowCache()
    rng = np.random.default_rng(13)
    all_ids = (
        pd.to_numeric(df_hist["GAME_ID"], errors="coerce")
        .dropna()
        .astype("int64")
        .unique()
    )
    teams = [int(t) for t in df_hist["TEAM_ID"].unique()]
    dates = [
        pd.Timestamp("2021-10-01") + pd.Timedelta(days=int(d))
        for d in rng.integers(0, 3 * 365 + 30, 40)
    ]
    # Each "horizon" redraws a few players' sets and keeps the rest, as a new
    # report changes only some memberships.
    sets = {
        (team, player): (
            set(rng.choice(all_ids, int(rng.integers(0, 80)), replace=False)),
            set(rng.choice(all_ids, int(rng.integers(0, 600)), replace=False)),
        )
        for team in teams
        for player in range(6)
    }
    queries = [
        (
            int(rng.choice(teams)),
            int(rng.integers(0, 6)),
            dates[int(rng.integers(0, 40))],
        )
        for _ in range(300)
    ]
    for horizon in range(4):
        for key in list(sets)[horizon::5]:
            sets[key] = (
                set(rng.choice(all_ids, int(rng.integers(0, 80)), replace=False)),
                sets[key][1],
            )
        shared = _TeamHistoryIndex(df_hist, cache)
        plain = _TeamHistoryIndex(df_hist)
        for team, player, before in queries:
            injured, available = sets[(team, player)]
            years = (
                (before.year - 1, before.year)
                if before.month < 10
                else (
                    before.year,
                    before.year + 1,
                )
            )
            expected = plain.player_effect(team, years, before, injured, available)
            actual = shared.player_effect(
                team, years, before, injured, available, player_key=player
            )
            assert repr(actual) == repr(expected)
    assert cache.hits > 0 and cache.misses > 0


def test_effect_cache_refuses_a_different_history():
    cache = RowCache()
    _TeamHistoryIndex(_history(seed=3), cache)
    with pytest.raises(ValueError):
        _TeamHistoryIndex(_history(seed=4), cache)


def test_team_history_index_unknown_team_is_empty():
    index = _TeamHistoryIndex(_history(50))
    result = index.player_effect(1, (2021, 2022), pd.Timestamp("2030-01-01"), {1}, None)
    assert repr(result) == repr(_empty_effect_result())


# ---- RowCache: players -----------------------------------------------------------


def test_row_cache_reproduces_every_horizon_exactly():
    team_games, players = _games()
    cache = RowCache()
    for state in _report_states(team_games):
        expected_team, expected_injured, expected_availability = _run(
            team_games, players, state
        )
        team, injured, availability = _run(team_games, players, state, cache)
        pd.testing.assert_frame_equal(team, expected_team, check_exact=True)
        assert injured == expected_injured
        assert availability == expected_availability
        assert list(availability) == list(expected_availability)

    # Rows away from the reported games never change, so most calls after the
    # first are served from the cache.
    n_states = len(_report_states(team_games))
    assert cache.hits >= (n_states - 1) * (len(team_games) - 6)
    assert cache.misses < 3 * len(team_games)


def test_row_cache_rebuilds_a_row_when_its_report_changes():
    team_games, players = _games()
    states = _report_states(team_games)
    cache = RowCache()
    _run(team_games, players, states[0], cache)
    misses = cache.misses
    _run(team_games, players, states[1], cache)
    # Two reported team-games changed, plus their opponents (roster membership
    # is read per game, so both sides of the game see the new report).
    assert cache.misses - misses >= 2


def test_row_cache_refuses_different_inputs():
    team_games, players = _games()
    state = _report_states(team_games)[0]
    cache = RowCache()
    _run(team_games, players, state, cache)
    with pytest.raises(ValueError, match="different inputs"):
        _run(team_games.iloc[:-2], players, state, cache)


# ---- RowCache: All-Star voting ------------------------------------------------

BOS, LAL, NYK, MIA = "1610612738", "1610612747", "1610612752", "1610612748"


def _all_star_inputs():
    day, earlier = pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-01")
    team_rows = pd.DataFrame(
        [
            (game, team, "22025", 2025, date)
            for game, date, teams in (
                ("e1", earlier, (BOS, LAL)),
                ("e2", earlier, (NYK, MIA)),
                ("x", day, (BOS, LAL)),
                ("y", day, (NYK, MIA)),
            )
            for team in teams
        ],
        columns=["GAME_ID", "TEAM_ID", "SEASON_ID", "SEASON_YEAR", "GAME_DATE"],
    )
    players = pd.DataFrame(
        [
            ("h1", "p1", BOS, "22025", 2025, "2026-01-10", 30, 25),
            ("h2", "p2", LAL, "22025", 2025, "2026-01-10", 32, 28),
            ("h3", "p3", NYK, "22025", 2025, "2026-01-10", 34, 22),
            ("h4", "p4", MIA, "22025", 2025, "2026-01-10", 28, 18),
        ],
        columns=[
            "GAME_ID",
            "PLAYER_ID",
            "TEAM_ID",
            "SEASON_ID",
            "SEASON_YEAR",
            "GAME_DATE",
            "MIN",
            "PTS",
        ],
    )
    votes = pd.DataFrame(
        [
            (2025, "p1", "Boston Celtics", 40, 1.5),
            (2025, "p2", "Los Angeles Lakers", 60, 2.5),
            (2025, "p3", "New York Knicks", 30, 3.5),
            (2025, "p4", "Miami Heat", 20, 4.5),
        ],
        columns=["season_year", "player_id", "team_name", "fan_votes", "score"],
    )
    return team_rows, players, votes


def _all_star_states():
    """Each state differs from the one before in one thing a row reads."""
    return [
        ({}, None),
        ({"x": {BOS: ["p1"]}}, None),  # the row's own game
        # Another game that day lists p1 for the Knicks: the Celtics row in
        # game x changes although game x's own entries did not.
        ({"y": {NYK: ["p1"]}}, None),
        ({}, {"x": {BOS: []}}),  # questionable group filed, empty
        ({}, {"x": {BOS: ["p2"]}}),
        ({}, {"x": {BOS: ["p1"]}}),
        ({}, None),
    ]


def test_all_star_row_cache_reproduces_every_horizon_exactly():
    from nba_ou.data_processing.all_star_voting.attach_all_star_voting_features import (
        add_all_star_voting_features,
    )

    team_rows, players, votes = _all_star_inputs()
    cache = RowCache()
    for injured, questionable in _all_star_states():
        expected = add_all_star_voting_features(
            team_rows, players, votes, injured, questionable
        )
        actual = add_all_star_voting_features(
            team_rows, players, votes, injured, questionable, row_cache=cache
        )
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert cache.hits > 0
