"""``pair_game``: observed guarding per game, reconciled with the stints."""

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.player_graph.pair_game import (
    COLUMNS,
    MATCHUP_COUNT_COLUMNS,
    NBA_SHARE_COLUMNS,
    build_pair_game,
    cofloor_seconds,
)

GAME = "0022300001"
HOME, AWAY = "100", "200"
H = [f"h{k}" for k in range(7)]
A = [f"a{k}" for k in range(6)]


def _stints():
    lineups = [
        (H[:5], A[:5], 300.0),
        ([*H[:4], H[5]], A[:5], 120.0),
        ([*H[1:5], H[6]], [*A[:4], A[5]], 60.0),
    ]
    return pd.DataFrame(
        {
            "game_id": GAME,
            "game_date": pd.Timestamp("2023-10-24"),
            "season_year": 2023,
            "home_team_id": HOME,
            "away_team_id": AWAY,
            "home_lineup": [np.array(h) for h, _, _ in lineups],
            "away_lineup": [np.array(a) for _, a, _ in lineups],
            "seconds": [s for _, _, s in lineups],
        }
    )


def _matchup(off, dfn, seconds, **counts):
    off_home = off.startswith("h")
    return {
        **dict.fromkeys(MATCHUP_COUNT_COLUMNS, 0),
        **dict.fromkeys(NBA_SHARE_COLUMNS, 0.1),
        "game_id": GAME,
        "home_team_id": HOME,
        "off_team_id": HOME if off_home else AWAY,
        "def_team_id": AWAY if off_home else HOME,
        "off_player_id": off,
        "def_player_id": dfn,
        "matchup_seconds": seconds,
        "partial_possessions": seconds / 15,
        **counts,
    }


def _matchups(*extra):
    rows = [
        _matchup("h0", "a0", 120.0, matchup_fga=3, player_points=4),
        _matchup("h0", "a1", 60.0),
        _matchup("a0", "h0", 90.0),
        *extra,
    ]
    return pd.DataFrame(rows)


def _row(pairs, off, dfn):
    found = pairs.loc[pairs.off_player_id.eq(off) & pairs.def_player_id.eq(dfn)]
    assert len(found) == 1
    return found.iloc[0]


def test_cofloor_sums_to_five_times_floor_time():
    stints = _stints()
    cofloor = cofloor_seconds(stints)
    for player in H:
        on = stints.loc[
            stints.home_lineup.map(lambda x, p=player: p in x), "seconds"
        ].sum()
        mine = cofloor.loc[cofloor.home_player_id.eq(player), "cofloor_seconds"].sum()
        assert mine == pytest.approx(5 * on)


def test_every_cofloor_pair_appears_once_per_direction():
    pairs = build_pair_game(_stints(), _matchups())
    assert list(pairs.columns) == list(COLUMNS)
    assert not pairs.duplicated(["game_id", "off_player_id", "def_player_id"]).any()
    shared = pairs.loc[pairs.cofloor_seconds.gt(0)]
    forward = set(zip(shared.off_player_id, shared.def_player_id, strict=True))
    assert forward == {(d, o) for o, d in forward}
    # h0 shared all 420 s of the first two stints with a0.
    assert _row(pairs, "h0", "a0").cofloor_seconds == 420.0
    assert _row(pairs, "a0", "h0").cofloor_seconds == 420.0


def test_rate_and_raw_columns():
    pairs = build_pair_game(_stints(), _matchups())
    row = _row(pairs, "h0", "a0")
    assert row.guard_rate == pytest.approx(120 / 420)
    assert row.matchup_fga == 3 and row.player_points == 4
    assert row.attacker_is_home and row.off_team_id == HOME
    assert row.has_matchup_row
    away = _row(pairs, "a0", "h0")
    assert not away.attacker_is_home and away.def_team_id == HOME


def test_shared_floor_without_a_matchup_row_is_a_real_zero():
    row = _row(build_pair_game(_stints(), _matchups()), "h2", "a3")
    assert not row.has_matchup_row
    assert row.cofloor_seconds == 480.0
    assert row.matchup_seconds == 0 and row.guard_rate == 0
    assert np.isnan(row.pct_total_time_both_on)


def test_matchup_without_shared_floor_is_kept_with_a_nan_rate():
    # h5 and a5 never shared a stint.
    pairs = build_pair_game(_stints(), _matchups(_matchup("a5", "h5", 4.0)))
    row = _row(pairs, "a5", "h5")
    assert row.cofloor_seconds == 0 and row.matchup_seconds == 4.0
    assert np.isnan(row.guard_rate)
    assert not row.attacker_is_home and row.off_team_id == AWAY


def test_rate_is_clipped_at_one_but_raw_seconds_are_kept():
    # h6 and a5 shared only the 60 s stint.
    pairs = build_pair_game(_stints(), _matchups(_matchup("h6", "a5", 75.0)))
    row = _row(pairs, "h6", "a5")
    assert row.guard_rate == 1.0 and row.matchup_seconds == 75.0
    assert pairs.guard_rate.dropna().le(1).all()


def test_games_with_only_one_source_are_skipped():
    other = _matchups().assign(game_id="0022300002")
    pairs = build_pair_game(_stints(), pd.concat([_matchups(), other]))
    assert set(pairs.game_id) == {GAME}
    assert build_pair_game(_stints(), _matchups().iloc[0:0]).empty
    assert build_pair_game(pd.DataFrame(), _matchups()).empty


def test_duplicate_matchup_rows_are_rejected():
    with pytest.raises(ValueError, match="more than one row"):
        build_pair_game(_stints(), _matchups(_matchup("h0", "a0", 1.0)))
