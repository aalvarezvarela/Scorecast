"""The player/availability column set is a profile, and the reduction holds.

The presence/absence block was 216 columns of which the training pipeline's own
correlation filter already deleted 120. This pins what the reduced profile emits
so the cut cannot silently grow back, and guards the two things the cut was NOT
allowed to break: the ids the availability-effect features read, and the four
injured slots the fresh-absence features sum over.
"""

import pandas as pd
import pytest
from nba_ou.data_processing.players.attach_player_features import (
    add_player_history_features,
)
from nba_ou.data_processing.players.feature_profile import (
    ACTIVE_PROFILE,
    COUNTING_STAT_COLS,
    LEGACY_PROFILE,
    RATE_STAT_COLS,
    REDUCED_PROFILE,
)
from nba_ou.data_processing.players.fresh_absence import STREAK_STAT_COLS

STATS = ["PTS", "PACE_PER40", "DEF_RATING", "OFF_RATING", "TS_PCT", "MIN"]
TEAM = "1610612738"


def _frames(n_players: int = 9, n_games: int = 5):
    df_team = pd.DataFrame(
        {
            "GAME_ID": [f"002250000{i}" for i in range(n_games)],
            "TEAM_ID": TEAM,
            "SEASON_ID": "22025",
            "SEASON_YEAR": 2025,
            "GAME_DATE": pd.date_range("2025-10-20", periods=n_games, freq="3D"),
        }
    )
    rows = []
    for game in df_team.itertuples():
        for p in range(n_players):
            rows.append(
                {
                    "GAME_ID": game.GAME_ID,
                    "TEAM_ID": TEAM,
                    "SEASON_ID": game.SEASON_ID,
                    "SEASON_YEAR": game.SEASON_YEAR,
                    "GAME_DATE": game.GAME_DATE,
                    "PLAYER_ID": 200000 + p,
                    "PLAYER_NAME": f"Player {p}",
                    "MIN": 34.0 - 3 * p,
                    "PTS": 28.0 - 2 * p,
                    "OFF_RATING": 118.0 - p,
                    "DEF_RATING": 104.0 + p,
                    "TS_PCT": 0.60 - 0.01 * p,
                    "PACE_PER40": 99.0 + p,
                }
            )
    df_players = pd.DataFrame(rows)
    df_injuries = pd.DataFrame(
        {
            "GAME_ID": [df_team.GAME_ID.iloc[3], df_team.GAME_ID.iloc[3]],
            "TEAM_ID": TEAM,
            "PLAYER_ID": [200000, 200001],
        }
    )
    return df_team, df_players, df_injuries


@pytest.fixture(scope="module")
def emitted() -> list[str]:
    df_team, df_players, df_injuries = _frames()
    out, _ = add_player_history_features(df_team, df_players, df_injuries, STATS)
    return list(out.columns)


def test_rate_statistics_lose_their_per_slot_columns(emitted):
    """Per-slot rate columns repeat each other (r=0.97 between slots 2 and 3)."""
    for stat in RATE_STAT_COLS:
        for slot in range(1, 7):
            assert f"TOP{slot}_PLAYER_{stat}_BEFORE" not in emitted
            assert f"TOP{slot}_INJURED_PLAYER_{stat}_BEFORE" not in emitted


def test_counting_statistics_keep_their_per_slot_columns(emitted):
    for stat in COUNTING_STAT_COLS:
        for slot in range(1, REDUCED_PROFILE.active_value_slots + 1):
            assert f"TOP{slot}_PLAYER_{stat}_BEFORE" in emitted
        # deep active slots are described by the team-level rolling stats
        assert f"TOP6_PLAYER_{stat}_BEFORE" not in emitted


def test_all_four_injured_slots_survive_for_fresh_absence(emitted):
    """fresh_absence.py sums over four slots; cutting them would redefine it."""
    for stat in STREAK_STAT_COLS:
        for slot in range(1, 5):
            assert f"TOP{slot}_INJURED_PLAYER_{stat}_BEFORE" in emitted, (stat, slot)
            assert f"TOP{slot}_INJURED_STREAK_{stat}_BEFORE" in emitted, (stat, slot)


def test_identifier_columns_the_availability_effects_read_are_kept(emitted):
    """These are consumed by add_top3_availability_effect_features_for_columns."""
    for slot in (1, 2, 3):
        assert f"TOP{slot}_PLAYER_ID_PTS_BEFORE" in emitted
        assert f"TOP{slot}_INJURED_PLAYER_ID_PTS_BEFORE" in emitted
    assert "TOP1_PLAYER_ID_MIN_BEFORE" in emitted
    assert "TOP1_INJURED_PLAYER_ID_MIN_BEFORE" in emitted


def test_rate_sums_are_gone_and_counting_sums_remain(emitted):
    """A sum of rates just re-counts the injured players (r=0.98)."""
    for stat in RATE_STAT_COLS:
        assert f"TOTAL_INJURED_PLAYER_{stat}_BEFORE" not in emitted
        assert f"AVG_INJURED_{stat}_BEFORE" not in emitted
    for stat in COUNTING_STAT_COLS:
        assert f"TOTAL_INJURED_PLAYER_{stat}_BEFORE" in emitted
        assert f"AVG_INJURED_{stat}_BEFORE" in emitted


def test_weighted_rate_aggregates_replace_the_per_slot_rates(emitted):
    for stat in REDUCED_PROFILE.weighted_rate_stats:
        assert f"ACTIVE_WEIGHTED_{stat}_BEFORE" in emitted
        assert f"INJURED_WEIGHTED_{stat}_BEFORE" in emitted


def _expected_weighted(players: range) -> tuple[float, float]:
    """(minutes-weighted rating, plain mean rating) for the given players."""
    ratings = [118.0 - p for p in players]
    minutes = [34.0 - 3 * p for p in players]
    weighted = sum(r * m for r, m in zip(ratings, minutes, strict=True)) / sum(minutes)
    return weighted, sum(ratings) / len(ratings)


def test_weighted_aggregate_is_minutes_weighted_not_a_plain_mean():
    """A 34-minute starter must count for more than a 10-minute reserve."""
    df_team, df_players, df_injuries = _frames()
    out, _ = add_player_history_features(df_team, df_players, df_injuries, STATS)
    out = out.sort_values("GAME_DATE").reset_index(drop=True)
    value = out["ACTIVE_WEIGHTED_OFF_RATING_BEFORE"]

    # Game 3 is the one with players 0 and 1 injured, so its active set differs.
    healthy_weighted, healthy_mean = _expected_weighted(range(9))
    injured_weighted, injured_mean = _expected_weighted(range(2, 9))
    for row, (expected, plain_mean) in (
        (1, (healthy_weighted, healthy_mean)),
        (2, (healthy_weighted, healthy_mean)),
        (3, (injured_weighted, injured_mean)),
        (4, (healthy_weighted, healthy_mean)),
    ):
        assert value.iloc[row] == pytest.approx(expected), row
        # OFF_RATING falls as minutes fall, so weighting lifts the aggregate
        # above the plain mean over the same players.
        assert value.iloc[row] > plain_mean, row


def test_bench_keeps_one_average_and_the_count(emitted):
    assert "BENCH_AVG_PTS_PER_MIN_BEFORE" in emitted
    assert "N_BENCH_PLAYERS_BEFORE" in emitted
    assert "BENCH_MAX_PTS_PER_MIN_BEFORE" not in emitted
    assert "BENCH_MAX_PACE_PER40_BEFORE" not in emitted
    assert "BENCH_AVG_PACE_PER40_BEFORE" not in emitted


def test_legacy_profile_still_describes_the_old_schema():
    """Pinned so a pre-reduction dataset stays reproducible for old bundles."""
    assert LEGACY_PROFILE.active_value_slots == 6
    assert LEGACY_PROFILE.weighted_rate_stats == ()
    for stat in RATE_STAT_COLS:
        assert LEGACY_PROFILE.emits_value(stat, 6, injured=False)
        assert stat in LEGACY_PROFILE.total_injured_stats
    assert len(LEGACY_PROFILE.bench_cols) == 5


def test_active_profile_is_the_reduced_one():
    assert ACTIVE_PROFILE is REDUCED_PROFILE
