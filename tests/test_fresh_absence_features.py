"""Tests for the fresh-absence (first game out) interaction features."""

import pandas as pd
import pytest
from nba_ou.data_processing.merged_home_away_data.add_features_after_merging import (
    add_fresh_absence_sums,
)
from nba_ou.data_processing.past_injuries.past_injuries import N_TOP_PLAYERS_INJURED
from nba_ou.data_processing.players.fresh_absence import (
    KEY_PLAYER_THRESHOLDS,
    STREAK_STAT_COLS,
    add_fresh_absence_features,
    fresh_absence_feature_names,
)


def _team_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a team-level frame with every top-N injured value/streak column."""
    base: dict[str, list] = {}
    for stat in STREAK_STAT_COLS:
        for i in range(1, N_TOP_PLAYERS_INJURED + 1):
            base[f"TOP{i}_INJURED_PLAYER_{stat}_BEFORE"] = [
                row.get(f"v{i}_{stat}", 0.0) for row in rows
            ]
            base[f"TOP{i}_INJURED_STREAK_{stat}_BEFORE"] = [
                row.get(f"s{i}_{stat}", 0) for row in rows
            ]
    return pd.DataFrame(base)


def test_flags_only_the_first_game_out():
    """The flag fires on streak 1 and on no other streak value."""
    threshold = KEY_PLAYER_THRESHOLDS["PTS"]
    df = _team_frame(
        [{"v1_PTS": threshold + 5, "s1_PTS": streak} for streak in (0, 1, 2, 3, 10)]
    )
    out = add_fresh_absence_features(df)
    first, second = (
        out["INJ_KEY_PLAYER_FIRST_GAME_OUT_PTS_BEFORE"].tolist(),
        out["INJ_KEY_PLAYER_SECOND_GAME_OUT_PTS_BEFORE"].tolist(),
    )
    assert first == [0, 1, 0, 0, 0]
    assert second == [0, 0, 1, 0, 0]


def test_threshold_separates_key_players_from_bench():
    """A player below the cut-off is not a 'key player', however fresh."""
    threshold = KEY_PLAYER_THRESHOLDS["MIN"]
    df = _team_frame(
        [
            {"v1_MIN": threshold - 0.1, "s1_MIN": 1},
            {"v1_MIN": threshold, "s1_MIN": 1},
            {"v1_MIN": threshold + 10, "s1_MIN": 1},
        ]
    )
    out = add_fresh_absence_features(df)
    assert out["INJ_KEY_PLAYER_FIRST_GAME_OUT_MIN_BEFORE"].tolist() == [0, 1, 1]
    # the continuous companion keeps the magnitude the flag discards
    assert out["INJ_TOP1_FIRST_GAME_OUT_MIN_BEFORE"].tolist() == [
        threshold - 0.1,
        threshold,
        threshold + 10,
    ]


def test_fresh_sum_adds_only_first_game_out_players():
    """Players out longer contribute nothing to tonight's news."""
    df = _team_frame(
        [
            {
                "v1_PTS": 20.0,
                "s1_PTS": 1,  # counts
                "v2_PTS": 10.0,
                "s2_PTS": 1,  # counts
                "v3_PTS": 8.0,
                "s3_PTS": 7,  # long-term absence, ignored
                "v4_PTS": 5.0,
                "s4_PTS": 0,  # nobody in this slot
            }
        ]
    )
    out = add_fresh_absence_features(df)
    assert out["INJ_FRESH_OUT_PTS_BEFORE"].tolist() == [30.0]


def test_minutes_and_points_are_independent():
    """A high-minute absence is picked up even when the top scorer is fit."""
    df = _team_frame(
        [
            {
                "v1_PTS": 4.0,
                "s1_PTS": 1,
                "v1_MIN": KEY_PLAYER_THRESHOLDS["MIN"] + 2,
                "s1_MIN": 1,
            }
        ]
    )
    out = add_fresh_absence_features(df)
    assert out["INJ_KEY_PLAYER_FIRST_GAME_OUT_PTS_BEFORE"].tolist() == [0]
    assert out["INJ_KEY_PLAYER_FIRST_GAME_OUT_MIN_BEFORE"].tolist() == [1]


def test_missing_values_are_treated_as_no_news():
    """NaN means 'no such injured player', which is 0, not an unknown."""
    df = _team_frame([{"v1_PTS": None, "s1_PTS": None}])
    out = add_fresh_absence_features(df)
    assert out["INJ_FRESH_OUT_PTS_BEFORE"].tolist() == [0.0]
    assert out["INJ_KEY_PLAYER_FIRST_GAME_OUT_PTS_BEFORE"].tolist() == [0]


def test_columns_exist_even_without_source_columns():
    """Schema stays stable when streak columns were never computed."""
    out = add_fresh_absence_features(pd.DataFrame({"GAME_ID": ["0021900001"]}))
    for stat in STREAK_STAT_COLS:
        for col in fresh_absence_feature_names(stat):
            assert col in out.columns
            assert out[col].tolist() == [0] or out[col].tolist() == [0.0]


def test_every_emitted_column_is_before_suffixed():
    """The leakage gate keeps columns by name, so the name must carry _BEFORE."""
    out = add_fresh_absence_features(_team_frame([{"v1_PTS": 20.0, "s1_PTS": 1}]))
    for stat in STREAK_STAT_COLS:
        for col in fresh_absence_feature_names(stat):
            assert col.endswith("_BEFORE")
            assert col in out.columns


def test_is_idempotent():
    """Re-running must refresh in place, not duplicate or stack columns."""
    df = _team_frame([{"v1_PTS": 20.0, "s1_PTS": 1}])
    once = add_fresh_absence_features(df)
    twice = add_fresh_absence_features(once)
    assert list(twice.columns).count("INJ_FRESH_OUT_PTS_BEFORE") == 1
    pd.testing.assert_frame_equal(once, twice[once.columns])


def test_sums_add_the_two_sides_rather_than_differencing():
    """Both teams' absences move a total the same way, so they add."""
    merged = pd.DataFrame(
        {
            "INJ_FRESH_OUT_PTS_BEFORE_TEAM_HOME": [20.0, 0.0, 12.0],
            "INJ_FRESH_OUT_PTS_BEFORE_TEAM_AWAY": [15.0, 0.0, 12.0],
            "INJ_KEY_PLAYER_FIRST_GAME_OUT_MIN_BEFORE_TEAM_HOME": [1, 0, 1],
            "INJ_KEY_PLAYER_FIRST_GAME_OUT_MIN_BEFORE_TEAM_AWAY": [0, 0, 1],
        }
    )
    out = add_fresh_absence_sums(merged)
    assert out["INJ_FRESH_OUT_PTS_SUM_BEFORE"].tolist() == [35.0, 0.0, 24.0]
    assert out["INJ_KEY_PLAYER_FIRST_GAME_OUT_MIN_SUM_BEFORE"].tolist() == [1, 0, 2]


def test_sums_ignore_unpaired_and_non_injury_columns():
    merged = pd.DataFrame(
        {
            "INJ_FRESH_OUT_PTS_BEFORE_TEAM_HOME": [1.0],
            "PTS_SEASON_BEFORE_AVG_TEAM_HOME": [100.0],
            "PTS_SEASON_BEFORE_AVG_TEAM_AWAY": [90.0],
        }
    )
    out = add_fresh_absence_sums(merged)
    assert not [c for c in out.columns if c.endswith("_SUM_BEFORE")]


def test_sums_are_idempotent():
    merged = pd.DataFrame(
        {
            "INJ_FRESH_OUT_PTS_BEFORE_TEAM_HOME": [20.0],
            "INJ_FRESH_OUT_PTS_BEFORE_TEAM_AWAY": [15.0],
        }
    )
    once = add_fresh_absence_sums(merged)
    twice = add_fresh_absence_sums(once)
    assert list(twice.columns).count("INJ_FRESH_OUT_PTS_SUM_BEFORE") == 1
    pd.testing.assert_frame_equal(once, twice[once.columns])


@pytest.mark.parametrize("stat", STREAK_STAT_COLS)
def test_streaks_are_computed_for_every_key_statistic(stat):
    """STREAK_STAT_COLS drives the streak columns attach_player_features emits."""
    from nba_ou.data_processing.players import attach_player_features

    assert attach_player_features.STREAK_STAT_COLS is STREAK_STAT_COLS
    assert stat in STREAK_STAT_COLS


# --------------------------------------------------------------------------
# End-to-end: the MIN streak columns must actually be produced by the pipeline
# function, not merely be consumable if they happen to exist.
# --------------------------------------------------------------------------

TEAM = "1610612738"
STAR, OTHER = 201939, 202691


def _pipeline_frames(injured_game_indices):
    """6 team games; STAR is injured on the given (0-based) game indices."""
    n = 6
    df_team = pd.DataFrame(
        {
            "GAME_ID": [f"002250000{i}" for i in range(n)],
            "TEAM_ID": TEAM,
            "SEASON_ID": "22025",
            "SEASON_YEAR": 2025,
            "GAME_DATE": pd.date_range("2025-10-20", periods=n, freq="3D"),
        }
    )
    rows = []
    for game in df_team.itertuples():
        for player_id, pts, minutes in ((STAR, 28.0, 36.0), (OTHER, 9.0, 18.0)):
            rows.append(
                {
                    "GAME_ID": game.GAME_ID,
                    "TEAM_ID": TEAM,
                    "SEASON_ID": game.SEASON_ID,
                    "SEASON_YEAR": game.SEASON_YEAR,
                    "GAME_DATE": game.GAME_DATE,
                    "PLAYER_ID": player_id,
                    "PLAYER_NAME": f"Player {player_id}",
                    "MIN": minutes,
                    "PTS": pts,
                }
            )
    df_players = pd.DataFrame(rows)
    df_injuries = pd.DataFrame(
        {
            "GAME_ID": [df_team.GAME_ID.iloc[i] for i in injured_game_indices],
            "TEAM_ID": TEAM,
            "PLAYER_ID": STAR,
        }
    )
    return df_team, df_players, df_injuries


def _run_pipeline(injured_game_indices):
    from nba_ou.data_processing.players.attach_player_features import (
        add_player_history_features,
    )

    df_team, df_players, df_injuries = _pipeline_frames(injured_game_indices)
    out, _ = add_player_history_features(
        df_team, df_players, df_injuries, stat_cols=["PTS", "MIN"]
    )
    return out.sort_values("GAME_DATE").reset_index(drop=True)


def test_pipeline_emits_minutes_streak_columns():
    """Regression: streaks used to be computed for PTS only.

    Without a MIN streak the minutes-based features can never fire, because
    they read TOP1_INJURED_STREAK_MIN_BEFORE.
    """
    out = _run_pipeline([3, 4])
    assert "TOP1_INJURED_STREAK_MIN_BEFORE" in out.columns
    assert "TOP1_INJURED_STREAK_PTS_BEFORE" in out.columns


def test_pipeline_minutes_streak_counts_consecutive_games():
    out = _run_pipeline([3, 4])
    streak = out["TOP1_INJURED_STREAK_MIN_BEFORE"].tolist()
    assert streak[3] == 1, f"first game out should be streak 1, got {streak}"
    assert streak[4] == 2, f"second game out should be streak 2, got {streak}"


def test_pipeline_produces_fresh_absence_features_for_both_statistics():
    out = _run_pipeline([3, 4])
    for stat in STREAK_STAT_COLS:
        for col in fresh_absence_feature_names(stat):
            assert col in out.columns, col
    first_out = out.index[out["TOP1_INJURED_STREAK_MIN_BEFORE"] == 1][0]
    assert out.loc[first_out, "INJ_KEY_PLAYER_FIRST_GAME_OUT_MIN_BEFORE"] == 1
    assert out.loc[first_out, "INJ_KEY_PLAYER_FIRST_GAME_OUT_PTS_BEFORE"] == 1
    second_out = out.index[out["TOP1_INJURED_STREAK_MIN_BEFORE"] == 2][0]
    assert out.loc[second_out, "INJ_KEY_PLAYER_FIRST_GAME_OUT_MIN_BEFORE"] == 0
    assert out.loc[second_out, "INJ_KEY_PLAYER_SECOND_GAME_OUT_MIN_BEFORE"] == 1
