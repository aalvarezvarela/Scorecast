"""Phase E: the minutes frame's temporal contract and the team constraint."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from nba_ou.data_processing.lineups.minutes_model import (
    FEATURE_COLUMNS,
    TEAM_MINUTES,
    apply_team_constraint,
    build_minutes_frame,
    walk_forward_minutes,
)

TEAM = "1610612739"
OTHER = "1610612738"


def _box(rows):
    return pd.DataFrame(
        [
            {
                "GAME_ID": str(g).zfill(10),
                "TEAM_ID": t,
                "GAME_DATE": d,
                "PLAYER_ID": p,
                "MIN": m,
                "START_POSITION": s,
            }
            for g, t, d, p, m, s in rows
        ]
    )


def _season(n=12, team=TEAM, start=1, roster=None):
    roster = roster or [
        ("a", 34.0, "G"),
        ("b", 30.0, "F"),
        ("c", 20.0, "C"),
        ("d", 10.0, ""),
    ]
    rows = []
    for index in range(start, start + n):
        for player, minutes, position in roster:
            rows.append(
                (index, team, f"2025-01-{index:02d}", player, minutes, position)
            )
    return rows


class TestBuildMinutesFrame:
    def test_features_come_only_from_earlier_games(self):
        frame = build_minutes_frame(_box(_season(n=8)))
        row = frame.loc[frame.GAME_ID.eq("0000000008") & frame.PLAYER_ID.eq("a")].iloc[
            0
        ]
        assert row.min_last_5 == pytest.approx(34.0)
        assert row.MIN == pytest.approx(34.0)
        assert row.team_game_number == 7

    def test_the_first_game_produces_no_rows(self):
        frame = build_minutes_frame(_box(_season(n=3)))
        assert "0000000001" not in set(frame.GAME_ID)

    def test_tonights_minutes_do_not_reach_tonights_features(self):
        """The mandatory leakage check."""
        clean = build_minutes_frame(_box(_season(n=8)))
        rows = _season(n=7)
        # Game 8 finishes with wildly different minutes.
        for player, position in (("a", "G"), ("b", "F"), ("c", "C"), ("d", "")):
            rows.append((8, TEAM, "2025-01-08", player, 1.0, position))
        polluted = build_minutes_frame(_box(rows))
        columns = list(FEATURE_COLUMNS)
        a = clean.loc[clean.GAME_ID.eq("0000000008")].sort_values("PLAYER_ID")
        b = polluted.loc[polluted.GAME_ID.eq("0000000008")].sort_values("PLAYER_ID")
        np.testing.assert_allclose(a[columns].to_numpy(), b[columns].to_numpy())

    def test_p_out_reaches_the_row_and_silence_means_available(self):
        frame = build_minutes_frame(
            _box(_season(n=8)), p_out={("0000000008", TEAM, "a"): 0.9}
        )
        by_player = frame.loc[frame.GAME_ID.eq("0000000008")].set_index("PLAYER_ID")
        assert by_player.loc["a", "p_out"] == pytest.approx(0.9)
        assert by_player.loc["b", "p_out"] == 0.0

    def test_a_teammates_expected_absence_shows_as_minutes_to_absorb(self):
        frame = build_minutes_frame(
            _box(_season(n=8)), p_out={("0000000008", TEAM, "a"): 0.95}
        )
        by_player = frame.loc[frame.GAME_ID.eq("0000000008")].set_index("PLAYER_ID")
        # a plays 34; his team-mates each see those 34 as up for grabs.
        assert by_player.loc["b", "team_minutes_missing"] == pytest.approx(34.0)
        # a himself is not absorbing his own minutes.
        assert by_player.loc["a", "team_minutes_missing"] == pytest.approx(0.0)

    def test_position_minutes_are_restricted_to_the_same_group(self):
        frame = build_minutes_frame(
            _box(_season(n=8)), p_out={("0000000008", TEAM, "a"): 0.95}
        )
        by_player = frame.loc[frame.GAME_ID.eq("0000000008")].set_index("PLAYER_ID")
        # a is a guard; b is a forward, so nothing of a's is in b's group.
        assert by_player.loc["b", "position_minutes_missing"] == pytest.approx(0.0)

    def test_starts_and_absences_are_counted(self):
        rows = _season(n=6)
        # Game 7: "a" does not play at all.
        for player, minutes, position in (
            ("b", 40.0, "F"),
            ("c", 25.0, "C"),
            ("d", 15.0, ""),
        ):
            rows.append((7, TEAM, "2025-01-07", player, minutes, position))
        rows.append((7, TEAM, "2025-01-07", "a", 0.0, ""))
        rows += _season(n=1, start=8)
        frame = build_minutes_frame(_box(rows))
        row = frame.loc[frame.GAME_ID.eq("0000000008") & frame.PLAYER_ID.eq("a")].iloc[
            0
        ]
        assert row.starts_season == 6
        # Zero: the absence was the immediately preceding game.
        assert row.games_since_absence == pytest.approx(0.0)
        # A player who has never missed one is far from it and stays ordered.
        never = frame.loc[
            frame.GAME_ID.eq("0000000008") & frame.PLAYER_ID.eq("b")
        ].iloc[0]
        assert never.games_since_absence > row.games_since_absence

    def test_rest_days_and_back_to_back(self):
        rows = _season(n=5)
        rows += _season(n=1, start=9)  # a four-day gap
        frame = build_minutes_frame(_box(rows))
        row = frame.loc[frame.GAME_ID.eq("0000000009")].iloc[0]
        assert row.rest_days == pytest.approx(4.0)

    def test_a_traded_player_leaves_his_old_roster(self):
        rows = _season(n=8)
        rows.append((20, OTHER, "2025-01-09", "a", 30.0, "G"))
        rows += _season(n=1, start=10)
        frame = build_minutes_frame(_box(rows))
        late = frame.loc[frame.GAME_ID.eq("0000000010")]
        assert "a" not in set(late.PLAYER_ID)

    def test_missing_columns_raise(self):
        with pytest.raises(ValueError, match="Player history is missing"):
            build_minutes_frame(pd.DataFrame({"GAME_ID": ["1"]}))


class TestTeamConstraint:
    def test_each_team_game_is_rescaled_to_240(self):
        frame = pd.DataFrame(
            {
                "GAME_ID": ["1"] * 3 + ["1"] * 2,
                "TEAM_ID": [TEAM] * 3 + [OTHER] * 2,
                "predicted_minutes": [30.0, 20.0, 10.0, 50.0, 50.0],
            }
        )
        scaled = apply_team_constraint(frame)
        totals = scaled.groupby([frame.GAME_ID, frame.TEAM_ID]).sum()
        assert all(total == pytest.approx(TEAM_MINUTES) for total in totals)

    def test_proportions_survive_the_rescale(self):
        frame = pd.DataFrame(
            {
                "GAME_ID": ["1"] * 2,
                "TEAM_ID": [TEAM] * 2,
                "predicted_minutes": [30.0, 10.0],
            }
        )
        scaled = apply_team_constraint(frame)
        assert scaled.iloc[0] / scaled.iloc[1] == pytest.approx(3.0)

    def test_a_team_predicted_to_play_nobody_becomes_nan_not_zero(self):
        frame = pd.DataFrame(
            {"GAME_ID": ["1"], "TEAM_ID": [TEAM], "predicted_minutes": [0.0]}
        )
        assert apply_team_constraint(frame).isna().all()

    def test_negative_predictions_are_clipped_before_scaling(self):
        frame = pd.DataFrame(
            {
                "GAME_ID": ["1"] * 2,
                "TEAM_ID": [TEAM] * 2,
                "predicted_minutes": [-5.0, 40.0],
            }
        )
        scaled = apply_team_constraint(frame)
        assert scaled.iloc[0] == pytest.approx(0.0)
        assert scaled.iloc[1] == pytest.approx(TEAM_MINUTES)


class TestWalkForward:
    def test_an_empty_frame_returns_an_empty_prediction(self):
        result = walk_forward_minutes(pd.DataFrame())
        assert result.empty

    def test_a_month_without_enough_history_is_left_unpredicted(self):
        frame = build_minutes_frame(_box(_season(n=10)))
        result = walk_forward_minutes(frame, min_train_rows=10_000)
        assert result.predicted_minutes.isna().all()

    def test_a_listed_out_player_is_projected_to_sit(self):
        frame = build_minutes_frame(_box(_season(n=25)))
        frame.loc[frame.PLAYER_ID.eq("a"), "p_out"] = 1.0
        result = walk_forward_minutes(frame, min_train_rows=20)
        predicted = result.loc[result.PLAYER_ID.eq("a"), "predicted_minutes"].dropna()
        assert (predicted == pytest.approx(0.0)).all()

    def test_minutes_given_play_is_exposed_unmultiplied(self):
        """A consumer that models availability itself must not apply p_out twice."""
        frame = build_minutes_frame(_box(_season(n=25)))
        frame.loc[frame.PLAYER_ID.eq("a"), "p_out"] = 1.0
        result = walk_forward_minutes(frame, min_train_rows=20)
        rows = result.loc[result.PLAYER_ID.eq("a")].dropna(
            subset=["minutes_given_play"]
        )
        assert (rows.predicted_minutes == pytest.approx(0.0)).all()
        # He is certain to sit, but were he to play he would play real minutes.
        assert (rows.minutes_given_play > 5.0).all()
