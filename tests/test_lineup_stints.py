import pandas as pd
import pytest
from nba_ou.data_processing.lineups.stints import (
    StintValidationError,
    build_game_stints,
    elapsed_ds,
    validate_game_stints,
)


def rotation(team, *, substitute=False, overtime=False):
    end = 31800 if overtime else 28800
    rows = [(team, i, 0, end) for i in range(1, 6)]
    if substitute:
        rows[4] = (team, 5, 0, 3000)
        rows.append((team, 6, 3000, end))
    return pd.DataFrame(rows, columns=["TEAM_ID", "PERSON_ID", "IN_TIME_REAL", "OUT_TIME_REAL"])


def event(number, period, clock, action, *, team=None, home="", away="", subtype=""):
    return dict(actionNumber=number, period=period, clock=clock, actionType=action,
                teamId=team, scoreHome=home, scoreAway=away, subType=subtype,
                isFieldGoal=action == "Made Shot", shotValue=2)


def test_midperiod_sub_and_same_clock_scoring_order():
    pbp = pd.DataFrame([
        event(1, 1, "PT07M00.00S", "Made Shot", team="H", home="2"),
        event(2, 1, "PT07M00.00S", "Substitution", team="H"),
        event(3, 1, "PT07M00.00S", "Made Shot", team="A", away="2"),
        event(4, 4, "PT00M00.00S", "period"),
    ])
    home, away = rotation("H", substitute=True), rotation("A")
    stints = build_game_stints(home, away, pbp)
    assert stints.iloc[0].home_pts == 2
    assert stints.iloc[1].away_pts == 2
    assert stints.iloc[0].home_lineup == ("1", "2", "3", "4", "5")
    assert stints.iloc[1].home_lineup == ("1", "2", "3", "4", "6")
    assert stints.seconds.sum() == 2880
    box = pd.DataFrame(
        [("H", str(i), 48 if i != 5 else 5) for i in range(1, 7)]
        + [("A", str(i), 48) for i in range(1, 6)],
        columns=["TEAM_ID", "PLAYER_ID", "MIN"],
    )
    box.loc[box.TEAM_ID.eq("H") & box.PLAYER_ID.eq("6"), "MIN"] = 43
    validate_game_stints(stints, home, away, home_points=2, away_points=2, box_minutes=box)


def test_period_break_and_overtime():
    home, away = rotation("H", overtime=True), rotation("A", overtime=True)
    pbp = pd.DataFrame([
        event(1, 3, "PT12M00.00S", "Made Shot", team="H", home="2"),
        event(2, 5, "PT00M00.00S", "period"),
    ])
    stints = build_game_stints(home, away, pbp)
    assert elapsed_ds(3, "PT12M00.00S") == 14400
    assert stints.end_ds.max() == 31800
    assert stints.seconds.sum() == 3180
    assert stints.loc[stints.period.eq(3), "home_pts"].sum() == 2


def test_points_mismatch_rejected():
    home, away = rotation("H"), rotation("A")
    stints = build_game_stints(
        home, away, pd.DataFrame([event(1, 4, "PT00M00.00S", "period")])
    )
    box = pd.DataFrame([("H", str(i), 48) for i in range(1, 6)]
                       + [("A", str(i), 48) for i in range(1, 6)],
                       columns=["TEAM_ID", "PLAYER_ID", "MIN"])
    with pytest.raises(StintValidationError, match="points_mismatch"):
        validate_game_stints(stints, home, away, home_points=100, away_points=99,
                             box_minutes=box)


def test_zero_length_rotation_and_free_throw_between_subs():
    home = rotation("H", substitute=True)
    home.loc[len(home)] = ["H", 7, 3000, 3000]
    away = rotation("A")
    pbp = pd.DataFrame([
        event(1, 1, "PT07M00.00S", "Substitution", team="H"),
        dict(event(2, 1, "PT07M00.00S", "Free Throw", team="H", home="1"),
             personId=6),
        event(3, 1, "PT07M00.00S", "Substitution", team="A"),
        event(4, 4, "PT00M00.00S", "period"),
    ])
    stints = build_game_stints(home, away, pbp)
    assert stints.home_pts.sum() == 1
    assert stints.loc[stints.home_pts.eq(1), "home_lineup"].iloc[0] == (
        "1", "2", "3", "4", "6"
    )


def test_overlapping_player_intervals_rejected():
    home = rotation("H")
    home.loc[len(home)] = ["H", 1, 100, 200]
    with pytest.raises(StintValidationError, match="overlapping_player_intervals"):
        build_game_stints(home, rotation("A"),
                          pd.DataFrame([event(1, 4, "PT00M00.00S", "period")]))


def test_v3_rebounds_use_description_counters():
    pbp = pd.DataFrame([
        dict(event(1, 1, "PT10M00.00S", "Rebound", team="H"),
             personId=1, description="Player REBOUND (Off:1 Def:0)",
             subType="Unknown"),
        dict(event(2, 1, "PT09M00.00S", "Rebound", team="H"),
             personId=1, description="Player REBOUND (Off:1 Def:1)",
             subType="Unknown"),
        event(3, 4, "PT00M00.00S", "period"),
    ])
    stints = build_game_stints(rotation("H"), rotation("A"), pbp)
    assert stints.home_oreb.sum() == 1
    assert stints.home_dreb.sum() == 1


def test_appended_corrected_action_uses_game_clock_before_action_number():
    pbp = pd.DataFrame([
        event(1, 1, "PT09M00.00S", "Made Shot", team="H", home="4"),
        event(100, 1, "PT10M00.00S", "Made Shot", team="H", home="2"),
        event(101, 4, "PT00M00.00S", "period", home="4"),
    ])
    stints = build_game_stints(rotation("H"), rotation("A"), pbp)
    assert stints.home_pts.sum() == 4
