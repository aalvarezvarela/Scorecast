import pandas as pd
import pytest
from nba_ou.data_processing.merged_home_away_data.merge_home_away import (
    merge_home_away_data,
)
from nba_ou.data_processing.players.roster_continuity import (
    IMMEDIATE_NET_ROSTER_MINUTES_COLUMN,
    IMMEDIATE_NEW_PLAYER_MINUTES_COLUMN,
    IMMEDIATE_ROSTER_CONTINUITY_COLUMN,
    NET_ROSTER_MINUTES_COLUMN,
    NEW_PLAYER_MINUTES_COLUMN,
    ROSTER_CONTINUITY_COLUMN,
    add_roster_continuity_feature,
)


def _team_rows(dates: list[str]) -> pd.DataFrame:
    rows = []
    for index, date in enumerate(dates, start=1):
        game_id = f"0022500{index:03d}"
        for team_id, home in [("A", True), ("B", False)]:
            rows.append(
                {
                    "GAME_ID": game_id,
                    "GAME_DATE": date,
                    "SEASON_YEAR": 2025,
                    "SEASON_ID": "22025",
                    "SEASON_TYPE": "Regular Season",
                    "TEAM_ID": team_id,
                    "HOME": home,
                    "IS_OVERTIME": False,
                }
            )
    return pd.DataFrame(rows)


def _player(
    game_id: str,
    date: str,
    season_year: int,
    team_id: str,
    player_id: str,
    minutes: float,
    season_prefix: str = "2",
) -> dict:
    return {
        "GAME_ID": game_id,
        "GAME_DATE": date,
        "SEASON_YEAR": season_year,
        "SEASON_ID": f"{season_prefix}{season_year}",
        "TEAM_ID": team_id,
        "PLAYER_ID": player_id,
        "MIN": minutes,
    }


def test_scheduled_game_weights_lost_players_and_keeps_injured_players() -> None:
    df_team = _team_rows(["2025-10-22"])
    game_id = df_team.iloc[0]["GAME_ID"]
    df_players = pd.DataFrame(
        [
            _player("0022400901", "2025-03-20", 2024, "A", "stays", 40),
            _player("0022400901", "2025-03-20", 2024, "A", "leaves", 110),
            _player("0022400901", "2025-03-20", 2024, "A", "injured", 20),
            _player("0022400901", "2025-03-20", 2024, "A", "retained_bench", 70),
            _player("0022400901", "2025-03-20", 2024, "B", "b_stable", 20),
            # Scheduled active-roster placeholders have no current-game minutes.
            # The injury dictionary supplies the current pregame assignment.
            _player(game_id, "2025-10-22", 2025, "B", "leaves", float("nan")),
        ]
    )
    # The traded player is healthy and therefore absent from the scheduled OUT
    # dictionary. Their scheduled placeholder must still update team membership.
    injured_dict = {game_id: {"A": ["injured"]}}

    result = add_roster_continuity_feature(
        df_team,
        df_players,
        injured_dict,
        scheduled_game_ids=[game_id],
    )

    team_a = result.loc[result["TEAM_ID"] == "A"].iloc[0]
    assert team_a[ROSTER_CONTINUITY_COLUMN] == pytest.approx(1 - 110 / 240)
    assert team_a[IMMEDIATE_ROSTER_CONTINUITY_COLUMN] == pytest.approx(1 - 110 / 240)
    team_b = result.loc[result["TEAM_ID"] == "B"].iloc[0]
    assert team_b[NEW_PLAYER_MINUTES_COLUMN] == pytest.approx(110 / 240)
    assert team_b[IMMEDIATE_NEW_PLAYER_MINUTES_COLUMN] == pytest.approx(110 / 240)
    assert team_a[NET_ROSTER_MINUTES_COLUMN] == pytest.approx(-110 / 240)
    assert team_a[IMMEDIATE_NET_ROSTER_MINUTES_COLUMN] == pytest.approx(-110 / 240)
    assert team_b[NET_ROSTER_MINUTES_COLUMN] == pytest.approx(110 / 240)
    assert team_b[IMMEDIATE_NET_ROSTER_MINUTES_COLUMN] == pytest.approx(110 / 240)


def test_scheduled_placeholder_requires_explicit_scheduled_game_id() -> None:
    df_team = _team_rows(["2025-10-22"])
    game_id = df_team.iloc[0]["GAME_ID"]
    df_players = pd.DataFrame(
        [
            _player("0022400901", "2025-03-20", 2024, "A", "healthy_trade", 40),
            _player(game_id, "2025-10-22", 2025, "B", "healthy_trade", float("nan")),
        ]
    )

    result = add_roster_continuity_feature(df_team, df_players, injured_dict={})

    continuity = result.loc[result["TEAM_ID"] == "A", ROSTER_CONTINUITY_COLUMN].iloc[0]
    assert continuity == 1.0


def test_official_roster_records_unreported_departures_and_new_signings():
    team = _team_rows(["2025-10-22"])
    game = team.GAME_ID.iloc[0]
    players = pd.DataFrame(
        [
            _player("0022400901", "2025-03-20", 2024, "A", "stays", 40),
            _player("0022400901", "2025-03-20", 2024, "A", "leaves", 110),
            _player("0022400901", "2025-03-20", 2024, "B", "baseline", 30),
        ]
    )
    result = add_roster_continuity_feature(
        team,
        players,
        {},
        current_rosters={
            (game, "A"): {"stays": "Stays"},
            (game, "B"): {"leaves": "Incoming", "baseline": "Baseline"},
        },
    )
    home = result.loc[result.TEAM_ID.eq("A")].iloc[0]
    away = result.loc[result.TEAM_ID.eq("B")].iloc[0]
    assert home[ROSTER_CONTINUITY_COLUMN] == pytest.approx(40 / 150)
    assert away[NEW_PLAYER_MINUTES_COLUMN] == pytest.approx(110 / 240)


def test_sequential_departures_cannot_force_artificial_zero_continuity() -> None:
    df_team = _team_rows(["2025-10-22"])
    df_players = pd.DataFrame(
        [
            _player("0022400801", "2025-03-20", 2024, "A", "lost_first", 120),
            _player("0022400801", "2025-03-20", 2024, "A", "stable", 40),
            _player("0022400851", "2025-04-20", 2024, "A", "lost_second", 120),
            _player("0022400851", "2025-04-20", 2024, "A", "stable", 40),
            _player("0022500001", "2025-10-01", 2025, "B", "lost_first", 30),
            _player("0022500002", "2025-10-02", 2025, "C", "lost_second", 30),
        ]
    )

    result = add_roster_continuity_feature(df_team, df_players, injured_dict={})
    team_a = result.loc[result["TEAM_ID"].eq("A")].iloc[0]

    # Old per-appearance weights summed the two 120-minute departures and then
    # clipped against 240. Their actual A contributions are 60 each, while the
    # retained player's contribution is 40: continuity = 40 / 160.
    assert team_a[ROSTER_CONTINUITY_COLUMN] == pytest.approx(0.25)
    assert team_a[IMMEDIATE_ROSTER_CONTINUITY_COLUMN] == pytest.approx(0.25)


def test_target_team_current_minutes_are_preferred_and_current_game_is_excluded() -> (
    None
):
    df_team = _team_rows(["2025-11-01"])
    target_game_id = df_team.iloc[0]["GAME_ID"]
    df_players = pd.DataFrame(
        [
            _player("0022400901", "2025-04-01", 2024, "A", "leaves", 50),
            _player("0022400901", "2025-04-01", 2024, "A", "stable", 40),
            _player("0022500981", "2025-10-20", 2025, "A", "leaves", 20),
            _player("0022500981", "2025-10-20", 2025, "A", "stable", 40),
            _player("0022500991", "2025-10-25", 2025, "B", "leaves", 30),
            _player(target_game_id, "2025-11-01", 2025, "B", "leaves", 99),
        ]
    )

    result = add_roster_continuity_feature(df_team, df_players, injured_dict={})

    continuity = result.loc[result["TEAM_ID"] == "A", ROSTER_CONTINUITY_COLUMN].iloc[0]
    assert continuity == pytest.approx(40 / (40 + 20))


def test_future_trade_does_not_change_earlier_historical_game() -> None:
    df_team = _team_rows(["2025-10-20", "2025-11-10"])
    later_game_id = df_team.loc[df_team["GAME_DATE"].eq("2025-11-10"), "GAME_ID"].iloc[
        0
    ]
    df_players = pd.DataFrame(
        [
            _player("0022400901", "2025-04-01", 2024, "A", "moves_later", 24),
            _player("0022400901", "2025-04-01", 2024, "A", "stable", 24),
            _player("0022500001", "2025-10-01", 2025, "A", "stable", 24),
            _player("0022500010", "2025-11-01", 2025, "B", "moves_later", 36),
            _player(later_game_id, "2025-11-10", 2025, "B", "moves_later", 48),
        ]
    )

    result = add_roster_continuity_feature(df_team, df_players, injured_dict={})
    team_a = result[result["TEAM_ID"].eq("A")].set_index("GAME_DATE")

    assert team_a.loc["2025-10-20", ROSTER_CONTINUITY_COLUMN] == 1.0
    assert team_a.loc["2025-11-10", ROSTER_CONTINUITY_COLUMN] == pytest.approx(0.5)
    assert team_a.loc["2025-10-20", IMMEDIATE_ROSTER_CONTINUITY_COLUMN] == 1.0
    assert team_a.loc[
        "2025-11-10", IMMEDIATE_ROSTER_CONTINUITY_COLUMN
    ] == pytest.approx(0.5)


def test_preseason_rows_do_not_create_roster_membership_or_minute_weight() -> None:
    df_team = _team_rows(["2025-10-22"])
    df_players = pd.DataFrame(
        [
            _player("0022400901", "2025-03-20", 2024, "A", "regular", 20),
            _player(
                "0012500001",
                "2025-10-10",
                2025,
                "B",
                "regular",
                100,
                season_prefix="1",
            ),
        ]
    )

    result = add_roster_continuity_feature(df_team, df_players, injured_dict={})

    continuity = result.loc[result["TEAM_ID"] == "A", ROSTER_CONTINUITY_COLUMN].iloc[0]
    assert continuity == 1.0


@pytest.mark.parametrize(
    ("postseason_game_id", "season_prefix"),
    [("0042400001", "4"), ("0052400001", "5")],
    ids=["playoffs", "play-in"],
)
def test_postseason_game_is_included_via_canonical_season_type_mapping(
    postseason_game_id: str, season_prefix: str
) -> None:
    df_team = _team_rows(["2025-10-22"])
    df_players = pd.DataFrame(
        [
            _player(
                postseason_game_id,
                "2025-05-01",
                2024,
                "A",
                "postseason_trade",
                40,
                season_prefix=season_prefix,
            ),
            _player(
                postseason_game_id,
                "2025-05-01",
                2024,
                "A",
                "postseason_stable",
                60,
                season_prefix=season_prefix,
            ),
            _player("0022500009", "2025-10-01", 2025, "B", "postseason_trade", 20),
        ]
    )

    result = add_roster_continuity_feature(df_team, df_players, injured_dict={})

    continuity = result.loc[result["TEAM_ID"] == "A", ROSTER_CONTINUITY_COLUMN].iloc[0]
    assert continuity == pytest.approx(0.6)


def test_previous_season_injury_assignment_uses_extra_game_context() -> None:
    df_team = _team_rows(["2025-10-22"])
    target_game_id = df_team.iloc[0]["GAME_ID"]
    previous_injury_game = "0022400902"
    df_players = pd.DataFrame(
        [
            # Supplies the previous-season minute average but is before the
            # March 15 membership window.
            _player("0022400500", "2025-02-01", 2024, "A", "injury_only", 40),
            _player("0022400500", "2025-02-01", 2024, "A", "stable", 60),
        ]
    )
    df_game_context = pd.DataFrame(
        [
            {
                "GAME_ID": previous_injury_game,
                "GAME_DATE": "2025-03-20",
                "SEASON_YEAR": 2024,
                "SEASON_ID": "22024",
            }
        ]
    )
    injured_dict = {
        previous_injury_game: {"A": ["injury_only", "stable"]},
        target_game_id: {"B": ["injury_only"]},
    }

    result = add_roster_continuity_feature(
        df_team,
        df_players,
        injured_dict,
        df_game_context=df_game_context,
    )

    continuity = result.loc[result["TEAM_ID"] == "A", ROSTER_CONTINUITY_COLUMN].iloc[0]
    assert continuity == pytest.approx(0.6)


def test_immediate_continuity_ignores_trades_older_than_two_months() -> None:
    df_team = _team_rows(["2026-05-15"])
    df_players = pd.DataFrame(
        [
            _player("0022400901", "2025-04-01", 2024, "A", "old_trade", 30),
            _player("0022500501", "2026-02-01", 2025, "B", "old_trade", 30),
            _player("0022500701", "2026-04-01", 2025, "A", "stable", 20),
        ]
    )

    result = add_roster_continuity_feature(df_team, df_players, injured_dict={})
    team_a = result[result["TEAM_ID"].eq("A")].iloc[0]

    assert team_a[ROSTER_CONTINUITY_COLUMN] == pytest.approx(0.4)
    assert team_a[IMMEDIATE_ROSTER_CONTINUITY_COLUMN] == 1.0


def test_immediate_continuity_counts_trade_inside_two_month_window() -> None:
    df_team = _team_rows(["2026-05-15"])
    df_players = pd.DataFrame(
        [
            _player("0022500701", "2026-04-01", 2025, "A", "recent_trade", 30),
            _player("0022500701", "2026-04-01", 2025, "A", "stable", 30),
            _player("0022500751", "2026-04-20", 2025, "B", "recent_trade", 30),
        ]
    )

    result = add_roster_continuity_feature(df_team, df_players, injured_dict={})

    continuity = result.loc[
        result["TEAM_ID"] == "A", IMMEDIATE_ROSTER_CONTINUITY_COLUMN
    ].iloc[0]
    assert continuity == pytest.approx(0.5)


def test_immediate_offseason_window_extends_to_march_first() -> None:
    df_team = _team_rows(["2025-10-22"])
    df_players = pd.DataFrame(
        [
            _player("0022400851", "2025-03-05", 2024, "A", "summer_trade", 40),
            _player("0022400851", "2025-03-05", 2024, "A", "stable", 40),
            _player("0022500009", "2025-10-01", 2025, "B", "summer_trade", 20),
        ]
    )

    result = add_roster_continuity_feature(df_team, df_players, injured_dict={})

    continuity = result.loc[
        result["TEAM_ID"] == "A", IMMEDIATE_ROSTER_CONTINUITY_COLUMN
    ].iloc[0]
    assert continuity == pytest.approx(0.5)


def test_new_player_share_uses_minutes_from_previous_team_only() -> None:
    df_team = _team_rows(["2026-05-15"])
    df_players = pd.DataFrame(
        [
            _player("0022500701", "2026-04-01", 2025, "A", "new_player", 30),
            _player("0022500751", "2026-04-20", 2025, "B", "new_player", 10),
        ]
    )

    result = add_roster_continuity_feature(df_team, df_players, injured_dict={})

    share = result.loc[
        result["TEAM_ID"] == "B", IMMEDIATE_NEW_PLAYER_MINUTES_COLUMN
    ].iloc[0]
    assert share == pytest.approx(30 / 240)


def test_immediate_new_player_share_ignores_older_incorporations() -> None:
    df_team = _team_rows(["2026-05-15"])
    df_players = pd.DataFrame(
        [
            _player("0022400901", "2025-04-01", 2024, "A", "old_new_player", 40),
            _player("0022400901", "2025-04-01", 2024, "B", "b_stable", 20),
            _player("0022500501", "2026-02-01", 2025, "B", "old_new_player", 10),
            _player("0022500701", "2026-04-01", 2025, "B", "old_new_player", 10),
        ]
    )

    result = add_roster_continuity_feature(df_team, df_players, injured_dict={})
    team_b = result[result["TEAM_ID"].eq("B")].iloc[0]

    assert team_b[NEW_PLAYER_MINUTES_COLUMN] == pytest.approx(40 / 240)
    assert team_b[IMMEDIATE_NEW_PLAYER_MINUTES_COLUMN] == 0.0


def test_second_trade_uses_immediately_previous_team_for_incoming_minutes() -> None:
    df_team = _team_rows(["2026-05-15"])
    team_c = df_team.iloc[[0]].copy()
    team_c["TEAM_ID"] = "C"
    df_team = pd.concat([df_team, team_c], ignore_index=True)
    df_players = pd.DataFrame(
        [
            _player("0022400901", "2025-03-20", 2024, "A", "twice_traded", 40),
            _player("0022400901", "2025-03-20", 2024, "B", "b_stable", 5),
            _player("0022400901", "2025-03-20", 2024, "C", "c_baseline", 5),
            _player("0022500601", "2026-03-20", 2025, "A", "twice_traded", 15),
            _player("0022500601", "2026-03-20", 2025, "A", "a_stable", 15),
            _player("0022500701", "2026-04-01", 2025, "B", "twice_traded", 20),
            _player("0022500701", "2026-04-01", 2025, "B", "b_stable", 20),
            _player("0022500751", "2026-04-20", 2025, "C", "twice_traded", 10),
        ]
    )

    result = add_roster_continuity_feature(df_team, df_players, injured_dict={})
    teams = result.set_index("TEAM_ID")

    # A and B each retain half of their observed target-team minute value.
    for team_id in ["A", "B"]:
        assert teams.loc[team_id, ROSTER_CONTINUITY_COLUMN] == pytest.approx(0.5)
        assert teams.loc[team_id, IMMEDIATE_ROSTER_CONTINUITY_COLUMN] == pytest.approx(
            0.5
        )
        assert teams.loc[team_id, NEW_PLAYER_MINUTES_COLUMN] == 0.0
        assert teams.loc[team_id, IMMEDIATE_NEW_PLAYER_MINUTES_COLUMN] == 0.0

    # C receives the player from B, so only the 20-minute B average is incoming.
    assert teams.loc["C", ROSTER_CONTINUITY_COLUMN] == 1.0
    assert teams.loc["C", IMMEDIATE_ROSTER_CONTINUITY_COLUMN] == 1.0
    assert teams.loc["C", NEW_PLAYER_MINUTES_COLUMN] == pytest.approx(20 / 240)
    assert teams.loc["C", IMMEDIATE_NEW_PLAYER_MINUTES_COLUMN] == pytest.approx(
        20 / 240
    )
    assert teams.loc["C", NET_ROSTER_MINUTES_COLUMN] == pytest.approx(20 / 240)
    assert teams.loc["C", IMMEDIATE_NET_ROSTER_MINUTES_COLUMN] == pytest.approx(
        20 / 240
    )


def test_missing_previous_season_history_is_nan() -> None:
    df_team = _team_rows(["2025-10-22"])
    df_players = pd.DataFrame(
        [_player("0022500001", "2025-10-20", 2025, "A", "current", 20)]
    )

    result = add_roster_continuity_feature(df_team, df_players, injured_dict={})

    assert result[ROSTER_CONTINUITY_COLUMN].isna().all()


def test_feature_becomes_separate_home_and_away_columns_after_merge() -> None:
    df = pd.DataFrame(
        {
            "SEASON_ID": ["22025", "22025"],
            "GAME_ID": ["game", "game"],
            "GAME_DATE": [pd.Timestamp("2026-03-02")] * 2,
            "SEASON_TYPE": ["Regular Season", "Regular Season"],
            "SEASON_YEAR": [2025, 2025],
            "IS_OVERTIME": [0, 0],
            "HOME": [True, False],
            "TEAM_ID": ["A", "B"],
            "TEAM_CITY": ["Home", "Away"],
            "TEAM_ABBREVIATION": ["HOM", "AWY"],
            "TEAM_NAME": ["Home Team", "Away Team"],
            "MATCHUP": ["HOM vs. AWY", "AWY @ HOM"],
            "GAME_NUMBER": [1, 1],
            "OFF_RATING_SEASON_BEFORE_AVG": [100, 100],
            "TOP1_PLAYER_OFF_RATING_BEFORE": [100, 100],
            "TOP1_PLAYER_PTS_BEFORE": [20, 20],
            "PTS_SEASON_BEFORE_AVG": [110, 110],
            "PTS": [100, 90],
            "PF": [20, 18],
            ROSTER_CONTINUITY_COLUMN: [0.75, 0.5],
            IMMEDIATE_ROSTER_CONTINUITY_COLUMN: [0.9, 0.8],
            NEW_PLAYER_MINUTES_COLUMN: [0.2, 0.3],
            IMMEDIATE_NEW_PLAYER_MINUTES_COLUMN: [0.1, 0.15],
            NET_ROSTER_MINUTES_COLUMN: [-0.05, -0.2],
            IMMEDIATE_NET_ROSTER_MINUTES_COLUMN: [0.0, -0.05],
        }
    )

    merged = merge_home_away_data(df)

    assert merged[f"{ROSTER_CONTINUITY_COLUMN}_TEAM_HOME"].iloc[0] == 0.75
    assert merged[f"{ROSTER_CONTINUITY_COLUMN}_TEAM_AWAY"].iloc[0] == 0.5
    assert merged[f"{IMMEDIATE_ROSTER_CONTINUITY_COLUMN}_TEAM_HOME"].iloc[0] == 0.9
    assert merged[f"{IMMEDIATE_ROSTER_CONTINUITY_COLUMN}_TEAM_AWAY"].iloc[0] == 0.8
    assert merged[f"{NEW_PLAYER_MINUTES_COLUMN}_TEAM_HOME"].iloc[0] == 0.2
    assert merged[f"{NEW_PLAYER_MINUTES_COLUMN}_TEAM_AWAY"].iloc[0] == 0.3
    assert merged[f"{IMMEDIATE_NEW_PLAYER_MINUTES_COLUMN}_TEAM_HOME"].iloc[0] == 0.1
    assert merged[f"{IMMEDIATE_NEW_PLAYER_MINUTES_COLUMN}_TEAM_AWAY"].iloc[0] == 0.15
    assert merged[f"{NET_ROSTER_MINUTES_COLUMN}_TEAM_HOME"].iloc[0] == -0.05
    assert merged[f"{NET_ROSTER_MINUTES_COLUMN}_TEAM_AWAY"].iloc[0] == -0.2
    assert merged[f"{IMMEDIATE_NET_ROSTER_MINUTES_COLUMN}_TEAM_HOME"].iloc[0] == 0.0
    assert merged[f"{IMMEDIATE_NET_ROSTER_MINUTES_COLUMN}_TEAM_AWAY"].iloc[0] == -0.05
