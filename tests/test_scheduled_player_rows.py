import pandas as pd

from nba_ou.data_processing.players.players_statistics import (
    precompute_cumulative_avg_stat,
)
from nba_ou.data_processing.scheduled_games.merge_scheduled_with_existing_data import (
    standardize_and_merge_scheduled_games_to_players_data,
)


def test_scheduled_player_rows_keep_historical_season_bucket() -> None:
    df_players = pd.DataFrame(
        [
            {
                "GAME_ID": "0022500900",
                "GAME_DATE": "2026-03-15",
                "SEASON_ID": "22025",
                "SEASON_YEAR": 2025,
                "TEAM_ID": "1610612747",
                "PLAYER_ID": "201939",
                "PLAYER_NAME": "S. Curry",
                "START_POSITION": "G",
                "MIN": 35.0,
                "PTS": 25,
            }
        ]
    )

    scheduled_games = pd.DataFrame(
        [
            {
                "GAME_ID": "0022500999",
                "SEASON": "2025",
                "GAME_DATE_EST": "2026-03-17T00:00:00",
                "HOME_TEAM_ID": "1610612747",
                "VISITOR_TEAM_ID": "1610612744",
            }
        ]
    )

    scheduled_rows = standardize_and_merge_scheduled_games_to_players_data(
        scheduled_games, df_players
    )

    scheduled_row = scheduled_rows.loc[scheduled_rows["PLAYER_ID"] == "201939"].iloc[0]
    assert scheduled_row["SEASON_YEAR"] == 2025
    assert scheduled_rows["SEASON_YEAR"].dtype == df_players["SEASON_YEAR"].dtype
    assert scheduled_row["SEASON_ID"] == "22025"

    combined = pd.concat([df_players, scheduled_rows], ignore_index=True, sort=False)
    with_cum_avg = precompute_cumulative_avg_stat(combined, stat_col="PTS")

    scheduled_with_avg = with_cum_avg.loc[
        (with_cum_avg["GAME_ID"] == "0022500999")
        & (with_cum_avg["PLAYER_ID"] == "201939")
    ].iloc[0]
    assert scheduled_with_avg["PTS_CUM_AVG"] == 25


# --- only players already in tonight's season bucket get a placeholder -------
# Training reads a game's roster from box scores in that game's own season
# bucket (create_player_lookup). A placeholder for anyone else puts a player on
# tonight's roster that training would never have had there.

LAKERS = "1610612747"
WARRIORS = "1610612744"


def _row(game_id, date, season_id, season_year, player, team=LAKERS):
    return {
        "GAME_ID": game_id,
        "GAME_DATE": date,
        "SEASON_ID": season_id,
        "SEASON_YEAR": season_year,
        "TEAM_ID": team,
        "PLAYER_ID": player,
        "PLAYER_NAME": player,
        "START_POSITION": "G",
        "MIN": 20.0,
        "PTS": 10,
    }


def _tonight(game_id="0022500999", date="2026-03-17T00:00:00"):
    return pd.DataFrame(
        [{"GAME_ID": game_id, "SEASON": "2025", "GAME_DATE_EST": date,
          "HOME_TEAM_ID": LAKERS, "VISITOR_TEAM_ID": WARRIORS}]
    )


def test_a_player_last_seen_in_an_earlier_season_gets_no_placeholder() -> None:
    """Cam Reddish's last Lakers game was in 2024-25; on 2026-04-10 he was
    still being placed on their roster."""
    history = pd.DataFrame(
        [
            _row("0022500900", "2026-03-15", "22025", 2025, "current"),
            _row("0022401000", "2025-03-26", "22024", 2024, "gone_last_season"),
        ]
    )
    rows = standardize_and_merge_scheduled_games_to_players_data(_tonight(), history)
    assert set(rows["PLAYER_ID"]) == {"current"}


def test_a_preseason_camp_cut_gets_no_placeholder_for_a_regular_season_game() -> None:
    history = pd.DataFrame(
        [
            _row("0022500900", "2026-03-15", "22025", 2025, "current"),
            _row("0012500010", "2025-10-15", "12025", 2025, "camp_cut"),
        ]
    )
    rows = standardize_and_merge_scheduled_games_to_players_data(_tonight(), history)
    assert set(rows["PLAYER_ID"]) == {"current"}


def test_opening_night_builds_no_placeholders_so_the_lookup_falls_back() -> None:
    """Nobody has a regular-season row yet. Training then falls back to the
    preseason roster; a placeholder here would block that fallback."""
    history = pd.DataFrame(
        [
            _row("0012500010", "2025-10-15", "12025", 2025, "preseason_player"),
            _row("0022401000", "2025-04-13", "22024", 2024, "last_season_player"),
        ]
    )
    rows = standardize_and_merge_scheduled_games_to_players_data(
        _tonight("0022500001", "2025-10-21T00:00:00"), history
    )
    assert rows.empty
