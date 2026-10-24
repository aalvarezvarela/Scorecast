"""Box scores shared by historical availability and live roster tests."""

import pandas as pd

TEAM = "1610612738"


OTHER_TEAM = "1610612744"


SEASON_ID = "22024"


SEASON_YEAR = 2024


TARGET_GAME = "0022400003"


def _team_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GAME_ID": ["0022400001", "0022400002", TARGET_GAME],
            "TEAM_ID": [TEAM] * 3,
            "SEASON_ID": [SEASON_ID] * 3,
            "SEASON_YEAR": [SEASON_YEAR] * 3,
            "GAME_DATE": pd.to_datetime(["2024-10-20", "2024-10-22", "2024-10-24"]),
        }
    )


def _player_row(
    game_id: str,
    game_date: str,
    player_id: str,
    points: float | None,
    minutes: float,
    *,
    team_id: str = TEAM,
    comment: str = "",
) -> dict:
    return {
        "GAME_ID": game_id,
        "TEAM_ID": team_id,
        "SEASON_ID": SEASON_ID,
        "SEASON_YEAR": SEASON_YEAR,
        "GAME_DATE": pd.Timestamp(game_date),
        "PLAYER_ID": player_id,
        "PLAYER_NAME": f"Player {player_id}",
        "MIN": minutes,
        "PTS": points,
        "PACE_PER40": 99.0 if minutes > 0 else None,
        "COMMENT": comment,
    }


def _players(
    *,
    target_star_minutes: float,
    target_star_points: float | None,
    target_star_comment: str | None = None,
):
    rows = []
    for game_id, game_date in [
        ("0022400001", "2024-10-20"),
        ("0022400002", "2024-10-22"),
    ]:
        rows.extend(
            [
                _player_row(game_id, game_date, "star", 30.0, 32.0),
                _player_row(game_id, game_date, "rotation", 15.0, 24.0),
            ]
        )
    rows.extend(
        [
            _player_row(
                TARGET_GAME,
                "2024-10-24",
                "star",
                target_star_points,
                target_star_minutes,
                comment=(
                    target_star_comment
                    if target_star_comment is not None
                    else ("DNP - Coach's Decision" if target_star_minutes == 0 else "")
                ),
            ),
            _player_row(TARGET_GAME, "2024-10-24", "rotation", 15.0, 24.0),
        ]
    )
    return pd.DataFrame(rows)
